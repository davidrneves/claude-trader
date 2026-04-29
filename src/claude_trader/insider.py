"""Insider and SEC-filing data feeds for the analyst pipeline.

Five signals, all returning ``None`` on fetch failure (sentinel pattern: the
caller distinguishes "fetched, none found" from "fetch failed" so a bad
request is never silently treated as bullish or bearish):

1. cluster_buys   - openinsider.com/latest-cluster-buys
2. officer_buys   - openinsider.com/officer-purchases-25k
3. dilution_filings   - SEC submissions JSON (S-1, S-3, 424B5, F-1, F-3)
4. late_filings   - SEC submissions JSON (NT 10-K, NT 10-Q)
5. failures_to_deliver - SEC bi-monthly fails-to-deliver pipe-delimited CSVs

Networks limits:
- openinsider.com only serves HTTPS via a non-standard cert chain in some
  environments; HTTP is the documented fallback. We use HTTP.
- SEC requires a User-Agent identifying the operator. Provided by config.

File cache: per-tick launchd spawn means in-memory caches don't survive,
so all results are persisted at ``.state/insider_cache/...`` with TTLs.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
import requests.exceptions
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

log = structlog.get_logger()

_TRANSIENT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionError,
    TimeoutError,
)

_OPENINSIDER_BASE = "http://openinsider.com"
_SEC_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_FTD_PAGE = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
_SEC_FTD_DOWNLOAD_BASE = "https://www.sec.gov"

_DEFAULT_CACHE_DIR = Path(".state/insider_cache")

# Forms that signal share dilution risk for existing holders.
_DILUTION_FORMS = frozenset(
    {"S-1", "S-1/A", "S-3", "S-3/A", "424B3", "424B5", "F-1", "F-3"}
)

# NT 10-K / NT 10-Q indicate an issuer cannot file financials on time -
# strong accounting-risk signal. 8-K Item 4.02 (non-reliance) requires
# parsing the filing body, so it is left for a follow-up.
_LATE_FILING_FORMS = frozenset({"NT 10-K", "NT 10-Q", "NT 20-F", "NT 10-K/A"})


def _api_retry(func):
    """Tenacity wrapper matching executor._api_retry semantics.

    Mirrored here rather than imported to avoid a cross-module dependency
    on a private symbol; the policy stays in lockstep with executor.py.
    """
    return retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8) + wait_random(0, 1),
        before_sleep=lambda rs: log.warning(
            "insider_api_retry",
            attempt=rs.attempt_number,
            error=str(rs.outcome.exception()),
        ),
        reraise=True,
    )(func)


# -------- file cache --------


def _cache_path(cache_dir: Path, key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return cache_dir / f"{safe}.json"


def _read_cache(cache_dir: Path, key: str, ttl_seconds: int) -> Any | None:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl_seconds:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("insider_cache_read_failed", key=key, error=str(e))
        return None


def _write_cache(cache_dir: Path, key: str, data: Any) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, key)
    try:
        path.write_text(json.dumps(data, default=str))
    except OSError as e:
        log.warning("insider_cache_write_failed", key=key, error=str(e))


# -------- HTML row extractor (openinsider tables) --------


# Strip <td>...</td> attribute noise. Captures the first cell content found.
_CELL_RE = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _parse_tinytable(html: str) -> list[list[str]]:
    """Extract rows from openinsider's ``<table class="tinytable">``.

    Returns ``[]`` if the table is missing - openinsider serves a soft 200
    with an empty table when there are no matches, which we want callers to
    distinguish from a fetch error (handled at the get_* level).
    """
    start = html.find('class="tinytable"')
    if start < 0:
        return []
    end = html.find("</table>", start)
    if end < 0:
        return []
    table = html[start:end]
    rows = re.split(r"<tr[^>]*>", table)[1:]  # drop pre-row preamble
    parsed = []
    for raw in rows:
        cells = _CELL_RE.findall(raw)
        cleaned = [
            _TAG_RE.sub("", c).replace("&nbsp;", " ").strip() for c in cells
        ]
        parsed.append(cleaned)
    return parsed


def _extract_ticker(cell: str) -> str:
    """openinsider wraps the ticker in JS onmouseover/UnTip noise. The plain
    ticker is whatever trails after the last ``>``."""
    if ">" in cell:
        return cell.rsplit(">", 1)[-1].strip()
    return cell.strip()


def _parse_int(text: str) -> int | None:
    """Parse '+2,000' / '-1,500' / '13,500' style numbers; return None on miss."""
    if not text:
        return None
    cleaned = text.replace(",", "").replace("+", "").replace("$", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        return None


# -------- InsiderFeed --------


class InsiderFeed:
    """Fetches insider-trading and SEC-filing signals for a ticker.

    Every public method returns ``None`` on fetch failure (transient retries
    exhausted, network blocked, parse error). An empty list / empty result
    means "fetched successfully, no events". Callers MUST treat ``None``
    differently to avoid mistaking a failed lookup for an absent signal.
    """

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | None = None,
        timeout: float = 10.0,
    ) -> None:
        _validate_user_agent(user_agent)
        self._user_agent = user_agent
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    # ---- Low-level HTTP ----

    @_api_retry
    def _get(self, url: str) -> requests.Response:
        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()
        return resp

    # ---- 1. Cluster buys ----

    def get_cluster_buys(
        self, ticker: str, lookback_days: int = 14
    ) -> list[dict] | None:
        """Recent cluster buys (multiple insiders, same ticker, same window).

        Returns a list of cluster events for ``ticker`` in the last
        ``lookback_days``, or ``None`` on fetch failure.
        """
        cache_key = "cluster_buys_index"
        events = _read_cache(self._cache_dir, cache_key, ttl_seconds=6 * 3600)
        if events is None:
            try:
                resp = self._get(f"{_OPENINSIDER_BASE}/latest-cluster-buys")
            except _TRANSIENT_ERRORS as e:
                log.warning("cluster_buys_fetch_failed", error=str(e))
                return None
            except requests.exceptions.RequestException as e:
                log.warning("cluster_buys_fetch_failed", error=str(e))
                return None
            events = self._parse_openinsider_rows(resp.text, has_industry=True)
            if events is None:
                return None
            _write_cache(self._cache_dir, cache_key, events)
        return _filter_recent(events, ticker.upper(), lookback_days)

    # ---- 2. Officer buys ----

    def get_officer_buys(
        self, ticker: str, lookback_days: int = 14
    ) -> list[dict] | None:
        """Recent CEO/CFO/Director purchases >= $25k for ``ticker``.

        Only buys with title in {CEO, CFO, COO, Pres, Dir, 10%} are kept,
        excluding routine 10b5-1 sales. Returns ``None`` on fetch failure.
        """
        cache_key = "officer_buys_index"
        events = _read_cache(self._cache_dir, cache_key, ttl_seconds=6 * 3600)
        if events is None:
            try:
                resp = self._get(f"{_OPENINSIDER_BASE}/officer-purchases-25k")
            except _TRANSIENT_ERRORS as e:
                log.warning("officer_buys_fetch_failed", error=str(e))
                return None
            except requests.exceptions.RequestException as e:
                log.warning("officer_buys_fetch_failed", error=str(e))
                return None
            events = self._parse_openinsider_rows(resp.text, has_industry=False)
            if events is None:
                return None
            _write_cache(self._cache_dir, cache_key, events)
        recent = _filter_recent(events, ticker.upper(), lookback_days)
        return recent

    @staticmethod
    def _parse_openinsider_rows(
        html: str, has_industry: bool
    ) -> list[dict] | None:
        """Parse openinsider ``tinytable`` rows into a list of normalized dicts.

        Returns ``None`` on parse error (broken HTML, schema change). The
        layout differs slightly between cluster-buys (has Industry + Ins
        count) and officer-purchases (has Insider Name + Title), captured
        by ``has_industry``.
        """
        rows = _parse_tinytable(html)
        if not rows:
            # Empty table is a valid result (no events); but a missing table
            # almost always signals a layout change - treat as failure.
            if 'class="tinytable"' not in html:
                return None
            return []
        # Drop the header row.
        data_rows = rows[1:] if rows else []
        events: list[dict] = []
        for r in data_rows:
            if has_industry:
                # 0:X, 1:Filing Date, 2:Trade Date, 3:Ticker, 4:Company,
                # 5:Industry, 6:Ins (count), 7:Trade Type, 8:Price,
                # 9:Qty, 10:Owned, 11:dOwn, 12:Value
                if len(r) < 13:
                    continue
                ticker = _extract_ticker(r[3])
                trade_date = r[2]
                trade_type = r[7]
                if "P - Purchase" not in trade_type:
                    continue
                events.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "filing_date": r[1],
                        "insider_count": _parse_int(r[6]) or 0,
                        "qty": _parse_int(r[9]),
                        "value": _parse_int(r[12]),
                        "kind": "cluster_buy",
                    }
                )
            else:
                # 0:X, 1:Filing Date, 2:Trade Date, 3:Ticker, 4:Company,
                # 5:Insider Name, 6:Title, 7:Trade Type, 8:Price, 9:Qty,
                # 10:Owned, 11:dOwn, 12:Value, ...
                if len(r) < 13:
                    continue
                ticker = _extract_ticker(r[3])
                trade_date = r[2]
                trade_type = r[7]
                title = r[6]
                if "P - Purchase" not in trade_type:
                    continue
                if not _is_meaningful_title(title):
                    continue
                events.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "filing_date": r[1],
                        "insider_name": r[5],
                        "title": title,
                        "qty": _parse_int(r[9]),
                        "value": _parse_int(r[12]),
                        "kind": "officer_buy",
                    }
                )
        return events

    # ---- 3 & 4. SEC filings (dilution + late filings share submissions JSON) ----

    def _load_cik_map(self) -> dict[str, str] | None:
        """Load the SEC ticker->CIK map, returning ``None`` only when the
        fetch or parse fails. A successfully loaded map missing a ticker is
        not the same as a failed lookup."""
        cache_key = "cik_map"
        mapping = _read_cache(self._cache_dir, cache_key, ttl_seconds=24 * 3600)
        if mapping is not None:
            return mapping
        try:
            resp = self._get(_SEC_TICKERS_URL)
        except (
            requests.exceptions.RequestException,
            *_TRANSIENT_ERRORS,
        ) as e:
            log.warning("cik_map_fetch_failed", error=str(e))
            return None
        try:
            raw = resp.json()
        except json.JSONDecodeError as e:
            log.warning("cik_map_parse_failed", error=str(e))
            return None
        mapping = {
            str(v["ticker"]).upper(): f"{int(v['cik_str']):010d}"
            for v in raw.values()
        }
        _write_cache(self._cache_dir, cache_key, mapping)
        return mapping

    def _get_recent_filings(
        self, ticker: str, ttl_seconds: int = 24 * 3600
    ) -> list[dict] | None:
        cache_key = f"submissions_{ticker.upper()}"
        cached = _read_cache(self._cache_dir, cache_key, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached
        mapping = self._load_cik_map()
        if mapping is None:
            # CIK map fetch failed -> we genuinely don't know the ticker's
            # filing state. Surface as None, NOT as "no filings found".
            return None
        cik = mapping.get(ticker.upper())
        if not cik:
            log.info("ticker_not_in_cik_map", ticker=ticker)
            return []
        try:
            resp = self._get(f"{_SEC_SUBMISSIONS_BASE}/CIK{cik}.json")
        except (requests.exceptions.RequestException, *_TRANSIENT_ERRORS) as e:
            log.warning("submissions_fetch_failed", ticker=ticker, error=str(e))
            return None
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            log.warning("submissions_parse_failed", ticker=ticker, error=str(e))
            return None
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        filings = [
            {
                "form": forms[i] if i < len(forms) else "",
                "filing_date": dates[i] if i < len(dates) else "",
                "accession": accessions[i] if i < len(accessions) else "",
            }
            for i in range(min(len(forms), len(dates), len(accessions)))
        ]
        _write_cache(self._cache_dir, cache_key, filings)
        return filings

    def get_dilution_filings(
        self, ticker: str, lookback_days: int = 30
    ) -> list[dict] | None:
        """Recent dilution-risk filings (S-1, S-3, 424B5, F-1, F-3) for ticker.

        Returns ``None`` on fetch failure, ``[]`` if no qualifying filings.
        """
        filings = self._get_recent_filings(ticker)
        if filings is None:
            return None
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)
        out = []
        for f in filings:
            if f["form"] not in _DILUTION_FORMS:
                continue
            try:
                fd = date.fromisoformat(f["filing_date"])
            except ValueError:
                continue
            if fd < cutoff:
                continue
            out.append({**f, "kind": "dilution_filing"})
        return out

    def get_late_filings(
        self, ticker: str, lookback_days: int = 14
    ) -> list[dict] | None:
        """Recent NT 10-K / NT 10-Q late-filing notices for ticker.

        These signal an issuer cannot file financials on time - a strong
        accounting-risk signal. Returns ``None`` on fetch failure.
        """
        filings = self._get_recent_filings(ticker)
        if filings is None:
            return None
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)
        out = []
        for f in filings:
            if f["form"] not in _LATE_FILING_FORMS:
                continue
            try:
                fd = date.fromisoformat(f["filing_date"])
            except ValueError:
                continue
            if fd < cutoff:
                continue
            out.append({**f, "kind": "late_filing"})
        return out

    # ---- 5. Failures to Deliver ----

    def _get_latest_ftd_filename(self) -> str | None:
        """Scrape the SEC FTD page for the most recent ``cnsfailsYYYYMM[a|b]``.

        The full file list is parsed from a single page; we keep only the
        most recent two halves (current + previous) so the dataset stays
        bounded.
        """
        cache_key = "ftd_filename_index"
        filenames = _read_cache(self._cache_dir, cache_key, ttl_seconds=24 * 3600)
        if filenames is None:
            try:
                resp = self._get(_SEC_FTD_PAGE)
            except (requests.exceptions.RequestException, *_TRANSIENT_ERRORS) as e:
                log.warning("ftd_index_fetch_failed", error=str(e))
                return None
            filenames = re.findall(
                r'href="(/files/data/fails-deliver-data/cnsfails\d{6}[ab]\.zip)"',
                resp.text,
            )
            if not filenames:
                log.warning("ftd_index_no_files_found")
                return None
            _write_cache(self._cache_dir, cache_key, filenames)
        return filenames[0] if filenames else None

    def get_failures_to_deliver(
        self,
        ticker: str,
        min_aggregate_qty: int = 50_000,
    ) -> list[dict] | None:
        """Aggregate FTD records for ``ticker`` in the most recent published period.

        ``min_aggregate_qty`` is the threshold over which the result is
        flagged as elevated. This returns the raw per-day records; the
        scoring layer applies the threshold.
        """
        href = self._get_latest_ftd_filename()
        if href is None:
            return None
        filename = href.rsplit("/", 1)[-1]  # cnsfailsYYYYMMa.zip
        period = filename.replace("cnsfails", "").replace(".zip", "")
        cache_key = f"ftd_{ticker.upper()}_{period}"
        cached = _read_cache(self._cache_dir, cache_key, ttl_seconds=14 * 24 * 3600)
        if cached is not None:
            return cached

        zip_cache = self._cache_dir / f"cnsfails_{period}.zip"
        if not zip_cache.exists():
            try:
                resp = self._get(f"{_SEC_FTD_DOWNLOAD_BASE}{href}")
            except (requests.exceptions.RequestException, *_TRANSIENT_ERRORS) as e:
                log.warning("ftd_zip_fetch_failed", error=str(e))
                return None
            zip_cache.parent.mkdir(parents=True, exist_ok=True)
            zip_cache.write_bytes(resp.content)

        try:
            records = _filter_ftd_zip(zip_cache, ticker.upper())
        except (zipfile.BadZipFile, OSError, csv.Error) as e:
            log.warning("ftd_parse_failed", ticker=ticker, error=str(e))
            return None
        _write_cache(self._cache_dir, cache_key, records)
        return records

    # ---- Composite ----

    def get_full_signals(self, ticker: str) -> dict:
        """Run all 5 fetchers and return a single dict with per-signal results.

        Each value is either the fetched list or ``None`` (failure). The
        scoring layer treats ``None`` as 'unknown' - never as bullish or
        bearish - and adjusts confidence accordingly.
        """
        return {
            "cluster_buys": self.get_cluster_buys(ticker),
            "officer_buys": self.get_officer_buys(ticker),
            "dilution_filings": self.get_dilution_filings(ticker),
            "late_filings": self.get_late_filings(ticker),
            "failures_to_deliver": self.get_failures_to_deliver(ticker),
        }


# -------- helpers --------


# SEC EDGAR rejects requests with no contact email or with placeholder-y
# User-Agents (rate-limit / block). The feature is opt-in, so we fail loudly
# at construction rather than at the first fetch.
_PLACEHOLDER_UA_HINTS = ("example.com", "test@", "your-email", "todo")


def _validate_user_agent(ua: str) -> None:
    if not ua or not ua.strip():
        raise ValueError(
            "InsiderFeed requires a non-empty user_agent identifying a real "
            "contact email (SEC EDGAR access policy)."
        )
    lowered = ua.lower()
    for hint in _PLACEHOLDER_UA_HINTS:
        if hint in lowered:
            raise ValueError(
                f"InsiderFeed user_agent appears to be a placeholder "
                f"(contains {hint!r}). Set a real contact email."
            )


_MEANINGFUL_TITLES = ("CEO", "CFO", "COO", "Pres", "Dir", "10%", "Chair")


def _is_meaningful_title(title: str) -> bool:
    if not title:
        return False
    return any(t in title for t in _MEANINGFUL_TITLES)


def _filter_recent(
    events: list[dict], ticker: str, lookback_days: int
) -> list[dict]:
    """Filter to ``ticker`` and trade_date within ``lookback_days`` of today."""
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)
    out = []
    for e in events:
        if e.get("ticker", "").upper() != ticker:
            continue
        td = e.get("trade_date", "")
        # Trade date may be 'YYYY-MM-DD' or empty in some rare rows.
        try:
            d = date.fromisoformat(td[:10])
        except ValueError:
            continue
        if d < cutoff:
            continue
        out.append(e)
    return out


def _filter_ftd_zip(zip_path: Path, ticker: str) -> list[dict]:
    """Open the cached FTD zip and return all rows for ``ticker``.

    Pipe-delimited columns: SETTLEMENT DATE | CUSIP | SYMBOL | QTY (FAILS) |
    DESCRIPTION | PRICE.
    """
    out: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if not members:
            return out
        with zf.open(members[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text, delimiter="|")
            for row in reader:
                if (row.get("SYMBOL") or "").strip().upper() != ticker:
                    continue
                qty_raw = (row.get("QUANTITY (FAILS)") or "").strip()
                price_raw = (row.get("PRICE") or "").strip()
                try:
                    qty = int(qty_raw)
                except ValueError:
                    continue
                try:
                    price = float(price_raw) if price_raw else None
                except ValueError:
                    price = None
                out.append(
                    {
                        "settlement_date": (
                            row.get("SETTLEMENT DATE") or ""
                        ).strip(),
                        "qty": qty,
                        "price": price,
                    }
                )
    return out
