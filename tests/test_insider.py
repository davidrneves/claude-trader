"""Tests for the insider/SEC signal feed.

All HTTP is mocked. The fixtures embed the smallest realistic openinsider
table, SEC submissions JSON, and FTD CSV needed to exercise each parser.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests.exceptions

from claude_trader.insider import InsiderFeed


# -------- HTML / JSON fixtures --------


def _cluster_html_with_aapl(today_iso: str) -> str:
    """openinsider /latest-cluster-buys-style HTML with one AAPL cluster row."""
    return f"""<html><body>
<table width="100%" cellpadding="0" cellspacing="0" border="0" class="tinytable">
<tr><th>X</th><th>Filing&nbsp;Date</th><th>Trade&nbsp;Date</th><th>Ticker</th>
<th>Company&nbsp;Name</th><th>Industry</th><th>Ins</th><th>Trade&nbsp;Type</th>
<th>Price</th><th>Qty</th><th>Owned</th><th>&Delta;Own</th><th>Value</th></tr>
<tr><td></td><td>{today_iso} 16:00:00</td><td>{today_iso}</td>
<td><a onmouseover="ToolTip(p,'X', DELAY, 1)" onmouseout="UnTip()">AAPL</a></td>
<td>Apple Inc</td><td>Tech</td><td>5</td><td>P - Purchase</td>
<td>$170.00</td><td>+10,000</td><td>50,000</td><td>+25%</td><td>+$1,700,000</td></tr>
<tr><td></td><td>{today_iso} 15:00:00</td><td>{today_iso}</td>
<td><a>NVDA</a></td><td>Nvidia</td><td>Tech</td><td>2</td><td>P - Purchase</td>
<td>$800.00</td><td>+1,000</td><td>5,000</td><td>+25%</td><td>+$800,000</td></tr>
</table></body></html>"""


def _officer_html_with_msft(today_iso: str) -> str:
    return f"""<html><body>
<table class="tinytable">
<tr><th>X</th><th>Filing&nbsp;Date</th><th>Trade&nbsp;Date</th><th>Ticker</th>
<th>Company</th><th>Insider&nbsp;Name</th><th>Title</th><th>Trade&nbsp;Type</th>
<th>Price</th><th>Qty</th><th>Owned</th><th>&Delta;Own</th><th>Value</th>
<th>1d</th><th>1w</th><th>1m</th><th>6m</th></tr>
<tr><td></td><td>{today_iso}</td><td>{today_iso}</td>
<td><a>MSFT</a></td><td>Microsoft</td><td>Nadella Satya</td>
<td>CEO</td><td>P - Purchase</td><td>$400.00</td><td>+500</td>
<td>200,000</td><td>+0%</td><td>+$200,000</td>
<td></td><td></td><td></td><td></td></tr>
<tr><td></td><td>{today_iso}</td><td>{today_iso}</td>
<td><a>NOISE</a></td><td>Noise Inc</td><td>Random Person</td>
<td>Other</td><td>P - Purchase</td><td>$10.00</td><td>+100</td>
<td>1,000</td><td>+10%</td><td>+$1,000</td>
<td></td><td></td><td></td><td></td></tr>
</table></body></html>"""


def _sec_submissions(forms: list[str], dates: list[str]) -> dict:
    return {
        "filings": {
            "recent": {
                "form": forms,
                "filingDate": dates,
                "accessionNumber": [f"0000000000-00-{i:06d}" for i in range(len(forms))],
            }
        }
    }


def _ticker_map(mapping: dict[str, int]) -> dict:
    return {
        str(i): {"ticker": tkr, "cik_str": cik, "title": tkr}
        for i, (tkr, cik) in enumerate(mapping.items())
    }


def _ftd_zip_bytes(rows: list[tuple[str, str, int, float]]) -> bytes:
    """Build an in-memory FTD zip matching the SEC pipe-delimited format."""
    lines = ["SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE"]
    for date_str, sym, qty, price in rows:
        lines.append(f"{date_str}|999999999|{sym}|{qty}|{sym} TEST|{price}")
    body = "\n".join(lines).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("cnsfails202604a", body)
    return buf.getvalue()


# -------- Fixtures --------


@pytest.fixture
def feed(tmp_path: Path) -> InsiderFeed:
    return InsiderFeed(
        user_agent="claude-trader-tests/0.0 (ci@anthropic.test)",
        cache_dir=tmp_path / "cache",
    )


def _today() -> str:
    return date.today().isoformat()


# -------- User-Agent validation --------


class TestUserAgentValidation:
    def test_empty_ua_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="user_agent"):
            InsiderFeed(user_agent="", cache_dir=tmp_path)

    def test_whitespace_ua_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="user_agent"):
            InsiderFeed(user_agent="   ", cache_dir=tmp_path)

    def test_placeholder_email_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="placeholder"):
            InsiderFeed(
                user_agent="claude-trader/0.7.5 (ops@example.com)",
                cache_dir=tmp_path,
            )

    def test_real_contact_accepted(self, tmp_path: Path) -> None:
        # Should not raise
        InsiderFeed(
            user_agent="claude-trader/1.0 (jane.doe@my-company.com)",
            cache_dir=tmp_path,
        )


def _stub_response(status: int = 200, *, text: str = "", body: bytes = b"", json_data=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.text = text
    resp.content = body
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


# -------- Cluster buys --------


class TestClusterBuys:
    def test_returns_recent_aapl_cluster(self, feed: InsiderFeed) -> None:
        with patch.object(
            feed,
            "_get",
            return_value=_stub_response(text=_cluster_html_with_aapl(_today())),
        ):
            result = feed.get_cluster_buys("AAPL")
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["insider_count"] == 5
        assert result[0]["kind"] == "cluster_buy"

    def test_filters_to_requested_ticker(self, feed: InsiderFeed) -> None:
        with patch.object(
            feed,
            "_get",
            return_value=_stub_response(text=_cluster_html_with_aapl(_today())),
        ):
            assert feed.get_cluster_buys("MSFT") == []

    def test_lookback_window_excludes_old_events(self, feed: InsiderFeed) -> None:
        old = (date.today() - timedelta(days=30)).isoformat()
        with patch.object(
            feed,
            "_get",
            return_value=_stub_response(text=_cluster_html_with_aapl(old)),
        ):
            assert feed.get_cluster_buys("AAPL", lookback_days=14) == []

    def test_returns_none_on_fetch_failure(self, feed: InsiderFeed) -> None:
        with patch.object(
            feed,
            "_get",
            side_effect=requests.exceptions.ConnectionError("network down"),
        ):
            assert feed.get_cluster_buys("AAPL") is None

    def test_returns_none_on_layout_change(self, feed: InsiderFeed) -> None:
        """If the page no longer contains class=tinytable, treat as failure -
        layout change should not silently look like 'no events'."""
        with patch.object(
            feed,
            "_get",
            return_value=_stub_response(text="<html>no table here</html>"),
        ):
            assert feed.get_cluster_buys("AAPL") is None

    def test_uses_cache_on_second_call(self, feed: InsiderFeed) -> None:
        stub = _stub_response(text=_cluster_html_with_aapl(_today()))
        with patch.object(feed, "_get", return_value=stub) as mock_get:
            feed.get_cluster_buys("AAPL")
            feed.get_cluster_buys("AAPL")
            assert mock_get.call_count == 1


# -------- Officer buys --------


class TestOfficerBuys:
    def test_keeps_meaningful_titles(self, feed: InsiderFeed) -> None:
        with patch.object(
            feed,
            "_get",
            return_value=_stub_response(text=_officer_html_with_msft(_today())),
        ):
            result = feed.get_officer_buys("MSFT")
        assert result is not None
        assert len(result) == 1
        assert result[0]["title"] == "CEO"
        assert result[0]["kind"] == "officer_buy"

    def test_filters_unmeaningful_titles(self, feed: InsiderFeed) -> None:
        """Non-officer titles (e.g., 'Other') must not appear."""
        with patch.object(
            feed,
            "_get",
            return_value=_stub_response(text=_officer_html_with_msft(_today())),
        ):
            assert feed.get_officer_buys("NOISE") == []

    def test_returns_none_on_failure(self, feed: InsiderFeed) -> None:
        with patch.object(
            feed,
            "_get",
            side_effect=requests.exceptions.Timeout("slow"),
        ):
            assert feed.get_officer_buys("MSFT") is None


# -------- SEC filings: dilution and late --------


class TestDilutionFilings:
    def _patch_sec(self, feed: InsiderFeed, forms: list[str], dates: list[str]):
        ticker_resp = _stub_response(
            json_data=_ticker_map({"AAPL": 320193}), text="ignored"
        )
        sub_resp = _stub_response(json_data=_sec_submissions(forms, dates))
        return patch.object(feed, "_get", side_effect=[ticker_resp, sub_resp])

    def test_finds_recent_424b5(self, feed: InsiderFeed) -> None:
        today = date.today().isoformat()
        with self._patch_sec(feed, ["8-K", "424B5"], [today, today]):
            result = feed.get_dilution_filings("AAPL")
        assert result is not None
        assert len(result) == 1
        assert result[0]["form"] == "424B5"

    def test_excludes_old_filings(self, feed: InsiderFeed) -> None:
        old = (date.today() - timedelta(days=120)).isoformat()
        with self._patch_sec(feed, ["S-3"], [old]):
            assert feed.get_dilution_filings("AAPL", lookback_days=30) == []

    def test_returns_none_on_submissions_failure(self, feed: InsiderFeed) -> None:
        ticker_resp = _stub_response(
            json_data=_ticker_map({"AAPL": 320193})
        )
        with patch.object(
            feed,
            "_get",
            side_effect=[
                ticker_resp,
                requests.exceptions.ConnectionError("down"),
            ],
        ):
            assert feed.get_dilution_filings("AAPL") is None

    def test_returns_empty_when_ticker_unknown(self, feed: InsiderFeed) -> None:
        """Unmapped ticker must NOT return None - we know there are no filings."""
        ticker_resp = _stub_response(json_data=_ticker_map({"AAPL": 320193}))
        with patch.object(feed, "_get", return_value=ticker_resp):
            assert feed.get_dilution_filings("UNKNOWNTICKER") == []


class TestLateFilings:
    def test_finds_nt_10k(self, feed: InsiderFeed) -> None:
        today = date.today().isoformat()
        ticker_resp = _stub_response(json_data=_ticker_map({"XYZ": 999999}))
        sub_resp = _stub_response(json_data=_sec_submissions(["NT 10-K"], [today]))
        with patch.object(feed, "_get", side_effect=[ticker_resp, sub_resp]):
            result = feed.get_late_filings("XYZ")
        assert result is not None
        assert len(result) == 1
        assert result[0]["form"] == "NT 10-K"
        assert result[0]["kind"] == "late_filing"


# -------- Failures to deliver --------


class TestFailuresToDeliver:
    def test_filters_zip_to_ticker(self, feed: InsiderFeed) -> None:
        page_resp = _stub_response(
            text='<a href="/files/data/fails-deliver-data/cnsfails202604a.zip">x</a>'
        )
        zip_resp = _stub_response(
            body=_ftd_zip_bytes(
                [
                    ("20260415", "AAPL", 5000, 200.0),
                    ("20260415", "MSFT", 1000, 300.0),
                    ("20260416", "AAPL", 7500, 201.0),
                ]
            )
        )
        with patch.object(feed, "_get", side_effect=[page_resp, zip_resp]):
            result = feed.get_failures_to_deliver("AAPL")
        assert result is not None
        assert len(result) == 2
        assert all(r["qty"] >= 5000 for r in result)

    def test_returns_none_when_index_unavailable(self, feed: InsiderFeed) -> None:
        with patch.object(
            feed,
            "_get",
            side_effect=requests.exceptions.ConnectionError("down"),
        ):
            assert feed.get_failures_to_deliver("AAPL") is None

    def test_returns_none_when_no_files_listed(self, feed: InsiderFeed) -> None:
        page_resp = _stub_response(text="<html>no links here</html>")
        with patch.object(feed, "_get", return_value=page_resp):
            assert feed.get_failures_to_deliver("AAPL") is None


# -------- Composite --------


class TestFullSignals:
    def test_partial_failure_marks_unknown_not_bullish(
        self, feed: InsiderFeed
    ) -> None:
        """When some fetchers fail, the failed ones must surface as None
        rather than being silently treated as 'no events found'."""
        # Make every fetch fail
        with patch.object(
            feed,
            "_get",
            side_effect=requests.exceptions.ConnectionError("down"),
        ):
            result = feed.get_full_signals("AAPL")
        assert all(v is None for v in result.values())

    def test_caching_persists_to_disk(self, tmp_path: Path) -> None:
        feed = InsiderFeed(user_agent="ct/0.0 (ci@anthropic.test)", cache_dir=tmp_path / "c")
        with patch.object(
            feed,
            "_get",
            return_value=_stub_response(text=_cluster_html_with_aapl(_today())),
        ):
            feed.get_cluster_buys("AAPL")
        # New feed instance, same cache dir - should hit the disk cache
        feed2 = InsiderFeed(user_agent="ct/0.0 (ci@anthropic.test)", cache_dir=tmp_path / "c")
        with patch.object(feed2, "_get") as never_called:
            feed2.get_cluster_buys("AAPL")
            assert never_called.call_count == 0


# -------- HTTP retry behavior --------


class TestHTTPRetry:
    def test_retries_on_transient_error(self, feed: InsiderFeed) -> None:
        good = _stub_response(text=_cluster_html_with_aapl(_today()))
        with patch.object(
            feed._session,
            "get",
            side_effect=[
                requests.exceptions.ConnectionError("flap 1"),
                good,
            ],
        ) as mock_get:
            result = feed.get_cluster_buys("AAPL")
            # Tenacity retried, so we got a non-None result and 2 calls.
            assert result is not None
            assert mock_get.call_count == 2

    def test_gives_up_after_three_attempts(self, feed: InsiderFeed) -> None:
        with patch.object(
            feed._session,
            "get",
            side_effect=requests.exceptions.ConnectionError("permanent"),
        ) as mock_get:
            assert feed.get_cluster_buys("AAPL") is None
            assert mock_get.call_count == 3


# -------- json import is used in cache layer --------


def test_cache_round_trip_through_disk(tmp_path: Path) -> None:
    feed = InsiderFeed(user_agent="ct/0.0 (ci@anthropic.test)", cache_dir=tmp_path)
    today = _today()
    with patch.object(
        feed,
        "_get",
        return_value=_stub_response(text=_cluster_html_with_aapl(today)),
    ):
        feed.get_cluster_buys("AAPL")
    cache_files = list(tmp_path.glob("*.json"))
    assert any("cluster_buys_index" in p.name for p in cache_files)
    raw = json.loads(cache_files[0].read_text())
    assert isinstance(raw, list)
