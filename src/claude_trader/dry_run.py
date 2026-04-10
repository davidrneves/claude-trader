"""Dry-run validation - check connectivity and run analysis without trading.

Usage: uv run python -m claude_trader --dry-run
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import structlog
from alpaca.data.timeframe import TimeFrame

from claude_trader.analyst import Analyst, GeminiError
from claude_trader.config import Settings
from claude_trader.executor import AlpacaExecutor
from claude_trader.news import NewsFeed
from claude_trader.risk import RiskConfig, RiskManager
from claude_trader.strategy import EMAMomentumStrategy

log = structlog.get_logger()


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    elapsed_ms: float


@dataclass
class DryRunReport:
    validations: list[ValidationResult] = field(default_factory=list)
    symbols_analyzed: list[dict] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(v.passed for v in self.validations)


def _check_alpaca(settings: Settings) -> ValidationResult:
    """Validate Alpaca API connectivity by fetching account info."""
    start = time.monotonic()
    try:
        risk_config = RiskConfig.from_settings(settings)
        risk_mgr = RiskManager(risk_config, portfolio_value=Decimal("0"))
        executor = AlpacaExecutor(settings, risk_mgr)
        account = executor.get_account()
        elapsed = (time.monotonic() - start) * 1000
        return ValidationResult(
            name="Alpaca API",
            passed=True,
            message=f"Connected. Equity: ${account['equity']}",
            elapsed_ms=round(elapsed, 1),
        )
    except (ConnectionError, ValueError, OSError, KeyError, RuntimeError) as e:
        elapsed = (time.monotonic() - start) * 1000
        return ValidationResult(
            name="Alpaca API",
            passed=False,
            message=f"Failed: {e}",
            elapsed_ms=round(elapsed, 1),
        )


def _check_market_data(settings: Settings) -> ValidationResult:
    """Validate market data access by fetching bars for the first watchlist symbol."""
    start = time.monotonic()
    try:
        risk_config = RiskConfig.from_settings(settings)
        risk_mgr = RiskManager(risk_config, portfolio_value=Decimal("0"))
        executor = AlpacaExecutor(settings, risk_mgr)
        symbol = settings.watchlist[0]
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        bars = executor.get_bars(symbol, TimeFrame.Day, start=start_date)
        elapsed = (time.monotonic() - start) * 1000
        has_data = hasattr(bars, "df") and not bars.df.empty
        return ValidationResult(
            name="Market Data",
            passed=has_data,
            message=f"Fetched bars for {symbol}"
            if has_data
            else f"No data for {symbol}",
            elapsed_ms=round(elapsed, 1),
        )
    except (ConnectionError, ValueError, OSError, KeyError, RuntimeError) as e:
        elapsed = (time.monotonic() - start) * 1000
        return ValidationResult(
            name="Market Data",
            passed=False,
            message=f"Failed: {e}",
            elapsed_ms=round(elapsed, 1),
        )


def _check_news_feed(settings: Settings) -> ValidationResult:
    """Validate news feed access."""
    start = time.monotonic()
    try:
        news = NewsFeed(
            api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key
        )
        symbol = settings.watchlist[0]
        headlines = news.get_headlines(symbol, limit=5)
        elapsed = (time.monotonic() - start) * 1000
        return ValidationResult(
            name="News Feed",
            passed=True,
            message=f"Fetched {len(headlines)} headlines for {symbol}",
            elapsed_ms=round(elapsed, 1),
        )
    except (ConnectionError, ValueError, OSError, KeyError, RuntimeError) as e:
        elapsed = (time.monotonic() - start) * 1000
        return ValidationResult(
            name="News Feed",
            passed=False,
            message=f"Failed: {e}",
            elapsed_ms=round(elapsed, 1),
        )


def _check_gemini(settings: Settings) -> ValidationResult:
    """Validate Gemini API connectivity if key is configured."""
    if not settings.gemini_api_key:
        return ValidationResult(
            name="Gemini API",
            passed=True,
            message="Skipped (no API key configured)",
            elapsed_ms=0.0,
        )
    start = time.monotonic()
    try:
        analyst = Analyst(api_key=settings.gemini_api_key)
        analyst.analyze_sentiment("TEST", ["Market rallies on strong earnings"])
        elapsed = (time.monotonic() - start) * 1000
        return ValidationResult(
            name="Gemini API",
            passed=True,
            message="Connected and responding",
            elapsed_ms=round(elapsed, 1),
        )
    except (GeminiError, ConnectionError, ValueError, OSError, RuntimeError) as e:
        elapsed = (time.monotonic() - start) * 1000
        return ValidationResult(
            name="Gemini API",
            passed=False,
            message=f"Failed: {e}",
            elapsed_ms=round(elapsed, 1),
        )


def _run_analysis_cycle(settings: Settings) -> list[dict]:
    """Run a full analysis cycle for each watchlist symbol without trading."""
    risk_config = RiskConfig.from_settings(settings)
    risk_mgr = RiskManager(risk_config, portfolio_value=Decimal("0"))
    executor = AlpacaExecutor(settings, risk_mgr)
    strategy = EMAMomentumStrategy(ema_period=settings.ema_period)
    analyst = Analyst(api_key=settings.gemini_api_key)

    results = []
    for symbol in settings.watchlist:
        try:
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
            bars = executor.get_bars(symbol, TimeFrame.Day, start=start_date)

            if not hasattr(bars, "df") or bars.df.empty:
                results.append({"symbol": symbol, "error": "No price data"})
                continue

            df = bars.df
            closes = df["close"].tolist()
            current_price = closes[-1]

            would_buy = strategy.should_buy(symbol, current_price, closes)
            would_sell = strategy.should_sell(symbol, current_price, closes)

            signal = "N/A"
            score = 0.0
            if settings.gemini_api_key:
                ohlcv = [
                    {
                        "date": str(idx),
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                    }
                    for idx, row in df.iloc[-30:].iterrows()
                ]
                analysis = analyst.full_analysis(symbol=symbol, prices=ohlcv)
                signal = analysis.final_signal.value
                score = analysis.combined_score

            results.append(
                {
                    "symbol": symbol,
                    "signal": signal,
                    "score": score,
                    "would_buy": would_buy,
                    "would_sell": would_sell,
                }
            )
        except (
            ConnectionError,
            ValueError,
            OSError,
            KeyError,
            RuntimeError,
            GeminiError,
        ) as e:
            log.warning("dry_run_symbol_error", symbol=symbol, error=str(e))
            results.append({"symbol": symbol, "error": str(e)})

    return results


def _print_report(report: DryRunReport) -> None:
    """Print a formatted dry-run report."""
    print()
    print("=" * 60)
    print("  DRY-RUN VALIDATION REPORT")
    print("=" * 60)
    print(f"  {'Check':<20} {'Status':<10} {'Time':<10} {'Details'}")
    print("-" * 60)

    for v in report.validations:
        status = "PASS" if v.passed else "FAIL"
        marker = "  " if v.passed else "X "
        print(
            f"  {marker}{v.name:<18} {status:<10} {v.elapsed_ms:>6.0f}ms  {v.message}"
        )

    print("-" * 60)

    if report.symbols_analyzed:
        print()
        print("  SIGNAL SUMMARY")
        print("-" * 60)
        print(f"  {'Symbol':<8} {'Signal':<12} {'Score':<8} {'Buy?':<6} {'Sell?'}")
        print("-" * 60)
        for s in report.symbols_analyzed:
            if "error" in s:
                print(f"  {s['symbol']:<8} ERROR: {s['error']}")
            else:
                print(
                    f"  {s['symbol']:<8} {s['signal']:<12} {s['score']:<8.3f} "
                    f"{'Yes' if s['would_buy'] else 'No':<6} "
                    f"{'Yes' if s['would_sell'] else 'No'}"
                )

    print("-" * 60)
    if report.all_passed:
        print("  RESULT: All validations passed - system ready")
    else:
        failed = sum(1 for v in report.validations if not v.passed)
        print(f"  RESULT: {failed} validation(s) failed - check configuration")
    print("=" * 60)
    print()


def run_dry_run(settings: Settings) -> DryRunReport:
    """Execute full dry-run validation sequence."""
    report = DryRunReport()

    log.info("dry_run_starting")

    report.validations.append(_check_alpaca(settings))
    report.validations.append(_check_market_data(settings))
    report.validations.append(_check_news_feed(settings))
    report.validations.append(_check_gemini(settings))

    # Only run analysis cycle if basic connectivity checks pass
    connectivity_ok = all(v.passed for v in report.validations)
    if connectivity_ok:
        report.symbols_analyzed = _run_analysis_cycle(settings)
    else:
        log.warning("dry_run_skip_analysis", reason="connectivity checks failed")

    _print_report(report)

    log.info("dry_run_complete", all_passed=report.all_passed)
    return report
