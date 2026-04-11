"""Entry point: uv run python -m claude_trader

Modes:
  (default)     Run scheduler loop during market hours
  --backtest    Run backtest against historical data
  --dry-run     Validate connectivity without trading
  --summary     Send daily summary only
  --graduation  Show graduation dashboard
  --metrics     Show detailed metrics
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timedelta

import structlog

from claude_trader.bot import TradingBot
from claude_trader.config import Settings
from claude_trader.constants import ET, MARKET_CLOSE, MARKET_OPEN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Claude Trading Bot",
        prog="python -m claude_trader",
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run backtest against historical data"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Backtest start date (YYYY-MM-DD, default: 6 months ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Backtest initial capital (default: 100000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate connectivity without trading"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Send daily summary only"
    )
    parser.add_argument(
        "--graduation", action="store_true", help="Show graduation dashboard"
    )
    parser.add_argument("--metrics", action="store_true", help="Show detailed metrics")
    parser.add_argument("--journal", action="store_true", help="Browse trade journal")
    parser.add_argument(
        "--symbol", type=str, default=None, help="Filter journal by symbol"
    )
    parser.add_argument(
        "--side", type=str, default=None, help="Filter journal by side (buy/sell)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable WebSocket trade update streaming",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Override cycle interval in minutes",
    )
    return parser.parse_args()


def confirm_live_trading(settings: Settings) -> None:
    """Prompt for confirmation if live trading is enabled."""
    if settings.alpaca_paper_trade:
        return
    confirm = input("WARNING: Live trading mode! Type 'CONFIRM LIVE' to proceed: ")
    if confirm != "CONFIRM LIVE":
        print("Aborted. Set ALPACA_PAPER_TRADE=true for paper trading.")
        sys.exit(1)


def log_startup_summary(settings: Settings) -> None:
    """Log key configuration at startup."""
    structlog.get_logger().info(
        "bot_starting",
        paper_mode=settings.alpaca_paper_trade,
        watchlist=settings.watchlist,
        max_position_pct=str(settings.max_position_pct),
        stop_loss_pct=str(settings.stop_loss_pct),
        max_daily_loss_pct=str(settings.max_daily_loss_pct),
        cycle_interval=settings.cycle_interval_minutes,
        gemini=bool(settings.gemini_api_key),
        telegram=bool(settings.telegram_bot_token),
    )


async def run_scheduler(
    bot: TradingBot, settings: Settings, interval_minutes: int
) -> None:
    """Run the trading loop during market hours with graceful shutdown."""
    log = structlog.get_logger()
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    daily_summary_sent = False
    current_date = datetime.now(ET).date()

    while not shutdown.is_set():
        now_et = datetime.now(ET)

        # Reset daily flag on date change
        if now_et.date() != current_date:
            current_date = now_et.date()
            daily_summary_sent = False
            bot.reset_daily()

        is_weekday = now_et.weekday() < 5

        if is_weekday and MARKET_OPEN <= now_et.time() < MARKET_CLOSE:
            bot.run_once()
            daily_summary_sent = False
        elif is_weekday and now_et.time() >= MARKET_CLOSE and not daily_summary_sent:
            bot.run_daily_summary()
            daily_summary_sent = True

        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval_minutes * 60)
        except TimeoutError:
            pass  # Normal wake-up

    log.info("scheduler_stopped")


def _run_backtest(args: argparse.Namespace, settings: Settings) -> int:
    """Fetch historical data and run backtest."""
    from decimal import Decimal

    from claude_trader.backtest import (
        BacktestConfig,
        BacktestEngine,
        print_backtest_report,
    )
    from claude_trader.executor import AlpacaExecutor
    from claude_trader.risk import RiskConfig, RiskManager

    log = structlog.get_logger()

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start or (datetime.now() - timedelta(days=180)).strftime(
        "%Y-%m-%d"
    )

    config = BacktestConfig(
        symbols=settings.watchlist,
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal(str(args.capital)),
        ema_period=settings.ema_period,
        stop_loss_pct=settings.stop_loss_pct,
        trailing_stop_pct=settings.trailing_stop_pct,
        max_position_pct=settings.max_position_pct,
    )

    risk_config = RiskConfig.from_settings(settings)
    risk_manager = RiskManager(risk_config, portfolio_value=Decimal(str(args.capital)))
    executor = AlpacaExecutor(settings, risk_manager)

    log.info(
        "backtest_starting",
        symbols=config.symbols,
        start=start_date,
        end=end_date,
        capital=str(config.initial_capital),
    )

    bars = BacktestEngine.fetch_backtest_data(
        executor, config.symbols, start_date, end_date
    )
    empty = [s for s in config.symbols if not bars.get(s)]
    if empty:
        log.warning("backtest_missing_data", symbols=empty)

    engine = BacktestEngine(config)
    result = engine.run(bars)
    print_backtest_report(result)
    return 0


def main() -> int:
    args = parse_args()

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    try:
        settings = Settings()
    except (ValueError, OSError) as e:
        print(f"Configuration error: {e}")
        print("Ensure .env file exists with required variables. See .env.example")
        return 1

    confirm_live_trading(settings)
    log_startup_summary(settings)

    if args.backtest:
        return _run_backtest(args, settings)

    if args.dry_run:
        from claude_trader.dry_run import run_dry_run

        report = run_dry_run(settings)
        return 0 if report.all_passed else 1

    if args.journal:
        from claude_trader.journal import print_journal

        print_journal(
            log_path=settings.trades_log_path,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            side=args.side,
        )
        return 0

    bot = TradingBot(settings)

    if args.graduation:
        from scripts.graduation import print_graduation

        from claude_trader.performance import PerformanceTracker

        tracker = PerformanceTracker(settings.snapshots_path)
        print_graduation(tracker)
    elif args.metrics:
        from scripts.graduation import print_metrics

        from claude_trader.performance import PerformanceTracker

        tracker = PerformanceTracker(settings.snapshots_path)
        print_metrics(tracker)
    elif args.summary:
        bot.run_daily_summary()
    else:
        interval = args.interval or settings.cycle_interval_minutes
        if args.no_stream:
            asyncio.run(run_scheduler(bot, settings, interval))
        else:
            from claude_trader.streaming import TradeUpdateListener

            listener = TradeUpdateListener(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=settings.alpaca_paper_trade,
            )

            async def run_with_stream():
                await asyncio.gather(
                    run_scheduler(bot, settings, interval),
                    listener.run(),
                    return_exceptions=True,
                )

            asyncio.run(run_with_stream())

    return 0


if __name__ == "__main__":
    sys.exit(main())
