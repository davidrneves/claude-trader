#!/usr/bin/env python3
"""Entry point for the Claude Trading Bot.

Usage:
  uv run python scripts/run.py           # Single trading cycle
  uv run python scripts/run.py --summary # End-of-day summary only
"""

import argparse
import sys

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

from claude_trader.bot import TradingBot
from claude_trader.config import Settings


def main():
    parser = argparse.ArgumentParser(description="Claude Trading Bot")
    parser.add_argument(
        "--summary", action="store_true", help="Send daily summary only"
    )
    args = parser.parse_args()

    log = structlog.get_logger()

    try:
        settings = Settings()
    except Exception as e:
        print(f"Configuration error: {e}")
        print("Ensure .env file exists with required variables. See .env.example")
        sys.exit(1)

    if not settings.alpaca_paper_trade:
        confirm = input("WARNING: Live trading mode! Type 'CONFIRM LIVE' to proceed: ")
        if confirm != "CONFIRM LIVE":
            print("Aborted. Set ALPACA_PAPER_TRADE=true for paper trading.")
            sys.exit(1)

    log.info(
        "bot_starting",
        paper_mode=settings.alpaca_paper_trade,
        watchlist=settings.watchlist,
        gemini=bool(settings.gemini_api_key),
        telegram=bool(settings.telegram_bot_token),
    )

    bot = TradingBot(settings)

    if args.summary:
        bot.run_daily_summary()
    else:
        summary = bot.run_once()
        log.info("bot_finished", summary=summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
