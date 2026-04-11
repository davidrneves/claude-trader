"""Tests for the __main__ scheduler entry point."""

import asyncio
from datetime import datetime, time
from unittest.mock import MagicMock, patch

import pytest

from claude_trader.__main__ import (
    confirm_live_trading,
    log_startup_summary,
    parse_args,
    run_scheduler,
)


# --- parse_args ---


def test_parse_args_defaults():
    with patch("sys.argv", ["prog"]):
        args = parse_args()
    assert args.backtest is False
    assert args.dry_run is False
    assert args.summary is False
    assert args.graduation is False
    assert args.metrics is False
    assert args.interval is None
    assert args.start is None
    assert args.end is None
    assert args.capital == 100000


def test_parse_args_backtest():
    with patch(
        "sys.argv",
        [
            "prog",
            "--backtest",
            "--start",
            "2025-01-01",
            "--end",
            "2025-06-30",
            "--capital",
            "50000",
        ],
    ):
        args = parse_args()
    assert args.backtest is True
    assert args.start == "2025-01-01"
    assert args.end == "2025-06-30"
    assert args.capital == 50000


def test_parse_args_dry_run():
    with patch("sys.argv", ["prog", "--dry-run"]):
        args = parse_args()
    assert args.dry_run is True


def test_parse_args_interval():
    with patch("sys.argv", ["prog", "--interval", "5"]):
        args = parse_args()
    assert args.interval == 5


# --- confirm_live_trading ---


def test_confirm_live_paper_mode(mock_settings):
    mock_settings.alpaca_paper_trade = True
    confirm_live_trading(mock_settings)  # Should return without prompting


def test_confirm_live_rejects(mock_settings):
    mock_settings.alpaca_paper_trade = False
    with patch("builtins.input", return_value="no"):
        with pytest.raises(SystemExit):
            confirm_live_trading(mock_settings)


def test_confirm_live_accepts(mock_settings):
    mock_settings.alpaca_paper_trade = False
    with patch("builtins.input", return_value="CONFIRM LIVE"):
        confirm_live_trading(mock_settings)  # Should not raise


# --- log_startup_summary ---


def test_log_startup_summary(mock_settings):
    mock_settings.cycle_interval_minutes = 15
    with patch("claude_trader.__main__.structlog") as mock_structlog:
        mock_logger = MagicMock()
        mock_structlog.get_logger.return_value = mock_logger
        log_startup_summary(mock_settings)
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args
        assert call_kwargs[0][0] == "bot_starting"


# --- run_scheduler ---


@pytest.mark.asyncio
async def test_scheduler_immediate_shutdown(mock_settings):
    bot = MagicMock()
    mock_settings.cycle_interval_minutes = 15

    with patch("claude_trader.__main__.asyncio.Event") as mock_event_cls:
        event = MagicMock()
        event.is_set.return_value = True
        event.wait = MagicMock(return_value=asyncio.Future())
        mock_event_cls.return_value = event

        with patch("claude_trader.__main__.asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value = MagicMock()
            await run_scheduler(bot, mock_settings, 15)

    bot.run_once.assert_not_called()


@pytest.mark.asyncio
async def test_scheduler_calls_run_once(mock_settings):
    bot = MagicMock()
    mock_settings.cycle_interval_minutes = 15
    call_count = 0

    # Create a real event for shutdown control
    shutdown = asyncio.Event()

    with patch("claude_trader.__main__.asyncio.Event", return_value=shutdown):
        with patch("claude_trader.__main__.asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value = MagicMock()

            # Mock datetime to return market hours on a weekday (Monday)
            mock_now = MagicMock()
            mock_now.weekday.return_value = 0  # Monday
            mock_now.time.return_value = time(10, 0)  # 10:00 AM ET
            mock_now.date.return_value = datetime(2026, 1, 5).date()

            with patch("claude_trader.__main__.datetime") as mock_dt:
                mock_dt.now.return_value = mock_now
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

                # After first run_once, set shutdown
                def run_once_then_stop():
                    nonlocal call_count
                    call_count += 1
                    shutdown.set()

                bot.run_once.side_effect = run_once_then_stop

                await run_scheduler(bot, mock_settings, 15)

    assert bot.run_once.call_count == 1


@pytest.mark.asyncio
async def test_scheduler_resets_on_date_change(mock_settings):
    bot = MagicMock()
    mock_settings.cycle_interval_minutes = 15

    shutdown = asyncio.Event()
    iteration = 0

    with patch("claude_trader.__main__.asyncio.Event", return_value=shutdown):
        with patch("claude_trader.__main__.asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value = MagicMock()

            day1 = datetime(2026, 1, 5)  # Monday
            day2 = datetime(2026, 1, 6)  # Tuesday

            mock_now_day1 = MagicMock()
            mock_now_day1.weekday.return_value = 0
            mock_now_day1.time.return_value = time(10, 0)
            mock_now_day1.date.return_value = day1.date()

            mock_now_day2 = MagicMock()
            mock_now_day2.weekday.return_value = 1
            mock_now_day2.time.return_value = time(10, 0)
            mock_now_day2.date.return_value = day2.date()

            with patch("claude_trader.__main__.datetime") as mock_dt:
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

                def advance_day():
                    nonlocal iteration
                    iteration += 1
                    if iteration == 1:
                        # Switch to day2 for next loop iteration
                        mock_dt.now.return_value = mock_now_day2
                    else:
                        shutdown.set()

                mock_dt.now.return_value = mock_now_day1
                bot.run_once.side_effect = advance_day

                # Use interval=0 so asyncio.wait_for times out immediately
                # between iterations instead of sleeping 15 minutes
                await run_scheduler(bot, mock_settings, 0)

    bot.reset_daily.assert_called_once()
