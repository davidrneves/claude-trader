"""Shared test fixtures."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from claude_trader.config import Settings


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.alpaca_api_key = "test-key"
    settings.alpaca_secret_key = "test-secret"
    settings.alpaca_paper_trade = True
    settings.gemini_api_key = ""
    settings.max_position_pct = Decimal("0.02")
    settings.stop_loss_pct = Decimal("0.08")
    settings.trailing_stop_pct = Decimal("0.05")
    settings.max_daily_loss_pct = Decimal("0.03")
    settings.max_drawdown_pct = Decimal("0.10")
    settings.max_consecutive_losses = 3
    settings.max_open_positions = 5
    settings.ema_period = 20
    settings.telegram_bot_token = ""
    settings.telegram_chat_id = ""
    settings.obsidian_log_path = MagicMock()
    settings.trades_log_path = MagicMock()
    settings.snapshots_path = MagicMock()
    settings.watchlist = ["AAPL", "MSFT"]
    return settings
