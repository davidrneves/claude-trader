"""Tests for Settings configuration."""

from decimal import Decimal
from pathlib import Path

import pytest

from claude_trader.config import Settings


def _settings_no_dotenv(**overrides):
    """Create Settings without reading .env file."""
    return Settings(_env_file=None, **overrides)


class TestSettingsDefaults:
    def test_paper_trade_default_true(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        s = _settings_no_dotenv()
        assert s.alpaca_paper_trade is True

    def test_risk_defaults(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        s = _settings_no_dotenv()
        assert s.max_position_pct == Decimal("0.02")
        assert s.stop_loss_pct == Decimal("0.08")
        assert s.trailing_stop_pct == Decimal("0.05")
        assert s.max_daily_loss_pct == Decimal("0.03")
        assert s.max_drawdown_pct == Decimal("0.10")
        assert s.max_consecutive_losses == 3
        assert s.max_open_positions == 5

    def test_ema_period_default(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        s = _settings_no_dotenv()
        assert s.ema_period == 20

    def test_gemini_key_defaults_empty(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        s = _settings_no_dotenv()
        assert s.gemini_api_key == ""

    def test_telegram_defaults_empty(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        s = _settings_no_dotenv()
        assert s.telegram_bot_token == ""
        assert s.telegram_chat_id == ""

    def test_default_watchlist(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        s = _settings_no_dotenv()
        assert "AAPL" in s.watchlist
        assert len(s.watchlist) == 10

    def test_log_paths_are_paths(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        s = _settings_no_dotenv()
        assert isinstance(s.trades_log_path, Path)
        assert isinstance(s.snapshots_path, Path)


class TestSettingsRequired:
    def test_missing_alpaca_key_raises(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(Exception):
            _settings_no_dotenv()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "my-key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "my-secret")
        monkeypatch.setenv("EMA_PERIOD", "50")
        s = _settings_no_dotenv()
        assert s.alpaca_api_key == "my-key"
        assert s.ema_period == 50
