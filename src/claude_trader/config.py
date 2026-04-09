"""Configuration management via Pydantic Settings.

All sensitive values come from environment variables or .env file.
Paper trading is the default - live trading requires explicit opt-in.
"""

from decimal import Decimal
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Alpaca ---
    alpaca_api_key: str = Field(description="Alpaca API key")
    alpaca_secret_key: str = Field(description="Alpaca secret key")
    alpaca_paper_trade: bool = Field(
        default=True,
        description="Paper trading mode. Set to false for live (requires confirmation).",
    )
    # --- Gemini ---
    gemini_api_key: str = Field(
        default="", description="Google Gemini API key for analysis agents"
    )

    # --- Risk Parameters ---
    max_position_pct: Decimal = Field(
        default=Decimal("0.02"), description="Max % of portfolio per trade"
    )
    stop_loss_pct: Decimal = Field(
        default=Decimal("0.08"), description="Stop loss % below entry"
    )
    trailing_stop_pct: Decimal = Field(
        default=Decimal("0.05"), description="Trailing stop trail %"
    )
    max_daily_loss_pct: Decimal = Field(
        default=Decimal("0.03"), description="Max daily loss before halt"
    )
    max_drawdown_pct: Decimal = Field(
        default=Decimal("0.10"), description="Max total drawdown before halt"
    )
    max_consecutive_losses: int = Field(
        default=3, description="Circuit breaker threshold"
    )
    max_open_positions: int = Field(default=5, description="Max concurrent positions")

    # --- Strategy ---
    watchlist: list[str] = Field(
        default=[
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "SPY",
            "QQQ",
            "AMD",
        ],
        description="Symbols to monitor",
    )
    ema_period: int = Field(default=20, description="EMA period for momentum strategy")

    # --- Telegram ---
    telegram_bot_token: str = Field(
        default="", description="Telegram bot token for notifications"
    )
    telegram_chat_id: str = Field(
        default="", description="Telegram chat ID for notifications"
    )

    # --- Obsidian ---
    obsidian_log_path: Path = Field(
        default=Path.home() / "Obsidian" / "1. Projects" / "claude-trader" / "logs",
        description="Obsidian vault path for daily trade logs",
    )

    # --- Logging ---
    log_level: str = Field(default="INFO")
    trades_log_path: Path = Field(default=Path("trades.jsonl"))
