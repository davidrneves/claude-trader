"""Rule-based risk management - zero LLM dependency.

Research finding: Risk management is the #1 factor determining trading system
robustness, not model intelligence. Every rule here is hard-coded logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

import structlog

from claude_trader.constants import MARKET_CLOSE, MARKET_OPEN

if TYPE_CHECKING:
    from claude_trader.config import Settings

log = structlog.get_logger()


@dataclass
class RiskConfig:
    """All risk parameters - configurable but with safe defaults."""

    max_position_pct: Decimal = Decimal("0.02")  # 2% of portfolio per trade
    stop_loss_pct: Decimal = Decimal("0.08")  # 8% below entry
    trailing_stop_pct: Decimal = Decimal("0.05")  # 5% trail
    max_daily_loss_pct: Decimal = Decimal("0.03")  # 3% daily loss -> halt
    max_drawdown_pct: Decimal = Decimal("0.10")  # 10% total drawdown -> halt
    max_consecutive_losses: int = 3  # Circuit breaker threshold
    max_open_positions: int = 5
    banned_minutes_open: int = 15  # Minutes after open to avoid
    banned_minutes_close: int = 15  # Minutes before close to avoid

    @classmethod
    def from_settings(cls, settings: Settings) -> RiskConfig:
        """Build RiskConfig from application Settings (single source of truth)."""
        return cls(
            max_position_pct=settings.max_position_pct,
            stop_loss_pct=settings.stop_loss_pct,
            trailing_stop_pct=settings.trailing_stop_pct,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            max_drawdown_pct=settings.max_drawdown_pct,
            max_consecutive_losses=settings.max_consecutive_losses,
            max_open_positions=settings.max_open_positions,
        )


@dataclass
class TradeRequest:
    """A proposed trade that must pass risk checks before execution."""

    symbol: str
    side: Literal["buy", "sell"]
    price: Decimal
    qty: int


@dataclass
class RiskCheckResult:
    """Result of a risk check - approved or rejected with reason."""

    approved: bool
    reason: str = ""
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


class RiskManager:
    """Pure rule-based risk manager. No LLM calls, no ML models."""

    def __init__(self, config: RiskConfig, portfolio_value: Decimal) -> None:
        self.config = config
        self.portfolio_value = portfolio_value
        self._daily_pnl = Decimal("0")
        self._current_drawdown = Decimal("0")
        self._consecutive_losses = 0
        self.open_positions = 0

    def calculate_position_size(self, symbol: str, price: Decimal) -> int:
        """Max shares to buy within position size limit."""
        max_value = self.portfolio_value * self.config.max_position_pct
        if price <= 0:
            return 0
        return int(max_value / price)

    def calculate_stop_loss(self, entry_price: Decimal) -> Decimal:
        """Fixed stop-loss price below entry."""
        return entry_price * (1 - self.config.stop_loss_pct)

    def calculate_trailing_stop(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        current_floor: Decimal | None,
    ) -> Decimal:
        """Trailing stop - floor only goes up, never down."""
        new_floor = current_price * (1 - self.config.trailing_stop_pct)
        if current_floor is None:
            return new_floor
        return max(current_floor, new_floor)

    def check_trade(
        self,
        trade: TradeRequest,
        market_time: time | None = None,
    ) -> RiskCheckResult:
        """Run all risk checks on a proposed trade. ALL must pass."""
        passed = []
        failed = []

        # Basic validation
        if trade.qty <= 0:
            return RiskCheckResult(
                approved=False, reason="Invalid quantity: must be > 0"
            )
        if trade.price <= 0:
            return RiskCheckResult(approved=False, reason="Invalid price: must be > 0")

        # Sells are always allowed (closing positions reduces risk)
        if trade.side == "sell":
            return RiskCheckResult(approved=True, reason="Sell orders always approved")

        # --- Buy-side checks ---

        # 1. Daily loss limit
        daily_loss_limit = self.portfolio_value * self.config.max_daily_loss_pct
        if abs(self._daily_pnl) >= daily_loss_limit and self._daily_pnl < 0:
            failed.append("daily_loss")
            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss limit hit: {self._daily_pnl} exceeds -{daily_loss_limit}",
                checks_failed=failed,
            )
        passed.append("daily_loss")

        # 2. Max drawdown
        if self._current_drawdown >= self.config.max_drawdown_pct:
            failed.append("drawdown")
            return RiskCheckResult(
                approved=False,
                reason=f"Max drawdown exceeded: {self._current_drawdown:.1%} >= {self.config.max_drawdown_pct:.1%}",
                checks_failed=failed,
            )
        passed.append("drawdown")

        # 3. Circuit breaker (consecutive losses)
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            failed.append("circuit_breaker")
            return RiskCheckResult(
                approved=False,
                reason=f"Circuit breaker: {self._consecutive_losses} consecutive losses",
                checks_failed=failed,
            )
        passed.append("circuit_breaker")

        # 4. Max open positions
        if self.open_positions >= self.config.max_open_positions:
            failed.append("max_positions")
            return RiskCheckResult(
                approved=False,
                reason=f"Max open positions reached: {self.open_positions}/{self.config.max_open_positions}",
                checks_failed=failed,
            )
        passed.append("max_positions")

        # 5. Position size limit
        if self.portfolio_value > 0:
            trade_value = trade.price * trade.qty
            max_trade_value = self.portfolio_value * self.config.max_position_pct
            if trade_value > max_trade_value:
                failed.append("position_size")
                return RiskCheckResult(
                    approved=False,
                    reason=f"Position size exceeds limit: {trade_value} > {max_trade_value} ({self.config.max_position_pct:.0%} of portfolio)",
                    checks_passed=passed,
                    checks_failed=failed,
                )
        passed.append("position_size")

        # 6. Banned hours
        if market_time is not None:
            open_minutes = (
                MARKET_OPEN.hour * 60
                + MARKET_OPEN.minute
                + self.config.banned_minutes_open
            )
            close_minutes = (
                MARKET_CLOSE.hour * 60
                + MARKET_CLOSE.minute
                - self.config.banned_minutes_close
            )
            open_cutoff = time(open_minutes // 60, open_minutes % 60)
            close_cutoff = time(close_minutes // 60, close_minutes % 60)
            if market_time < open_cutoff or market_time >= close_cutoff:
                failed.append("banned_hours")
                return RiskCheckResult(
                    approved=False,
                    reason=f"Banned trading hours: {market_time} (allowed {open_cutoff}-{close_cutoff})",
                    checks_failed=failed,
                )
        passed.append("banned_hours")

        return RiskCheckResult(
            approved=True, reason="All checks passed", checks_passed=passed
        )

    def record_daily_pnl(self, pnl: Decimal) -> None:
        """Update today's P&L. Called after each trade closes."""
        self._daily_pnl += pnl
        log.info("daily_pnl_updated", daily_pnl=str(self._daily_pnl))

    def record_drawdown(self, drawdown_pct: Decimal) -> None:
        """Update current drawdown from peak. Called on portfolio updates."""
        self._current_drawdown = drawdown_pct
        log.info("drawdown_updated", drawdown=str(drawdown_pct))

    def record_trade_result(self, profit: Decimal) -> None:
        """Track consecutive wins/losses for circuit breaker."""
        if profit < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        log.info(
            "trade_result_recorded",
            profit=str(profit),
            consecutive_losses=self._consecutive_losses,
        )

    def get_risk_state(self) -> dict:
        """Return current risk state for external consumers (e.g. snapshots)."""
        return {
            "open_positions": self.open_positions,
            "consecutive_losses": self._consecutive_losses,
            "circuit_breaker_triggered": (
                self._consecutive_losses >= self.config.max_consecutive_losses
            ),
        }

    def reset_daily(self) -> None:
        """Reset daily counters. Called at start of each trading day."""
        self._daily_pnl = Decimal("0")
        self._consecutive_losses = 0
        log.info("daily_reset")
