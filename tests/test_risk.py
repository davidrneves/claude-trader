"""Tests for the risk management module - the #1 priority."""

from datetime import time
from decimal import Decimal

import pytest

from claude_trader.risk import (
    RiskConfig,
    RiskManager,
    TradeRequest,
)


@pytest.fixture
def config():
    return RiskConfig(
        max_position_pct=Decimal("0.02"),
        stop_loss_pct=Decimal("0.08"),
        trailing_stop_pct=Decimal("0.05"),
        max_daily_loss_pct=Decimal("0.03"),
        max_drawdown_pct=Decimal("0.10"),
        max_consecutive_losses=3,
        max_open_positions=5,
        banned_minutes_open=15,
        banned_minutes_close=15,
    )


@pytest.fixture
def risk_manager(config):
    return RiskManager(config, portfolio_value=Decimal("10000"))


class TestPositionSizing:
    def test_max_position_size(self, risk_manager):
        qty = risk_manager.calculate_position_size(symbol="AAPL", price=Decimal("150"))
        assert qty == 1

    def test_position_size_expensive_stock(self, risk_manager):
        qty = risk_manager.calculate_position_size(
            symbol="BRK.A", price=Decimal("500000")
        )
        assert qty == 0

    def test_position_size_cheap_stock(self, risk_manager):
        qty = risk_manager.calculate_position_size(symbol="F", price=Decimal("10"))
        assert qty == 20


class TestStopLoss:
    def test_stop_loss_price(self, risk_manager):
        stop = risk_manager.calculate_stop_loss(entry_price=Decimal("100"))
        assert stop == Decimal("92")

    def test_stop_loss_with_decimal_price(self, risk_manager):
        stop = risk_manager.calculate_stop_loss(entry_price=Decimal("47.50"))
        expected = Decimal("47.50") * (1 - Decimal("0.08"))
        assert stop == expected


class TestTrailingStop:
    def test_trailing_stop_initial(self, risk_manager):
        trail = risk_manager.calculate_trailing_stop(
            entry_price=Decimal("100"),
            current_price=Decimal("100"),
            current_floor=None,
        )
        assert trail == Decimal("95")

    def test_trailing_stop_moves_up(self, risk_manager):
        trail = risk_manager.calculate_trailing_stop(
            entry_price=Decimal("100"),
            current_price=Decimal("110"),
            current_floor=Decimal("95"),
        )
        assert trail == Decimal("104.50")

    def test_trailing_stop_never_moves_down(self, risk_manager):
        trail = risk_manager.calculate_trailing_stop(
            entry_price=Decimal("100"),
            current_price=Decimal("95"),
            current_floor=Decimal("104.50"),
        )
        assert trail == Decimal("104.50")


class TestDailyLossLimit:
    def test_no_trades_allowed_after_daily_loss(self, risk_manager):
        risk_manager.record_daily_pnl(Decimal("-300"))
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert not result.approved
        assert "daily loss" in result.reason.lower()

    def test_trades_allowed_within_limit(self, risk_manager):
        risk_manager.record_daily_pnl(Decimal("-100"))
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert result.approved


class TestMaxDrawdown:
    def test_halt_on_max_drawdown(self, risk_manager):
        risk_manager.record_drawdown(Decimal("0.11"))
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert not result.approved
        assert "drawdown" in result.reason.lower()

    def test_allow_within_drawdown(self, risk_manager):
        risk_manager.record_drawdown(Decimal("0.05"))
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert result.approved


class TestCircuitBreaker:
    def test_halt_after_consecutive_losses(self, risk_manager):
        for _ in range(3):
            risk_manager.record_trade_result(profit=Decimal("-10"))
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert not result.approved
        assert "consecutive" in result.reason.lower()

    def test_reset_after_win(self, risk_manager):
        risk_manager.record_trade_result(profit=Decimal("-10"))
        risk_manager.record_trade_result(profit=Decimal("-10"))
        risk_manager.record_trade_result(profit=Decimal("20"))
        risk_manager.record_trade_result(profit=Decimal("-10"))
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert result.approved


class TestMaxOpenPositions:
    def test_reject_when_at_max(self, risk_manager):
        risk_manager.open_positions = 5
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert not result.approved
        assert "positions" in result.reason.lower()

    def test_allow_sell_when_at_max(self, risk_manager):
        risk_manager.open_positions = 5
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="sell", price=Decimal("150"), qty=1)
        )
        assert result.approved

    def test_allow_buy_under_max(self, risk_manager):
        risk_manager.open_positions = 4
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert result.approved


class TestBannedHours:
    def test_reject_during_market_open(self, risk_manager):
        market_time = time(9, 35)
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1),
            market_time=market_time,
        )
        assert not result.approved
        assert "banned" in result.reason.lower()

    def test_reject_during_market_close(self, risk_manager):
        market_time = time(15, 50)
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1),
            market_time=market_time,
        )
        assert not result.approved
        assert "banned" in result.reason.lower()

    def test_allow_during_normal_hours(self, risk_manager):
        market_time = time(11, 0)
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1),
            market_time=market_time,
        )
        assert result.approved


class TestTradeRequestValidation:
    def test_zero_quantity_rejected(self, risk_manager):
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=0)
        )
        assert not result.approved

    def test_negative_price_rejected(self, risk_manager):
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("-10"), qty=1)
        )
        assert not result.approved


class TestResetDaily:
    def test_daily_reset_clears_pnl(self, risk_manager):
        risk_manager.record_daily_pnl(Decimal("-300"))
        risk_manager.reset_daily()
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert result.approved

    def test_daily_reset_preserves_drawdown(self, risk_manager):
        risk_manager.record_drawdown(Decimal("0.11"))
        risk_manager.reset_daily()
        result = risk_manager.check_trade(
            TradeRequest(symbol="AAPL", side="buy", price=Decimal("150"), qty=1)
        )
        assert not result.approved
