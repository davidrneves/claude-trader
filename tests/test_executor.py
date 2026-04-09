"""Tests for the Alpaca executor."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from claude_trader.risk import RiskConfig, RiskManager


class FakeOrder:
    def __init__(self, order_id="order-123"):
        self.id = order_id


@pytest.fixture
def risk_manager():
    config = RiskConfig()
    return RiskManager(config, portfolio_value=Decimal("10000"))


class TestExecutorBuy:
    def test_risk_rejection_returns_none(self, risk_manager):
        """Buy is rejected when risk manager disapproves."""
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._risk = risk_manager
            executor._client = MagicMock()

            # Force risk rejection by maxing positions
            risk_manager.open_positions = 5
            result = executor.buy("AAPL", Decimal("150"), qty=1)
            assert result is None
            executor._client.submit_order.assert_not_called()

    def test_zero_qty_returns_none(self, risk_manager):
        """Buy with zero calculated qty returns None."""
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._risk = risk_manager
            executor._client = MagicMock()

            result = executor.buy("BRK.A", Decimal("999999"), qty=None)
            assert result is None

    def test_successful_buy_with_stop_loss(self, risk_manager):
        """Buy succeeds and sets stop-loss."""
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._risk = risk_manager
            executor._client = MagicMock()
            executor._client.submit_order.return_value = FakeOrder("buy-001")

            result = executor.buy("AAPL", Decimal("100"), qty=1)
            assert result is not None
            assert result["symbol"] == "AAPL"
            assert result["qty"] == 1
            assert result["order_id"] == "buy-001"
            assert "stop_order_id" in result
            assert executor._client.submit_order.call_count == 2

    def test_stop_loss_failure_still_returns_buy(self, risk_manager):
        """If stop-loss fails, buy still succeeds but without stop info."""
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._risk = risk_manager
            executor._client = MagicMock()

            # First call (buy) succeeds, second call (stop) fails
            executor._client.submit_order.side_effect = [
                FakeOrder("buy-001"),
                Exception("stop-loss API error"),
            ]

            result = executor.buy("AAPL", Decimal("100"), qty=1)
            assert result is not None
            assert result["order_id"] == "buy-001"
            assert "stop_order_id" not in result


class TestExecutorSell:
    def test_sell_decrements_positions(self, risk_manager):
        """Sell reduces open_positions count."""
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._risk = risk_manager
            executor._client = MagicMock()
            executor._client.submit_order.return_value = FakeOrder("sell-001")

            risk_manager.open_positions = 3
            result = executor.sell("AAPL", 10)
            assert result is not None
            assert result["order_id"] == "sell-001"
            assert risk_manager.open_positions == 2

    def test_sell_floor_at_zero(self, risk_manager):
        """open_positions never goes negative."""
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._risk = risk_manager
            executor._client = MagicMock()
            executor._client.submit_order.return_value = FakeOrder()

            risk_manager.open_positions = 0
            executor.sell("AAPL", 10)
            assert risk_manager.open_positions == 0
