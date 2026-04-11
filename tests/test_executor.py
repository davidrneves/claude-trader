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


class TestUpdateStopLoss:
    def test_update_cancels_old_and_submits_new(self, risk_manager):
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._risk = risk_manager
            executor._client = MagicMock()
            executor._client.submit_order.return_value = FakeOrder("stop-new")

            result = executor.update_stop_loss(
                "AAPL", 10, Decimal("104.50"), "stop-old"
            )
            executor._client.cancel_order_by_id.assert_called_once_with("stop-old")
            executor._client.submit_order.assert_called_once()
            assert result is not None
            assert result["stop_order_id"] == "stop-new"
            assert result["stop_price"] == 104.5

    def test_update_handles_cancel_failure(self, risk_manager):
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._risk = risk_manager
            executor._client = MagicMock()
            executor._client.cancel_order_by_id.side_effect = Exception("not found")
            executor._client.submit_order.return_value = FakeOrder("stop-new")

            result = executor.update_stop_loss(
                "AAPL", 10, Decimal("104.50"), "stop-old"
            )
            assert result is not None
            assert result["stop_order_id"] == "stop-new"


class TestCancelStopLoss:
    def test_cancel_calls_api(self, risk_manager):
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._client = MagicMock()

            executor.cancel_stop_loss("stop-123")
            executor._client.cancel_order_by_id.assert_called_once_with("stop-123")

    def test_cancel_handles_error(self, risk_manager):
        from claude_trader.executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, *a, **kw: None):
            executor = AlpacaExecutor.__new__(AlpacaExecutor)
            executor._client = MagicMock()
            executor._client.cancel_order_by_id.side_effect = Exception("fail")

            # Should not raise
            executor.cancel_stop_loss("stop-123")
