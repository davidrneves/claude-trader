"""Tests for the Alpaca WebSocket trade update listener."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_trader.streaming import TradeUpdateListener


@pytest.fixture
def mock_stream():
    with patch("claude_trader.streaming.TradingStream") as mock:
        mock.return_value = MagicMock()
        mock.return_value.stop_ws = AsyncMock()
        yield mock


class TestTradeUpdateListener:
    def test_subscribes_to_trade_updates(self, mock_stream):
        TradeUpdateListener(api_key="k", secret_key="s", paper=True)
        mock_stream.return_value.subscribe_trade_updates.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_fill_calls_callback(self, mock_stream):
        cb = MagicMock()
        listener = TradeUpdateListener(
            api_key="k", secret_key="s", paper=True, on_fill=cb
        )

        data = MagicMock()
        data.event = "fill"
        data.order.symbol = "AAPL"

        await listener._handle_trade_update(data)
        cb.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_handle_fill_no_callback(self, mock_stream):
        listener = TradeUpdateListener(api_key="k", secret_key="s", paper=True)

        data = MagicMock()
        data.event = "fill"
        data.order.symbol = "AAPL"

        await listener._handle_trade_update(data)  # should not raise

    @pytest.mark.asyncio
    async def test_handle_canceled_logs_warning(self, mock_stream):
        listener = TradeUpdateListener(api_key="k", secret_key="s", paper=True)

        data = MagicMock()
        data.event = "canceled"
        data.order.symbol = "TSLA"
        data.order.id = "order-123"

        await listener._handle_trade_update(data)  # should not raise

    @pytest.mark.asyncio
    async def test_stop_calls_stop_ws(self, mock_stream):
        listener = TradeUpdateListener(api_key="k", secret_key="s", paper=True)
        await listener.stop()
        mock_stream.return_value.stop_ws.assert_awaited_once()
