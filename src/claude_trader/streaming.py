"""Real-time trade update listener via Alpaca WebSocket.

Runs alongside the polling scheduler to provide immediate notification
when orders fill, stops trigger, or orders are cancelled/rejected.
Does NOT replace the polling analysis cycle.
"""

import structlog
from alpaca.trading.stream import TradingStream

log = structlog.get_logger()


class TradeUpdateListener:
    """Listens for Alpaca trade update events and dispatches to handlers."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        on_fill=None,
    ) -> None:
        self._stream = TradingStream(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self._on_fill = on_fill
        self._stream.subscribe_trade_updates(self._handle_trade_update)

    async def _handle_trade_update(self, data) -> None:
        """Dispatch trade update events to registered callbacks."""
        event = data.event if hasattr(data, "event") else "unknown"
        order = data.order if hasattr(data, "order") else None
        symbol = order.symbol if order and hasattr(order, "symbol") else "unknown"

        log.info("trade_update", trade_event=str(event), symbol=symbol)

        if str(event) == "fill" and self._on_fill:
            self._on_fill(data)
        elif str(event) in ("canceled", "expired", "rejected"):
            log.warning(
                "order_event",
                trade_event=str(event),
                symbol=symbol,
                order_id=str(order.id) if order and hasattr(order, "id") else "unknown",
            )

    async def run(self) -> None:
        """Start the WebSocket listener."""
        log.info("streaming_started")
        await self._stream._run_forever()

    async def stop(self) -> None:
        """Stop the WebSocket listener gracefully."""
        log.info("streaming_stopping")
        await self._stream.stop_ws()
