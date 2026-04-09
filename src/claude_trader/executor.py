"""Alpaca API execution wrapper.

Thin layer over alpaca-py. Every order passes through the risk manager
before reaching the API. No direct trading without risk approval.
"""

from decimal import Decimal

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from claude_trader.config import Settings
from claude_trader.risk import RiskManager, TradeRequest

log = structlog.get_logger()


class AlpacaExecutor:
    """Executes trades via Alpaca API with mandatory risk checks."""

    def __init__(self, settings: Settings, risk_manager: RiskManager) -> None:
        self._settings = settings
        self._risk = risk_manager
        self._client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.alpaca_paper_trade,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

    def get_account(self) -> dict:
        """Get account balance, equity, and buying power."""
        account = self._client.get_account()
        return {
            "equity": Decimal(str(account.equity)),
            "cash": Decimal(str(account.cash)),
            "buying_power": Decimal(str(account.buying_power)),
            "portfolio_value": Decimal(str(account.portfolio_value)),
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        positions = self._client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": int(p.qty),
                "avg_entry": Decimal(str(p.avg_entry_price)),
                "current_price": Decimal(str(p.current_price)),
                "unrealized_pnl": Decimal(str(p.unrealized_pl)),
                "side": str(p.side),
            }
            for p in positions
        ]

    def buy(
        self,
        symbol: str,
        price: Decimal,
        qty: int | None = None,
        market_time=None,
    ) -> dict | None:
        """Buy shares with automatic stop-loss. Returns order info or None if rejected."""
        if qty is None:
            qty = self._risk.calculate_position_size(symbol, price)
        if qty == 0:
            log.warning("buy_rejected_zero_qty", symbol=symbol, price=str(price))
            return None

        trade_req = TradeRequest(symbol=symbol, side="buy", price=price, qty=qty)
        check = self._risk.check_trade(trade_req, market_time=market_time)

        if not check.approved:
            log.warning("buy_rejected_risk", symbol=symbol, reason=check.reason)
            return None

        order = self._client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        log.info("buy_executed", symbol=symbol, qty=qty, order_id=str(order.id))

        # Set stop-loss
        stop_price = round(float(self._risk.calculate_stop_loss(price)), 2)
        stop_order = self._client.submit_order(
            StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=stop_price,
            )
        )
        log.info("stop_loss_set", symbol=symbol, stop_price=stop_price)

        self._risk.open_positions += 1
        return {
            "order_id": str(order.id),
            "stop_order_id": str(stop_order.id),
            "symbol": symbol,
            "qty": qty,
            "stop_price": stop_price,
        }

    def sell(self, symbol: str, qty: int) -> dict | None:
        """Sell shares (market order)."""
        order = self._client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        log.info("sell_executed", symbol=symbol, qty=qty, order_id=str(order.id))
        self._risk.open_positions = max(0, self._risk.open_positions - 1)
        return {"order_id": str(order.id), "symbol": symbol, "qty": qty}

    def set_trailing_stop(self, symbol: str, qty: int, trail_pct: float) -> dict | None:
        """Set a trailing stop order."""
        order = self._client.submit_order(
            TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=trail_pct * 100,
            )
        )
        log.info("trailing_stop_set", symbol=symbol, trail_pct=trail_pct)
        return {"order_id": str(order.id), "symbol": symbol, "trail_pct": trail_pct}

    def get_bars(self, symbol: str, timeframe: TimeFrame, start: str, end: str | None = None):
        """Get historical price bars for analysis."""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        return self._data_client.get_stock_bars(request)
