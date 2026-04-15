"""Alpaca API execution wrapper.

Thin layer over alpaca-py. Every order passes through the risk manager
before reaching the API. No direct trading without risk approval.
"""

import json
from decimal import Decimal

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    OrderStatus,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    StopOrderRequest,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from claude_trader.config import Settings
from claude_trader.risk import RiskManager, TradeRequest

log = structlog.get_logger()


def _parse_existing_stop_from_error(error_str: str) -> str | None:
    """Extract existing stop order ID from Alpaca rejection error.

    When Alpaca rejects a stop order with code 40310000 ("insufficient qty
    available"), the error JSON includes a `related_orders` list with the
    IDs of existing orders that hold the shares. Return the first ID so
    the caller can adopt it instead of creating a new one.
    """
    try:
        data = json.loads(error_str)
    except json.JSONDecodeError, TypeError:
        return None
    if data.get("code") != 40310000:
        return None
    related = data.get("related_orders", [])
    if related:
        return str(related[0])
    existing_id = data.get("existing_order_id")
    if existing_id:
        return str(existing_id)
    return None


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

    # Statuses that indicate a stop order is still active (not filled/cancelled).
    # Includes HELD for OTO child legs and SUSPENDED for halted stocks.
    _ACTIVE_STOP_STATUSES = {
        OrderStatus.NEW,
        OrderStatus.HELD,
        OrderStatus.ACCEPTED,
        OrderStatus.PENDING_NEW,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_REPLACE,
        OrderStatus.SUSPENDED,
    }

    def get_open_stop_orders(self, symbol: str) -> list[dict]:
        """Get active stop/stop_limit sell orders for a symbol.

        Queries ALL statuses and filters client-side so that OTO child
        legs (status=HELD) are included alongside standalone stops.
        """
        try:
            orders = self._client.get_orders(
                GetOrdersRequest(
                    symbols=[symbol],
                    side=OrderSide.SELL,
                    status=QueryOrderStatus.ALL,
                )
            )
            return [
                {"order_id": str(o.id), "stop_price": float(o.stop_price)}
                for o in orders
                if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
                and o.stop_price is not None
                and o.status in self._ACTIVE_STOP_STATUSES
            ]
        except Exception as e:
            log.warning("get_stop_orders_failed", symbol=symbol, error=str(e))
            return []

    def _submit_market_order(self, symbol: str, qty: int, side: OrderSide):
        """Submit a market order, return the order object."""
        return self._client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )

    def set_stop_loss(self, symbol: str, qty: int, entry_price: Decimal) -> dict | None:
        """Set a stop-loss order. Returns order info or None on failure.

        If Alpaca rejects because shares are already held by an existing
        stop order, returns the existing order ID with ``adopted=True``
        so the caller can track it instead of giving up.
        """
        stop_price = round(float(self._risk.calculate_stop_loss(entry_price)), 2)
        try:
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
            return {"stop_order_id": str(stop_order.id), "stop_price": stop_price}
        except Exception as e:
            error_str = str(e)
            log.error("stop_loss_failed", symbol=symbol, error=error_str)
            existing_id = _parse_existing_stop_from_error(error_str)
            if existing_id:
                log.info(
                    "existing_stop_adopted_from_error",
                    symbol=symbol,
                    order_id=existing_id,
                )
                return {
                    "stop_order_id": existing_id,
                    "stop_price": None,
                    "adopted": True,
                }
            return None

    def buy(
        self,
        symbol: str,
        price: Decimal,
        qty: int | None = None,
        market_time=None,
    ) -> dict | None:
        """Buy shares with automatic stop-loss via OTO order.

        Uses Alpaca's One-Triggers-Other (OTO) order class to atomically
        submit a market buy with an attached stop-loss sell. This avoids
        the wash trade rejection that occurs when placing separate orders.
        """
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

        stop_price = round(float(self._risk.calculate_stop_loss(price)), 2)
        order = self._client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.OTO,
                stop_loss=StopLossRequest(stop_price=stop_price),
            )
        )
        log.info("buy_executed", symbol=symbol, qty=qty, order_id=str(order.id))

        self._risk.open_positions += 1
        result = {
            "order_id": str(order.id),
            "symbol": symbol,
            "qty": qty,
        }

        # Extract stop leg order ID from OTO response
        if order.legs:
            result["stop_order_id"] = str(order.legs[0].id)
            result["stop_price"] = stop_price
            log.info("stop_loss_set", symbol=symbol, stop_price=stop_price)
        else:
            log.warning("oto_no_stop_leg", symbol=symbol)

        return result

    def sell(self, symbol: str, qty: int) -> dict | None:
        """Sell shares (market order)."""
        order = self._submit_market_order(symbol, qty, OrderSide.SELL)
        log.info("sell_executed", symbol=symbol, qty=qty, order_id=str(order.id))
        self._risk.open_positions = max(0, self._risk.open_positions - 1)
        return {"order_id": str(order.id), "symbol": symbol, "qty": qty}

    def update_stop_loss(
        self, symbol: str, qty: int, new_stop_price: Decimal, old_order_id: str
    ) -> dict | None:
        """Replace a GTC stop-loss order with updated trailing stop price."""
        try:
            self._client.cancel_order_by_id(old_order_id)
        except Exception as e:
            log.warning(
                "cancel_stop_failed", symbol=symbol, order_id=old_order_id, error=str(e)
            )

        stop_price = round(float(new_stop_price), 2)
        try:
            stop_order = self._client.submit_order(
                StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_price,
                )
            )
            log.info("trailing_stop_updated", symbol=symbol, new_stop=stop_price)
            return {"stop_order_id": str(stop_order.id), "stop_price": stop_price}
        except Exception as e:
            log.error("trailing_stop_update_failed", symbol=symbol, error=str(e))
            return None

    def cancel_stop_loss(self, order_id: str) -> None:
        """Cancel a GTC stop-loss order (cleanup when position sold)."""
        try:
            self._client.cancel_order_by_id(order_id)
            log.info("stop_loss_cancelled", order_id=order_id)
        except Exception as e:
            log.warning("cancel_stop_failed", order_id=order_id, error=str(e))

    def get_bars(
        self, symbol: str, timeframe: TimeFrame, start: str, end: str | None = None
    ):
        """Get historical price bars for analysis."""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        return self._data_client.get_stock_bars(request)


def df_to_bar_dicts(df, window: int | None = None) -> list[dict]:
    """Convert an Alpaca OHLCV DataFrame to a list of bar dicts.

    Handles both single-index (timestamp) and MultiIndex (symbol, timestamp)
    DataFrames from the Alpaca SDK.

    Args:
        df: pandas DataFrame with OHLCV columns.
        window: if set, only return the last N bars.
    """
    subset = df.iloc[-window:] if window else df
    bars = []
    for idx, row in subset.iterrows():
        # Alpaca returns MultiIndex (symbol, timestamp) for single-symbol queries
        ts = idx[1] if isinstance(idx, tuple) else idx
        bars.append(
            {
                "date": ts.strftime("%Y-%m-%d")
                if hasattr(ts, "strftime")
                else str(ts)[:10],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
        )
    return bars
