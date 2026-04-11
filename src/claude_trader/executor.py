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
        """Set a stop-loss order. Returns order info or None on failure."""
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
            log.error("stop_loss_failed", symbol=symbol, error=str(e))
            return None

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

        order = self._submit_market_order(symbol, qty, OrderSide.BUY)
        log.info("buy_executed", symbol=symbol, qty=qty, order_id=str(order.id))

        stop_result = self.set_stop_loss(symbol, qty, price)
        if not stop_result:
            log.warning("buy_without_stop_loss", symbol=symbol, qty=qty)

        self._risk.open_positions += 1
        result = {
            "order_id": str(order.id),
            "symbol": symbol,
            "qty": qty,
        }
        if stop_result:
            result["stop_order_id"] = stop_result["stop_order_id"]
            result["stop_price"] = stop_result["stop_price"]
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
