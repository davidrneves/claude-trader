"""Backtest engine - replay historical bars through EMA momentum strategy.

Simulates trades using the same strategy and risk parameters as live trading,
but against historical OHLCV data. No API calls during simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from claude_trader.performance import PerformanceTracker
from claude_trader.strategy import EMAMomentumStrategy

if TYPE_CHECKING:
    from concurrent.futures import Executor

log = structlog.get_logger()


@dataclass
class BacktestConfig:
    """Parameters for a backtest run."""

    symbols: list[str]
    start_date: str
    end_date: str
    initial_capital: Decimal = Decimal("100000")
    ema_period: int = 20
    stop_loss_pct: Decimal = Decimal("0.08")
    trailing_stop_pct: Decimal = Decimal("0.05")
    max_position_pct: Decimal = Decimal("0.02")


@dataclass
class BacktestTrade:
    """A single trade executed during backtest."""

    symbol: str
    side: str
    date: str
    price: float
    qty: int
    pnl: float | None = None


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    config: BacktestConfig
    trades: list[BacktestTrade]
    equity_curve: list[float]
    total_return_pct: float
    sharpe_ratio: float | None
    max_drawdown_pct: float
    win_rate: float
    trade_count: int
    total_pnl: float


@dataclass
class _SimPosition:
    """Internal: tracks an open simulated position."""

    symbol: str
    entry_price: float
    qty: int
    entry_date: str
    stop_loss: float
    trailing_stop_floor: float


class BacktestEngine:
    """Replay historical bars through EMA momentum strategy with stop management."""

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._strategy = EMAMomentumStrategy(ema_period=config.ema_period)
        self._positions: dict[str, _SimPosition] = {}
        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[float] = []
        self._cash: float = float(config.initial_capital)

    def run(self, bars: dict[str, list[dict]]) -> BacktestResult:
        """Run backtest over historical bars.

        Args:
            bars: mapping of symbol -> list of bar dicts sorted by date.
                  Each bar must have: date, open, high, low, close, volume.

        Returns:
            BacktestResult with trades, equity curve, and computed metrics.
        """
        # Build date-indexed lookups per symbol
        bars_by_date: dict[str, dict[str, dict]] = {}
        all_dates: set[str] = set()
        for symbol, symbol_bars in bars.items():
            bars_by_date[symbol] = {}
            for bar in symbol_bars:
                d = bar["date"]
                bars_by_date[symbol][d] = bar
                all_dates.add(d)

        sorted_dates = sorted(all_dates)

        for current_date in sorted_dates:
            self._strategy.reset_daily()

            for symbol in self._config.symbols:
                if current_date not in bars_by_date.get(symbol, {}):
                    continue

                bar = bars_by_date[symbol][current_date]
                closes = [b["close"] for b in bars[symbol] if b["date"] <= current_date]

                # Update trailing stops for open positions
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    new_floor = bar["close"] * (
                        1 - float(self._config.trailing_stop_pct)
                    )
                    if new_floor > pos.trailing_stop_floor:
                        pos.trailing_stop_floor = new_floor

                # Check stop-loss and trailing-stop exits
                if symbol in self._positions:
                    if self._check_stops(symbol, bar):
                        continue

                # Check EMA sell signal
                if symbol in self._positions:
                    if self._strategy.should_sell(symbol, bar["close"], closes):
                        self._close_position(
                            symbol, bar["close"], current_date, "ema_sell"
                        )
                        continue

                # Check EMA buy signal for symbols we don't hold
                if symbol not in self._positions:
                    if self._strategy.should_buy(
                        symbol, bar["close"], closes, analysis=None
                    ):
                        self._open_position(symbol, bar["close"], current_date)
                        self._strategy.record_trade(symbol)

            # Record end-of-day equity
            equity = self._cash
            for sym, pos in self._positions.items():
                latest_date = current_date
                if sym in bars_by_date and latest_date in bars_by_date[sym]:
                    equity += pos.qty * bars_by_date[sym][latest_date]["close"]
                else:
                    equity += pos.qty * pos.entry_price
            self._equity_curve.append(equity)

        # Force-close remaining positions at last available price
        for symbol in list(self._positions.keys()):
            last_bar = bars[symbol][-1] if bars.get(symbol) else None
            if last_bar:
                self._close_position(
                    symbol, last_bar["close"], last_bar["date"], "force_close"
                )

        return self._build_result()

    def _check_stops(self, symbol: str, bar: dict) -> bool:
        """Check if stop-loss or trailing-stop triggers on this bar.

        Uses bar's low price to detect intra-day stop hits.
        Returns True if position was closed.
        """
        pos = self._positions[symbol]
        low = bar["low"]

        if low <= pos.stop_loss:
            self._close_position(symbol, pos.stop_loss, bar["date"], "stop_loss")
            return True

        if low <= pos.trailing_stop_floor:
            self._close_position(
                symbol, pos.trailing_stop_floor, bar["date"], "trailing_stop"
            )
            return True

        return False

    def _open_position(self, symbol: str, price: float, date: str) -> None:
        """Open a new position respecting max_position_pct sizing."""
        max_cost = self._cash * float(self._config.max_position_pct)
        if price <= 0:
            return
        qty = int(max_cost / price)
        if qty < 1:
            qty = 1

        cost = qty * price
        if cost > self._cash:
            return

        self._cash -= cost
        stop_loss = price * (1 - float(self._config.stop_loss_pct))
        trailing_floor = price * (1 - float(self._config.trailing_stop_pct))

        self._positions[symbol] = _SimPosition(
            symbol=symbol,
            entry_price=price,
            qty=qty,
            entry_date=date,
            stop_loss=stop_loss,
            trailing_stop_floor=trailing_floor,
        )

        self._trades.append(
            BacktestTrade(
                symbol=symbol,
                side="buy",
                date=date,
                price=price,
                qty=qty,
            )
        )

        log.debug("backtest_buy", symbol=symbol, price=price, qty=qty, date=date)

    def _close_position(
        self, symbol: str, price: float, date: str, reason: str
    ) -> None:
        """Close an open position and record PnL."""
        pos = self._positions.pop(symbol)
        pnl = (price - pos.entry_price) * pos.qty
        self._cash += pos.qty * price

        self._trades.append(
            BacktestTrade(
                symbol=symbol,
                side="sell",
                date=date,
                price=price,
                qty=pos.qty,
                pnl=pnl,
            )
        )

        log.debug(
            "backtest_sell",
            symbol=symbol,
            price=price,
            qty=pos.qty,
            pnl=round(pnl, 2),
            reason=reason,
            date=date,
        )

    def _build_result(self) -> BacktestResult:
        """Compute final metrics and return BacktestResult."""
        initial = float(self._config.initial_capital)
        final_equity = self._equity_curve[-1] if self._equity_curve else initial

        total_return = PerformanceTracker._compute_cumulative_return(
            initial, final_equity
        )
        max_dd = PerformanceTracker._compute_max_drawdown(
            self._equity_curve if self._equity_curve else [initial],
        )

        # Compute Sharpe from daily equity returns
        daily_returns = []
        for i in range(1, len(self._equity_curve)):
            prev = self._equity_curve[i - 1]
            if prev > 0:
                daily_returns.append((self._equity_curve[i] - prev) / prev)
        sharpe = PerformanceTracker._compute_sharpe(daily_returns)

        # Win rate from closed trades (sells with pnl)
        closed = [t for t in self._trades if t.side == "sell" and t.pnl is not None]
        wins = sum(1 for t in closed if t.pnl > 0)
        win_rate = (wins / len(closed) * 100) if closed else 0.0
        total_pnl = sum(t.pnl for t in closed if t.pnl is not None)
        trade_count = len(closed)

        return BacktestResult(
            config=self._config,
            trades=self._trades,
            equity_curve=self._equity_curve,
            total_return_pct=round(total_return, 4),
            sharpe_ratio=sharpe,
            max_drawdown_pct=round(max_dd, 4),
            win_rate=round(win_rate, 2),
            trade_count=trade_count,
            total_pnl=round(total_pnl, 2),
        )

    @staticmethod
    def fetch_backtest_data(
        executor: Executor, symbols: list[str], start: str, end: str
    ) -> dict[str, list[dict]]:
        """Fetch historical bars via executor for each symbol.

        Args:
            executor: an object with get_bars(symbol, start, end) method.
            symbols: list of ticker symbols.
            start: start date string (YYYY-MM-DD).
            end: end date string (YYYY-MM-DD).

        Returns:
            dict mapping symbol to list of bar dicts.
        """
        bars: dict[str, list[dict]] = {}
        for symbol in symbols:
            bars[symbol] = executor.get_bars(symbol, start, end)
        return bars
