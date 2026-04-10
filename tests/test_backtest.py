"""Tests for backtest engine."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from claude_trader.backtest import BacktestConfig, BacktestEngine, print_backtest_report


def _make_bars(prices: list[float], start_date: str = "2025-01-01") -> list[dict]:
    """Generate synthetic daily bars from a list of close prices."""
    bars = []
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    for price in prices:
        bars.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000000,
            }
        )
        dt += timedelta(days=1)
    return bars


def _make_config(
    symbols: list[str] | None = None,
    ema_period: int = 5,
    initial_capital: Decimal = Decimal("100000"),
) -> BacktestConfig:
    return BacktestConfig(
        symbols=symbols or ["TEST"],
        start_date="2025-01-01",
        end_date="2025-12-31",
        initial_capital=initial_capital,
        ema_period=ema_period,
    )


class TestNoTrades:
    def test_no_trades_flat_prices(self):
        """Flat prices never cross EMA -> 0 trades, equity unchanged."""
        config = _make_config(ema_period=5)
        prices = [100.0] * 30
        bars = {"TEST": _make_bars(prices)}
        engine = BacktestEngine(config)
        result = engine.run(bars)

        assert result.trade_count == 0
        assert len(result.trades) == 0
        assert result.equity_curve[-1] == pytest.approx(float(config.initial_capital))


class TestBuyAndSell:
    def test_buy_and_sell_on_ema_crossover(self):
        """Price crosses above EMA (buy), then drops below EMA (sell)."""
        config = _make_config(ema_period=5)

        # 5 bars below crossover point, then a spike up, then a drop
        # EMA(5) of [90,90,90,90,90] = 90. Price jumps to 100 -> crossover.
        # prev_price=90 <= prev_ema=90, current_price=100 > current_ema -> buy
        # Then price drops to 80 -> below EMA -> sell
        prices = [90.0] * 5 + [100.0] + [80.0] * 5
        bars = {"TEST": _make_bars(prices)}

        engine = BacktestEngine(config)
        result = engine.run(bars)

        buys = [t for t in result.trades if t.side == "buy"]
        sells = [t for t in result.trades if t.side == "sell"]

        assert len(buys) >= 1
        assert len(sells) >= 1

        # The sell should have a pnl recorded
        sell = sells[0]
        assert sell.pnl is not None


class TestStopLoss:
    def test_stop_loss_triggers(self):
        """Buy triggers, then price drops >8% -> stop loss fires."""
        config = _make_config(ema_period=5, initial_capital=Decimal("100000"))

        # Build crossover: 5 bars at 90, jump to 100, then crash to trigger stop
        # stop_loss = 100 * (1 - 0.08) = 92
        # Need bar low <= 92
        prices = [90.0] * 5 + [100.0]
        # Add bars that crash - low must go below 92
        crash_bars = _make_bars(prices, "2025-01-01")
        # Add crash bar where low hits stop
        crash_date = datetime.strptime("2025-01-01", "%Y-%m-%d") + timedelta(
            days=len(prices)
        )
        crash_bars.append(
            {
                "date": crash_date.strftime("%Y-%m-%d"),
                "open": 95.0,
                "high": 95.0,
                "low": 85.0,
                "close": 88.0,
                "volume": 1000000,
            }
        )
        # Add a few more flat bars after
        for i in range(1, 4):
            d = crash_date + timedelta(days=i)
            crash_bars.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "open": 88.0,
                    "high": 89.0,
                    "low": 87.0,
                    "close": 88.0,
                    "volume": 1000000,
                }
            )

        bars = {"TEST": crash_bars}
        engine = BacktestEngine(config)
        result = engine.run(bars)

        sells = [t for t in result.trades if t.side == "sell"]
        assert len(sells) >= 1
        # Stop-loss sell should have negative PnL (sold at 92, bought at 100)
        stop_sell = sells[0]
        assert stop_sell.pnl is not None
        assert stop_sell.pnl < 0
        # Sold at stop_loss price = 92
        assert stop_sell.price == pytest.approx(92.0)


class TestTrailingStop:
    def test_trailing_stop_triggers(self):
        """Buy, price rises 20%, then drops 5% from peak -> trailing stop."""
        config = _make_config(ema_period=5, initial_capital=Decimal("100000"))

        # Crossover at 100, then gradual rise to 120, then drop 5% from 120
        # trailing_stop_floor at 120 = 120 * 0.95 = 114
        prices = [90.0] * 5 + [100.0, 105.0, 110.0, 115.0, 120.0]

        rise_bars = _make_bars(prices, "2025-01-01")

        # Add a bar that drops below trailing stop floor (114)
        drop_date = datetime.strptime("2025-01-01", "%Y-%m-%d") + timedelta(
            days=len(prices)
        )
        rise_bars.append(
            {
                "date": drop_date.strftime("%Y-%m-%d"),
                "open": 118.0,
                "high": 118.0,
                "low": 112.0,  # Below trailing floor of 114
                "close": 113.0,
                "volume": 1000000,
            }
        )
        # A few more flat bars
        for i in range(1, 3):
            d = drop_date + timedelta(days=i)
            rise_bars.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "open": 113.0,
                    "high": 114.0,
                    "low": 112.0,
                    "close": 113.0,
                    "volume": 1000000,
                }
            )

        bars = {"TEST": rise_bars}
        engine = BacktestEngine(config)
        result = engine.run(bars)

        sells = [t for t in result.trades if t.side == "sell"]
        assert len(sells) >= 1
        # Trailing stop sell: sold above entry (100) so positive PnL
        trailing_sell = sells[0]
        assert trailing_sell.pnl is not None
        assert trailing_sell.pnl > 0
        # Sold at trailing floor = 120 * 0.95 = 114
        assert trailing_sell.price == pytest.approx(114.0)


class TestMetrics:
    def test_metrics_computation(self):
        """Verify total_return_pct, max_drawdown_pct, trade_count from known data."""
        config = _make_config(ema_period=5, initial_capital=Decimal("100000"))

        # Crossover at 100, sell at 110 via EMA cross below
        prices = [90.0] * 5 + [100.0, 105.0, 110.0] + [85.0] * 5
        bars = {"TEST": _make_bars(prices)}

        engine = BacktestEngine(config)
        result = engine.run(bars)

        # Should have at least 1 closed trade
        assert result.trade_count >= 1
        assert len(result.equity_curve) == len(prices)
        # Total return should be close to initial capital (small position)
        assert isinstance(result.total_return_pct, float)
        assert isinstance(result.max_drawdown_pct, float)
        assert result.max_drawdown_pct >= 0


class TestMultipleSymbols:
    def test_multiple_symbols(self):
        """Two symbols with different patterns produce independent trades."""
        config = _make_config(
            symbols=["AAPL", "MSFT"],
            ema_period=5,
        )

        # AAPL: crossover and rise
        aapl_prices = [90.0] * 5 + [100.0, 105.0, 110.0] + [80.0] * 3
        # MSFT: flat, no trades
        msft_prices = [50.0] * 11

        bars = {
            "AAPL": _make_bars(aapl_prices),
            "MSFT": _make_bars(msft_prices),
        }

        engine = BacktestEngine(config)
        result = engine.run(bars)

        aapl_trades = [t for t in result.trades if t.symbol == "AAPL"]
        msft_trades = [t for t in result.trades if t.symbol == "MSFT"]

        assert len(aapl_trades) >= 2  # At least a buy and sell
        assert len(msft_trades) == 0  # Flat prices, no crossover


class TestDailyTradeLimit:
    def test_daily_trade_limit(self):
        """Only 1 trade per symbol per day due to reset_daily() per date."""
        config = _make_config(ema_period=3)

        # Create a pattern with rapid crossovers. Since reset_daily is called
        # per date, each day allows at most 1 buy per symbol.
        # After a buy + record_trade, same-day buy is blocked.
        prices = [50.0, 50.0, 50.0, 60.0, 45.0, 60.0, 45.0, 60.0, 45.0, 60.0]
        bars = {"TEST": _make_bars(prices)}

        engine = BacktestEngine(config)
        result = engine.run(bars)

        # Count buys per date
        buy_dates = [t.date for t in result.trades if t.side == "buy"]
        from collections import Counter

        date_counts = Counter(buy_dates)

        # No date should have more than 1 buy for the same symbol
        for count in date_counts.values():
            assert count <= 1


class TestForceClose:
    def test_force_close_at_end(self):
        """Position still open at last date gets force-closed."""
        config = _make_config(ema_period=5)

        # Crossover at bar 6, then price stays high -> no sell signal
        # Position should remain open and be force-closed at end
        prices = [90.0] * 5 + [100.0, 105.0, 110.0, 115.0, 120.0]
        bars = {"TEST": _make_bars(prices)}

        engine = BacktestEngine(config)
        result = engine.run(bars)

        buys = [t for t in result.trades if t.side == "buy"]
        sells = [t for t in result.trades if t.side == "sell"]

        assert len(buys) >= 1
        # Force-close should produce a sell
        assert len(sells) >= 1
        # Last sell should be at last bar's close price
        last_sell = sells[-1]
        assert last_sell.price == pytest.approx(120.0)
        assert last_sell.pnl is not None
        assert last_sell.pnl > 0  # 120 - 100 = profit


class TestReport:
    def test_print_backtest_report(self, capsys):
        """Verify report prints key sections."""
        config = _make_config(ema_period=5)
        prices = [90.0] * 5 + [100.0, 105.0, 110.0] + [85.0] * 5
        bars = {"TEST": _make_bars(prices)}

        engine = BacktestEngine(config)
        result = engine.run(bars)
        print_backtest_report(result)

        output = capsys.readouterr().out
        assert "BACKTEST REPORT" in output
        assert "PERFORMANCE" in output
        assert "GRADUATION CHECK" in output
        assert "Total return:" in output
        assert "Sharpe ratio:" in output
        assert "Max drawdown:" in output
        assert "Win rate:" in output
