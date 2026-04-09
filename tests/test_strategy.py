"""Tests for EMA momentum strategy."""

from datetime import date

import pytest

from claude_trader.strategy import EMAMomentumStrategy, calculate_ema


class TestCalculateEma:
    def test_insufficient_data(self):
        assert calculate_ema([1, 2], 5) == []

    def test_exact_period(self):
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        ema = calculate_ema(prices, 5)
        assert len(ema) == 1
        assert ema[0] == pytest.approx(12.0)

    def test_ema_length(self):
        prices = list(range(1, 31))
        ema = calculate_ema(prices, 20)
        assert len(ema) == 11  # 30 - 20 + 1

    def test_ema_follows_trend(self):
        prices = list(range(1, 31))
        ema = calculate_ema(prices, 5)
        for i in range(1, len(ema)):
            assert ema[i] > ema[i - 1]


class TestDailyTradeLimit:
    def test_blocks_second_trade_same_symbol(self):
        strategy = EMAMomentumStrategy(ema_period=3)
        strategy.record_trade("AAPL")
        prices = [10.0, 11.0, 12.0, 9.0, 13.0]
        assert strategy.should_buy("AAPL", 13.0, prices) is False

    def test_allows_different_symbol(self):
        strategy = EMAMomentumStrategy(ema_period=3)
        strategy.record_trade("AAPL")
        assert strategy._daily_trades.get("MSFT", 0) == 0


class TestDailyReset:
    def test_reset_clears_trades(self):
        strategy = EMAMomentumStrategy(ema_period=3)
        strategy.record_trade("AAPL")
        strategy.reset_daily()
        assert strategy._daily_trades == {}

    def test_auto_reset_on_date_change(self):
        strategy = EMAMomentumStrategy(ema_period=3)
        strategy.record_trade("AAPL")
        assert strategy._daily_trades.get("AAPL") == 1

        # Simulate date change
        strategy._last_reset_date = date(2020, 1, 1)
        strategy._ensure_daily_reset()
        assert strategy._daily_trades == {}
        assert strategy._last_reset_date == date.today()


class TestShouldSell:
    def test_sell_below_ema(self):
        strategy = EMAMomentumStrategy(ema_period=3)
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        assert strategy.should_sell("AAPL", 8.0, prices) is True

    def test_no_sell_above_ema(self):
        strategy = EMAMomentumStrategy(ema_period=3)
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        assert strategy.should_sell("AAPL", 20.0, prices) is False

    def test_no_sell_insufficient_data(self):
        strategy = EMAMomentumStrategy(ema_period=20)
        prices = [10.0, 11.0]
        assert strategy.should_sell("AAPL", 5.0, prices) is False
