"""Tests for EMA momentum strategy."""

from datetime import date

import pytest

from claude_trader.analyst import MultiAgentAnalysis, Signal
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


class TestShouldBuy:
    def test_buy_on_ema_crossover_with_positive_sentiment(self):
        """Price crosses above EMA with positive analysis -> buy."""
        strategy = EMAMomentumStrategy(ema_period=3)
        # Build prices where prev was below EMA and current crosses above.
        # EMA(3) of [10, 9, 8, 7] gives declining EMA, then a jump to 15.
        prices = [10.0, 9.0, 8.0, 7.0, 15.0]
        analysis = MultiAgentAnalysis(
            symbol="AAPL",
            combined_score=0.5,
            final_signal=Signal.BUY,
            agreement_count=3,
            reasoning="test",
        )
        assert strategy.should_buy("AAPL", 15.0, prices, analysis) is True

    def test_no_buy_below_ema(self):
        strategy = EMAMomentumStrategy(ema_period=3)
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        assert strategy.should_buy("AAPL", 5.0, prices) is False

    def test_no_buy_above_ema_without_crossover(self):
        """Price above EMA for >5 bars (beyond lookback) - no buy."""
        strategy = EMAMomentumStrategy(ema_period=3)
        # Steadily rising for 8 bars - crossover happened well beyond 5-bar lookback
        prices = [5.0, 6.0, 7.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        analysis = MultiAgentAnalysis(
            symbol="AAPL",
            combined_score=0.8,
            final_signal=Signal.STRONG_BUY,
            agreement_count=4,
            reasoning="test",
        )
        assert strategy.should_buy("AAPL", 16.0, prices, analysis) is False

    def test_buy_on_recent_crossover_within_lookback(self):
        """Crossover 3 bars ago, price still above EMA - should buy."""
        strategy = EMAMomentumStrategy(ema_period=3)
        # EMA(3) starts from index 2. Prices below EMA, then cross 3 bars ago.
        prices = [10.0, 9.0, 8.0, 7.0, 12.0, 13.0, 14.0]
        analysis = MultiAgentAnalysis(
            symbol="AAPL",
            combined_score=0.5,
            final_signal=Signal.BUY,
            agreement_count=3,
            reasoning="test",
        )
        assert strategy.should_buy("AAPL", 14.0, prices, analysis) is True

    def test_no_buy_stale_crossover_beyond_lookback(self):
        """Crossover 7+ bars ago (beyond 5-bar lookback) - no buy."""
        strategy = EMAMomentumStrategy(ema_period=3)
        # Cross happens early, then 7+ bars of steady rise above EMA
        prices = [10.0, 9.0, 8.0, 7.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
        analysis = MultiAgentAnalysis(
            symbol="AAPL",
            combined_score=0.8,
            final_signal=Signal.STRONG_BUY,
            agreement_count=4,
            reasoning="test",
        )
        assert strategy.should_buy("AAPL", 18.0, prices, analysis) is False

    def test_no_buy_low_sentiment(self):
        """Even with EMA crossover, low sentiment blocks buy."""
        strategy = EMAMomentumStrategy(ema_period=3)
        prices = [10.0, 9.0, 8.0, 7.0, 15.0]
        analysis = MultiAgentAnalysis(
            symbol="AAPL",
            combined_score=-0.5,
            final_signal=Signal.SELL,
            agreement_count=3,
            reasoning="test",
        )
        assert strategy.should_buy("AAPL", 15.0, prices, analysis) is False


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
