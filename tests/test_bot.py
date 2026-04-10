"""Tests for the trading bot orchestrator."""

from datetime import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from claude_trader.analyst import MultiAgentAnalysis, Signal
from claude_trader.bot import TradingBot


@pytest.fixture
def bot(mock_settings):
    with (
        patch("claude_trader.bot.AlpacaExecutor"),
        patch("claude_trader.bot.NewsFeed"),
        patch("claude_trader.bot.TelegramNotifier"),
        patch("claude_trader.bot.ObsidianLogger"),
        patch("claude_trader.bot.TradeLogger"),
    ):
        return TradingBot(mock_settings)


class TestRunOnceMarketClosed:
    def test_returns_market_closed(self, bot):
        bot._is_market_open = MagicMock(return_value=False)
        result = bot.run_once()
        assert "market_closed" in result["actions"]
        assert result["trades"] == []

    def test_does_not_call_executor(self, bot):
        bot._is_market_open = MagicMock(return_value=False)
        bot.run_once()
        bot._executor.get_account.assert_not_called()


class TestRunOnceSellSignals:
    def test_executes_sell_on_signal(self, bot):
        bot._is_market_open = MagicMock(return_value=True)
        bot._get_market_time = MagicMock(return_value=time(11, 0))
        bot._executor.get_account.return_value = {
            "equity": Decimal("10000"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10000"),
        }
        bot._executor.get_positions.return_value = [
            {
                "symbol": "AAPL",
                "qty": 10,
                "avg_entry": Decimal("100"),
                "current_price": Decimal("90"),
                "unrealized_pnl": Decimal("-100"),
                "side": "long",
            },
        ]
        bot._strategy.should_sell = MagicMock(return_value=True)
        bot._executor.sell.return_value = {
            "order_id": "sell-1",
            "symbol": "AAPL",
            "qty": 10,
        }
        bot._get_price_bars = MagicMock(return_value=([90.0], []))

        result = bot.run_once()
        assert any("sold" in a for a in result["actions"])


class TestRunOnceBuySignals:
    def test_no_signals_when_strategy_says_no(self, bot):
        bot._is_market_open = MagicMock(return_value=True)
        bot._get_market_time = MagicMock(return_value=time(11, 0))
        bot._executor.get_account.return_value = {
            "equity": Decimal("10000"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10000"),
        }
        bot._executor.get_positions.return_value = []
        bot._strategy.should_sell = MagicMock(return_value=False)
        bot._strategy.should_buy = MagicMock(return_value=False)
        bot._get_price_bars = MagicMock(return_value=([100.0], []))

        result = bot.run_once()
        assert "no_signals" in result["actions"]
        assert result["trades"] == []


class TestRunOnceExceptionHandling:
    def test_exception_in_symbol_does_not_crash(self, bot):
        bot._is_market_open = MagicMock(return_value=True)
        bot._get_market_time = MagicMock(return_value=time(11, 0))
        bot._executor.get_account.return_value = {
            "equity": Decimal("10000"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10000"),
        }
        bot._executor.get_positions.return_value = []
        bot._get_price_bars = MagicMock(side_effect=Exception("API down"))

        # Should not raise
        result = bot.run_once()
        assert isinstance(result, dict)


class TestDfToBarDicts:
    def test_extracts_correct_fields(self):
        import pandas as pd

        from claude_trader.executor import df_to_bar_dicts

        df = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [100, 200],
            },
            index=["2024-01-01", "2024-01-02"],
        )

        result = df_to_bar_dicts(df, window=2)
        assert len(result) == 2
        assert result[0]["open"] == 1.0
        assert result[1]["close"] == 2.2
        assert "date" in result[0]


class TestExtractAgentScores:
    def test_none_analysis_returns_empty(self):
        assert TradingBot._extract_agent_scores(None) == {}

    def test_extracts_scores(self):
        from claude_trader.analyst import SentimentResult, TechnicalResult

        analysis = MultiAgentAnalysis(
            symbol="AAPL",
            sentiment=SentimentResult(
                score=0.5, signal=Signal.BUY, reasoning="ok", key_factors=[]
            ),
            technical=TechnicalResult(
                score=0.6, signal=Signal.BUY, pattern="up", reasoning="ok"
            ),
            combined_score=0.55,
            final_signal=Signal.BUY,
            agreement_count=3,
            reasoning="test",
        )
        scores = TradingBot._extract_agent_scores(analysis)
        assert scores["combined"] == 0.55
        assert scores["sentiment"] == 0.5
        assert scores["agreement"] == 3
