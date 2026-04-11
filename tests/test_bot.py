"""Tests for the trading bot orchestrator."""

from datetime import date, time, timedelta
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
        patch("claude_trader.bot.BotStateStore"),
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


class TestDrawdownTracking:
    def test_update_portfolio_tracks_peak_equity(self, bot):
        bot._executor.get_account.return_value = {
            "equity": Decimal("10000"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10000"),
        }
        bot._executor.get_positions.return_value = []
        bot._update_portfolio()
        assert bot._peak_equity == Decimal("10000")

        bot._executor.get_account.return_value = {
            "equity": Decimal("10500"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10500"),
        }
        bot._update_portfolio()
        assert bot._peak_equity == Decimal("10500")

    def test_update_portfolio_calls_record_drawdown(self, bot):
        # Set peak first
        bot._executor.get_account.return_value = {
            "equity": Decimal("10000"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10000"),
        }
        bot._executor.get_positions.return_value = []
        bot._update_portfolio()

        # Now equity drops to 9000 -> 10% drawdown
        bot._executor.get_account.return_value = {
            "equity": Decimal("9000"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("9000"),
        }
        bot._risk.record_drawdown = MagicMock()
        bot._update_portfolio()
        bot._risk.record_drawdown.assert_called_with(Decimal("0.1"))

    def test_no_drawdown_when_equity_rises(self, bot):
        bot._executor.get_account.return_value = {
            "equity": Decimal("10000"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10000"),
        }
        bot._executor.get_positions.return_value = []
        bot._update_portfolio()

        bot._executor.get_account.return_value = {
            "equity": Decimal("10500"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10500"),
        }
        bot._risk.record_drawdown = MagicMock()
        bot._update_portfolio()
        bot._risk.record_drawdown.assert_called_with(Decimal("0"))


class TestDailyReset:
    def test_reset_daily_calls_both(self, bot):
        bot._risk.reset_daily = MagicMock()
        bot._strategy.reset_daily = MagicMock()
        bot.reset_daily()
        bot._risk.reset_daily.assert_called_once()
        bot._strategy.reset_daily.assert_called_once()

    def test_run_once_resets_on_date_change(self, bot):
        yesterday = date.today() - timedelta(days=1)
        bot._last_trading_date = yesterday
        bot._is_market_open = MagicMock(return_value=True)
        bot._get_market_time = MagicMock(return_value=time(11, 0))
        bot._executor.get_account.return_value = {
            "equity": Decimal("10000"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10000"),
        }
        bot._executor.get_positions.return_value = []
        bot._get_price_bars = MagicMock(return_value=([], []))
        bot._risk.reset_daily = MagicMock()
        bot._strategy.reset_daily = MagicMock()

        bot.run_once()
        bot._risk.reset_daily.assert_called_once()
        bot._strategy.reset_daily.assert_called_once()


class TestTrailingStops:
    def test_trailing_stop_floor_updates_on_price_rise(self, bot):
        bot._trailing_stops["AAPL"] = {
            "floor": Decimal("95"),
            "stop_order_id": "stop-old",
        }
        bot._risk.calculate_trailing_stop = MagicMock(return_value=Decimal("104.50"))
        bot._executor.update_stop_loss = MagicMock(
            return_value={"stop_order_id": "stop-new", "stop_price": 104.50}
        )
        bot._strategy.should_sell = MagicMock(return_value=False)
        bot._get_price_bars = MagicMock(return_value=([110.0], []))

        positions = [
            {
                "symbol": "AAPL",
                "qty": 10,
                "avg_entry": Decimal("100"),
                "current_price": Decimal("110"),
                "unrealized_pnl": Decimal("100"),
                "side": "long",
            }
        ]
        bot._scan_and_execute_sells(
            {
                "actions": [],
                "analyses": [],
                "trades": [],
            },
            positions,
        )

        assert bot._trailing_stops["AAPL"]["floor"] == Decimal("104.50")
        bot._executor.update_stop_loss.assert_called_once_with(
            "AAPL", 10, Decimal("104.50"), "stop-old"
        )

    def test_trailing_stop_triggers_sell(self, bot):
        bot._trailing_stops["AAPL"] = {
            "floor": Decimal("104.50"),
            "stop_order_id": "stop-1",
        }
        # Return same floor (price dropped, floor doesn't move)
        bot._risk.calculate_trailing_stop = MagicMock(return_value=Decimal("98.80"))
        bot._strategy.should_sell = MagicMock(return_value=False)
        bot._get_price_bars = MagicMock(return_value=([104.0], []))
        bot._executor.sell = MagicMock(
            return_value={"order_id": "sell-1", "symbol": "AAPL", "qty": 10}
        )
        bot._executor.cancel_stop_loss = MagicMock()
        bot._risk.record_trade_result = MagicMock()
        bot._risk.record_daily_pnl = MagicMock()

        summary = {"actions": [], "analyses": [], "trades": []}
        positions = [
            {
                "symbol": "AAPL",
                "qty": 10,
                "avg_entry": Decimal("100"),
                "current_price": Decimal("104"),
                "unrealized_pnl": Decimal("40"),
                "side": "long",
            }
        ]
        bot._scan_and_execute_sells(summary, positions)
        bot._executor.sell.assert_called_once_with("AAPL", 10)

    def test_trailing_stop_cleanup_on_sell(self, bot):
        bot._trailing_stops["AAPL"] = {
            "floor": Decimal("104.50"),
            "stop_order_id": "stop-1",
        }
        bot._risk.calculate_trailing_stop = MagicMock(return_value=Decimal("98.80"))
        bot._strategy.should_sell = MagicMock(return_value=True)
        bot._get_price_bars = MagicMock(return_value=([104.0], []))
        bot._executor.sell = MagicMock(
            return_value={"order_id": "sell-1", "symbol": "AAPL", "qty": 10}
        )
        bot._executor.cancel_stop_loss = MagicMock()
        bot._risk.record_trade_result = MagicMock()
        bot._risk.record_daily_pnl = MagicMock()

        summary = {"actions": [], "analyses": [], "trades": []}
        positions = [
            {
                "symbol": "AAPL",
                "qty": 10,
                "avg_entry": Decimal("100"),
                "current_price": Decimal("104"),
                "unrealized_pnl": Decimal("40"),
                "side": "long",
            }
        ]
        bot._scan_and_execute_sells(summary, positions)
        assert "AAPL" not in bot._trailing_stops
        bot._executor.cancel_stop_loss.assert_called_once_with("stop-1")

    def test_buy_initializes_trailing_stop(self, bot):
        bot._is_market_open = MagicMock(return_value=True)
        bot._get_market_time = MagicMock(return_value=time(11, 0))
        bot._get_price_bars = MagicMock(return_value=([100.0], [{"close": 100.0}]))
        bot._strategy.should_buy = MagicMock(return_value=True)
        bot._executor.buy = MagicMock(
            return_value={
                "order_id": "buy-1",
                "symbol": "AAPL",
                "qty": 2,
                "stop_order_id": "stop-1",
            }
        )
        bot._risk.calculate_trailing_stop = MagicMock(return_value=Decimal("95"))

        summary = {"actions": [], "analyses": [], "trades": []}
        bot._process_buy_candidate("AAPL", summary, time(11, 0))

        assert "AAPL" in bot._trailing_stops
        assert bot._trailing_stops["AAPL"]["floor"] == Decimal("95")
        assert bot._trailing_stops["AAPL"]["stop_order_id"] == "stop-1"


class TestStatePersistence:
    def test_state_saved_after_run_once(self, bot):
        bot._is_market_open = MagicMock(return_value=True)
        bot._get_market_time = MagicMock(return_value=time(11, 0))
        bot._executor.get_account.return_value = {
            "equity": Decimal("10000"),
            "cash": Decimal("5000"),
            "buying_power": Decimal("5000"),
            "portfolio_value": Decimal("10000"),
        }
        bot._executor.get_positions.return_value = []
        bot._get_price_bars = MagicMock(return_value=([], []))

        bot.run_once()
        bot._state_store.save.assert_called_once()

    def test_state_not_saved_when_market_closed(self, bot):
        bot._is_market_open = MagicMock(return_value=False)
        bot.run_once()
        bot._state_store.save.assert_not_called()

    def test_load_restores_peak_equity(self, mock_settings):
        with (
            patch("claude_trader.bot.AlpacaExecutor"),
            patch("claude_trader.bot.NewsFeed"),
            patch("claude_trader.bot.TelegramNotifier"),
            patch("claude_trader.bot.ObsidianLogger"),
            patch("claude_trader.bot.TradeLogger"),
            patch("claude_trader.bot.BotStateStore") as mock_store_cls,
        ):
            mock_store_cls.return_value.load.return_value = {
                "peak_equity": Decimal("15000"),
                "trailing_stops": {
                    "AAPL": {"floor": Decimal("142"), "stop_order_id": "s-1"},
                },
                "last_trading_date": "2026-04-10",
            }
            bot = TradingBot(mock_settings)

        assert bot._peak_equity == Decimal("15000")
        assert bot._trailing_stops["AAPL"]["floor"] == Decimal("142")
        assert bot._last_trading_date == date(2026, 4, 10)
