"""Tests for Telegram notifier."""

from unittest.mock import MagicMock

from claude_trader.notifier import TelegramNotifier


class TestTelegramDisabled:
    def test_no_token_disables(self):
        n = TelegramNotifier(bot_token="", chat_id="123")
        assert n._enabled is False

    def test_no_chat_id_disables(self):
        n = TelegramNotifier(bot_token="tok", chat_id="")
        assert n._enabled is False

    def test_send_skips_when_disabled(self):
        n = TelegramNotifier(bot_token="", chat_id="")
        n._send("test")  # should not raise


class TestTradeAlert:
    def test_buy_alert_format(self):
        n = TelegramNotifier(bot_token="", chat_id="")
        n._send = MagicMock()
        n._enabled = True

        n.trade_alert(
            symbol="AAPL", side="buy", qty=10, price=150.0, rationale="EMA crossover"
        )

        n._send.assert_called_once()
        msg = n._send.call_args[0][0]
        assert "BUY AAPL" in msg
        assert "150.00" in msg
        assert "EMA crossover" in msg

    def test_sell_alert_format(self):
        n = TelegramNotifier(bot_token="", chat_id="")
        n._send = MagicMock()
        n._enabled = True

        n.trade_alert(
            symbol="MSFT", side="sell", qty=5, price=300.0, rationale="Stop loss"
        )

        msg = n._send.call_args[0][0]
        assert "SELL MSFT" in msg

    def test_agent_scores_included(self):
        n = TelegramNotifier(bot_token="", chat_id="")
        n._send = MagicMock()
        n._enabled = True

        n.trade_alert(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            rationale="test",
            agent_scores={"combined": 0.75, "agreement": 3},
        )

        msg = n._send.call_args[0][0]
        assert "0.75" in msg
        assert "3" in msg


class TestDailySummary:
    def test_summary_format(self):
        n = TelegramNotifier(bot_token="", chat_id="")
        n._send = MagicMock()
        n._enabled = True

        n.daily_summary(
            date="2026-04-10",
            equity="10000",
            daily_pnl="150",
            trades_count=3,
            positions=[
                {
                    "symbol": "AAPL",
                    "qty": 10,
                    "avg_entry": "145.00",
                    "unrealized_pnl": "50",
                }
            ],
        )

        msg = n._send.call_args[0][0]
        assert "2026-04-10" in msg
        assert "$10000" in msg
        assert "AAPL" in msg

    def test_empty_positions(self):
        n = TelegramNotifier(bot_token="", chat_id="")
        n._send = MagicMock()
        n._enabled = True

        n.daily_summary(
            date="2026-04-10",
            equity="10000",
            daily_pnl="0",
            trades_count=0,
            positions=[],
        )

        msg = n._send.call_args[0][0]
        assert "None" in msg
