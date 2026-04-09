"""Tests for trade logger."""

import json
from datetime import datetime, timezone

import pytest

from claude_trader.logger import TradeLogger


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "trades.jsonl"


@pytest.fixture
def logger(log_path):
    return TradeLogger(log_path)


class TestLogTrade:
    def test_writes_jsonl_entry(self, logger, log_path):
        logger.log_trade(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            rationale="test",
            order_id="order-1",
        )
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["symbol"] == "AAPL"
        assert entry["side"] == "buy"
        assert entry["qty"] == 10
        assert entry["price"] == 150.0
        assert entry["order_id"] == "order-1"

    def test_appends_multiple_entries(self, logger, log_path):
        logger.log_trade(symbol="AAPL", side="buy", qty=10, price=150.0, rationale="r1")
        logger.log_trade(symbol="MSFT", side="sell", qty=5, price=300.0, rationale="r2")
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_includes_agent_scores(self, logger, log_path):
        logger.log_trade(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            rationale="test",
            agent_scores={"combined": 0.75, "agreement": 4},
        )
        entry = json.loads(log_path.read_text().strip())
        assert entry["agent_scores"]["combined"] == 0.75

    def test_timestamp_is_iso_utc(self, logger, log_path):
        logger.log_trade(symbol="AAPL", side="buy", qty=1, price=100.0, rationale="t")
        entry = json.loads(log_path.read_text().strip())
        ts = datetime.fromisoformat(entry["timestamp"])
        assert ts.tzinfo is not None


class TestGetDailySummary:
    def test_empty_file(self, logger):
        summary = logger.get_daily_summary()
        assert summary["total_trades"] == 0

    def test_counts_today_trades(self, logger, log_path):
        today = datetime.now(timezone.utc).date().isoformat()
        entries = [
            {"timestamp": f"{today}T10:00:00+00:00", "side": "buy", "symbol": "AAPL"},
            {"timestamp": f"{today}T11:00:00+00:00", "side": "sell", "symbol": "AAPL"},
            {"timestamp": "2020-01-01T10:00:00+00:00", "side": "buy", "symbol": "OLD"},
        ]
        with open(log_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        summary = logger.get_daily_summary()
        assert summary["total_trades"] == 2
        assert summary["buys"] == 1
        assert summary["sells"] == 1
