"""Tests for bot state persistence."""

import json
from decimal import Decimal

from claude_trader.state import BotStateStore


class TestLoad:
    def test_missing_file_returns_empty(self, tmp_path):
        store = BotStateStore(tmp_path / "missing.json")
        assert store.load() == {}

    def test_corrupt_json_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json{{{")
        store = BotStateStore(path)
        assert store.load() == {}

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        store = BotStateStore(path)
        assert store.load() == {}


class TestSaveAndLoad:
    def test_roundtrip_peak_equity(self, tmp_path):
        path = tmp_path / "state.json"
        store = BotStateStore(path)

        store.save({"peak_equity": Decimal("10500.50")})
        loaded = store.load()
        assert loaded["peak_equity"] == Decimal("10500.50")

    def test_roundtrip_trailing_stops(self, tmp_path):
        path = tmp_path / "state.json"
        store = BotStateStore(path)

        stops = {
            "AAPL": {"floor": Decimal("142.50"), "stop_order_id": "order-abc"},
            "MSFT": {"floor": Decimal("380.00"), "stop_order_id": None},
        }
        store.save({"trailing_stops": stops})
        loaded = store.load()

        assert loaded["trailing_stops"]["AAPL"]["floor"] == Decimal("142.50")
        assert loaded["trailing_stops"]["AAPL"]["stop_order_id"] == "order-abc"
        assert loaded["trailing_stops"]["MSFT"]["stop_order_id"] is None

    def test_roundtrip_last_trading_date(self, tmp_path):
        path = tmp_path / "state.json"
        store = BotStateStore(path)

        store.save({"last_trading_date": "2026-04-11"})
        loaded = store.load()
        assert loaded["last_trading_date"] == "2026-04-11"

    def test_roundtrip_full_state(self, tmp_path):
        path = tmp_path / "state.json"
        store = BotStateStore(path)

        state = {
            "peak_equity": Decimal("25000"),
            "trailing_stops": {
                "NVDA": {"floor": Decimal("900.00"), "stop_order_id": "stop-1"},
            },
            "last_trading_date": "2026-04-10",
        }
        store.save(state)
        loaded = store.load()

        assert loaded["peak_equity"] == Decimal("25000")
        assert loaded["trailing_stops"]["NVDA"]["floor"] == Decimal("900.00")
        assert loaded["last_trading_date"] == "2026-04-10"


class TestAtomicWrite:
    def test_file_not_corrupted_on_overwrite(self, tmp_path):
        path = tmp_path / "state.json"
        store = BotStateStore(path)

        store.save({"peak_equity": Decimal("1000")})
        store.save({"peak_equity": Decimal("2000")})

        loaded = store.load()
        assert loaded["peak_equity"] == Decimal("2000")

    def test_ignores_unknown_keys(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"peak_equity": "500", "unknown_key": "ignored"}))
        store = BotStateStore(path)
        loaded = store.load()
        assert loaded["peak_equity"] == Decimal("500")
        assert "unknown_key" not in loaded
