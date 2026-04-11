"""Tests for the trade journal module."""

import json
from pathlib import Path

from claude_trader.journal import (
    compute_journal_stats,
    filter_trades,
    format_trade_table,
    print_journal,
    read_trades,
)

TRADE_1 = {
    "timestamp": "2026-04-10T14:30:00+00:00",
    "symbol": "AAPL",
    "side": "buy",
    "qty": 10,
    "price": 150.5,
    "rationale": "EMA crossover",
    "order_id": "order-1",
}
TRADE_2 = {
    "timestamp": "2026-04-11T10:00:00+00:00",
    "symbol": "MSFT",
    "side": "sell",
    "qty": 5,
    "price": 420.0,
    "rationale": "Trailing stop triggered after extended rally beyond resistance",
    "order_id": "order-2",
}


def _write_jsonl(path: Path, entries: list) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# --- read_trades ---


class TestReadTrades:
    def test_reads_valid_jsonl(self, tmp_path: Path):
        p = tmp_path / "trades.jsonl"
        _write_jsonl(p, [TRADE_1, TRADE_2])
        result = read_trades(p)
        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[1]["symbol"] == "MSFT"

    def test_skips_malformed_lines(self, tmp_path: Path):
        p = tmp_path / "trades.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(TRADE_1) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps(TRADE_2) + "\n")
        result = read_trades(p)
        assert len(result) == 2

    def test_missing_file_returns_empty(self, tmp_path: Path):
        p = tmp_path / "nonexistent.jsonl"
        assert read_trades(p) == []


# --- filter_trades ---


class TestFilterTrades:
    def test_filter_by_symbol(self):
        trades = [TRADE_1, TRADE_2]
        result = filter_trades(trades, symbol="AAPL")
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"

    def test_filter_by_side(self):
        trades = [TRADE_1, TRADE_2]
        result = filter_trades(trades, side="buy")
        assert len(result) == 1
        assert result[0]["side"] == "buy"

    def test_filter_by_date_range(self):
        trades = [TRADE_1, TRADE_2]
        result = filter_trades(trades, start_date="2026-04-11", end_date="2026-04-11")
        assert len(result) == 1
        assert result[0]["symbol"] == "MSFT"

    def test_no_filters_returns_all(self):
        trades = [TRADE_1, TRADE_2]
        result = filter_trades(trades)
        assert len(result) == 2


# --- format_trade_table ---


class TestFormatTradeTable:
    def test_formats_trades(self):
        table = format_trade_table([TRADE_1, TRADE_2])
        assert "Timestamp" in table
        assert "AAPL" in table
        assert "MSFT" in table
        assert "buy" in table
        assert "sell" in table

    def test_empty_list(self):
        assert format_trade_table([]) == "  No trades found."


# --- compute_journal_stats ---


class TestComputeStats:
    def test_correct_counts(self):
        stats = compute_journal_stats([TRADE_1, TRADE_2])
        assert stats["total_trades"] == 2
        assert stats["buys"] == 1
        assert stats["sells"] == 1
        assert stats["symbols"] == ["AAPL", "MSFT"]


# --- print_journal ---


class TestPrintJournal:
    def test_output_contains_sections(self, tmp_path: Path, capsys):
        p = tmp_path / "trades.jsonl"
        _write_jsonl(p, [TRADE_1])
        print_journal(p)
        captured = capsys.readouterr().out
        assert "TRADE JOURNAL" in captured
        assert "AAPL" in captured
