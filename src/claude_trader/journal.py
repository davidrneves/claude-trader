"""Trade journal - CLI for reviewing trade history.

Reads trades.jsonl and displays formatted trade data for manual review,
a graduation requirement before going live.
"""

import json
from pathlib import Path


def read_trades(log_path: Path) -> list[dict]:
    """Read all trades from JSONL file, skipping malformed lines."""
    if not log_path.exists():
        return []
    trades = []
    with open(log_path) as f:
        for line in f:
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return trades


def filter_trades(
    trades: list[dict],
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    side: str | None = None,
) -> list[dict]:
    """Filter trades by symbol, side, and date range."""
    result = trades
    if symbol:
        symbol_upper = symbol.upper()
        result = [t for t in result if t.get("symbol") == symbol_upper]
    if side:
        side_lower = side.lower()
        result = [t for t in result if t.get("side") == side_lower]
    if start_date:
        result = [t for t in result if t.get("timestamp", "") >= start_date]
    if end_date:
        result = [t for t in result if t.get("timestamp", "")[:10] <= end_date]
    return result


def compute_journal_stats(trades: list[dict]) -> dict:
    """Compute summary stats for a list of trades."""
    symbols = sorted({t.get("symbol", "") for t in trades})
    return {
        "total_trades": len(trades),
        "buys": sum(1 for t in trades if t.get("side") == "buy"),
        "sells": sum(1 for t in trades if t.get("side") == "sell"),
        "symbols": symbols,
    }


def format_trade_table(trades: list[dict]) -> str:
    """Format trades as an ASCII table."""
    if not trades:
        return "  No trades found."

    header = f"  {'Timestamp':<28} {'Side':<6} {'Symbol':<8} {'Qty':>5} {'Price':>10}  {'Rationale':<40}"
    sep = "  " + "-" * (len(header) - 2)
    lines = [sep, header, sep]
    for t in trades:
        rationale = t.get("rationale", "")
        if len(rationale) > 40:
            rationale = rationale[:37] + "..."
        lines.append(
            f"  {t.get('timestamp', ''):<28} {t.get('side', ''):<6} "
            f"{t.get('symbol', ''):<8} {t.get('qty', ''):>5} "
            f"{t.get('price', ''):>10}  {rationale:<40}"
        )
    lines.append(sep)
    return "\n".join(lines)


def print_journal(
    log_path: Path,
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    side: str | None = None,
) -> None:
    """Print the full trade journal with filters and stats."""
    trades = read_trades(log_path)
    filtered = filter_trades(
        trades, symbol=symbol, start_date=start_date, end_date=end_date, side=side
    )

    print("\n  TRADE JOURNAL")
    print("  " + "=" * 40)

    active_filters = []
    if symbol:
        active_filters.append(f"symbol={symbol.upper()}")
    if side:
        active_filters.append(f"side={side.lower()}")
    if start_date:
        active_filters.append(f"from={start_date}")
    if end_date:
        active_filters.append(f"to={end_date}")
    if active_filters:
        print(f"  Filters: {', '.join(active_filters)}")
    else:
        print("  Filters: none")

    stats = compute_journal_stats(filtered)
    print(
        f"  Total: {stats['total_trades']} | "
        f"Buys: {stats['buys']} | "
        f"Sells: {stats['sells']} | "
        f"Symbols: {', '.join(stats['symbols']) or '-'}"
    )
    print()
    print(format_trade_table(filtered))
    print()
