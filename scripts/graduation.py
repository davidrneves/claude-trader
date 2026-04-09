#!/usr/bin/env python3
"""Graduation dashboard - check if paper trading meets go-live criteria.

Usage:
  uv run python scripts/graduation.py
  uv run python scripts/graduation.py --metrics   # Show detailed metrics only
"""

import argparse
import sys
from pathlib import Path

from claude_trader.performance import PerformanceTracker


def print_graduation(tracker: PerformanceTracker) -> bool:
    """Print graduation dashboard. Returns True if all criteria pass."""
    result = tracker.check_graduation()

    print()
    print("=" * 60)
    print("  PAPER TRADING GRADUATION DASHBOARD")
    print("=" * 60)
    print(f"  Period: {result.first_date} to {result.last_date}")
    print(f"  Trading days: {result.trading_days}")
    print("-" * 60)
    print(f"  {'Criterion':<30} {'Value':<15} {'Target':<12} {'Status'}")
    print("-" * 60)

    for c in result.criteria:
        status = "PASS" if c.passed else "FAIL"
        marker = "  " if c.passed else "X "
        print(
            f"  {marker}{c.name:<28} {c.current_value:<15} {c.threshold:<12} {status}"
        )

    print("-" * 60)
    if result.all_passed:
        print("  RESULT: ALL CRITERIA MET - Ready for live trading review")
    else:
        failed = sum(1 for c in result.criteria if not c.passed)
        print(f"  RESULT: {failed} criteria not met - Continue paper trading")
    print("=" * 60)
    print()

    return result.all_passed


def print_metrics(tracker: PerformanceTracker) -> None:
    """Print detailed performance metrics."""
    metrics = tracker.get_metrics()

    if not metrics:
        print("No snapshots found. Run the bot first to collect data.")
        return

    print()
    print("=" * 50)
    print("  PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  Period:            {metrics.first_date} to {metrics.last_date}")
    print(f"  Trading days:      {metrics.trading_days}")
    print(f"  Cumulative return: {metrics.cumulative_return_pct:.2f}%")
    sharpe_str = (
        f"{metrics.sharpe_ratio:.3f}"
        if metrics.sharpe_ratio is not None
        else "N/A (need 2+ days)"
    )
    print(f"  Sharpe ratio:      {sharpe_str}")
    print(f"  Max drawdown:      {metrics.max_drawdown_pct:.2f}%")
    print(f"  Total trades:      {metrics.total_trades}")
    print(f"  Circuit breakers:  {metrics.circuit_breaker_days_last_7} (last 7 days)")
    print("=" * 50)
    print()


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Graduation Dashboard")
    parser.add_argument(
        "--metrics", action="store_true", help="Show detailed metrics only"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("snapshots.jsonl"),
        help="Path to snapshots file",
    )
    args = parser.parse_args()

    tracker = PerformanceTracker(args.path)

    if args.metrics:
        print_metrics(tracker)
    else:
        print_graduation(tracker)

    return 0


if __name__ == "__main__":
    sys.exit(main())
