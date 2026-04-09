"""Performance tracking and graduation criteria evaluation.

Records daily portfolio snapshots to JSONL and computes metrics needed
to evaluate the 6 graduation criteria before going live:
1. 30+ days paper trading
2. Positive cumulative return
3. Sharpe ratio > 0.5
4. Max drawdown < 10%
5. No circuit breaker in last 7 days
6. Manual review of all trades
"""

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import structlog

from claude_trader.logger import DecimalEncoder

log = structlog.get_logger()


@dataclass
class DailySnapshot:
    """One row per trading day, persisted to snapshots.jsonl."""

    date: str
    equity: str
    cash: str
    portfolio_value: str
    daily_pnl: str
    cumulative_return_pct: float
    positions_count: int
    trades_count: int
    buys: int
    sells: int
    max_drawdown_pct: float
    consecutive_losses: int
    circuit_breaker_triggered: bool


@dataclass
class GraduationCriterion:
    """A single pass/fail criterion for live trading graduation."""

    name: str
    threshold: str
    current_value: str
    passed: bool


@dataclass
class GraduationResult:
    """Full graduation evaluation."""

    criteria: list[GraduationCriterion]
    all_passed: bool
    trading_days: int
    first_date: str
    last_date: str


@dataclass
class PerformanceMetrics:
    """Computed metrics from daily snapshots."""

    trading_days: int
    cumulative_return_pct: float
    sharpe_ratio: float | None
    max_drawdown_pct: float
    circuit_breaker_days_last_7: int
    total_trades: int
    win_rate: float | None
    first_date: str
    last_date: str


class PerformanceTracker:
    """Records daily snapshots and computes graduation metrics.

    Pure computation from JSONL - no Alpaca API calls.
    """

    def __init__(self, snapshots_path: Path) -> None:
        self._path = snapshots_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record_snapshot(
        self,
        account: dict,
        trades_today: dict,
        risk_state: dict,
    ) -> DailySnapshot:
        """Append a daily snapshot. Call once per trading day after the last cycle."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Compute cumulative return from first snapshot
        snapshots = self._read_snapshots()
        first_equity = Decimal(snapshots[0]["equity"]) if snapshots else None
        current_equity = Decimal(str(account["equity"]))

        if first_equity and first_equity > 0:
            cumulative_return = float(
                (current_equity - first_equity) / first_equity * 100
            )
        else:
            cumulative_return = 0.0

        # Compute max drawdown from full equity curve
        equities = [float(s["equity"]) for s in snapshots] + [float(current_equity)]
        max_dd = self._compute_max_drawdown(equities)

        # Compute daily P&L from previous snapshot
        if snapshots:
            prev_equity = Decimal(snapshots[-1]["equity"])
            daily_pnl = str(current_equity - prev_equity)
        else:
            daily_pnl = "0"

        snapshot = DailySnapshot(
            date=today,
            equity=str(current_equity),
            cash=str(account["cash"]),
            portfolio_value=str(account["portfolio_value"]),
            daily_pnl=daily_pnl,
            cumulative_return_pct=round(cumulative_return, 4),
            positions_count=risk_state.get("open_positions", 0),
            trades_count=trades_today.get("total_trades", 0),
            buys=trades_today.get("buys", 0),
            sells=trades_today.get("sells", 0),
            max_drawdown_pct=round(max_dd, 4),
            consecutive_losses=risk_state.get("consecutive_losses", 0),
            circuit_breaker_triggered=risk_state.get(
                "circuit_breaker_triggered", False
            ),
        )

        # Deduplicate: overwrite if same date already exists
        updated = False
        new_snapshots = []
        for s in snapshots:
            if s["date"] == today:
                new_snapshots.append(asdict(snapshot))
                updated = True
            else:
                new_snapshots.append(s)
        if not updated:
            new_snapshots.append(asdict(snapshot))

        self._write_snapshots(new_snapshots)
        log.info(
            "snapshot_recorded",
            date=today,
            equity=str(current_equity),
            daily_pnl=daily_pnl,
            cumulative_return=f"{cumulative_return:.2f}%",
        )
        return snapshot

    def get_metrics(self) -> PerformanceMetrics | None:
        """Compute all performance metrics from snapshots."""
        snapshots = self._read_snapshots()
        if not snapshots:
            return None

        equities = [float(s["equity"]) for s in snapshots]
        daily_returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                daily_returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

        # Sharpe ratio (annualized, assuming 252 trading days)
        sharpe = None
        if len(daily_returns) >= 2:
            mean_ret = sum(daily_returns) / len(daily_returns)
            std_ret = math.sqrt(
                sum((r - mean_ret) ** 2 for r in daily_returns)
                / (len(daily_returns) - 1)
            )
            if std_ret > 0:
                sharpe = round((mean_ret / std_ret) * math.sqrt(252), 3)

        # Max drawdown
        max_dd = self._compute_max_drawdown(equities)

        # Circuit breaker in last 7 days
        recent = snapshots[-7:] if len(snapshots) >= 7 else snapshots
        cb_days = sum(1 for s in recent if s.get("circuit_breaker_triggered", False))

        # Cumulative return
        first_eq = equities[0]
        last_eq = equities[-1]
        cum_return = ((last_eq - first_eq) / first_eq * 100) if first_eq > 0 else 0.0

        total_trades = sum(s.get("trades_count", 0) for s in snapshots)

        return PerformanceMetrics(
            trading_days=len(snapshots),
            cumulative_return_pct=round(cum_return, 4),
            sharpe_ratio=sharpe,
            max_drawdown_pct=round(max_dd, 4),
            circuit_breaker_days_last_7=cb_days,
            total_trades=total_trades,
            win_rate=None,
            first_date=snapshots[0]["date"],
            last_date=snapshots[-1]["date"],
        )

    def check_graduation(self) -> GraduationResult:
        """Evaluate all 6 graduation criteria. Returns pass/fail for each."""
        metrics = self.get_metrics()

        if not metrics:
            return GraduationResult(
                criteria=[
                    GraduationCriterion("Paper trading days", ">= 30", "0", False),
                    GraduationCriterion("Cumulative return", "> 0%", "N/A", False),
                    GraduationCriterion("Sharpe ratio", "> 0.5", "N/A", False),
                    GraduationCriterion("Max drawdown", "< 10%", "N/A", False),
                    GraduationCriterion(
                        "No circuit breaker (7d)", "0 days", "N/A", False
                    ),
                    GraduationCriterion("Manual review", "completed", "pending", False),
                ],
                all_passed=False,
                trading_days=0,
                first_date="N/A",
                last_date="N/A",
            )

        c1 = GraduationCriterion(
            "Paper trading days",
            ">= 30",
            str(metrics.trading_days),
            metrics.trading_days >= 30,
        )
        c2 = GraduationCriterion(
            "Cumulative return",
            "> 0%",
            f"{metrics.cumulative_return_pct:.2f}%",
            metrics.cumulative_return_pct > 0,
        )
        c3 = GraduationCriterion(
            "Sharpe ratio",
            "> 0.5",
            f"{metrics.sharpe_ratio:.3f}"
            if metrics.sharpe_ratio is not None
            else "N/A",
            metrics.sharpe_ratio is not None and metrics.sharpe_ratio > 0.5,
        )
        c4 = GraduationCriterion(
            "Max drawdown",
            "< 10%",
            f"{metrics.max_drawdown_pct:.2f}%",
            metrics.max_drawdown_pct < 10.0,
        )
        c5 = GraduationCriterion(
            "No circuit breaker (7d)",
            "0 days",
            f"{metrics.circuit_breaker_days_last_7} days",
            metrics.circuit_breaker_days_last_7 == 0,
        )
        c6 = GraduationCriterion(
            "Manual review",
            "completed",
            "pending",
            False,
        )

        criteria = [c1, c2, c3, c4, c5, c6]
        all_passed = all(c.passed for c in criteria)

        return GraduationResult(
            criteria=criteria,
            all_passed=all_passed,
            trading_days=metrics.trading_days,
            first_date=metrics.first_date,
            last_date=metrics.last_date,
        )

    def get_daily_pnl(self) -> str:
        """Get today's P&L from the latest snapshot, or '0' if none."""
        snapshots = self._read_snapshots()
        if not snapshots:
            return "0"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for s in reversed(snapshots):
            if s["date"] == today:
                return s.get("daily_pnl", "0")
        return "0"

    def _read_snapshots(self) -> list[dict]:
        """Read all snapshots from JSONL."""
        if not self._path.exists():
            return []
        snapshots = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    snapshots.append(json.loads(line))
        return snapshots

    def _write_snapshots(self, snapshots: list[dict]) -> None:
        """Rewrite the full snapshots file (needed for dedup)."""
        with open(self._path, "w") as f:
            for s in snapshots:
                f.write(json.dumps(s, cls=DecimalEncoder) + "\n")

    @staticmethod
    def _compute_max_drawdown(equities: list[float]) -> float:
        """Compute max drawdown percentage from an equity curve."""
        if len(equities) < 2:
            return 0.0
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        return max_dd
