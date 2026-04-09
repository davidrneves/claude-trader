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
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TypedDict

import structlog

from claude_trader.logger import DecimalEncoder

log = structlog.get_logger()

# Graduation thresholds (match CLAUDE.md)
MIN_TRADING_DAYS = 30
MIN_SHARPE_RATIO = 0.5
MAX_DRAWDOWN_PCT = 10.0


class AccountSnapshot(TypedDict):
    equity: str
    cash: str
    portfolio_value: str


class TradesSummary(TypedDict):
    total_trades: int
    buys: int
    sells: int


class RiskState(TypedDict):
    open_positions: int
    consecutive_losses: int
    circuit_breaker_triggered: bool


@dataclass
class DailySnapshot:
    """One row per trading day, persisted to snapshots.jsonl."""

    date: str
    equity: str
    daily_pnl: str
    cumulative_return_pct: float
    trades_count: int
    max_drawdown_pct: float
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
        account: AccountSnapshot,
        trades_today: TradesSummary,
        risk_state: RiskState,
    ) -> DailySnapshot:
        """Append a daily snapshot. Call once per trading day after the last cycle."""
        today = self._today()
        snapshots = self._read_snapshots()
        current_equity = Decimal(str(account["equity"]))

        first_equity = float(snapshots[0]["equity"]) if snapshots else None
        cumulative_return = self._compute_cumulative_return(
            first_equity, float(current_equity)
        )

        equities = [float(s["equity"]) for s in snapshots] + [float(current_equity)]
        max_dd = self._compute_max_drawdown(equities)

        if snapshots:
            prev_equity = Decimal(snapshots[-1]["equity"])
            daily_pnl = str(current_equity - prev_equity)
        else:
            daily_pnl = "0"

        snapshot = DailySnapshot(
            date=today,
            equity=str(current_equity),
            daily_pnl=daily_pnl,
            cumulative_return_pct=round(cumulative_return, 4),
            trades_count=trades_today.get("total_trades", 0),
            max_drawdown_pct=round(max_dd, 4),
            circuit_breaker_triggered=risk_state.get(
                "circuit_breaker_triggered", False
            ),
        )

        new_snapshots = self._deduplicate_snapshots(snapshots, today, asdict(snapshot))
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

        sharpe = self._compute_sharpe(daily_returns)
        max_dd = self._compute_max_drawdown(equities)

        recent = snapshots[-7:] if len(snapshots) >= 7 else snapshots
        cb_days = sum(1 for s in recent if s.get("circuit_breaker_triggered", False))

        cum_return = self._compute_cumulative_return(equities[0], equities[-1])
        total_trades = sum(s.get("trades_count", 0) for s in snapshots)

        return PerformanceMetrics(
            trading_days=len(snapshots),
            cumulative_return_pct=round(cum_return, 4),
            sharpe_ratio=sharpe,
            max_drawdown_pct=round(max_dd, 4),
            circuit_breaker_days_last_7=cb_days,
            total_trades=total_trades,
            first_date=snapshots[0]["date"],
            last_date=snapshots[-1]["date"],
        )

    def check_graduation(self) -> GraduationResult:
        """Evaluate all 6 graduation criteria. Returns pass/fail for each."""
        metrics = self.get_metrics()

        if not metrics:
            criteria_defs = [
                ("Paper trading days", f">= {MIN_TRADING_DAYS}", "0"),
                ("Cumulative return", "> 0%", "N/A"),
                ("Sharpe ratio", f"> {MIN_SHARPE_RATIO}", "N/A"),
                ("Max drawdown", f"< {MAX_DRAWDOWN_PCT}%", "N/A"),
                ("No circuit breaker (7d)", "0 days", "N/A"),
                ("Manual review", "completed", "pending"),
            ]
            return GraduationResult(
                criteria=[
                    GraduationCriterion(name, thresh, val, False)
                    for name, thresh, val in criteria_defs
                ],
                all_passed=False,
                trading_days=0,
                first_date="N/A",
                last_date="N/A",
            )

        sharpe_value = (
            f"{metrics.sharpe_ratio:.3f}" if metrics.sharpe_ratio is not None else "N/A"
        )
        sharpe_passed = (
            metrics.sharpe_ratio is not None and metrics.sharpe_ratio > MIN_SHARPE_RATIO
        )

        criteria_defs = [
            (
                "Paper trading days",
                f">= {MIN_TRADING_DAYS}",
                str(metrics.trading_days),
                metrics.trading_days >= MIN_TRADING_DAYS,
            ),
            (
                "Cumulative return",
                "> 0%",
                f"{metrics.cumulative_return_pct:.2f}%",
                metrics.cumulative_return_pct > 0,
            ),
            (
                "Sharpe ratio",
                f"> {MIN_SHARPE_RATIO}",
                sharpe_value,
                sharpe_passed,
            ),
            (
                "Max drawdown",
                f"< {MAX_DRAWDOWN_PCT}%",
                f"{metrics.max_drawdown_pct:.2f}%",
                metrics.max_drawdown_pct < MAX_DRAWDOWN_PCT,
            ),
            (
                "No circuit breaker (7d)",
                "0 days",
                f"{metrics.circuit_breaker_days_last_7} days",
                metrics.circuit_breaker_days_last_7 == 0,
            ),
            (
                "Manual review",
                "completed",
                "pending",
                False,
            ),
        ]

        criteria = [
            GraduationCriterion(name, thresh, val, passed)
            for name, thresh, val, passed in criteria_defs
        ]

        return GraduationResult(
            criteria=criteria,
            all_passed=all(c.passed for c in criteria),
            trading_days=metrics.trading_days,
            first_date=metrics.first_date,
            last_date=metrics.last_date,
        )

    def get_daily_pnl(self) -> str:
        """Get today's P&L from the latest snapshot, or '0' if none."""
        snapshots = self._read_snapshots()
        if not snapshots:
            return "0"
        today = self._today()
        for s in reversed(snapshots):
            if s["date"] == today:
                return s.get("daily_pnl", "0")
        return "0"

    def _read_snapshots(self) -> list[dict]:
        """Read all snapshots from JSONL, skipping malformed lines."""
        if not self._path.exists():
            return []
        snapshots = []
        with open(self._path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    snapshots.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("malformed_snapshot_line", line_num=line_num)
        return snapshots

    def _write_snapshots(self, snapshots: list[dict]) -> None:
        """Atomically rewrite the snapshots file (write to temp, then replace)."""
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                for s in snapshots:
                    f.write(json.dumps(s, cls=DecimalEncoder) + "\n")
            os.replace(tmp, self._path)
        except BaseException:
            os.unlink(tmp)
            raise

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _deduplicate_snapshots(
        existing: list[dict], today: str, new_snapshot: dict
    ) -> list[dict]:
        """Replace existing entry for today, or append if new."""
        updated = False
        result = []
        for s in existing:
            if s["date"] == today:
                result.append(new_snapshot)
                updated = True
            else:
                result.append(s)
        if not updated:
            result.append(new_snapshot)
        return result

    @staticmethod
    def _compute_cumulative_return(
        first_equity: float | None, current_equity: float
    ) -> float:
        if not first_equity or first_equity <= 0:
            return 0.0
        return (current_equity - first_equity) / first_equity * 100

    @staticmethod
    def _compute_sharpe(daily_returns: list[float]) -> float | None:
        """Annualized Sharpe ratio (252 trading days)."""
        if len(daily_returns) < 2:
            return None
        mean_ret = sum(daily_returns) / len(daily_returns)
        std_ret = math.sqrt(
            sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        )
        if std_ret <= 0:
            return None
        return round((mean_ret / std_ret) * math.sqrt(252), 3)

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
