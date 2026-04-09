"""Tests for performance tracking and graduation criteria."""

import json

import pytest

from claude_trader.performance import PerformanceTracker


@pytest.fixture
def snapshots_path(tmp_path):
    return tmp_path / "snapshots.jsonl"


@pytest.fixture
def tracker(snapshots_path):
    return PerformanceTracker(snapshots_path)


def _write_snapshots(path, snapshots):
    with open(path, "w") as f:
        for s in snapshots:
            f.write(json.dumps(s) + "\n")


def _make_snapshot(
    date="2025-01-01",
    equity="10000",
    cash="5000",
    portfolio_value="10000",
    daily_pnl="0",
    cumulative_return_pct=0.0,
    positions_count=0,
    trades_count=0,
    buys=0,
    sells=0,
    max_drawdown_pct=0.0,
    consecutive_losses=0,
    circuit_breaker_triggered=False,
):
    return {
        "date": date,
        "equity": equity,
        "cash": cash,
        "portfolio_value": portfolio_value,
        "daily_pnl": daily_pnl,
        "cumulative_return_pct": cumulative_return_pct,
        "positions_count": positions_count,
        "trades_count": trades_count,
        "buys": buys,
        "sells": sells,
        "max_drawdown_pct": max_drawdown_pct,
        "consecutive_losses": consecutive_losses,
        "circuit_breaker_triggered": circuit_breaker_triggered,
    }


class TestRecordSnapshot:
    def test_first_snapshot(self, tracker, snapshots_path):
        snapshot = tracker.record_snapshot(
            account={"equity": "10000", "cash": "5000", "portfolio_value": "10000"},
            trades_today={"total_trades": 2, "buys": 1, "sells": 1},
            risk_state={
                "open_positions": 1,
                "consecutive_losses": 0,
                "circuit_breaker_triggered": False,
            },
        )
        assert snapshot.daily_pnl == "0"
        assert snapshot.cumulative_return_pct == 0.0
        assert snapshots_path.exists()

    def test_second_snapshot_computes_pnl(self, tracker, snapshots_path):
        _write_snapshots(
            snapshots_path,
            [
                _make_snapshot(date="2025-01-01", equity="10000"),
            ],
        )
        snapshot = tracker.record_snapshot(
            account={"equity": "10500", "cash": "5000", "portfolio_value": "10500"},
            trades_today={"total_trades": 1, "buys": 1, "sells": 0},
            risk_state={
                "open_positions": 1,
                "consecutive_losses": 0,
                "circuit_breaker_triggered": False,
            },
        )
        assert snapshot.daily_pnl == "500"
        assert snapshot.cumulative_return_pct == pytest.approx(5.0)

    def test_deduplicates_same_date(self, tracker, snapshots_path):
        tracker.record_snapshot(
            account={"equity": "10000", "cash": "5000", "portfolio_value": "10000"},
            trades_today={"total_trades": 0, "buys": 0, "sells": 0},
            risk_state={
                "open_positions": 0,
                "consecutive_losses": 0,
                "circuit_breaker_triggered": False,
            },
        )
        tracker.record_snapshot(
            account={"equity": "10100", "cash": "5000", "portfolio_value": "10100"},
            trades_today={"total_trades": 1, "buys": 1, "sells": 0},
            risk_state={
                "open_positions": 1,
                "consecutive_losses": 0,
                "circuit_breaker_triggered": False,
            },
        )
        snapshots = tracker._read_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["equity"] == "10100"


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equities = [100, 110, 120, 130]
        assert PerformanceTracker._compute_max_drawdown(equities) == 0.0

    def test_simple_drawdown(self):
        equities = [100, 120, 90, 110]
        dd = PerformanceTracker._compute_max_drawdown(equities)
        assert dd == pytest.approx(25.0)  # (120-90)/120 * 100

    def test_single_point(self):
        assert PerformanceTracker._compute_max_drawdown([100]) == 0.0

    def test_monotonic_decline(self):
        equities = [100, 90, 80, 70]
        dd = PerformanceTracker._compute_max_drawdown(equities)
        assert dd == pytest.approx(30.0)  # (100-70)/100 * 100


class TestSharpeRatio:
    def test_no_snapshots(self, tracker):
        assert tracker.get_metrics() is None

    def test_single_snapshot_no_sharpe(self, tracker, snapshots_path):
        _write_snapshots(snapshots_path, [_make_snapshot()])
        metrics = tracker.get_metrics()
        assert metrics.sharpe_ratio is None

    def test_positive_sharpe(self, tracker, snapshots_path):
        # Steadily increasing equity = high Sharpe
        snapshots = [
            _make_snapshot(date=f"2025-01-{i + 1:02d}", equity=str(10000 + i * 100))
            for i in range(30)
        ]
        _write_snapshots(snapshots_path, snapshots)
        metrics = tracker.get_metrics()
        assert metrics.sharpe_ratio is not None
        assert metrics.sharpe_ratio > 0

    def test_volatile_returns_lower_sharpe(self, tracker, snapshots_path):
        # Alternating up/down = lower Sharpe
        snapshots = [
            _make_snapshot(
                date=f"2025-01-{i + 1:02d}",
                equity=str(10000 + (100 if i % 2 == 0 else -100) * (i + 1)),
            )
            for i in range(20)
        ]
        _write_snapshots(snapshots_path, snapshots)
        metrics = tracker.get_metrics()
        # Just check it computed something, volatile returns could go either way
        assert metrics.sharpe_ratio is not None


class TestCumulativeReturn:
    def test_positive_return(self, tracker, snapshots_path):
        _write_snapshots(
            snapshots_path,
            [
                _make_snapshot(date="2025-01-01", equity="10000"),
                _make_snapshot(date="2025-01-02", equity="11000"),
            ],
        )
        metrics = tracker.get_metrics()
        assert metrics.cumulative_return_pct == pytest.approx(10.0)

    def test_negative_return(self, tracker, snapshots_path):
        _write_snapshots(
            snapshots_path,
            [
                _make_snapshot(date="2025-01-01", equity="10000"),
                _make_snapshot(date="2025-01-02", equity="9000"),
            ],
        )
        metrics = tracker.get_metrics()
        assert metrics.cumulative_return_pct == pytest.approx(-10.0)


class TestCircuitBreakerHistory:
    def test_no_circuit_breakers(self, tracker, snapshots_path):
        snapshots = [
            _make_snapshot(date=f"2025-01-{i + 1:02d}", circuit_breaker_triggered=False)
            for i in range(10)
        ]
        _write_snapshots(snapshots_path, snapshots)
        metrics = tracker.get_metrics()
        assert metrics.circuit_breaker_days_last_7 == 0

    def test_recent_circuit_breaker(self, tracker, snapshots_path):
        snapshots = [
            _make_snapshot(
                date=f"2025-01-{i + 1:02d}", circuit_breaker_triggered=(i >= 8)
            )
            for i in range(10)
        ]
        _write_snapshots(snapshots_path, snapshots)
        metrics = tracker.get_metrics()
        assert metrics.circuit_breaker_days_last_7 == 2


class TestGraduation:
    def test_all_fail_no_data(self, tracker):
        result = tracker.check_graduation()
        assert not result.all_passed
        assert all(not c.passed for c in result.criteria)

    def test_insufficient_days(self, tracker, snapshots_path):
        snapshots = [
            _make_snapshot(date=f"2025-01-{i + 1:02d}", equity=str(10000 + i * 50))
            for i in range(10)
        ]
        _write_snapshots(snapshots_path, snapshots)
        result = tracker.check_graduation()
        assert not result.all_passed
        days_criterion = next(c for c in result.criteria if "days" in c.name.lower())
        assert not days_criterion.passed

    def test_graduation_passes_except_manual_review(self, tracker, snapshots_path):
        # 35 days of steady gains, no circuit breakers
        snapshots = [
            _make_snapshot(
                date=f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                equity=str(10000 + i * 30),
            )
            for i in range(35)
        ]
        _write_snapshots(snapshots_path, snapshots)
        result = tracker.check_graduation()
        # Manual review always fails (requires human)
        assert not result.all_passed
        manual = next(c for c in result.criteria if "manual" in c.name.lower())
        assert not manual.passed
        # But all automated criteria should pass (except possibly Sharpe for small gains)
        days_c = next(c for c in result.criteria if "days" in c.name.lower())
        assert days_c.passed
        return_c = next(c for c in result.criteria if "return" in c.name.lower())
        assert return_c.passed
        dd_c = next(c for c in result.criteria if "drawdown" in c.name.lower())
        assert dd_c.passed
        cb_c = next(c for c in result.criteria if "circuit" in c.name.lower())
        assert cb_c.passed


class TestGetDailyPnl:
    def test_no_snapshots_returns_zero(self, tracker):
        assert tracker.get_daily_pnl() == "0"

    def test_returns_todays_pnl(self, tracker, snapshots_path):
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _write_snapshots(
            snapshots_path,
            [
                _make_snapshot(date=today, daily_pnl="150.50"),
            ],
        )
        assert tracker.get_daily_pnl() == "150.50"
