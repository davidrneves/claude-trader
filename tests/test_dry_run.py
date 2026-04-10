"""Tests for the dry-run validation module."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from claude_trader.dry_run import (
    DryRunReport,
    ValidationResult,
    _check_alpaca,
    _check_gemini,
    _check_market_data,
    _check_news_feed,
    _run_analysis_cycle,
    run_dry_run,
)


@pytest.fixture
def mock_settings(mock_settings):
    """Extend conftest mock_settings with cycle_interval_minutes."""
    mock_settings.cycle_interval_minutes = 15
    return mock_settings


# --- _check_alpaca ---


@patch("claude_trader.dry_run.AlpacaExecutor")
@patch("claude_trader.dry_run.RiskManager")
@patch("claude_trader.dry_run.RiskConfig")
def test_check_alpaca_success(mock_rc, mock_rm, mock_exec, mock_settings):
    mock_exec.return_value.get_account.return_value = {
        "equity": Decimal("10000"),
        "cash": Decimal("5000"),
        "buying_power": Decimal("10000"),
        "portfolio_value": Decimal("10000"),
    }
    result = _check_alpaca(mock_settings)
    assert result.passed is True
    assert "Connected" in result.message
    assert result.elapsed_ms >= 0


@patch("claude_trader.dry_run.AlpacaExecutor")
@patch("claude_trader.dry_run.RiskManager")
@patch("claude_trader.dry_run.RiskConfig")
def test_check_alpaca_failure(mock_rc, mock_rm, mock_exec, mock_settings):
    mock_exec.return_value.get_account.side_effect = ConnectionError("timeout")
    result = _check_alpaca(mock_settings)
    assert result.passed is False
    assert "timeout" in result.message


# --- _check_market_data ---


@patch("claude_trader.dry_run.AlpacaExecutor")
@patch("claude_trader.dry_run.RiskManager")
@patch("claude_trader.dry_run.RiskConfig")
def test_check_market_data(mock_rc, mock_rm, mock_exec, mock_settings):
    mock_bars = MagicMock()
    mock_bars.df.empty = False
    mock_exec.return_value.get_bars.return_value = mock_bars
    result = _check_market_data(mock_settings)
    assert result.passed is True
    assert "AAPL" in result.message


# --- _check_news_feed ---


@patch("claude_trader.dry_run.NewsFeed")
def test_check_news_feed(mock_news_cls, mock_settings):
    mock_news_cls.return_value.get_headlines.return_value = ["Headline 1", "Headline 2"]
    result = _check_news_feed(mock_settings)
    assert result.passed is True
    assert "2 headlines" in result.message


# --- _check_gemini ---


def test_check_gemini_skip(mock_settings):
    mock_settings.gemini_api_key = ""
    result = _check_gemini(mock_settings)
    assert result.passed is True
    assert "Skipped" in result.message
    assert result.elapsed_ms == 0.0


# --- _run_analysis_cycle ---


@patch("claude_trader.dry_run.Analyst")
@patch("claude_trader.dry_run.AlpacaExecutor")
@patch("claude_trader.dry_run.RiskManager")
@patch("claude_trader.dry_run.RiskConfig")
def test_analysis_cycle(mock_rc, mock_rm, mock_exec, mock_analyst, mock_settings):
    mock_settings.gemini_api_key = ""

    # Build mock bars with a DataFrame-like object
    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.__getitem__ = lambda self, key: MagicMock(
        tolist=lambda: [100.0 + i for i in range(30)]
    )
    mock_df.iloc.__getitem__ = lambda self, key: MagicMock(iterrows=lambda: iter([]))
    mock_bars = MagicMock()
    mock_bars.df = mock_df

    mock_exec.return_value.get_bars.return_value = mock_bars

    results = _run_analysis_cycle(mock_settings)
    assert len(results) == len(mock_settings.watchlist)
    for r in results:
        assert "symbol" in r
        if "error" not in r:
            assert "signal" in r
            assert "score" in r
            assert "would_buy" in r
            assert "would_sell" in r


# --- DryRunReport ---


def test_dry_run_report_all_pass():
    report = DryRunReport(
        validations=[
            ValidationResult(name="A", passed=True, message="ok", elapsed_ms=1.0),
            ValidationResult(name="B", passed=True, message="ok", elapsed_ms=2.0),
        ]
    )
    assert report.all_passed is True


def test_dry_run_report_with_failure():
    report = DryRunReport(
        validations=[
            ValidationResult(name="A", passed=True, message="ok", elapsed_ms=1.0),
            ValidationResult(name="B", passed=False, message="fail", elapsed_ms=2.0),
        ]
    )
    assert report.all_passed is False


# --- run_dry_run integration ---


@patch("claude_trader.dry_run._run_analysis_cycle")
@patch("claude_trader.dry_run._check_gemini")
@patch("claude_trader.dry_run._check_news_feed")
@patch("claude_trader.dry_run._check_market_data")
@patch("claude_trader.dry_run._check_alpaca")
@patch("claude_trader.dry_run._print_report")
def test_run_dry_run_all_pass(
    mock_print,
    mock_alpaca,
    mock_market,
    mock_news,
    mock_gemini,
    mock_cycle,
    mock_settings,
):
    ok = ValidationResult(name="test", passed=True, message="ok", elapsed_ms=1.0)
    mock_alpaca.return_value = ok
    mock_market.return_value = ok
    mock_news.return_value = ok
    mock_gemini.return_value = ok
    mock_cycle.return_value = [
        {
            "symbol": "AAPL",
            "signal": "hold",
            "score": 0.0,
            "would_buy": False,
            "would_sell": False,
        }
    ]

    report = run_dry_run(mock_settings)
    assert report.all_passed is True
    assert len(report.validations) == 4
    assert len(report.symbols_analyzed) == 1
    mock_print.assert_called_once()
