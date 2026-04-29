"""Tests for Analyst.analyze_insider and the SignalAggregator gate-bonus."""

from __future__ import annotations

import pytest

from claude_trader.analyst import (
    Analyst,
    DebateResult,
    FundamentalResult,
    InsiderResult,
    SentimentResult,
    Signal,
    SignalAggregator,
    TechnicalResult,
)


@pytest.fixture
def analyst() -> Analyst:
    return Analyst(api_key="")  # No Gemini calls in these tests


# -------- analyze_insider scoring --------


class TestAnalyzeInsider:
    def test_none_signals_returns_neutral(self, analyst: Analyst) -> None:
        result = analyst.analyze_insider("AAPL", None)
        assert result.score == 0.0
        assert result.signals_unknown == ["all"]

    def test_qualifying_cluster_buy_lifts_score(self, analyst: Analyst) -> None:
        signals = {
            "cluster_buys": [{"insider_count": 5}],
            "officer_buys": [],
            "dilution_filings": [],
            "late_filings": [],
            "failures_to_deliver": [],
        }
        result = analyst.analyze_insider("AAPL", signals)
        assert result.score == pytest.approx(0.4)
        assert result.signals_seen["cluster_buys"] == 1

    def test_low_insider_count_does_not_qualify(self, analyst: Analyst) -> None:
        """A cluster with <3 insiders should NOT score - the threshold is
        deliberate: 1-2 insiders buying together is noise, 3+ is signal."""
        signals = {
            "cluster_buys": [{"insider_count": 2}],
            "officer_buys": [],
            "dilution_filings": [],
            "late_filings": [],
            "failures_to_deliver": [],
        }
        result = analyst.analyze_insider("AAPL", signals)
        assert result.score == 0.0
        assert result.signals_seen["cluster_buys"] == 0

    def test_dilution_lowers_score(self, analyst: Analyst) -> None:
        signals = {
            "cluster_buys": [],
            "officer_buys": [],
            "dilution_filings": [{"form": "424B5"}],
            "late_filings": [],
            "failures_to_deliver": [],
        }
        result = analyst.analyze_insider("XYZ", signals)
        assert result.score == pytest.approx(-0.3)

    def test_late_filing_strong_negative(self, analyst: Analyst) -> None:
        signals = {
            "cluster_buys": [],
            "officer_buys": [],
            "dilution_filings": [],
            "late_filings": [{"form": "NT 10-K"}],
            "failures_to_deliver": [],
        }
        result = analyst.analyze_insider("XYZ", signals)
        assert result.score == pytest.approx(-0.5)

    def test_high_ftd_volume_flagged(self, analyst: Analyst) -> None:
        signals = {
            "cluster_buys": [],
            "officer_buys": [],
            "dilution_filings": [],
            "late_filings": [],
            "failures_to_deliver": [
                {"qty": 3_000_000},
                {"qty": 4_000_000},
            ],
        }
        result = analyst.analyze_insider("XYZ", signals)
        assert result.score == pytest.approx(-0.6)

    def test_score_clamps_to_minus_one(self, analyst: Analyst) -> None:
        """Pile up enough negative signals to verify the [-1, +1] clamp."""
        signals = {
            "cluster_buys": [],
            "officer_buys": [],
            "dilution_filings": [
                {"form": "S-3"},
                {"form": "424B5"},
                {"form": "S-1"},
            ],
            "late_filings": [
                {"form": "NT 10-K"},
                {"form": "NT 10-Q"},
            ],
            "failures_to_deliver": [{"qty": 6_000_000}],
        }
        result = analyst.analyze_insider("XYZ", signals)
        assert result.score == pytest.approx(-1.0)

    def test_failed_fetches_excluded_not_treated_as_zero(
        self, analyst: Analyst
    ) -> None:
        """Failed fetches must be marked unknown; never scored as 0 because
        that would silently look bullish in a sea of negatives."""
        signals = {
            "cluster_buys": None,
            "officer_buys": None,
            "dilution_filings": [{"form": "S-3"}],
            "late_filings": None,
            "failures_to_deliver": None,
        }
        result = analyst.analyze_insider("XYZ", signals)
        assert "cluster_buys" in result.signals_unknown
        assert "officer_buys" in result.signals_unknown
        assert "late_filings" in result.signals_unknown
        assert "failures_to_deliver" in result.signals_unknown
        # Only dilution scored
        assert result.score == pytest.approx(-0.3)


# -------- SignalAggregator gate bonus --------


@pytest.fixture
def aggregator() -> SignalAggregator:
    return SignalAggregator(
        sentiment_weight=0.3,
        technical_weight=0.4,
        fundamental_weight=0.3,
        min_agreement=3,
    )


def _flat_inputs():
    """All-zero analyst signals so the only thing moving combined_score is
    the insider gate bonus."""
    return dict(
        sentiment=SentimentResult(
            score=0.0, signal=Signal.HOLD, reasoning="x", key_factors=[]
        ),
        technical=TechnicalResult(
            score=0.0, signal=Signal.HOLD, pattern="x", reasoning="x"
        ),
        fundamental=FundamentalResult(
            score=0.0, signal=Signal.HOLD, reasoning="x"
        ),
        debate=DebateResult(
            bull_score=0.5,
            bear_score=0.5,
            verdict=Signal.HOLD,
            bull_argument="x",
            bear_argument="x",
        ),
    )


class TestGateBonus:
    def test_no_insider_means_no_bonus(
        self, aggregator: SignalAggregator
    ) -> None:
        result = aggregator.aggregate(symbol="AAPL", **_flat_inputs())
        assert result.insider_bonus == 0.0
        assert result.insider is None

    def test_positive_insider_adds_max_one_tenth(
        self, aggregator: SignalAggregator
    ) -> None:
        insider = InsiderResult(score=1.0, reasoning="strong")
        result = aggregator.aggregate(
            symbol="AAPL",
            insider=insider,
            **_flat_inputs(),
        )
        assert result.insider_bonus == pytest.approx(0.1)
        assert result.combined_score == pytest.approx(0.1)

    def test_negative_insider_subtracts_at_most_one_tenth(
        self, aggregator: SignalAggregator
    ) -> None:
        insider = InsiderResult(score=-1.0, reasoning="bad")
        result = aggregator.aggregate(
            symbol="AAPL",
            insider=insider,
            **_flat_inputs(),
        )
        assert result.insider_bonus == pytest.approx(-0.1)
        assert result.combined_score == pytest.approx(-0.1)

    def test_insider_score_scales_linearly(
        self, aggregator: SignalAggregator
    ) -> None:
        insider = InsiderResult(score=0.4, reasoning="ok")
        result = aggregator.aggregate(
            symbol="AAPL",
            insider=insider,
            **_flat_inputs(),
        )
        assert result.insider_bonus == pytest.approx(0.04)

    def test_insider_cannot_dominate_strong_negative_composite(
        self, aggregator: SignalAggregator
    ) -> None:
        """Strong bearish analyst signals must not be flipped to bullish by
        a +0.1 insider bonus - the gate bonus is a tiebreaker, not an axis."""
        bearish = dict(
            sentiment=SentimentResult(
                score=-0.8, signal=Signal.STRONG_SELL, reasoning="x", key_factors=[]
            ),
            technical=TechnicalResult(
                score=-0.8, signal=Signal.STRONG_SELL, pattern="x", reasoning="x"
            ),
            fundamental=FundamentalResult(
                score=-0.8, signal=Signal.STRONG_SELL, reasoning="x"
            ),
            debate=DebateResult(
                bull_score=0.0,
                bear_score=1.0,
                verdict=Signal.STRONG_SELL,
                bull_argument="x",
                bear_argument="x",
            ),
        )
        insider = InsiderResult(score=1.0, reasoning="cluster bought")
        result = aggregator.aggregate(symbol="AAPL", insider=insider, **bearish)
        # -0.8 weighted * 0.7 + (-1.0) * 0.3 + 0.1 = ~-0.76 - still strongly negative.
        assert result.combined_score < -0.5
        assert result.final_signal in (Signal.SELL, Signal.STRONG_SELL)
