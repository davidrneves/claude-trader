"""Tests for multi-agent analysis: debate, aggregation, contrarian filter."""

import pytest

from claude_trader.analyst import (
    Signal,
    SentimentResult,
    TechnicalResult,
    FundamentalResult,
    DebateResult,
    MultiAgentAnalysis,
    SignalAggregator,
)


@pytest.fixture
def aggregator():
    return SignalAggregator(
        sentiment_weight=0.3,
        technical_weight=0.4,
        fundamental_weight=0.3,
        min_agreement=3,
        min_agent_count=4,
    )


# --- Signal Aggregation ---

class TestSignalAggregation:
    def test_unanimous_buy(self, aggregator):
        result = aggregator.aggregate(
            symbol="AAPL",
            sentiment=SentimentResult(score=0.7, signal=Signal.BUY, reasoning="Positive news", key_factors=["earnings beat"]),
            technical=TechnicalResult(score=0.8, signal=Signal.BUY, pattern="bullish crossover", reasoning="EMA cross"),
            fundamental=FundamentalResult(score=0.6, signal=Signal.BUY, reasoning="Undervalued", pe_ratio=18.5, revenue_growth=0.12),
            debate=DebateResult(bull_score=0.7, bear_score=0.3, verdict=Signal.BUY, bull_argument="Strong momentum", bear_argument="Overextended short-term"),
        )
        assert result.final_signal in (Signal.BUY, Signal.STRONG_BUY)
        assert result.agreement_count == 4
        assert result.combined_score > 0.5

    def test_unanimous_sell(self, aggregator):
        result = aggregator.aggregate(
            symbol="AAPL",
            sentiment=SentimentResult(score=-0.6, signal=Signal.SELL, reasoning="Negative", key_factors=[]),
            technical=TechnicalResult(score=-0.7, signal=Signal.SELL, pattern="bearish", reasoning="Below EMA"),
            fundamental=FundamentalResult(score=-0.5, signal=Signal.SELL, reasoning="Overvalued", pe_ratio=45.0, revenue_growth=-0.05),
            debate=DebateResult(bull_score=0.2, bear_score=0.8, verdict=Signal.SELL, bull_argument="Weak", bear_argument="Strong bearish case"),
        )
        assert result.final_signal in (Signal.SELL, Signal.STRONG_SELL)
        assert result.agreement_count == 4

    def test_mixed_signals_hold(self, aggregator):
        result = aggregator.aggregate(
            symbol="AAPL",
            sentiment=SentimentResult(score=0.5, signal=Signal.BUY, reasoning="Positive", key_factors=[]),
            technical=TechnicalResult(score=-0.3, signal=Signal.SELL, pattern="mixed", reasoning="Choppy"),
            fundamental=FundamentalResult(score=0.1, signal=Signal.HOLD, reasoning="Fair value", pe_ratio=25.0, revenue_growth=0.03),
            debate=DebateResult(bull_score=0.5, bear_score=0.5, verdict=Signal.HOLD, bull_argument="Some upside", bear_argument="Some downside"),
        )
        # With only 1 buy out of 4 agents, should not reach agreement
        assert result.final_signal == Signal.HOLD
        assert result.agreement_count < 3

    def test_insufficient_agreement_defaults_to_hold(self, aggregator):
        result = aggregator.aggregate(
            symbol="AAPL",
            sentiment=SentimentResult(score=0.8, signal=Signal.STRONG_BUY, reasoning="Very bullish", key_factors=[]),
            technical=TechnicalResult(score=-0.6, signal=Signal.SELL, pattern="bearish", reasoning="Downtrend"),
            fundamental=FundamentalResult(score=0.3, signal=Signal.BUY, reasoning="Slightly undervalued", pe_ratio=20.0, revenue_growth=0.08),
            debate=DebateResult(bull_score=0.4, bear_score=0.6, verdict=Signal.SELL, bull_argument="Sentiment strong", bear_argument="Technicals weak"),
        )
        # 2 buy, 2 sell -> no agreement of 3 -> hold
        assert result.final_signal == Signal.HOLD


# --- Contrarian Filter ---

class TestContrarianFilter:
    def test_contrarian_buy_negative_sentiment_positive_technical(self, aggregator):
        """Research finding: buying on negative sentiment + positive technical outperforms."""
        result = aggregator.aggregate(
            symbol="AAPL",
            sentiment=SentimentResult(score=-0.6, signal=Signal.SELL, reasoning="Fear in market", key_factors=["selloff"]),
            technical=TechnicalResult(score=0.7, signal=Signal.BUY, pattern="oversold bounce", reasoning="RSI oversold"),
            fundamental=FundamentalResult(score=0.4, signal=Signal.BUY, reasoning="Fundamentals intact", pe_ratio=18.0, revenue_growth=0.10),
            debate=DebateResult(bull_score=0.6, bear_score=0.4, verdict=Signal.BUY, bull_argument="Contrarian opportunity", bear_argument="Sentiment risk"),
        )
        assert result.contrarian_signal is True
        # Contrarian filter should boost confidence
        assert result.combined_score > 0

    def test_no_contrarian_when_both_positive(self, aggregator):
        result = aggregator.aggregate(
            symbol="AAPL",
            sentiment=SentimentResult(score=0.5, signal=Signal.BUY, reasoning="Bullish", key_factors=[]),
            technical=TechnicalResult(score=0.6, signal=Signal.BUY, pattern="uptrend", reasoning="Trending up"),
            fundamental=FundamentalResult(score=0.4, signal=Signal.BUY, reasoning="OK", pe_ratio=22.0, revenue_growth=0.06),
            debate=DebateResult(bull_score=0.7, bear_score=0.3, verdict=Signal.BUY, bull_argument="Strong", bear_argument="Weak"),
        )
        assert result.contrarian_signal is False


# --- Debate Mechanism ---

class TestDebateScoring:
    def test_debate_result_model(self):
        d = DebateResult(
            bull_score=0.7,
            bear_score=0.3,
            verdict=Signal.BUY,
            bull_argument="Strong earnings growth and momentum",
            bear_argument="Valuation stretched but manageable",
        )
        assert d.bull_score > d.bear_score
        assert d.verdict == Signal.BUY

    def test_debate_bear_wins(self):
        d = DebateResult(
            bull_score=0.2,
            bear_score=0.8,
            verdict=Signal.SELL,
            bull_argument="Some support level nearby",
            bear_argument="Broken trend, declining volume, sector rotation",
        )
        assert d.bear_score > d.bull_score
        assert d.verdict == Signal.SELL


# --- Fundamental Analysis ---

class TestFundamentalResult:
    def test_fundamental_result_model(self):
        f = FundamentalResult(
            score=0.5,
            signal=Signal.BUY,
            reasoning="Undervalued relative to sector",
            pe_ratio=15.0,
            revenue_growth=0.15,
        )
        assert f.pe_ratio == 15.0
        assert f.revenue_growth == 0.15

    def test_fundamental_no_data(self):
        f = FundamentalResult(
            score=0.0,
            signal=Signal.HOLD,
            reasoning="No data",
        )
        assert f.pe_ratio is None
        assert f.revenue_growth is None


# --- MultiAgentAnalysis full model ---

class TestMultiAgentAnalysisModel:
    def test_full_model(self):
        m = MultiAgentAnalysis(
            symbol="AAPL",
            sentiment=SentimentResult(score=0.5, signal=Signal.BUY, reasoning="ok", key_factors=[]),
            technical=TechnicalResult(score=0.6, signal=Signal.BUY, pattern="up", reasoning="ok"),
            fundamental=FundamentalResult(score=0.4, signal=Signal.BUY, reasoning="ok"),
            debate=DebateResult(bull_score=0.6, bear_score=0.4, verdict=Signal.BUY, bull_argument="yes", bear_argument="no"),
            combined_score=0.5,
            final_signal=Signal.BUY,
            agreement_count=4,
            contrarian_signal=False,
            reasoning="All agents agree",
        )
        assert m.symbol == "AAPL"
        assert m.agreement_count == 4
