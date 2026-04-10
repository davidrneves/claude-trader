"""Integration tests for analyst agents - calls real Gemini API.

Skipped unless GEMINI_API_KEY is set. Tests STRUCTURE not VALUES
since model output is non-deterministic.
"""

import os

import pytest

from claude_trader.analyst import (
    Analyst,
    DebateResult,
    FundamentalResult,
    MultiAgentAnalysis,
    SentimentResult,
    Signal,
    TechnicalResult,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY", ""),
    reason="GEMINI_API_KEY not set",
)

KNOWN_SYMBOL = "AAPL"

HEADLINES = [
    "Apple reports record Q4 revenue beating analyst expectations",
    "iPhone sales decline 5% in key China market amid competition",
    "Apple announces $100B stock buyback program",
    "Analysts downgrade Apple citing slowing services growth",
    "Apple Vision Pro sales disappoint in first full quarter",
]

PRICES = [
    {
        "date": f"2025-03-{d:02d}",
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "volume": v,
    }
    for d, o, h, lo, c, v in [
        (3, 170.00, 172.50, 169.00, 171.80, 48_000_000),
        (4, 171.80, 173.00, 170.50, 172.30, 42_000_000),
        (5, 172.30, 174.00, 171.00, 173.50, 51_000_000),
        (6, 173.50, 175.20, 172.80, 174.90, 55_000_000),
        (7, 174.90, 176.00, 174.00, 175.60, 47_000_000),
        (10, 175.60, 177.30, 174.50, 176.80, 53_000_000),
        (11, 176.80, 178.00, 175.20, 177.40, 49_000_000),
        (12, 177.40, 179.50, 176.00, 178.90, 58_000_000),
        (13, 178.90, 180.00, 177.50, 179.20, 52_000_000),
        (14, 179.20, 180.50, 178.00, 179.80, 46_000_000),
        (17, 179.80, 181.00, 178.50, 180.50, 50_000_000),
        (18, 180.50, 182.30, 179.00, 181.70, 54_000_000),
        (19, 181.70, 183.00, 180.50, 182.40, 48_000_000),
        (20, 182.40, 183.50, 181.00, 182.00, 43_000_000),
        (21, 182.00, 183.00, 180.00, 180.50, 56_000_000),
        (24, 180.50, 181.50, 179.00, 179.80, 47_000_000),
        (25, 179.80, 181.00, 178.50, 180.20, 44_000_000),
        (26, 180.20, 182.00, 179.50, 181.50, 51_000_000),
        (27, 181.50, 183.00, 180.00, 182.80, 53_000_000),
        (28, 182.80, 184.00, 181.50, 183.50, 57_000_000),
        (31, 183.50, 185.00, 182.00, 184.20, 49_000_000),
        (1, 184.20, 186.00, 183.00, 185.50, 62_000_000),
        (2, 185.50, 187.00, 184.50, 186.30, 55_000_000),
        (3, 186.30, 188.00, 185.00, 187.00, 51_000_000),
        (4, 187.00, 188.50, 186.00, 187.80, 48_000_000),
    ]
]

FINANCIALS = {
    "pe_ratio": 28.5,
    "revenue_growth": 0.08,
    "profit_margin": 0.26,
    "debt_to_equity": 1.8,
    "free_cash_flow": 100_000_000_000,
}


@pytest.fixture(scope="module")
def analyst():
    return Analyst(api_key=os.environ["GEMINI_API_KEY"])


@pytest.mark.integration
class TestSentimentAgent:
    def test_valid_result(self, analyst):
        result = analyst.analyze_sentiment(KNOWN_SYMBOL, HEADLINES)
        assert isinstance(result, SentimentResult)
        assert -1.0 <= result.score <= 1.0
        assert isinstance(result.signal, Signal)
        assert len(result.reasoning) > 0
        assert isinstance(result.key_factors, list)

    def test_empty_headlines_fallback(self, analyst):
        result = analyst.analyze_sentiment(KNOWN_SYMBOL, [])
        assert isinstance(result, SentimentResult)
        assert result.score == 0.0
        assert result.signal == Signal.HOLD


@pytest.mark.integration
class TestTechnicalAgent:
    def test_valid_result(self, analyst):
        result = analyst.analyze_technical(KNOWN_SYMBOL, PRICES)
        assert isinstance(result, TechnicalResult)
        assert -1.0 <= result.score <= 1.0
        assert isinstance(result.signal, Signal)
        assert len(result.pattern) > 0
        assert len(result.reasoning) > 0

    def test_empty_prices_fallback(self, analyst):
        result = analyst.analyze_technical(KNOWN_SYMBOL, [])
        assert isinstance(result, TechnicalResult)
        assert result.score == 0.0
        assert result.signal == Signal.HOLD
        assert result.pattern == "no_data"


@pytest.mark.integration
class TestFundamentalAgent:
    def test_valid_result_with_financials(self, analyst):
        result = analyst.analyze_fundamental(KNOWN_SYMBOL, FINANCIALS)
        assert isinstance(result, FundamentalResult)
        assert -1.0 <= result.score <= 1.0
        assert isinstance(result.signal, Signal)
        assert len(result.reasoning) > 0

    def test_valid_result_without_financials(self, analyst):
        result = analyst.analyze_fundamental(KNOWN_SYMBOL)
        assert isinstance(result, FundamentalResult)
        assert -1.0 <= result.score <= 1.0
        assert isinstance(result.signal, Signal)
        assert len(result.reasoning) > 0


@pytest.mark.integration
class TestDebate:
    def test_valid_result(self, analyst):
        sentiment = analyst.analyze_sentiment(KNOWN_SYMBOL, HEADLINES)
        technical = analyst.analyze_technical(KNOWN_SYMBOL, PRICES)
        fundamental = analyst.analyze_fundamental(KNOWN_SYMBOL, FINANCIALS)
        result = analyst.run_debate(KNOWN_SYMBOL, sentiment, technical, fundamental)
        assert isinstance(result, DebateResult)
        assert 0.0 <= result.bull_score <= 1.0
        assert 0.0 <= result.bear_score <= 1.0
        assert isinstance(result.verdict, Signal)
        assert len(result.bull_argument) > 0
        assert len(result.bear_argument) > 0


@pytest.mark.integration
class TestFullAnalysisPipeline:
    def test_complete_analysis(self, analyst):
        result = analyst.full_analysis(
            symbol=KNOWN_SYMBOL,
            headlines=HEADLINES,
            prices=PRICES,
            financials=FINANCIALS,
        )
        assert isinstance(result, MultiAgentAnalysis)
        assert result.symbol == KNOWN_SYMBOL
        assert isinstance(result.sentiment, SentimentResult)
        assert isinstance(result.technical, TechnicalResult)
        assert isinstance(result.fundamental, FundamentalResult)
        assert isinstance(result.debate, DebateResult)
        assert -1.0 <= result.combined_score <= 1.0
        assert isinstance(result.final_signal, Signal)
        assert result.agreement_count >= 0
        assert isinstance(result.contrarian_signal, bool)
        assert len(result.reasoning) > 0
