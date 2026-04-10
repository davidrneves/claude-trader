"""Multi-agent analysis system - the hybrid approach.

Uses Gemini as the LLM backend for analysis agents. Four specialized agents:
- Sentiment Agent: news/social media sentiment scoring
- Technical Agent: price pattern and indicator analysis
- Fundamental Agent: valuation and financial health
- Bull/Bear Debate: structured opposing arguments

Research finding: multi-agent frameworks outperform single-agent.
Research finding: contrarian sentiment often outperforms naive following.
"""

import json
from collections.abc import Callable
from enum import Enum
from typing import TypeVar

import structlog
from google import genai
from pydantic import BaseModel, Field

log = structlog.get_logger()

ANALYSIS_MODEL = "gemini-3-flash-preview"

T = TypeVar("T", bound=BaseModel)


class GeminiError(Exception):
    """Base exception for Gemini API errors."""


class GeminiParseError(GeminiError):
    """Gemini returned a response that couldn't be parsed as JSON."""


class Signal(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


# --- Agent Result Models ---


class SentimentResult(BaseModel):
    score: float = Field(ge=-1.0, le=1.0, description="-1 bearish to +1 bullish")
    signal: Signal
    reasoning: str
    key_factors: list[str] = Field(default_factory=list)


class TechnicalResult(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    signal: Signal
    pattern: str = Field(description="Primary pattern identified")
    support_level: float | None = None
    resistance_level: float | None = None
    reasoning: str


class FundamentalResult(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    signal: Signal
    reasoning: str
    pe_ratio: float | None = None
    revenue_growth: float | None = None


class DebateResult(BaseModel):
    bull_score: float = Field(ge=0.0, le=1.0, description="Strength of bull case")
    bear_score: float = Field(ge=0.0, le=1.0, description="Strength of bear case")
    verdict: Signal
    bull_argument: str
    bear_argument: str


class MultiAgentAnalysis(BaseModel):
    symbol: str
    sentiment: SentimentResult | None = None
    technical: TechnicalResult | None = None
    fundamental: FundamentalResult | None = None
    debate: DebateResult | None = None
    combined_score: float = Field(ge=-1.0, le=1.0)
    final_signal: Signal
    agreement_count: int = Field(description="How many agents agree on direction")
    contrarian_signal: bool = Field(
        default=False, description="True if contrarian filter triggered"
    )
    reasoning: str


# --- Signal Aggregator ---


def _signal_direction(signal: Signal) -> str:
    if signal in (Signal.STRONG_BUY, Signal.BUY):
        return "buy"
    if signal in (Signal.STRONG_SELL, Signal.SELL):
        return "sell"
    return "neutral"


class SignalAggregator:
    """Combines signals from multiple agents with minimum agreement threshold."""

    def __init__(
        self,
        sentiment_weight: float = 0.3,
        technical_weight: float = 0.4,
        fundamental_weight: float = 0.3,
        min_agreement: int = 3,
    ) -> None:
        self.sentiment_weight = sentiment_weight
        self.technical_weight = technical_weight
        self.fundamental_weight = fundamental_weight
        self.min_agreement = min_agreement

    def aggregate(
        self,
        symbol: str,
        sentiment: SentimentResult,
        technical: TechnicalResult,
        fundamental: FundamentalResult,
        debate: DebateResult,
    ) -> MultiAgentAnalysis:
        weighted_score = (
            sentiment.score * self.sentiment_weight
            + technical.score * self.technical_weight
            + fundamental.score * self.fundamental_weight
        )

        debate_score = debate.bull_score - debate.bear_score
        combined = weighted_score * 0.7 + debate_score * 0.3

        contrarian = sentiment.score < -0.3 and technical.score > 0.3
        if contrarian:
            combined = combined + 0.15
            log.info(
                "contrarian_signal",
                symbol=symbol,
                sentiment=sentiment.score,
                technical=technical.score,
            )

        combined = max(-1.0, min(1.0, combined))

        directions = [
            _signal_direction(sentiment.signal),
            _signal_direction(technical.signal),
            _signal_direction(fundamental.signal),
            _signal_direction(debate.verdict),
        ]
        buy_count = directions.count("buy")
        sell_count = directions.count("sell")
        agreement_count = max(buy_count, sell_count)
        majority_direction = (
            "buy"
            if buy_count > sell_count
            else "sell"
            if sell_count > buy_count
            else "neutral"
        )

        if agreement_count >= self.min_agreement and majority_direction == "buy":
            final_signal = Signal.STRONG_BUY if combined >= 0.5 else Signal.BUY
        elif agreement_count >= self.min_agreement and majority_direction == "sell":
            final_signal = Signal.STRONG_SELL if combined <= -0.5 else Signal.SELL
        else:
            final_signal = Signal.HOLD

        reasoning_parts = [
            f"Sentiment: {sentiment.score:.2f} ({sentiment.signal.value})",
            f"Technical: {technical.score:.2f} ({technical.signal.value})",
            f"Fundamental: {fundamental.score:.2f} ({fundamental.signal.value})",
            f"Debate: bull={debate.bull_score:.2f} bear={debate.bear_score:.2f} -> {debate.verdict.value}",
            f"Agreement: {agreement_count}/{len(directions)} {majority_direction}",
        ]
        if contrarian:
            reasoning_parts.append(
                "CONTRARIAN: negative sentiment + positive technical"
            )

        return MultiAgentAnalysis(
            symbol=symbol,
            sentiment=sentiment,
            technical=technical,
            fundamental=fundamental,
            debate=debate,
            combined_score=round(combined, 3),
            final_signal=final_signal,
            agreement_count=agreement_count,
            contrarian_signal=contrarian,
            reasoning=" | ".join(reasoning_parts),
        )


# --- Gemini-Powered Agents ---


def _call_gemini(client: genai.Client, prompt: str) -> dict:
    """Make a Gemini API call and parse JSON response.

    Raises GeminiError or GeminiParseError on failure.
    """
    try:
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=8192,
                temperature=0.2,
            ),
        )
    except Exception as e:
        raise GeminiError(f"API call failed: {e}") from e
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise GeminiParseError(f"Failed to parse response: {e}") from e


class Analyst:
    """Multi-agent market analyst powered by Gemini. Analysis only - no trade execution."""

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key) if api_key else None
        self._aggregator = SignalAggregator()

    def _analyze_with_fallback(
        self,
        symbol: str,
        prompt_fn: Callable[[], str],
        result_cls: type[T],
        fallback_kwargs: dict,
        log_prefix: str,
    ) -> T:
        """Run a Gemini analysis with consistent error handling and fallback."""
        if not self._client:
            return result_cls(**fallback_kwargs)

        try:
            data = _call_gemini(self._client, prompt_fn())
            return result_cls(**data)
        except (GeminiError, GeminiParseError) as e:
            log.warning(
                "agent_api_failed", agent=log_prefix, symbol=symbol, error=str(e)
            )
        except (KeyError, TypeError, ValueError) as e:
            log.warning(
                "agent_parse_failed", agent=log_prefix, symbol=symbol, error=str(e)
            )
        return result_cls(**fallback_kwargs)

    def analyze_sentiment(self, symbol: str, headlines: list[str]) -> SentimentResult:
        fallback = {
            "score": 0.0,
            "signal": Signal.HOLD,
            "reasoning": "No data available",
            "key_factors": [],
        }
        if not headlines:
            return SentimentResult(**fallback)

        headlines_text = "\n".join(f"- {h}" for h in headlines[:20])

        def prompt() -> str:
            return f"""Analyze the sentiment of these news headlines for {symbol}.

Headlines:
{headlines_text}

IMPORTANT: Research shows contrarian sentiment often outperforms naive following.
Consider whether the market has already priced in the sentiment.

Return a JSON object with these exact fields:
- "score": float between -1.0 (very bearish) and 1.0 (very bullish)
- "signal": one of "strong_buy", "buy", "hold", "sell", "strong_sell"
- "reasoning": brief explanation (1-2 sentences)
- "key_factors": list of 2-3 key factors driving the sentiment"""

        return self._analyze_with_fallback(
            symbol, prompt, SentimentResult, fallback, "sentiment"
        )

    def analyze_technical(self, symbol: str, prices: list[dict]) -> TechnicalResult:
        fallback = {
            "score": 0.0,
            "signal": Signal.HOLD,
            "pattern": "no_data",
            "reasoning": "No data",
        }
        if not prices:
            return TechnicalResult(**fallback)

        price_text = "\n".join(
            f"  {p.get('date', 'N/A')}: O={p.get('open'):.2f} H={p.get('high'):.2f} L={p.get('low'):.2f} C={p.get('close'):.2f} V={p.get('volume', 0)}"
            for p in prices[-30:]
        )

        def prompt() -> str:
            return f"""Analyze the technical indicators for {symbol}.

Recent OHLCV data:
{price_text}

Consider: trend direction, momentum, support/resistance, chart patterns.

Return a JSON object:
- "score": float -1.0 to 1.0
- "signal": one of "strong_buy", "buy", "hold", "sell", "strong_sell"
- "pattern": primary pattern identified (string)
- "support_level": nearest support price (float or null)
- "resistance_level": nearest resistance price (float or null)
- "reasoning": 1-2 sentences"""

        return self._analyze_with_fallback(
            symbol, prompt, TechnicalResult, fallback, "technical"
        )

    def analyze_fundamental(
        self, symbol: str, financials: dict | None = None
    ) -> FundamentalResult:
        fallback = {"score": 0.0, "signal": Signal.HOLD, "reasoning": "No API key"}

        context = ""
        if financials:
            context = f"\nFinancial data:\n{json.dumps(financials, indent=2)}"

        def prompt() -> str:
            return f"""Analyze the fundamental valuation of {symbol}.{context}

Consider: P/E ratio relative to sector, revenue growth trends, margins, debt levels.
If no specific data is provided, use your knowledge of the company's recent financials.

Return a JSON object:
- "score": float -1.0 (very overvalued) to 1.0 (very undervalued)
- "signal": one of "strong_buy", "buy", "hold", "sell", "strong_sell"
- "reasoning": 1-2 sentences
- "pe_ratio": estimated P/E ratio (float or null)
- "revenue_growth": estimated YoY revenue growth as decimal (float or null)"""

        return self._analyze_with_fallback(
            symbol, prompt, FundamentalResult, fallback, "fundamental"
        )

    def run_debate(
        self,
        symbol: str,
        sentiment: SentimentResult,
        technical: TechnicalResult,
        fundamental: FundamentalResult,
    ) -> DebateResult:
        fallback = {
            "bull_score": 0.5,
            "bear_score": 0.5,
            "verdict": Signal.HOLD,
            "bull_argument": "No data",
            "bear_argument": "No data",
        }

        context = f"""Symbol: {symbol}
Sentiment: score={sentiment.score:.2f}, signal={sentiment.signal.value}, reasoning="{sentiment.reasoning}"
Technical: score={technical.score:.2f}, signal={technical.signal.value}, pattern="{technical.pattern}", reasoning="{technical.reasoning}"
Fundamental: score={fundamental.score:.2f}, signal={fundamental.signal.value}, reasoning="{fundamental.reasoning}" """

        def prompt() -> str:
            return f"""You are a trading debate judge. Given the analyst reports below,
construct the strongest Bull case and the strongest Bear case for {symbol},
then score each and render a verdict.

{context}

Rules:
- Each argument must cite specific data from the analyst reports
- Score each case 0.0 (very weak) to 1.0 (very strong)
- Verdict should reflect which case is more compelling overall

Return a JSON object:
- "bull_score": float 0.0-1.0
- "bear_score": float 0.0-1.0
- "verdict": one of "strong_buy", "buy", "hold", "sell", "strong_sell"
- "bull_argument": 2-3 sentence bull case
- "bear_argument": 2-3 sentence bear case"""

        return self._analyze_with_fallback(
            symbol, prompt, DebateResult, fallback, "debate"
        )

    def full_analysis(
        self,
        symbol: str,
        headlines: list[str] | None = None,
        prices: list[dict] | None = None,
        financials: dict | None = None,
    ) -> MultiAgentAnalysis:
        """Run the full multi-agent pipeline: 3 analysts + debate + aggregation."""
        sentiment = self.analyze_sentiment(symbol, headlines or [])
        technical = self.analyze_technical(symbol, prices or [])
        fundamental = self.analyze_fundamental(symbol, financials)
        debate = self.run_debate(symbol, sentiment, technical, fundamental)

        result = self._aggregator.aggregate(
            symbol=symbol,
            sentiment=sentiment,
            technical=technical,
            fundamental=fundamental,
            debate=debate,
        )

        log.info(
            "multi_agent_analysis_complete",
            symbol=symbol,
            final_signal=result.final_signal.value,
            combined_score=result.combined_score,
            agreement=result.agreement_count,
            contrarian=result.contrarian_signal,
        )
        return result
