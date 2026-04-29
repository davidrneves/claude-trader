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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

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


class InsiderResult(BaseModel):
    """Rule-based aggregation of openinsider + SEC filing signals.

    Score is a deterministic mapping from signal counts/thresholds, not an
    LLM call - the underlying data is structured and noisy LLM scoring would
    just add latency and variance.
    """

    score: float = Field(ge=-1.0, le=1.0, description="-1 bearish to +1 bullish")
    signals_seen: dict[str, int] = Field(
        default_factory=dict,
        description="Per-signal count of qualifying events. None values omitted.",
    )
    signals_unknown: list[str] = Field(
        default_factory=list,
        description="Signals whose fetch failed - excluded from scoring.",
    )
    reasoning: str = ""


class MultiAgentAnalysis(BaseModel):
    symbol: str
    sentiment: SentimentResult | None = None
    technical: TechnicalResult | None = None
    fundamental: FundamentalResult | None = None
    debate: DebateResult | None = None
    insider: InsiderResult | None = None
    combined_score: float = Field(ge=-1.0, le=1.0)
    final_signal: Signal
    agreement_count: int = Field(description="How many agents agree on direction")
    contrarian_signal: bool = Field(
        default=False, description="True if contrarian filter triggered"
    )
    insider_bonus: float = Field(
        default=0.0, description="Gate-bonus applied from insider signals"
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
        insider: InsiderResult | None = None,
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

        # Insider gate-bonus: max +/-0.1 nudge so insider data can break ties
        # but cannot dominate the sentiment+technical+fundamental composite.
        # Sign follows insider.score, magnitude scales linearly. Skipped
        # entirely when insider is None (feature flag off or fetcher disabled).
        insider_bonus = 0.0
        if insider is not None:
            insider_bonus = max(-0.1, min(0.1, insider.score * 0.1))
            combined = combined + insider_bonus

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
        if insider is not None and insider_bonus != 0.0:
            reasoning_parts.append(
                f"Insider: {insider.score:+.2f} (bonus {insider_bonus:+.2f})"
            )

        return MultiAgentAnalysis(
            symbol=symbol,
            sentiment=sentiment,
            technical=technical,
            fundamental=fundamental,
            debate=debate,
            insider=insider,
            combined_score=round(combined, 3),
            final_signal=final_signal,
            agreement_count=agreement_count,
            contrarian_signal=contrarian,
            insider_bonus=round(insider_bonus, 3),
            reasoning=" | ".join(reasoning_parts),
        )


# --- Gemini-Powered Agents ---


def _log_retry(retry_state) -> None:
    log.warning(
        "gemini_retry",
        attempt=retry_state.attempt_number,
        error=str(retry_state.outcome.exception()),
    )


@retry(
    retry=retry_if_exception_type(GeminiError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=16) + wait_random(0, 2),
    before_sleep=_log_retry,
    reraise=True,
)
def _call_gemini(client: genai.Client, prompt: str) -> dict:
    """Make a Gemini API call and parse JSON response.

    Retries up to 4 times on transient errors (connection resets, timeouts)
    with exponential backoff (2-16s) plus jitter to avoid thundering herd.
    Does NOT retry parse errors (GeminiParseError) since those won't self-heal.

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
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                raise GeminiParseError(
                    f"Expected JSON object, got {type(data).__name__}"
                )
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

    def analyze_insider(
        self, symbol: str, signals: dict | None
    ) -> InsiderResult:
        """Score insider/SEC signals deterministically into a [-1, +1] composite.

        Inputs are the per-signal results from ``InsiderFeed.get_full_signals``:
        each value is a list of qualifying events or ``None`` (fetch failure).
        Failed fetches are recorded in ``signals_unknown`` and excluded from
        scoring - they are treated as absence of information, not as bullish
        or bearish.

        Scoring rules (all magnitudes chosen so a single positive cluster buy
        is ~+0.4 and a single dilution + late filing is ~-0.6, leaving room
        for compound effects without saturating):
        - cluster_buys with insider_count >= 3:   +0.4 each (cap +0.8)
        - officer_buys (CEO/CFO/Dir/10%/Pres):    +0.2 each (cap +0.6)
        - dilution filings within window:         -0.3 each (cap -0.6)
        - late filings (NT 10-K/Q):               -0.5 each (cap -0.8)
        - failures_to_deliver: aggregate qty in last published period
            >  500,000 shares: -0.3
            > 5,000,000 shares: -0.6 (cap)
        Final score is clamped to [-1.0, +1.0].
        """
        if signals is None:
            return InsiderResult(
                score=0.0,
                signals_unknown=["all"],
                reasoning="Insider feed unavailable",
            )

        score = 0.0
        seen: dict[str, int] = {}
        unknown: list[str] = []
        parts: list[str] = []

        cluster = signals.get("cluster_buys")
        if cluster is None:
            unknown.append("cluster_buys")
        else:
            qualifying = [c for c in cluster if (c.get("insider_count") or 0) >= 3]
            seen["cluster_buys"] = len(qualifying)
            bonus = min(0.8, 0.4 * len(qualifying))
            if bonus:
                score += bonus
                parts.append(f"cluster_buys={len(qualifying)} (+{bonus:.2f})")

        officer = signals.get("officer_buys")
        if officer is None:
            unknown.append("officer_buys")
        else:
            seen["officer_buys"] = len(officer)
            bonus = min(0.6, 0.2 * len(officer))
            if bonus:
                score += bonus
                parts.append(f"officer_buys={len(officer)} (+{bonus:.2f})")

        dilution = signals.get("dilution_filings")
        if dilution is None:
            unknown.append("dilution_filings")
        else:
            seen["dilution_filings"] = len(dilution)
            penalty = max(-0.6, -0.3 * len(dilution))
            if penalty:
                score += penalty
                parts.append(f"dilution_filings={len(dilution)} ({penalty:+.2f})")

        late = signals.get("late_filings")
        if late is None:
            unknown.append("late_filings")
        else:
            seen["late_filings"] = len(late)
            penalty = max(-0.8, -0.5 * len(late))
            if penalty:
                score += penalty
                parts.append(f"late_filings={len(late)} ({penalty:+.2f})")

        ftds = signals.get("failures_to_deliver")
        if ftds is None:
            unknown.append("failures_to_deliver")
        else:
            seen["failures_to_deliver"] = len(ftds)
            total_qty = sum((r.get("qty") or 0) for r in ftds)
            ftd_penalty = 0.0
            if total_qty > 5_000_000:
                ftd_penalty = -0.6
            elif total_qty > 500_000:
                ftd_penalty = -0.3
            if ftd_penalty:
                score += ftd_penalty
                parts.append(f"ftd_qty={total_qty} ({ftd_penalty:+.2f})")

        score = max(-1.0, min(1.0, score))
        reasoning = "; ".join(parts) if parts else "no qualifying insider signals"
        if unknown:
            reasoning += f" | unknown: {','.join(unknown)}"

        return InsiderResult(
            score=round(score, 3),
            signals_seen=seen,
            signals_unknown=unknown,
            reasoning=reasoning,
        )

    def full_analysis(
        self,
        symbol: str,
        headlines: list[str] | None = None,
        prices: list[dict] | None = None,
        financials: dict | None = None,
        insider_signals: dict | None = None,
    ) -> MultiAgentAnalysis:
        """Run the full multi-agent pipeline: 3 analysts + debate + aggregation.

        ``insider_signals`` is the raw output of ``InsiderFeed.get_full_signals``
        when the insider feature flag is enabled, otherwise None.
        """
        sentiment = self.analyze_sentiment(symbol, headlines or [])
        technical = self.analyze_technical(symbol, prices or [])
        fundamental = self.analyze_fundamental(symbol, financials)
        debate = self.run_debate(symbol, sentiment, technical, fundamental)
        insider = (
            self.analyze_insider(symbol, insider_signals)
            if insider_signals is not None
            else None
        )

        result = self._aggregator.aggregate(
            symbol=symbol,
            sentiment=sentiment,
            technical=technical,
            fundamental=fundamental,
            debate=debate,
            insider=insider,
        )

        log.info(
            "multi_agent_analysis_complete",
            symbol=symbol,
            final_signal=result.final_signal.value,
            combined_score=result.combined_score,
            agreement=result.agreement_count,
            contrarian=result.contrarian_signal,
            insider_bonus=result.insider_bonus,
        )
        return result
