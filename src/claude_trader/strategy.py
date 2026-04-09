"""EMA Momentum Strategy - the only profitable survivor in research testing.

Simple strategy: buy when price crosses above EMA with positive sentiment,
sell when price crosses below EMA or trailing stop hit. Fewer trades = better.
"""

from decimal import Decimal

import structlog

from claude_trader.analyst import AnalysisResult, Signal

log = structlog.get_logger()


def calculate_ema(prices: list[float], period: int) -> list[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return []

    multiplier = 2 / (period + 1)
    ema = [sum(prices[:period]) / period]

    for price in prices[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])

    return ema


class EMAMomentumStrategy:
    """Buy above EMA + positive sentiment, sell below EMA."""

    def __init__(self, ema_period: int = 20) -> None:
        self.ema_period = ema_period
        self._daily_trades: dict[str, int] = {}  # symbol -> trade count today

    def should_buy(
        self,
        symbol: str,
        current_price: float,
        prices: list[float],
        analysis: AnalysisResult | None = None,
    ) -> bool:
        """Determine if we should buy this symbol."""
        # Max 1 trade per symbol per day
        if self._daily_trades.get(symbol, 0) >= 1:
            log.info("strategy_skip_daily_limit", symbol=symbol)
            return False

        ema_values = calculate_ema(prices, self.ema_period)
        if not ema_values:
            return False

        current_ema = ema_values[-1]
        price_above_ema = current_price > current_ema

        if not price_above_ema:
            return False

        # Check if price just crossed above EMA (wasn't above yesterday)
        if len(ema_values) >= 2 and len(prices) >= self.ema_period + 1:
            prev_price = prices[-2]
            prev_ema = ema_values[-2]
            just_crossed = prev_price <= prev_ema and current_price > current_ema
        else:
            just_crossed = price_above_ema

        if not just_crossed:
            return False

        # Require positive sentiment from analysis
        if analysis and analysis.combined_score < 0.1:
            log.info("strategy_skip_low_sentiment", symbol=symbol, score=analysis.combined_score)
            return False

        log.info(
            "strategy_buy_signal",
            symbol=symbol,
            price=current_price,
            ema=round(current_ema, 2),
            sentiment=analysis.combined_score if analysis else "N/A",
        )
        return True

    def should_sell(
        self,
        symbol: str,
        current_price: float,
        prices: list[float],
        entry_price: float | None = None,
    ) -> bool:
        """Determine if we should sell this symbol."""
        ema_values = calculate_ema(prices, self.ema_period)
        if not ema_values:
            return False

        current_ema = ema_values[-1]
        price_below_ema = current_price < current_ema

        if price_below_ema:
            log.info("strategy_sell_signal", symbol=symbol, price=current_price, ema=round(current_ema, 2))
            return True

        return False

    def record_trade(self, symbol: str) -> None:
        """Record that a trade was made for daily limit tracking."""
        self._daily_trades[symbol] = self._daily_trades.get(symbol, 0) + 1

    def reset_daily(self) -> None:
        """Reset daily trade counters."""
        self._daily_trades.clear()
