"""EMA Momentum Strategy - the only profitable survivor in research testing.

Simple strategy: buy when price crosses above EMA with positive sentiment,
sell when price crosses below EMA or trailing stop hit. Fewer trades = better.
"""

from datetime import date

import structlog

from claude_trader.analyst import MultiAgentAnalysis

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
        self._daily_trades: dict[str, int] = {}
        self._last_reset_date: date = date.today()

    def _ensure_daily_reset(self) -> None:
        """Auto-reset trade counters when the date changes."""
        today = date.today()
        if today != self._last_reset_date:
            self._daily_trades.clear()
            self._last_reset_date = today

    def should_buy(
        self,
        symbol: str,
        current_price: float,
        prices: list[float],
        analysis: MultiAgentAnalysis | None = None,
    ) -> bool:
        """Determine if we should buy this symbol."""
        self._ensure_daily_reset()
        if self._daily_trades.get(symbol, 0) >= 1:
            log.info("strategy_skip_daily_limit", symbol=symbol)
            return False

        ema_values = calculate_ema(prices, self.ema_period)
        if not ema_values:
            return False

        current_ema = ema_values[-1]
        price_above_ema = current_price > current_ema

        if not price_above_ema:
            log.debug(
                "strategy_skip_below_ema",
                symbol=symbol,
                price=current_price,
                ema=round(current_ema, 2),
            )
            return False

        # Check if price crossed above EMA within the last 5 bars.
        # ema_values is tail-aligned with prices, so ema[-k] pairs with prices[-k].
        if len(ema_values) >= 2:
            just_crossed = False
            max_transitions = min(5, len(ema_values) - 1)
            for i in range(1, max_transitions + 1):
                prev_p = prices[-(i + 1)]
                prev_e = ema_values[-(i + 1)]
                curr_p = prices[-i]
                curr_e = ema_values[-i]
                if prev_p <= prev_e and curr_p > curr_e:
                    just_crossed = True
                    break
        else:
            just_crossed = price_above_ema

        if not just_crossed:
            log.debug(
                "strategy_skip_no_crossover",
                symbol=symbol,
                price=current_price,
                ema=round(current_ema, 2),
            )
            return False

        # Require positive sentiment from analysis
        if analysis and analysis.combined_score < 0.1:
            log.info(
                "strategy_skip_low_sentiment",
                symbol=symbol,
                score=analysis.combined_score,
            )
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
    ) -> bool:
        """Determine if we should sell this symbol."""
        ema_values = calculate_ema(prices, self.ema_period)
        if not ema_values:
            return False

        current_ema = ema_values[-1]
        price_below_ema = current_price < current_ema

        if price_below_ema:
            log.info(
                "strategy_sell_signal",
                symbol=symbol,
                price=current_price,
                ema=round(current_ema, 2),
            )
            return True

        return False

    def record_trade(self, symbol: str) -> None:
        """Record that a trade was made for daily limit tracking."""
        self._daily_trades[symbol] = self._daily_trades.get(symbol, 0) + 1

    def reset_daily(self) -> None:
        """Reset daily trade counters."""
        self._daily_trades.clear()
