"""Main bot orchestrator - the trading loop.

Multi-agent pipeline with Phase 4 integrations:
1. Check market hours
2. Risk pre-check (circuit breakers, daily loss)
3. For each watchlist symbol:
   a. Fetch news headlines (Alpaca News API)
   b. Run 3 analyst agents (sentiment, technical, fundamental)
   c. Run Bull/Bear debate on analyst output
   d. Aggregate signals with agreement threshold + contrarian filter
4. Filter through risk manager
5. Execute approved trades
6. Log to JSONL + Obsidian + Telegram
"""

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import structlog

from claude_trader.analyst import Analyst, Signal
from claude_trader.config import Settings
from claude_trader.executor import AlpacaExecutor
from claude_trader.logger import TradeLogger
from claude_trader.news import NewsFeed
from claude_trader.notifier import TelegramNotifier
from claude_trader.obsidian import ObsidianLogger
from claude_trader.risk import RiskConfig, RiskManager
from claude_trader.strategy import EMAMomentumStrategy

log = structlog.get_logger()

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


class TradingBot:
    """Orchestrates the full multi-agent trading pipeline."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        risk_config = RiskConfig(
            max_position_pct=settings.max_position_pct,
            stop_loss_pct=settings.stop_loss_pct,
            trailing_stop_pct=settings.trailing_stop_pct,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            max_drawdown_pct=settings.max_drawdown_pct,
            max_consecutive_losses=settings.max_consecutive_losses,
            max_open_positions=settings.max_open_positions,
        )

        self._risk = RiskManager(risk_config, portfolio_value=Decimal("0"))
        self._executor = AlpacaExecutor(settings, self._risk)
        self._analyst = Analyst(api_key=settings.gemini_api_key)
        self._strategy = EMAMomentumStrategy(ema_period=settings.ema_period)
        self._logger = TradeLogger(settings.trades_log_path)
        self._news = NewsFeed(api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key)
        self._telegram = TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self._obsidian = ObsidianLogger(vault_path=settings.obsidian_log_path)

    def _get_market_time(self) -> time:
        return datetime.now(ET).time()

    def _is_market_open(self) -> bool:
        now = self._get_market_time()
        return MARKET_OPEN <= now < MARKET_CLOSE

    def _update_portfolio(self) -> None:
        account = self._executor.get_account()
        self._risk.portfolio_value = account["portfolio_value"]
        positions = self._executor.get_positions()
        self._risk.open_positions = len(positions)
        log.info(
            "portfolio_updated",
            equity=str(account["equity"]),
            positions=self._risk.open_positions,
        )

    def _get_price_bars(self, symbol: str) -> tuple[list[float], list[dict]]:
        from alpaca.data.timeframe import TimeFrame

        bars = self._executor.get_bars(
            symbol,
            TimeFrame.Day,
            start=(datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
        )
        if not hasattr(bars, "df") or bars.df.empty:
            return [], []

        df = bars.df
        closes = df["close"].tolist()
        ohlcv = [
            {"date": str(d), "open": o, "high": h, "low": l, "close": c, "volume": v}
            for d, o, h, l, c, v in zip(
                df.index.tolist()[-30:], df["open"].tolist()[-30:], df["high"].tolist()[-30:],
                df["low"].tolist()[-30:], df["close"].tolist()[-30:], df["volume"].tolist()[-30:],
            )
        ]
        return closes, ohlcv

    def run_once(self) -> dict:
        """Execute one trading cycle with multi-agent analysis."""
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": [],
            "analyses": [],
            "trades": [],
        }

        # 1. Market hours check
        if not self._is_market_open():
            log.info("market_closed")
            summary["actions"].append("market_closed")
            return summary

        market_time = self._get_market_time()

        # 2. Update portfolio and risk state
        self._update_portfolio()

        # 3. Check existing positions for sell signals
        positions = self._executor.get_positions()
        for pos in positions:
            symbol = pos["symbol"]
            current_price = float(pos["current_price"])
            closes, _ = self._get_price_bars(symbol)
            if not closes:
                closes = [current_price]

            if self._strategy.should_sell(symbol, current_price, closes):
                result = self._executor.sell(symbol, pos["qty"])
                if result:
                    pnl = Decimal(str(current_price)) - pos["avg_entry"]
                    self._risk.record_trade_result(profit=pnl * pos["qty"])
                    self._risk.record_daily_pnl(pnl * pos["qty"])
                    trade_info = {
                        "symbol": symbol, "side": "sell", "qty": pos["qty"],
                        "price": current_price, "rationale": "EMA crossover below",
                    }
                    self._logger.log_trade(**trade_info, order_id=result["order_id"])
                    self._telegram.trade_alert(**trade_info)
                    summary["actions"].append(f"sold {pos['qty']} {symbol}")
                    summary["trades"].append(trade_info)

        # 4. Scan watchlist for buy signals with multi-agent analysis
        for symbol in self._settings.watchlist:
            if any(p["symbol"] == symbol for p in positions):
                continue

            try:
                closes, ohlcv = self._get_price_bars(symbol)
                if not closes:
                    continue

                current_price = closes[-1]
                analysis = None

                # Run full multi-agent pipeline if Gemini key available
                if self._settings.gemini_api_key:
                    headlines = self._news.get_headlines(symbol, limit=10)
                    analysis = self._analyst.full_analysis(
                        symbol=symbol,
                        headlines=headlines,
                        prices=ohlcv,
                    )
                    summary["analyses"].append({
                        "symbol": symbol,
                        "signal": analysis.final_signal.value,
                        "score": analysis.combined_score,
                        "agreement": analysis.agreement_count,
                        "contrarian": analysis.contrarian_signal,
                    })

                    if analysis.final_signal == Signal.HOLD:
                        log.info("multi_agent_hold", symbol=symbol, agreement=analysis.agreement_count)
                        continue

                # EMA crossover + agent confirmation
                if self._strategy.should_buy(symbol, current_price, closes, analysis):
                    result = self._executor.buy(
                        symbol=symbol,
                        price=Decimal(str(current_price)),
                        market_time=market_time,
                    )
                    if result:
                        self._strategy.record_trade(symbol)
                        agent_scores = {}
                        if analysis:
                            agent_scores = {
                                "combined": analysis.combined_score,
                                "sentiment": analysis.sentiment.score if analysis.sentiment else None,
                                "technical": analysis.technical.score if analysis.technical else None,
                                "fundamental": analysis.fundamental.score if analysis.fundamental else None,
                                "debate_bull": analysis.debate.bull_score if analysis.debate else None,
                                "debate_bear": analysis.debate.bear_score if analysis.debate else None,
                                "agreement": analysis.agreement_count,
                                "contrarian": analysis.contrarian_signal,
                            }
                        trade_info = {
                            "symbol": symbol, "side": "buy", "qty": result["qty"],
                            "price": current_price,
                            "rationale": analysis.reasoning if analysis else "EMA crossover (no analysis)",
                        }
                        self._logger.log_trade(**trade_info, agent_scores=agent_scores, order_id=result["order_id"])
                        self._telegram.trade_alert(**trade_info, agent_scores=agent_scores)
                        summary["actions"].append(f"bought {result['qty']} {symbol}")
                        summary["trades"].append(trade_info)

            except Exception as e:
                log.error("symbol_scan_error", symbol=symbol, error=str(e))

        if not summary["actions"]:
            summary["actions"].append("no_signals")

        # 5. Write Obsidian daily log
        try:
            account = self._executor.get_account()
            self._obsidian.write_daily_log(
                equity=str(account["equity"]),
                cash=str(account["cash"]),
                daily_pnl="0",  # TODO: track cumulative daily P&L
                positions=self._executor.get_positions(),
                trades=summary["trades"],
                analyses=summary["analyses"],
            )
        except Exception as e:
            log.warning("obsidian_log_failed", error=str(e))

        log.info("cycle_complete", actions=summary["actions"], analyses_count=len(summary["analyses"]))
        return summary

    def run_daily_summary(self) -> None:
        """Generate and send end-of-day summary."""
        account = self._executor.get_account()
        positions = self._executor.get_positions()
        daily_log = self._logger.get_daily_summary()

        self._telegram.daily_summary(
            date=datetime.now(ET).strftime("%Y-%m-%d"),
            equity=str(account["equity"]),
            daily_pnl="0",
            trades_count=daily_log["total_trades"],
            positions=positions,
        )
        log.info("daily_summary_sent")
