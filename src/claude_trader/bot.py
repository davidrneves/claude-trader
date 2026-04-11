"""Main bot orchestrator - the trading loop.

Multi-agent pipeline:
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

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import structlog
from alpaca.data.timeframe import TimeFrame

from claude_trader.analyst import Analyst, MultiAgentAnalysis, Signal
from claude_trader.config import Settings
from claude_trader.constants import ET, MARKET_CLOSE, MARKET_OPEN
from claude_trader.executor import AlpacaExecutor, df_to_bar_dicts
from claude_trader.logger import TradeLogger
from claude_trader.news import NewsFeed
from claude_trader.notifier import TelegramNotifier
from claude_trader.obsidian import ObsidianLogger
from claude_trader.performance import PerformanceTracker
from claude_trader.risk import RiskConfig, RiskManager
from claude_trader.strategy import EMAMomentumStrategy

log = structlog.get_logger()


class TradingBot:
    """Orchestrates the full multi-agent trading pipeline."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        risk_config = RiskConfig.from_settings(settings)

        self._risk = RiskManager(risk_config, portfolio_value=Decimal("0"))
        self._executor = AlpacaExecutor(settings, self._risk)
        self._analyst = Analyst(api_key=settings.gemini_api_key)
        self._strategy = EMAMomentumStrategy(ema_period=settings.ema_period)
        self._logger = TradeLogger(settings.trades_log_path)
        self._news = NewsFeed(
            api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key
        )
        self._telegram = TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self._obsidian = ObsidianLogger(vault_path=settings.obsidian_log_path)
        self._performance = PerformanceTracker(settings.snapshots_path)
        self._peak_equity: Decimal = Decimal("0")
        self._last_trading_date: date | None = None
        self._trailing_stops: dict[str, dict] = {}

    def _get_market_time(self) -> time:
        return datetime.now(ET).time()

    def _is_market_open(self) -> bool:
        now = self._get_market_time()
        return MARKET_OPEN <= now < MARKET_CLOSE

    def _update_portfolio(self) -> None:
        account = self._executor.get_account()
        self._risk.portfolio_value = account["portfolio_value"]
        equity = account["equity"]
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - equity) / self._peak_equity
            self._risk.record_drawdown(drawdown_pct)
        positions = self._executor.get_positions()
        self._risk.open_positions = len(positions)
        log.info(
            "portfolio_updated",
            equity=str(account["equity"]),
            positions=self._risk.open_positions,
        )

    def _get_price_bars(self, symbol: str) -> tuple[list[float], list[dict]]:
        bars = self._executor.get_bars(
            symbol,
            TimeFrame.Day,
            start=(datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
        )
        if not hasattr(bars, "df") or bars.df.empty:
            return [], []

        df = bars.df
        closes = df["close"].tolist()
        ohlcv = df_to_bar_dicts(df, window=30)
        return closes, ohlcv

    def _scan_and_execute_sells(self, summary: dict, positions: list[dict]) -> None:
        """Check existing positions for trailing stop and EMA sell signals."""
        for pos in positions:
            symbol = pos["symbol"]
            current_price = float(pos["current_price"])
            entry_price = pos["avg_entry"]

            # Update trailing stop floor
            trailing_sell = False
            if symbol in self._trailing_stops:
                stored = self._trailing_stops[symbol]
                new_floor = self._risk.calculate_trailing_stop(
                    entry_price, Decimal(str(current_price)), stored["floor"]
                )
                if new_floor > stored["floor"]:
                    stored["floor"] = new_floor
                    # Update the Alpaca GTC stop order
                    if stored.get("stop_order_id"):
                        updated = self._executor.update_stop_loss(
                            symbol, pos["qty"], new_floor, stored["stop_order_id"]
                        )
                        if updated:
                            stored["stop_order_id"] = updated["stop_order_id"]

                # Check if price hit trailing stop
                if Decimal(str(current_price)) <= stored["floor"]:
                    trailing_sell = True
                    log.info(
                        "trailing_stop_triggered",
                        symbol=symbol,
                        price=current_price,
                        floor=str(stored["floor"]),
                    )

            # Check EMA sell signal
            closes, _ = self._get_price_bars(symbol)
            if not closes:
                closes = [current_price]
            ema_sell = self._strategy.should_sell(symbol, current_price, closes)

            if trailing_sell or ema_sell:
                result = self._executor.sell(symbol, pos["qty"])
                if result:
                    # Cancel the GTC stop order since we sold manually
                    if symbol in self._trailing_stops:
                        stop_id = self._trailing_stops[symbol].get("stop_order_id")
                        if stop_id:
                            self._executor.cancel_stop_loss(stop_id)
                        del self._trailing_stops[symbol]

                    pnl = Decimal(str(current_price)) - entry_price
                    self._risk.record_trade_result(profit=pnl * pos["qty"])
                    self._risk.record_daily_pnl(pnl * pos["qty"])
                    rationale = (
                        "trailing stop triggered"
                        if trailing_sell
                        else "EMA crossover below"
                    )
                    trade_info = {
                        "symbol": symbol,
                        "side": "sell",
                        "qty": pos["qty"],
                        "price": current_price,
                        "rationale": rationale,
                    }
                    self._record_trade(trade_info, summary, order_id=result["order_id"])

    def _analyze_symbol(
        self, symbol: str, ohlcv: list[dict]
    ) -> MultiAgentAnalysis | None:
        """Run multi-agent analysis pipeline for a symbol. Returns None if no API key."""
        if not self._settings.gemini_api_key:
            return None

        headlines = self._news.get_headlines(symbol, limit=10)
        return self._analyst.full_analysis(
            symbol=symbol,
            headlines=headlines,
            prices=ohlcv,
        )

    def _scan_and_execute_buys(
        self, summary: dict, positions: list[dict], market_time: time
    ) -> None:
        """Scan watchlist for buy signals with multi-agent analysis."""
        held_symbols = {p["symbol"] for p in positions}

        for symbol in self._settings.watchlist:
            if symbol in held_symbols:
                continue
            try:
                self._process_buy_candidate(symbol, summary, market_time)
            except ConnectionError as e:
                log.error("api_connection_error", symbol=symbol, error=str(e))
            except ValueError as e:
                log.error("data_error", symbol=symbol, error=str(e))
            except Exception as e:
                log.error(
                    "symbol_scan_error",
                    symbol=symbol,
                    error=str(e),
                    error_type=type(e).__name__,
                )

    def _process_buy_candidate(
        self, symbol: str, summary: dict, market_time: time
    ) -> None:
        """Evaluate a single symbol for buy opportunity."""
        closes, ohlcv = self._get_price_bars(symbol)
        if not closes:
            return

        current_price = closes[-1]
        analysis = self._analyze_symbol(symbol, ohlcv)

        if analysis:
            summary["analyses"].append(
                {
                    "symbol": symbol,
                    "signal": analysis.final_signal.value,
                    "score": analysis.combined_score,
                    "agreement": analysis.agreement_count,
                    "contrarian": analysis.contrarian_signal,
                }
            )
            if analysis.final_signal == Signal.HOLD:
                log.info(
                    "multi_agent_hold",
                    symbol=symbol,
                    agreement=analysis.agreement_count,
                )
                return

        if not self._strategy.should_buy(symbol, current_price, closes, analysis):
            return

        result = self._executor.buy(
            symbol=symbol,
            price=Decimal(str(current_price)),
            market_time=market_time,
        )
        if not result:
            return

        self._strategy.record_trade(symbol)

        # Initialize trailing stop tracking
        entry_price = Decimal(str(current_price))
        initial_floor = self._risk.calculate_trailing_stop(
            entry_price, entry_price, None
        )
        self._trailing_stops[symbol] = {
            "floor": initial_floor,
            "stop_order_id": result.get("stop_order_id"),
        }

        agent_scores = self._extract_agent_scores(analysis)
        trade_info = {
            "symbol": symbol,
            "side": "buy",
            "qty": result["qty"],
            "price": current_price,
            "rationale": analysis.reasoning
            if analysis
            else "EMA crossover (no analysis)",
        }
        self._record_trade(
            trade_info, summary, order_id=result["order_id"], agent_scores=agent_scores
        )

    def _record_trade(
        self,
        trade_info: dict,
        summary: dict,
        order_id: str,
        agent_scores: dict | None = None,
    ) -> None:
        """Record a trade to logger, telegram, and summary."""
        self._logger.log_trade(
            **trade_info, agent_scores=agent_scores, order_id=order_id
        )
        self._telegram.trade_alert(**trade_info, agent_scores=agent_scores)
        past = "sold" if trade_info["side"] == "sell" else "bought"
        summary["actions"].append(f"{past} {trade_info['qty']} {trade_info['symbol']}")
        summary["trades"].append(trade_info)

    @staticmethod
    def _extract_agent_scores(analysis: MultiAgentAnalysis | None) -> dict:
        """Extract agent scores dict from analysis result."""
        if not analysis:
            return {}
        return {
            "combined": analysis.combined_score,
            "sentiment": analysis.sentiment.score if analysis.sentiment else None,
            "technical": analysis.technical.score if analysis.technical else None,
            "fundamental": analysis.fundamental.score if analysis.fundamental else None,
            "debate_bull": analysis.debate.bull_score if analysis.debate else None,
            "debate_bear": analysis.debate.bear_score if analysis.debate else None,
            "agreement": analysis.agreement_count,
            "contrarian": analysis.contrarian_signal,
        }

    def _record_snapshot_and_log(self, summary: dict) -> None:
        """Record performance snapshot and write Obsidian daily log."""
        account = self._executor.get_account()
        trades_today = self._logger.get_daily_summary()
        snapshot = None

        try:
            snapshot = self._performance.record_snapshot(
                account=account,
                trades_today=trades_today,
                risk_state=self._risk.get_risk_state(),
            )
        except Exception as e:
            log.error("snapshot_failed", error=str(e))

        try:
            self._obsidian.write_daily_log(
                equity=str(account["equity"]),
                cash=str(account["cash"]),
                daily_pnl=snapshot.daily_pnl if snapshot else "0",
                positions=self._executor.get_positions(),
                trades=summary["trades"],
                analyses=summary["analyses"],
            )
        except Exception as e:
            log.error("obsidian_log_failed", error=str(e))

    def reset_daily(self) -> None:
        """Reset daily counters for risk manager and strategy."""
        self._risk.reset_daily()
        self._strategy.reset_daily()
        log.info("bot_daily_reset")

    def run_once(self) -> dict:
        """Execute one trading cycle with multi-agent analysis."""
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": [],
            "analyses": [],
            "trades": [],
        }

        if not self._is_market_open():
            log.info("market_closed")
            summary["actions"].append("market_closed")
            return summary

        today = datetime.now(ET).date()
        if self._last_trading_date is not None and today != self._last_trading_date:
            self.reset_daily()
        self._last_trading_date = today

        market_time = self._get_market_time()
        self._update_portfolio()

        positions = self._executor.get_positions()
        self._scan_and_execute_sells(summary, positions)
        self._scan_and_execute_buys(summary, positions, market_time)

        if not summary["actions"]:
            summary["actions"].append("no_signals")

        self._record_snapshot_and_log(summary)
        log.info(
            "cycle_complete",
            actions=summary["actions"],
            analyses_count=len(summary["analyses"]),
        )
        return summary

    def run_daily_summary(self) -> None:
        """Generate and send end-of-day summary."""
        account = self._executor.get_account()
        positions = self._executor.get_positions()
        daily_log = self._logger.get_daily_summary()
        daily_pnl = self._performance.get_daily_pnl()

        self._telegram.daily_summary(
            date=datetime.now(ET).strftime("%Y-%m-%d"),
            equity=str(account["equity"]),
            daily_pnl=daily_pnl,
            trades_count=daily_log["total_trades"],
            positions=positions,
        )
        log.info("daily_summary_sent")
