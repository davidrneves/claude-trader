"""Telegram notifications for trade alerts, P&L, and circuit breakers."""

import asyncio

import structlog
import telegram

log = structlog.get_logger()


class TelegramNotifier:
    """Sends trade alerts and reports to Telegram."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot = telegram.Bot(token=bot_token) if bot_token else None
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

    def _send(self, text: str) -> None:
        if not self._enabled:
            log.debug("telegram_disabled", message=text[:50])
            return
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                loop.create_task(
                    self._bot.send_message(
                        chat_id=self._chat_id,
                        text=text,
                        parse_mode="Markdown",
                    )
                )
            else:
                asyncio.run(
                    self._bot.send_message(
                        chat_id=self._chat_id,
                        text=text,
                        parse_mode="Markdown",
                    )
                )
        except Exception as e:
            log.warning("telegram_send_failed", error=str(e))

    def trade_alert(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        rationale: str,
        agent_scores: dict | None = None,
    ) -> None:
        emoji = "🟢" if side == "buy" else "🔴"
        msg = f"""{emoji} *{side.upper()} {symbol}*
Qty: {qty} @ ${price:.2f}
Rationale: {rationale}"""
        if agent_scores:
            msg += f"\nScore: {agent_scores.get('combined', 'N/A')} | Agreement: {agent_scores.get('agreement', 'N/A')}/4"
        self._send(msg)

    def daily_summary(
        self,
        date: str,
        equity: str,
        daily_pnl: str,
        trades_count: int,
        positions: list[dict],
    ) -> None:
        pos_text = (
            "\n".join(
                f"  {p['symbol']}: {p['qty']} @ ${p['avg_entry']} (P&L: ${p['unrealized_pnl']})"
                for p in positions
            )
            or "  None"
        )

        msg = f"""📊 *Daily Summary - {date}*
Equity: ${equity}
Daily P&L: ${daily_pnl}
Trades today: {trades_count}

*Positions:*
{pos_text}"""
        self._send(msg)
