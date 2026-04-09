"""Obsidian daily trade log integration.

Generates markdown trade logs with frontmatter, P&L, and wiki-links
to the research knowledge graph.
"""

from datetime import datetime, timezone
from pathlib import Path

import structlog

log = structlog.get_logger()


class ObsidianLogger:
    """Writes daily trade logs to the Obsidian vault."""

    def __init__(self, vault_path: Path) -> None:
        self._base = vault_path
        self._base.mkdir(parents=True, exist_ok=True)

    def _today_path(self) -> Path:
        return self._base / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"

    def write_daily_log(
        self,
        equity: str,
        cash: str,
        daily_pnl: str,
        positions: list[dict],
        trades: list[dict],
        analyses: list[dict],
    ) -> Path:
        """Write or update today's trade log."""
        path = self._today_path()
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        time_now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        pos_table = "| Symbol | Qty | Entry | Current | P&L |\n|---|---|---|---|---|\n"
        for p in positions:
            pos_table += f"| {p['symbol']} | {p['qty']} | ${p['avg_entry']} | ${p['current_price']} | ${p['unrealized_pnl']} |\n"
        if not positions:
            pos_table += "| - | - | - | - | - |\n"

        trades_section = ""
        for t in trades:
            side_icon = "BUY" if t.get("side") == "buy" else "SELL"
            trades_section += f"- **{side_icon}** {t.get('symbol', '?')} x{t.get('qty', '?')} @ ${t.get('price', '?'):.2f} - {t.get('rationale', 'N/A')}\n"
        if not trades:
            trades_section = "- No trades executed\n"

        analysis_section = ""
        for a in analyses:
            analysis_section += f"- **{a['symbol']}**: {a['signal']} (score: {a['score']:.3f}, agreement: {a['agreement']}/4, contrarian: {a['contrarian']})\n"
        if not analyses:
            analysis_section = "- No analyses run\n"

        content = f"""---
type: trade-log
date: {date}
equity: "{equity}"
daily_pnl: "{daily_pnl}"
trades_count: {len(trades)}
tags: [claude-trader, trade-log]
---

# Trade Log - {date}

**Last updated**: {time_now}
**Equity**: ${equity} | **Cash**: ${cash} | **Daily P&L**: ${daily_pnl}

## Positions

{pos_table}

## Trades

{trades_section}

## Multi-Agent Analyses

{analysis_section}

## Links

- [[2026-04-09-claude-trading-bot/README|Research Dashboard]]
- [[q6-risk-management|Risk Management Rules]]
"""

        path.write_text(content)
        log.info("obsidian_log_written", path=str(path))
        return path
