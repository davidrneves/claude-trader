"""Structured trade logging to JSONL."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import structlog

log = structlog.get_logger()


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


class TradeLogger:
    """Logs every trade, rationale, and risk check to a JSONL file."""

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        rationale: str,
        risk_checks: dict | None = None,
        agent_scores: dict | None = None,
        order_id: str | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "rationale": rationale,
            "risk_checks": risk_checks or {},
            "agent_scores": agent_scores or {},
            "order_id": order_id,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry, cls=DecimalEncoder) + "\n")
        log.info("trade_logged", symbol=symbol, side=side, qty=qty)

    def log_risk_rejection(self, symbol: str, reason: str) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "risk_rejection",
            "symbol": symbol,
            "reason": reason,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry, cls=DecimalEncoder) + "\n")

    def get_daily_summary(self) -> dict:
        """Read today's trades and compute summary stats."""
        today = datetime.now(timezone.utc).date().isoformat()
        trades = []
        if self._path.exists():
            with open(self._path) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("timestamp", "").startswith(today) and "side" in entry:
                        trades.append(entry)
        return {
            "date": today,
            "total_trades": len(trades),
            "buys": sum(1 for t in trades if t["side"] == "buy"),
            "sells": sum(1 for t in trades if t["side"] == "sell"),
        }
