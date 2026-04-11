"""Bot state persistence across sessions.

Saves peak equity, trailing stop floors, and last trading date to a JSON
file so cron-invoked single-shot runs maintain safety state.
"""

import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path

import structlog

from claude_trader.logger import DecimalEncoder

log = structlog.get_logger()


class BotStateStore:
    """Read/write bot state to a JSON file with atomic writes."""

    def __init__(self, state_path: Path) -> None:
        self._path = state_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        """Load state from disk. Returns empty dict if missing or corrupt."""
        if not self._path.exists():
            return {}
        try:
            with open(self._path) as f:
                raw = json.loads(f.read())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("state_load_failed", error=str(e))
            return {}

        return self._deserialize(raw)

    def save(self, state: dict) -> None:
        """Atomically write state to disk."""
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(state, cls=DecimalEncoder, indent=2) + "\n")
            os.replace(tmp, self._path)
            log.info("state_saved")
        except BaseException:
            os.unlink(tmp)
            raise

    @staticmethod
    def _deserialize(raw: dict) -> dict:
        """Convert stored string values back to proper types."""
        result = {}

        if "peak_equity" in raw:
            result["peak_equity"] = Decimal(str(raw["peak_equity"]))

        if "last_trading_date" in raw:
            result["last_trading_date"] = raw["last_trading_date"]

        if "trailing_stops" in raw and isinstance(raw["trailing_stops"], dict):
            stops = {}
            for symbol, data in raw["trailing_stops"].items():
                stops[symbol] = {
                    "floor": Decimal(str(data["floor"])),
                    "stop_order_id": data.get("stop_order_id"),
                }
            result["trailing_stops"] = stops

        return result
