"""Shared constants used across modules."""

from datetime import time
from zoneinfo import ZoneInfo

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

ET = ZoneInfo("America/New_York")
