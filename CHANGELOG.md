# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.5] - 2026-04-27

### Fixed

- `get_open_stop_orders` returns `None` (instead of `[]`) on non-transient lookup failure so callers don't mistake a failed query for "no stops exist" and create a duplicate stop. Both `bot.py` callsites now defer reconciliation when the lookup result is unknown.
- `update_stop_loss` no longer submits a replacement when the cancel of the old order raised; the old order may still hold the shares and a duplicate stop would be wash-rejected. The next bot tick retries.
- All stop-price rounding now uses `Decimal.quantize(ROUND_HALF_UP)` via a single `_quantize_cents` helper, avoiding binary float rounding edge cases hit by `round(float(decimal), 2)`.
- `df_to_bar_dicts` now treats `window=0` explicitly instead of falling through to "return full DataFrame".

### Changed

- Refresh dependency lockfile via `uv lock --upgrade` (alpaca-py 0.43.2 → 0.43.3, google-genai 1.71.0 → 1.73.1, pydantic 2.12.5 → 2.13.3, cryptography 46.0.7 → 47.0.0, ruff 0.15.10 → 0.15.12 plus transitive bumps).
- Bootstrap Citadel agent harness configuration (`.claude/harness.json`, `.citadel/project.md`, `AGENTS.md`, `.mcp.json`) and gitignore machine-specific harness state.

## [0.7.4] - 2026-04-15

### Fixed

- Add tenacity-based retry with exponential backoff for `get_open_stop_orders` and `get_bars` to survive transient connection errors
- Wrap `_get_price_bars` call in sell-signal path so a failure doesn't crash the monitoring loop
- Move end-of-day summary launchd trigger from 21:15 to 20:00 to avoid DarkWake/sleep misses
- EMA lookback window, stop order detection, Gemini response parsing and retry

## [0.7.3] - 2026-04-14

### Fixed

- Adopt existing stop orders from Alpaca rejection errors instead of permanently giving up (parses `related_orders` and `existing_order_id` from error code 40310000)
- Replace permanent `stop_create_failed` flag with retry counter and 60-minute cooldown for transient stop creation failures
- Check for existing Alpaca stop orders during position reconciliation before attempting to create new ones

## [0.7.2] - 2026-04-14

### Fixed

- Detect existing Alpaca stop orders before creating duplicates during position reconciliation (fixes "insufficient qty available" error)
- Add debug logging for EMA strategy rejection paths (`strategy_skip_below_ema`, `strategy_skip_no_crossover`) to explain why strong_buy signals don't produce trades

## [0.7.0] - 2026-04-13

### Added

- Migrate scheduling from cron to launchd (fires missed jobs on wake from sleep)
- Persist bot state (peak equity, trailing stops) across sessions
- Alpaca WebSocket trade update listener for real-time fill notifications
- Trade journal CLI for manual review of historical trades

### Fixed

- Wire trailing stop execution into live trading loop
- Wire drawdown tracking into portfolio updates
- Add position size check in risk manager `check_trade`

## [0.6.0] - 2026-04-10

### Added

- Backtest CLI: `python -m claude_trader --backtest` with `--start`, `--end`, `--capital` flags
- Formatted backtest report with performance metrics, graduation check, and trade log
- Shared `df_to_bar_dicts()` helper for Alpaca DataFrame conversion

### Fixed

- Backtest date parsing: Alpaca MultiIndex `(symbol, timestamp)` now handled correctly with `strftime` instead of fragile string slicing
- `fetch_backtest_data` type hint and method signature now match real `AlpacaExecutor` interface

### Changed

- Extracted `df_to_bar_dicts()` from `TradingBot._extract_ohlcv` into `executor.py` (DRY: shared by bot and backtest)

## [0.5.0] - 2026-04-10

### Added

- Backtest engine: run EMA strategy against historical data with simulated stop-loss/trailing-stop, computing Sharpe ratio, max drawdown, and win rate
- Scheduler entry point (`python -m claude_trader`): async loop with SIGINT/SIGTERM handling, market-hours detection, configurable cycle interval
- Dry-run mode (`--dry-run`): validates Alpaca, Gemini, and news feed connectivity, reports signals without placing trades
- Analyst integration tests: real Gemini API tests (auto-skipped without API key) validating response structure for all 4 analysis agents
- README with architecture diagram, config reference, and SVG logo

### Changed

- Full codebase quality review: fix dependencies, security, dead code, DRY violations, and tests

## [0.4.0] - 2026-04-09

### Added

- Performance tracker and graduation dashboard
- Multi-agent analysis pipeline (sentiment, technical, fundamental + bull/bear debate)
- EMA momentum strategy with risk-gated execution
- Alpaca executor with automatic stop-loss orders
- Telegram notifications and Obsidian daily logs
- Initial project scaffold
