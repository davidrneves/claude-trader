# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
