# Claude Trader - Trading Rules Specification

## Architecture

Hybrid approach: Claude as analyst, rule-based execution and risk management.
Research finding: Risk management is the #1 factor, not model intelligence.

## Risk Rules (Non-Negotiable)

| Rule | Value | Override |
|------|-------|---------|
| Max position size | 2% of portfolio | config |
| Stop loss | 8% below entry | config |
| Trailing stop | 5% trail (floor only up) | config |
| Max daily loss | 3% -> halt all trading | config |
| Max drawdown | 10% -> halt all trading | config |
| Circuit breaker | 3 consecutive losses -> pause | config |
| Max open positions | 5 | config |
| Banned hours | First/last 15min of session | config |

## Safety Circuit Breakers

- **ALPACA_PAPER_TRADE=true is the default**. Live trading requires:
  1. Setting ALPACA_PAPER_TRADE=false explicitly
  2. Typing "CONFIRM LIVE" at the prompt
- If ALPACA_PAPER_TRADE=false is detected, ALWAYS warn the user
- Never execute a trade without risk manager approval
- Sells are always allowed (reduce exposure)

## Strategy: EMA Momentum

- Buy: price crosses above 20-day EMA + positive sentiment score
- Sell: price crosses below 20-day EMA or trailing stop triggers
- Max 1 trade per symbol per day
- Fewer trades = better (minimizes fee exposure)

## Analysis Agents

Claude acts as **analyst only**, never executor:
- Sentiment Agent: news headlines -> score (-1 to +1)
- Technical Agent: OHLCV data -> pattern + score
- Combined score: 40% sentiment + 60% technical
- Signal thresholds: >0.5 strong_buy, >0.2 buy, <-0.2 sell, <-0.5 strong_sell

## Security

- API keys in .env only, never in code
- No third-party trading skills/plugins (ClawHavoc risk)
- Official anthropic SDK only
- Paper trading for minimum 30 days before live

## Graduation to Live

ALL criteria must pass:
- [ ] 30+ days paper trading
- [ ] Positive cumulative return
- [ ] Sharpe ratio > 0.5
- [ ] Max drawdown < 10%
- [ ] No circuit breaker in last 7 days
- [ ] Manual review of all trades

## Citadel Harness

This project uses the [Citadel](https://github.com/SethGammon/Citadel) agent
orchestration harness. Configuration is in `.claude/harness.json`.
