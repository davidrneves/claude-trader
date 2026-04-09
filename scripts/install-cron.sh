#!/bin/bash
# Install cron jobs for autonomous trading bot operation.
# All times are in the system's local timezone.
# Adjust if your system is not set to US/Eastern.

PROJECT_DIR="$HOME/Workspace/Personal/claude-trader"
UV="$HOME/.local/bin/uv"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Remove existing claude-trader cron entries
crontab -l 2>/dev/null | grep -v "claude-trader" > /tmp/crontab_clean

# Add new entries (times in ET - adjust for your timezone)
cat >> /tmp/crontab_clean << CRON
# Claude Trader - Market open scan (9:45 AM ET)
45 9 * * 1-5 cd $PROJECT_DIR && $UV run python scripts/run.py >> $LOG_DIR/cron.log 2>&1
# Claude Trader - Midday check (12:00 PM ET)
0 12 * * 1-5 cd $PROJECT_DIR && $UV run python scripts/run.py >> $LOG_DIR/cron.log 2>&1
# Claude Trader - End-of-day summary (4:15 PM ET)
15 16 * * 1-5 cd $PROJECT_DIR && $UV run python scripts/run.py --summary >> $LOG_DIR/cron.log 2>&1
CRON

crontab /tmp/crontab_clean
rm /tmp/crontab_clean

echo "Cron jobs installed:"
crontab -l | grep claude-trader
