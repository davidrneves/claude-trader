#!/bin/bash
# DEPRECATED: Use install-launchd.sh instead.
# launchd fires missed jobs on wake from sleep; cron silently skips them.
# This script is kept for reference only.
#
# Original: Install cron jobs for autonomous trading bot operation.
# Times converted from ET to local timezone (WEST/WET, 5h offset).
# During DST transition weeks (~2 weeks/year), timing may shift by 1h;
# the bot's internal market-hours check handles this gracefully.

PROJECT_DIR="$HOME/Workspace/Personal/claude-trader"
UV="/opt/homebrew/bin/uv"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

if [ ! -x "$UV" ]; then
    echo "ERROR: uv not found at $UV"
    exit 1
fi

# Remove existing claude-trader cron entries
crontab -l 2>/dev/null | grep -v "claude-trader" > /tmp/crontab_clean

cat >> /tmp/crontab_clean << CRON
# claude-trader: Market open scan (9:45 ET = 14:45 local)
45 14 * * 1-5 cd $PROJECT_DIR && $UV run python scripts/run.py >> $LOG_DIR/cron.log 2>&1 # claude-trader
# claude-trader: Late morning (11:30 ET = 16:30 local)
30 16 * * 1-5 cd $PROJECT_DIR && $UV run python scripts/run.py >> $LOG_DIR/cron.log 2>&1 # claude-trader
# claude-trader: Midday (13:00 ET = 18:00 local)
0 18 * * 1-5 cd $PROJECT_DIR && $UV run python scripts/run.py >> $LOG_DIR/cron.log 2>&1 # claude-trader
# claude-trader: Afternoon (14:30 ET = 19:30 local)
30 19 * * 1-5 cd $PROJECT_DIR && $UV run python scripts/run.py >> $LOG_DIR/cron.log 2>&1 # claude-trader
# claude-trader: End-of-day summary (16:15 ET = 21:15 local)
15 21 * * 1-5 cd $PROJECT_DIR && $UV run python scripts/run.py --summary >> $LOG_DIR/cron.log 2>&1 # claude-trader
# claude-trader: Weekly log rotation (Sunday midnight)
0 0 * * 0 cd $PROJECT_DIR && mv $LOG_DIR/cron.log $LOG_DIR/cron.\$(date +\%Y\%m\%d).log 2>/dev/null # claude-trader
CRON

crontab /tmp/crontab_clean
rm /tmp/crontab_clean

echo "Cron jobs installed:"
crontab -l | grep claude-trader
