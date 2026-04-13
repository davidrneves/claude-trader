#!/bin/bash
# Install launchd agents for the claude-trader bot.
# Replaces the old cron-based scheduling (install-cron.sh).
#
# launchd advantages over cron:
# - Fires missed jobs on wake from sleep (coalesced into one run)
# - Survives reboots via launchctl enable
# - Built-in stdout/stderr log redirection
#
# The bot's internal _is_market_open() check handles weekends and
# off-hours gracefully, so we omit Weekday from StartCalendarInterval.

set -euo pipefail

PROJECT_DIR="$HOME/Workspace/Personal/claude-trader"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
LOG_DIR="$PROJECT_DIR/logs"
AGENTS_DIR="$HOME/Library/LaunchAgents"
UID_NUM=$(id -u)
DOMAIN="gui/$UID_NUM"

LABEL_TRADER="com.davidneves.claude-trader"
LABEL_SUMMARY="com.davidneves.claude-trader-summary"
PLIST_TRADER="$SCRIPTS_DIR/$LABEL_TRADER.plist"
PLIST_SUMMARY="$SCRIPTS_DIR/$LABEL_SUMMARY.plist"

# Verify plists exist
for plist in "$PLIST_TRADER" "$PLIST_SUMMARY"; do
    if [ ! -f "$plist" ]; then
        echo "ERROR: $plist not found"
        exit 1
    fi
done

# Verify uv is installed
if [ ! -x "/opt/homebrew/bin/uv" ]; then
    echo "ERROR: uv not found at /opt/homebrew/bin/uv"
    exit 1
fi

# Create logs directory
mkdir -p "$LOG_DIR"

# Rotate logs > 10MB
for logfile in "$LOG_DIR"/launchd*.log; do
    if [ -f "$logfile" ] && [ "$(stat -f%z "$logfile" 2>/dev/null || echo 0)" -gt 10485760 ]; then
        mv "$logfile" "${logfile%.log}.$(date +%Y%m%d).log"
        echo "Rotated: $logfile"
    fi
done

# Unload existing agents (ignore errors on first install)
echo "Removing existing agents..."
launchctl bootout "$DOMAIN/$LABEL_TRADER" 2>/dev/null || true
launchctl bootout "$DOMAIN/$LABEL_SUMMARY" 2>/dev/null || true

# Remove old cron entries
if crontab -l 2>/dev/null | grep -q "claude-trader"; then
    echo "Removing old cron entries..."
    crontab -l 2>/dev/null | { grep -v "claude-trader" || true; } | crontab -
    echo "Cron entries removed."
fi

# Copy plists to LaunchAgents
echo "Installing launchd agents..."
cp "$PLIST_TRADER" "$AGENTS_DIR/"
cp "$PLIST_SUMMARY" "$AGENTS_DIR/"

# Bootstrap (register) the agents
launchctl bootstrap "$DOMAIN" "$AGENTS_DIR/$LABEL_TRADER.plist"
launchctl bootstrap "$DOMAIN" "$AGENTS_DIR/$LABEL_SUMMARY.plist"

# Enable for reboot persistence
launchctl enable "$DOMAIN/$LABEL_TRADER"
launchctl enable "$DOMAIN/$LABEL_SUMMARY"

# Verify
echo ""
echo "Verification:"
echo "============="
for label in "$LABEL_TRADER" "$LABEL_SUMMARY"; do
    if launchctl print "$DOMAIN/$label" > /dev/null 2>&1; then
        echo "  $label: loaded"
    else
        echo "  $label: FAILED to load"
        exit 1
    fi
done

# Confirm cron is clean
if crontab -l 2>/dev/null | grep -q "claude-trader"; then
    echo "  WARNING: cron entries still present"
else
    echo "  cron: clean (no claude-trader entries)"
fi

echo ""
echo "Done. Schedule (local time, daily):"
echo "  14:45  Market open scan"
echo "  16:30  Late morning"
echo "  18:00  Midday"
echo "  19:30  Afternoon"
echo "  21:15  End-of-day summary"
echo ""
echo "Test with: launchctl kickstart -k $DOMAIN/$LABEL_TRADER"
echo "Logs at:   $LOG_DIR/launchd.log"
