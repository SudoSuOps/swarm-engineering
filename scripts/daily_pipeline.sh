#!/usr/bin/env bash
# SwarmSignal Daily Pipeline
#
# Collects live AI signals, generates a daily briefing, and optionally
# generates a weekly journal on Fridays.
#
# Usage:
#   ./scripts/daily_pipeline.sh                    # run daily
#   ./scripts/daily_pipeline.sh --journal           # force weekly journal
#
# Cron example (run daily at 06:00 UTC):
#   0 6 * * * /path/to/Swarm-Signal/scripts/daily_pipeline.sh >> /var/log/swarmsignal.log 2>&1

set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-llama}"
DATE=$(date -u +%Y-%m-%d)
DOW=$(date -u +%u)  # 1=Monday, 5=Friday

echo "=== SwarmSignal Pipeline — $DATE ==="

# Step 1: Collect signals
echo "Collecting signals..."
python3 -m src.collect_signals

# Step 2: Generate daily briefing
echo "Generating daily briefing..."
python3 -m src.generate_daily --backend "$BACKEND" --format briefing

# Step 3: Weekly journal on Fridays (or if --journal flag)
if [[ "$DOW" == "5" ]] || [[ "${1:-}" == "--journal" ]]; then
    echo "Generating weekly journal..."
    python3 -m src.generate_daily --backend "$BACKEND" --format journal
fi

echo "=== Pipeline complete — $DATE ==="
