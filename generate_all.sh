#!/bin/bash
cd "$(dirname "$0")"

find . -name "generate_report.py" | sort | while read -r script; do
    echo "--- $script ---"
    python3 "$script" || echo "  SKIPPED (error — data file may be missing)"
done
