#!/bin/bash
set -e
cd "$(dirname "$0")"

find . -name "generate_report.py" | sort | while read -r script; do
    python3 "$script"
done
