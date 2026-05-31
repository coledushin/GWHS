#!/bin/bash
set -e
cd "$(dirname "$0")"

python3 "Global/April CMA/generate_report.py"
python3 "Global/May Final/generate_report.py"
python3 "US/April CMA/generate_report.py"
python3 "US/May Final/generate_report.py"
