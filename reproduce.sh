#!/usr/bin/env bash
# Reproduce CGSD + CSpec on the five Table-1 datasets.
# Usage: bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "=== CGSD reproduction ==="
echo "Python: $($PYTHON --version)"

echo
echo "--- install deps ---"
$PYTHON -m pip install -q -r requirements.txt

echo
echo "--- smoke test ---"
$PYTHON scripts/smoke_test.py

echo
echo "--- full run (5 datasets, CSpec) ---"
$PYTHON -m cgsd.run_all --cluster cspec

echo
echo "=== Done. See results/table2_cgsd.csv ==="
