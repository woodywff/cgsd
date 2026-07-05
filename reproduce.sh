#!/usr/bin/env bash
# reproduce.sh — one-command end-to-end reproduction of the canonical
# CGSD algorithm (v2026-06-25):
#   - Sheaf-diffusion encoder (label-free, structural losses)
#   - Curvature-aware spectral clustering (G3)
#
# Usage:  bash reproduce.sh
# Expected runtime: ~30 sec on a single CPU.
# Expected output:  results/cgsd_pure_optimization_log.csv
# Expected mean NMI: ~0.107 (vs KMeans-only baseline ~0.091)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "=== cgsd: end-to-end reproduction (v2026-06-25) ==="
echo "Python: $($PYTHON --version)"
echo "Working dir: $(pwd)"

echo
echo "--- Step 1: install pinned dependencies ---"
$PYTHON -m pip install -q -r requirements.txt

echo
echo "--- Step 2: run 2-sec smoke test (Cora, 1 cfg × 1 seed, 30 epochs) ---"
$PYTHON scripts/smoke_test.py

echo
echo "--- Step 3: run full 5-dataset CGSD + curvature-spectral ---"
$PYTHON -m cgsd.run_all --cluster_strategy curv_spectral

echo
echo "=== Done. ==="
echo "See results/cgsd_pure_optimization_log.csv"
echo "Expected final mean NMI: ~0.107 (curv_spectral, 5 datasets × 1 seed)"