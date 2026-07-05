"""2-second smoke test: train 1 config × 1 seed on Cora only.

Verifies the whole cgsd pipeline runs end-to-end on a fresh
checkout. Expected runtime: ~2-5 s on a single CPU.

Usage (from repo root):
    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import time

# Make cgsd importable when invoked as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.dirname(HERE)
sys.path.insert(0, PKG_PARENT)

from cgsd.data import load_dataset
from cgsd.train import train_cgsd_pure
from cgsd.eval import evaluate_nmi


SMOKE_CFG = {
    "hidden": 64, "n_heads": 2, "dropout": 0.5, "lr": 0.005, "wd": 5e-4,
    "pretrain_epochs": 0, "struct_epochs": 30,
    "w_mod": 1.0, "w_col": 5.0, "w_rec": 0.5,
    "collapse_mode": "balance",
}


def main():
    t0 = time.time()
    print("=== cgsd smoke test (Cora, 1 cfg × 1 seed, 30 epochs) ===")
    name = "Cora"
    adj, features, labels, n_nodes, n_classes = load_dataset(name)
    print(f"Loaded {name}: N={n_nodes}, K={n_classes}, |E|={adj.nnz}")

    emb = train_cgsd_pure(name, SMOKE_CFG, seed=0)
    print(f"Embedding shape: {emb.shape}")
    m = evaluate_nmi(emb, labels, k=n_classes)
    print(f"Single-embedding NMI: {m['nmi']:.4f}, ARI: {m['ari']:.4f}")

    elapsed = time.time() - t0
    print(f"\nSmoke test completed in {elapsed:.1f}s")
    assert elapsed < 180, f"smoke test took {elapsed:.1f}s (>3 min budget)"
    print("OK")


if __name__ == "__main__":
    main()
