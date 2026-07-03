"""Smoke test: train CGSD on Cora for one seed."""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cgsd.data import load_dataset
from cgsd.eval import evaluate_nmi
from cgsd.train import train_cgsd

SMOKE_CFG = {
    "hidden": 64,
    "n_heads": 2,
    "dropout": 0.5,
    "lr": 0.005,
    "wd": 5e-4,
    "struct_epochs": 30,
    "w_mod": 1.0,
    "w_col": 5.0,
    "w_rec": 0.5,
    "collapse_mode": "balance",
}


def main():
    """Run a short Cora training pass."""
    t0 = time.time()
    print("=== CGSD smoke test (Cora, 1 seed, 30 epochs) ===")
    name = "Cora"
    _, _, labels, n_nodes, n_classes = load_dataset(name)
    print(f"Loaded {name}: N={n_nodes}, K={n_classes}")

    emb = train_cgsd(name, SMOKE_CFG, seed=0)
    print(f"Embedding shape: {emb.shape}")
    metrics = evaluate_nmi(emb, labels, k=n_classes)
    print(f"K-Means NMI: {metrics['nmi']:.4f}, ARI: {metrics['ari']:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    assert elapsed < 180, f"smoke test took {elapsed:.1f}s (>3 min)"
    print("OK")


if __name__ == "__main__":
    main()
