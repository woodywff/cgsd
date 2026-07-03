"""Run CGSD + CSpec on the five Table-1 datasets."""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from cgsd.cspec import cluster_cspec, cluster_kmeans
from cgsd.data import DATASETS, load_dataset
from cgsd.sheaf import compute_forman_ricci_curvature
from cgsd.train import train_cgsd

warnings.filterwarnings("ignore")

CANONICAL_CONFIG: dict = {
    "hidden": 64,
    "n_heads": 2,
    "dropout": 0.3,
    "lr": 0.005,
    "wd": 5e-4,
    "struct_epochs": 30,
    "w_mod": 1.0,
    "w_col": 5.0,
    "w_rec": 1.0,
    "collapse_mode": "balance",
}

CANONICAL_SEED: int = 0
CLUSTER_DEFAULT: str = "cspec"


def _train_and_cluster(
    name: str,
    cfg: dict,
    seed: int,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, int, dict, float]:
    """Train one embedding and cluster it."""
    t0 = time.time()
    emb = train_cgsd(name, cfg, seed=seed)
    train_time = time.time() - t0

    adj, _, labels, _, n_classes = load_dataset(name)
    coo = adj.tocoo()
    edge_index = np.stack([coo.row, coo.col])
    curvature = compute_forman_ricci_curvature(adj)

    t1 = time.time()
    if strategy == "cspec":
        pred = cluster_cspec(emb, n_classes, curvature, edge_index, seed=seed)
    elif strategy == "kmeans":
        pred = cluster_kmeans(emb, n_classes, seed=seed)
    else:
        raise ValueError(
            f"Unknown cluster strategy: {strategy!r}. Choices: {{kmeans, cspec}}"
        )
    cluster_time = time.time() - t1

    metrics = {
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
    }
    print(
        f"  [{name}/seed{seed}/{strategy}] "
        f"train={train_time:.1f}s cluster={cluster_time:.1f}s "
        f"NMI={metrics['nmi']:.4f} ARI={metrics['ari']:.4f}",
        flush=True,
    )
    return emb, labels, n_classes, metrics, train_time + cluster_time


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="CGSD + CSpec on Table-1 datasets")
    parser.add_argument(
        "--cluster",
        default=CLUSTER_DEFAULT,
        choices=["kmeans", "cspec"],
        help="Clusterer. Default: cspec.",
    )
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.abspath(os.path.join(out_dir, "table2_cgsd.csv"))

    rows: list[dict] = []
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"  WARNING: skipping {ds!r} (not in {DATASETS})")
            continue
        print(f"\n=== {ds} ===")
        _, _, n_classes, metrics, runtime = _train_and_cluster(
            ds, CANONICAL_CONFIG, args.seed, args.cluster
        )
        rows.append({
            "dataset": ds,
            "n_classes": n_classes,
            "cluster": args.cluster,
            "seed": args.seed,
            "nmi": metrics["nmi"],
            "ari": metrics["ari"],
            "runtime_s": runtime,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    print(f"\nMean NMI = {df['nmi'].mean():.4f}  (expected ~0.107 with cspec)")
    return df


if __name__ == "__main__":
    main()
