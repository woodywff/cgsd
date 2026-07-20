#!/usr/bin/env python3
"""Per-component ablation of CGSD's three structural losses (paper §5.7).

Runs 4 ablation variants × 5 heterophilic datasets × 5 random seeds = 100 cells
and writes mean NMI per cell to ``<cgsd>/results/ablation_perloss.csv``.

Variants
--------
FULL    w_mod=1.0  w_col=5.0  w_rec=1.0     (canonical CGSD-CSpec)
noLmod  w_mod=0.0  w_col=5.0  w_rec=1.0
noLcol  w_mod=1.0  w_col=0.0  w_rec=1.0
noLrec  w_mod=1.0  w_col=5.0  w_rec=0.0

Datasets: Cora, Cornell, Texas, Wisconsin, Chameleon  (paper Table 2).
Seeds:    [0, 1, 2, 3, 4]                           (matches Table 2 caption).

Clusterer: CSpec (curvature-aware spectral) with alpha=1.0, k=10 (canonical
defaults from ``cgsd.run_all.CANONICAL_CONFIG``). All cells use the
same clusterer so the ablation isolates the encoder-side losses.

Output
------
Default: ``<cgsd>/results/ablation_perloss.csv``
Columns: variant, dataset, seed, nmi, ari, runtime_s, train_time_s,
         cluster_time_s

Also writes ``<cgsd>/results/ablation_perloss_summary.csv`` (mean ± std
across 5 seeds per (variant, dataset)) and a Friedman test result on the
5-dataset means per variant.

Expected runtime: ~30-50 min on a single CPU (CGSD-pure training is the
bottleneck; clusterer is <1 s per cell).
"""
from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

# --------------------------------------------------------------- paths ----
_HERE = Path(__file__).resolve().parent                              # .../cgsd/scripts/
_RELEASE_ROOT = _HERE.parent                                          # .../cgsd/
sys.path.insert(0, str(_RELEASE_ROOT))                                # so 'cgsd' package imports

from cgsd.data import load_dataset                              # noqa: E402
from cgsd.train import train_cgsd_pure                          # noqa: E402
from cgsd.cluster_strategies import cluster_curvature_spectral  # noqa: E402
from cgsd.sheaf import compute_forman_ricci_curvature           # noqa: E402
from cgsd.eval import evaluate_nmi                              # noqa: E402

# ----------------------------------------------------------------- cfg ----
DATASETS = ["Cora", "Cornell", "Texas", "Wisconsin", "Chameleon"]
SEEDS = [0, 1, 2, 3, 4]

# Canonical CGSD hyperparameters from cgsd.run_all.CANONICAL_CONFIG,
# minus the loss weights (which the variants override below).
_BASE_CFG: dict = {
    "hidden": 64,
    "n_heads": 2,
    "dropout": 0.3,
    "lr": 0.005,
    "wd": 5e-4,
    "pretrain_epochs": 0,
    "struct_epochs": 30,
    "w_recon": 1.0,
    "use_recon": False,
    "kmeans_init": False,
    "collapse_mode": "balance",
}

# Ablation variants. w_mod=0 / w_col=0 / w_rec=0 disables that loss in
# ``cgsd.train._train_cgsd_pure_inner`` (line ~107:
# ``loss = w_mod * loss_mod + w_col * loss_col + w_rec * loss_rec``).
VARIANTS = {
    "FULL":   {"w_mod": 1.0, "w_col": 5.0, "w_rec": 1.0},
    "noLmod": {"w_mod": 0.0, "w_col": 5.0, "w_rec": 1.0},
    "noLcol": {"w_mod": 1.0, "w_col": 0.0, "w_rec": 1.0},
    "noLrec": {"w_mod": 1.0, "w_col": 5.0, "w_rec": 0.0},
}

# CSpec defaults (paper §5.1, line 376-378).
CSPEC_ALPHA = 1.0
CSPEC_K = 10

_DEFAULT_OUT = _RELEASE_ROOT / "results" / "ablation_perloss.csv"
_DEFAULT_SUMMARY_OUT = _RELEASE_ROOT / "results" / "ablation_perloss_summary.csv"
DEFAULT_FRIEDMAN_OUT = _RELEASE_ROOT / "results" / "ablation_perloss_friedman.csv"

CSV_COLUMNS = [
    "variant", "dataset", "seed",
    "nmi", "ari", "runtime_s", "train_time_s", "cluster_time_s",
]


# ------------------------------------------------------------- helpers ----
def run_one(variant: str, dataset: str, seed: int) -> dict[str, Any]:
    """Train one CGSD encoder (with the variant's loss weights) and cluster."""
    cfg = {**_BASE_CFG, **VARIANTS[variant]}

    t_global = time.time()
    t0 = time.time()
    try:
        emb = train_cgsd_pure(dataset, cfg, seed=seed)
    except Exception as exc:
        print(f"  [{variant:>6} on {dataset:<10} seed={seed}] TRAIN FAIL: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return {
            "variant": variant, "dataset": dataset, "seed": seed,
            "nmi": float("nan"), "ari": float("nan"),
            "runtime_s": time.time() - t_global,
            "train_time_s": time.time() - t0, "cluster_time_s": 0.0,
        }
    train_time = time.time() - t0

    # Load adjacency + labels (clustering is unsupervised but we still need
    # adj to build the curvature-weighted kNN graph and labels to score NMI).
    adj, _features, labels, _n_nodes, n_classes = load_dataset(dataset)
    coo = adj.tocoo()
    edge_index = (coo.row, coo.col)

    t1 = time.time()
    try:
        curvature = compute_forman_ricci_curvature(adj)
        # cluster_curvature_spectral expects edge_index as a 2xE ndarray.
        edge_index_arr = np.stack([coo.row, coo.col])  # noqa: F841
        pred = cluster_curvature_spectral(
            emb, n_classes, curvature, edge_index_arr,
            alpha=CSPEC_ALPHA, seed=seed, n_neighbors=CSPEC_K,
        )
        # NMI from sklearn directly (avoids the KMeans re-cluster inside
        # evaluate_nmi — we already have assignments).
        from sklearn.metrics import normalized_mutual_info_score
        from sklearn.metrics import adjusted_rand_score
        nmi = float(normalized_mutual_info_score(labels, pred))
        ari = float(adjusted_rand_score(labels, pred))
    except Exception as exc:
        print(f"  [{variant:>6} on {dataset:<10} seed={seed}] CLUSTER FAIL: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return {
            "variant": variant, "dataset": dataset, "seed": seed,
            "nmi": float("nan"), "ari": float("nan"),
            "runtime_s": time.time() - t_global,
            "train_time_s": train_time, "cluster_time_s": time.time() - t1,
        }
    cluster_time = time.time() - t1

    return {
        "variant": variant, "dataset": dataset, "seed": seed,
        "nmi": nmi, "ari": ari,
        "runtime_s": time.time() - t_global,
        "train_time_s": train_time, "cluster_time_s": cluster_time,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _print_progress(done: int, total: int, t_global: float, row: dict[str, Any]) -> None:
    elapsed = time.time() - t_global
    rate = done / max(elapsed, 1e-9)
    eta_min = (total - done) / max(rate, 1e-9) / 60.0
    nmi_str = f"{row['nmi']:.4f}" if row['nmi'] == row['nmi'] else "NaN"
    print(f"  [{done:3d}/{total}] {row['variant']:>6} on {row['dataset']:<10} "
          f"seed={row['seed']} NMI={nmi_str} "
          f"({row['runtime_s']:.1f}s) "
          f"rate={rate:.2f}/s ETA={eta_min:.1f} min")


def _summarise(rows: list[dict[str, Any]], summary_path: Path) -> dict:
    """Mean ± std NMI per (variant, dataset); also Friedman test.

    Returns a dict with raw means (used by the paper §5.7 LaTeX table).
    """
    import numpy as np
    from collections import defaultdict

    by_pair = defaultdict(list)
    for r in rows:
        if r["nmi"] == r["nmi"]:  # not NaN
            by_pair[(r["variant"], r["dataset"])].append(r["nmi"])

    summary_rows = []
    means = {}                # {variant: {dataset: mean}}
    for variant in VARIANTS:
        means[variant] = {}
        for dataset in DATASETS:
            vals = by_pair.get((variant, dataset), [])
            if vals:
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                means[variant][dataset] = m
                summary_rows.append({
                    "variant": variant, "dataset": dataset,
                    "mean_nmi": f"{m:.4f}",
                    "std_nmi": f"{s:.4f}",
                    "n_seeds": len(vals),
                })
            else:
                summary_rows.append({
                    "variant": variant, "dataset": dataset,
                    "mean_nmi": "NaN", "std_nmi": "NaN", "n_seeds": 0,
                })

    # Marginal means per variant (across datasets)
    for variant in VARIANTS:
        ds_means = [m for m in means[variant].values()]
        if ds_means:
            summary_rows.append({
                "variant": variant, "dataset": "__MEAN__",
                "mean_nmi": f"{np.mean(ds_means):.4f}",
                "std_nmi": f"{np.std(ds_means, ddof=1):.4f}" if len(ds_means) > 1 else "0.0000",
                "n_seeds": len(ds_means),
            })

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["variant", "dataset", "mean_nmi", "std_nmi", "n_seeds"],
        )
        w.writeheader()
        w.writerows(summary_rows)

    # Friedman test on the 5-dataset means (4 related samples).
    from scipy.stats import friedmanchisquare
    samples = [np.array([means[v][d] for d in DATASETS]) for v in VARIANTS]
    try:
        chi2, p = friedmanchisquare(*samples)
        friedman = {"statistic": float(chi2), "pvalue": float(p),
                    "n_datasets": len(DATASETS), "n_variants": len(VARIANTS)}
    except Exception as exc:
        friedman = {"statistic": float("nan"), "pvalue": float("nan"),
                    "error": f"{type(exc).__name__}: {exc}"}

    friedman_path = Path(os.environ.get("ABLATION_FRIEDMAN_OUT", str(DEFAULT_FRIEDMAN_OUT)))
    with friedman_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["statistic", "pvalue", "n_datasets", "n_variants", "error"])
        w.writeheader()
        w.writerow(friedman)
    print(f"\nFriedman test: chi2 = {friedman.get('statistic', float('nan')):.4f}, "
          f"p = {friedman.get('pvalue', float('nan')):.4g}")

    return {"means": means, "friedman": friedman}


# ---------------------------------------------------------------- main ----
def main() -> None:
    out_csv = Path(os.environ.get("ABLATION_OUT", str(_DEFAULT_OUT)))
    summary_csv = Path(os.environ.get(
        "ABLATION_SUMMARY_OUT",
        str(_RELEASE_ROOT / "results" / "ablation_perloss_summary.csv"),
    ))

    total = len(VARIANTS) * len(DATASETS) * len(SEEDS)
    print(f"=== Per-component ablation: {len(VARIANTS)} variants × "
          f"{len(DATASETS)} datasets × {len(SEEDS)} seeds = {total} runs ===")
    print(f"  Output CSV:     {out_csv}")
    print(f"  Summary CSV:    {summary_csv}")
    print(f"  CSpec config:   alpha={CSPEC_ALPHA}, k={CSPEC_K}")

    rows: list[dict[str, Any]] = []
    done = 0
    t_global = time.time()
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                row = run_one(variant, dataset, seed)
                rows.append(row)
                done += 1
                _write_csv(rows, out_csv)
                _print_progress(done, total, t_global, row)

    elapsed = time.time() - t_global
    print(f"\n=== DONE: {done} runs in {elapsed/60:.1f} min ===")

    summary = _summarise(rows, summary_csv)
    means = summary["means"]

    # Pretty-print the final table (paper Table 6).
    import numpy as np
    print("\n=== Mean NMI per (variant, dataset) across 5 seeds ===")
    header = "Variant".ljust(8) + "".join(d.ljust(12) for d in DATASETS) + "Mean".rjust(8)
    print(header)
    print("-" * len(header))
    for v in VARIANTS:
        row = v.ljust(8)
        vals = []
        for d in DATASETS:
            m = means[v].get(d, float("nan"))
            row += f"{m:.4f}".ljust(12)
            if m == m:
                vals.append(m)
        row += f"{np.mean(vals):.4f}".rjust(8) if vals else "—".rjust(8)
        print(row)


if __name__ == "__main__":
    main()