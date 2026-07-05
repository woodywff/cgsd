#!/usr/bin/env python
"""Run the full SBM h-sweep and write long-form CSV.

Iterates 8 h values × 10 SBM realizations × 5 CGSD seeds per cell
(400 cells), running both CSpec and K-Means on the same encoder
embedding per cell. Writes one row per cell to stdout or to the file
named by the SBM_SWEEP_OUT env var (default: results/sbm_sweep.csv).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

_PKG_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_PARENT))

from cgsd_release.synthetic import make_sbm
from cgsd_release.train import train_cgsd_pure_from_data
from cgsd_release.cluster_strategies import (
    cluster_kmeans, cluster_curvature_spectral,
)
from cgsd_release.sheaf import compute_forman_ricci_curvature
from sklearn.metrics import normalized_mutual_info_score


H_VALUES = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90]
N_REALIZATIONS = 10
N_SEEDS = 5
N_NODES = 800
N_CLASSES = 5
D_FEATS = 64
P_IN = 0.3

CFG = dict(
    hidden=64, n_heads=2, dropout=0.3, lr=0.01, wd=0.0,
    pretrain_epochs=0, struct_epochs=30,
    w_mod=1.0, w_col=0.5, w_rec=0.1,
    collapse_mode="balance", use_dgi=False, use_recon=False,
    kmeans_init=False,
)


def _realization_seed(h: float, r: int) -> int:
    return int(abs(hash(("sbm_real", h, r))) % (2 ** 31 - 1))


def _curvature_edge_index(A):
    coo = A.tocoo()
    return np.stack([coo.row, coo.col])


def run_one_cell(h, r, s, lr_override=None):
    cfg = dict(CFG)
    if lr_override is not None:
        cfg["lr"] = lr_override
    sbm_seed = _realization_seed(h, r)
    x, A, y = make_sbm(n=N_NODES, c=N_CLASSES, h=h, d=D_FEATS, p_in=P_IN, seed=sbm_seed)
    try:
        emb = train_cgsd_pure_from_data(A, x, n_classes=N_CLASSES, cfg=cfg, seed=s)
        curv = compute_forman_ricci_curvature(A)
        edge_index = _curvature_edge_index(A)
        cspec = cluster_curvature_spectral(emb, n_classes=N_CLASSES,
                                           curvature=curv, edge_index=edge_index, seed=s)
        km = cluster_kmeans(emb, n_classes=N_CLASSES, seed=s)
        return dict(h=h, realization=r, seed=s,
                    cspec_nmi=float(normalized_mutual_info_score(y, cspec)),
                    kmeans_nmi=float(normalized_mutual_info_score(y, km)),
                    failed=0)
    except Exception as e:
        print(f"[WARN] cell h={h} r={r} s={s} failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return dict(h=h, realization=r, seed=s,
                    cspec_nmi=float("nan"), kmeans_nmi=float("nan"),
                    failed=1)


def main():
    out_path = Path(os.environ.get(
        "SBM_SWEEP_OUT",
        str(_PKG_PARENT / "results" / "sbm_sweep.csv"),
    ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_cells = len(H_VALUES) * N_REALIZATIONS * N_SEEDS
    print(f"SBM sweep: {len(H_VALUES)} h × {N_REALIZATIONS} rel × {N_SEEDS} seeds = {n_cells} cells; output → {out_path}",
          file=sys.stderr)
    fieldnames = ["h", "realization", "seed", "cspec_nmi", "kmeans_nmi", "failed"]
    rows = []
    t0 = time.time()
    done = 0
    for h in H_VALUES:
        for r in range(N_REALIZATIONS):
            for s in range(N_SEEDS):
                rows.append(run_one_cell(h, r, s))
                done += 1
                if done % 20 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (n_cells - done) / max(rate, 1e-9)
                    print(f"[{done}/{n_cells}] {rate:.2f} cells/s, ETA {eta/60:.1f} min",
                          file=sys.stderr)
    with out_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()