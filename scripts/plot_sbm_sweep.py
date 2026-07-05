#!/usr/bin/env python
"""Read SBM sweep CSV, write line+SEM plot to PDF.

Usage:
    python plot_sbm_sweep.py [INPUT_CSV] [OUTPUT_PDF]

Defaults:
    INPUT_CSV   = results/sbm_sweep.csv
    OUTPUT_PDF  = figures/sbm_sweep.pdf
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_REPO = Path(__file__).resolve().parent.parent


def _aggregate(rows):
    """Group by h; return h, mean_cspec, sem_cspec, mean_km, sem_km."""
    by_h = defaultdict(lambda: {"cspec": [], "km": []})
    for row in rows:
        if int(row["failed"]) == 1:
            continue
        h = float(row["h"])
        by_h[h]["cspec"].append(float(row["cspec_nmi"]))
        by_h[h]["km"].append(float(row["kmeans_nmi"]))
    out_h, out_cspec_m, out_cspec_s, out_km_m, out_km_s = [], [], [], [], []
    for h in sorted(by_h):
        cspec = np.array(by_h[h]["cspec"])
        km = np.array(by_h[h]["km"])
        out_h.append(h)
        out_cspec_m.append(cspec.mean())
        out_cspec_s.append(cspec.std(ddof=1) / np.sqrt(len(cspec))
                            if len(cspec) > 1 else 0.0)
        out_km_m.append(km.mean())
        out_km_s.append(km.std(ddof=1) / np.sqrt(len(km))
                        if len(km) > 1 else 0.0)
    return (np.array(out_h),
            np.array(out_cspec_m), np.array(out_cspec_s),
            np.array(out_km_m), np.array(out_km_s))


def main(argv):
    in_csv = Path(argv[1]) if len(argv) > 1 else _REPO / "results" / "sbm_sweep.csv"
    out_pdf = Path(argv[2]) if len(argv) > 2 else _REPO / "figures" / "sbm_sweep.pdf"

    with in_csv.open() as f:
        rows = list(csv.DictReader(f))

    h, cspec_m, cspec_s, km_m, km_s = _aggregate(rows)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(h, cspec_m, "-", color="#3B82F6", lw=2, label="CGSD-CSpec (ours)")
    ax.fill_between(h, cspec_m - cspec_s, cspec_m + cspec_s,
                    color="#3B82F6", alpha=0.20)
    ax.plot(h, km_m, "--", color="#9CA3AF", lw=2, label=r"$K$-Means on encoder")
    ax.fill_between(h, km_m - km_s, km_m + km_s,
                    color="#9CA3AF", alpha=0.20)

    ax.set_xlabel(r"Heterophily ratio $h$", fontsize=11)
    ax.set_ylabel("NMI", fontsize=11)
    ax.set_xlim(0.0, 0.95)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="both", alpha=0.15)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("SBM h-sweep: NMI vs heterophily ratio",
                 fontsize=10, pad=8)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)