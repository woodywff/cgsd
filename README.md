# CGSD - Curvature-Guided Sheaf Diffusion

Truly-unsupervised community detection on heterophilic graphs.
Sheaf-diffusion encoder + curvature-aware spectral clustering (CSpec).

Paper: [Curvature-Guided Sheaf Diffusion for Unsupervised Community Detection on Heterophilic Graphs](https://arxiv.org/abs/2606.30249)

Source for: paper §5.1 (canonical column), §5.3 (SBM h-sweep),
            §5.1 (9-baseline columns via supplementary driver),
            §5.7 (per-component loss ablation).

## Install

```bash
pip install -r requirements.txt
```

Or run `bash reproduce.sh`, which installs dependencies automatically.

## Reproduction

All commands assume the working directory is the repository root.
All scripts create `results/` and `figures/` on first run.

| Paper claim | Command | Time | Output |
|---|---|---|---|
| §5.1 CSpec column (canonical CGSD, 5 datasets × 1 seed) | `bash reproduce.sh` | ~30 s | `results/cgsd_pure_optimization_log.csv` (mean NMI ≈ 0.107) |
| §5.3 SBM h-sweep (8 h × 10 realisations × 5 seeds = 400 cells) | `python scripts/run_sbm_sweep.py && python scripts/plot_sbm_sweep.py` | ~7 min | `results/sbm_sweep.csv`, `figures/sbm_sweep.pdf` |
| §5.1 baseline columns (9 baselines × 5 datasets × 5 seeds) | `python scripts/run_baselines_5seed.py` | ~5 min | `results/baselines_NMI_5seed.csv` |
| §5.7 per-loss ablation (4 variants × 5 datasets × 5 seeds = 100 cells) | `python scripts/run_ablation_perloss.py` | ~30-50 min | `results/ablation_perloss.csv`, `ablation_perloss_summary.csv`, `ablation_perloss_friedman.csv` |

Expected mean NMI on the 5 heterophilic benchmarks:
**CSpec ≈ 0.107** vs K-Means-only ≈ 0.091 (+18 % relative gain,
paired $t$-test $p = 0.011$ on 25 paired observations).

The baseline driver depends on `experiments/baselines_official.py`
from the parent project (vendored separately). Run `bash reproduce.sh`
first to install dependencies, then the baseline driver resolves.

The ablation driver holds the clusterer fixed at canonical CSpec
(α=1.0, k=10) and only varies the three encoder losses
(`w_mod` / `w_col` / `w_rec`), so the 4 variants isolate each loss's
contribution. A Friedman test across the 5-dataset means is written to
`results/ablation_perloss_friedman.csv`.

### Quick start (equivalent to `reproduce.sh`)

```bash
python scripts/smoke_test.py                              # 2-sec sanity check (Cora)
python -m cgsd.run_all --cluster_strategy curv_spectral   # full CGSD
```

Compare clusterers on the same encoder embedding:

```bash
python -m cgsd.run_all --cluster_strategy kmeans        # encoder only
python -m cgsd.run_all --cluster_strategy curv_spectral   # full CGSD (CSpec)
```

## Layout

```
cgsd/
  sheaf.py              # Forman–Ricci + sheaf-diffusion encoder
  losses.py             # modularity / anti-collapse / curvature-recon
  train.py              # label-free trainer
  cluster_strategies.py # CSpec (curvature-aware spectral clustering)
  data.py               # Cora, Cornell, Texas, Wisconsin, Chameleon
  eval.py               # K-Means + NMI protocol
  synthetic.py          # SBM graph generator (§5.3)
  run_all.py            # canonical entry point (§5.1)
scripts/
  smoke_test.py           # 2-sec end-to-end sanity check
  run_sbm_sweep.py        # SBM h-sweep (§5.3)
  plot_sbm_sweep.py       # plot h-sweep results
  run_baselines_5seed.py  # 9-baseline columns (§5.1, supplementary)
  run_ablation_perloss.py # per-component loss ablation (§5.7)
results/                # run outputs (created on first run)
figures/                # plots (created on first run)
reproduce.sh            # one-command canonical reproduction
```

## Config

| Hyperparameter | Value |
|----------------|-------|
| hidden / heads | 64 / 2 |
| dropout | 0.3 |
| epochs | 30 |
| lr / weight decay | 0.005 / 5e-4 |
| (w_mod, w_col, w_rec) | (1, 5, 1) |
| CSpec (α, k) | (1.0, 10) |

## License

MIT. See `LICENSE`.
