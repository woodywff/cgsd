# CGSD

**Curvature-Guided Sheaf Diffusion** — unsupervised community detection on heterophilic graphs.

Paper: [Curvature-Guided Sheaf Diffusion for Unsupervised Community Detection on Heterophilic Graphs](https://arxiv.org/abs/2606.30249) 

## Install

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
bash reproduce.sh
# or
python -m cgsd.run_all --cluster cspec
```

Output: `results/table2_cgsd.csv` (expected mean NMI ≈ 0.107).

Compare clusterers:

```bash
python -m cgsd.run_all --cluster kmeans   # encoder only
python -m cgsd.run_all --cluster cspec    # full CGSD
```

Smoke test:

```bash
python scripts/smoke_test.py
```

## Layout

```
cgsd/
  sheaf.py      # Forman–Ricci + sheaf-diffusion encoder
  losses.py     # modularity / anti-collapse / curvature-recon
  train.py      # label-free trainer
  cspec.py      # CSpec (curvature-aware spectral clustering)
  data.py       # Cora, Cornell, Texas, Wisconsin, Chameleon
  eval.py       # K-Means + NMI protocol
  run_all.py    # Table-1 entry point
scripts/        # smoke tests
results/        # run outputs
paper/          # PDF
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

MIT
