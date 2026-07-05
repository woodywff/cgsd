"""cgsd_release — truly-unsupervised CGSD, label-free, single-config.

This is the canonical open-source release of the CGSD paper code
(v2026-06-25, plus 2026-07-04 SBM §5.3 evidence):

    - SheafDiffusion architecture (label-free)
    - 3 structural losses (modularity, anti-collapse, curvature-recon)
    - 5-dataset loader (Cora / Cornell / Texas / Wisconsin / Chameleon)
    - Single-embedding trainer (returns the embedding, not NMI)
    - Curvature-aware spectral clustering (CSpec / G3): Ng-Jordan-Weiss
      on a Forman-Ricci-weighted kNN affinity of the embedding
    - Unified KMeans+NMI evaluation protocol
    - Pure-numpy SBM generator (paper §5.3)

Algorithm (no ensemble, no multi-config, no label-based selection):
    1. Train sheaf-diffusion encoder with structural losses (label-free)
    2. Cluster the embedding via curvature-aware spectral clustering
    3. Evaluate NMI/ARI against ground truth (for evaluation only)

The previous "60-emb label-free ensemble" approach has been retired.
The ensemble was a generic post-processing trick (silhouette-select +
L2/power-norm + concat + KMeans) that any embedding method could use —
not a CGSD-specific algorithmic win, and its fusion logic was neither
simpler nor more interpretable than the direct CSpec clustering now used.

Honest disclosure: the supervised cross-entropy version (which produced
the retracted 0.928 NMI) is NOT in this package. See
``stage-22/code/_deprecated/main_supervised_ce.py`` in the parent project
for archival reference only.
"""
__version__ = "2.1.0"   # 2026-07-04: SBM §5.3 mechanism evidence added
__all__ = [
    "sheaf",
    "losses",
    "data",
    "train",
    "cluster_strategies",
    "synthetic",   # pure-numpy SBM generator (paper §5.3)
    "eval",
    "run_all",
]
