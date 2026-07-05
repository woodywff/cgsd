"""Unified KMeans+NMI evaluation protocol.

This is the same protocol used in the paper's Table 1 (EVAL_PROTOCOL.md):
    KMeans(seed=0, n_init=10, k=n_classes) -> NMI vs ground truth.

This module is the SINGLE SOURCE OF TRUTH for evaluation in cgsd.
The release does not depend on the parent project's unified_eval.py —
this is a self-contained copy.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    normalized_mutual_info_score,
    v_measure_score,
)

# Protocol constants — DO NOT change without re-running every result.
KMEANS_SEED: int = 0
KMEANS_N_INIT: int = 10


def evaluate_nmi(
    embedding: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    k: int | None = None,
    n_init: int = KMEANS_N_INIT,
    seed: int = KMEANS_SEED,
) -> dict:
    """Run KMeans(embedding) -> cluster assignment, return clustering metrics.

    Parameters
    ----------
    embedding : (N, d) or (N,) array-like.
        If 1D, treated as cluster assignments directly (no KMeans).
    labels    : (N,) array-like of ground-truth labels.
    k         : number of clusters.  Defaults to labels.max() + 1.

    Returns
    -------
    dict with keys: nmi, ari, fmi, homo, compl, v_measure, k, n, pred
    """
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    embedding = np.asarray(embedding)
    labels = np.asarray(labels)

    n = labels.shape[0]
    if k is None:
        k = int(labels.max()) + 1

    if embedding.ndim == 1 or (embedding.ndim == 2 and embedding.shape[1] == 1):
        pred = embedding.reshape(-1).astype(int)
        unique = np.unique(pred)
        if not np.array_equal(unique, np.arange(k)):
            remap = {v: i for i, v in enumerate(unique)}
            pred = np.array([remap[v] for v in pred])
    else:
        if embedding.shape[0] != n:
            raise ValueError(
                f"embedding has {embedding.shape[0]} rows but labels has {n}"
            )
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        pred = km.fit_predict(embedding)

    return {
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
        "fmi": float(fowlkes_mallows_score(labels, pred)),
        "homo": float(homogeneity_score(labels, pred)),
        "compl": float(completeness_score(labels, pred)),
        "v_measure": float(v_measure_score(labels, pred)),
        "k": int(k),
        "n": int(n),
        "pred": pred,
    }
