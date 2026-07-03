"""K-Means + NMI evaluation protocol."""

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

KMEANS_SEED: int = 0
KMEANS_N_INIT: int = 10


def evaluate_nmi(
    embedding: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    k: int | None = None,
    n_init: int = KMEANS_N_INIT,
    seed: int = KMEANS_SEED,
) -> dict:
    """Cluster an embedding with K-Means and report NMI/ARI.

    Args:
        embedding: Node embeddings ``[N, d]``, or 1-D assignments.
        labels: Ground-truth labels ``[N]`` (evaluation only).
        k: Number of clusters. Defaults to ``labels.max() + 1``.
        n_init: K-Means restarts.
        seed: K-Means random seed.

    Returns:
        Dict with ``nmi``, ``ari``, and related metrics.
    """
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    embedding = np.asarray(embedding)
    labels = np.asarray(labels)

    n_nodes = labels.shape[0]
    if k is None:
        k = int(labels.max()) + 1

    if embedding.ndim == 1 or (embedding.ndim == 2 and embedding.shape[1] == 1):
        pred = embedding.reshape(-1).astype(int)
        unique = np.unique(pred)
        if not np.array_equal(unique, np.arange(k)):
            remap = {v: i for i, v in enumerate(unique)}
            pred = np.array([remap[v] for v in pred])
    else:
        if embedding.shape[0] != n_nodes:
            raise ValueError(
                f"embedding has {embedding.shape[0]} rows but labels has {n_nodes}"
            )
        pred = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(
            embedding
        )

    return {
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
        "fmi": float(fowlkes_mallows_score(labels, pred)),
        "homo": float(homogeneity_score(labels, pred)),
        "compl": float(completeness_score(labels, pred)),
        "v_measure": float(v_measure_score(labels, pred)),
        "k": int(k),
        "n": int(n_nodes),
        "pred": pred,
    }
