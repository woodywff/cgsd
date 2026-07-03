"""Clustering strategies: K-Means, spectral, and CSpec."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph


def cluster_kmeans(
    emb: np.ndarray,
    n_classes: int,
    seed: int = 0,
) -> np.ndarray:
    """K-Means on the embedding.

    Args:
        emb: Node embeddings, shape ``[N, d]``.
        n_classes: Number of communities.
        seed: K-Means random seed.

    Returns:
        Integer cluster assignments of length ``N``.
    """
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    return km.fit_predict(emb)


def cluster_spectral(
    emb: np.ndarray,
    n_classes: int,
    seed: int = 0,
    n_neighbors: int = 10,
) -> np.ndarray:
    """Ng-Jordan-Weiss spectral clustering on a k-NN affinity.

    Args:
        emb: Node embeddings, shape ``[N, d]``.
        n_classes: Number of communities.
        seed: K-Means random seed.
        n_neighbors: k-NN graph degree.

    Returns:
        Integer cluster assignments of length ``N``.
    """
    affinity = kneighbors_graph(
        emb,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
        n_jobs=1,
    )
    affinity = affinity.maximum(affinity.T).tocsr()
    return _njw(affinity, n_classes, seed)


def cluster_cspec(
    emb: np.ndarray,
    n_classes: int,
    curvature: np.ndarray,
    edge_index: np.ndarray,
    alpha: float = 1.0,
    seed: int = 0,
    n_neighbors: int = 10,
) -> np.ndarray:
    """Curvature-aware spectral clustering (CSpec).

    Builds a k-NN graph on the embedding, re-weights each affinity edge by
    ``σ(α κ_{e*})`` from the nearest original-graph edge, then runs NJW.

    Args:
        emb: Node embeddings, shape ``[N, d]``.
        n_classes: Number of communities.
        curvature: Forman-Ricci curvature per original edge, shape ``[E]``.
        edge_index: Original-graph edge index, shape ``[2, E]``.
        alpha: Curvature weight (default 1.0).
        seed: K-Means random seed.
        n_neighbors: k-NN graph degree (default 10).

    Returns:
        Integer cluster assignments of length ``N``.
    """
    affinity = kneighbors_graph(
        emb,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
        n_jobs=1,
    )
    affinity = affinity.maximum(affinity.T).tocsr()

    coo = affinity.tocoo()
    src, dst = coo.row, coo.col
    curv_lookup = {}
    edge_index = np.asarray(edge_index)
    for idx in range(edge_index.shape[1]):
        u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
        curv_lookup[(u, v)] = float(curvature[idx])
        curv_lookup[(v, u)] = float(curvature[idx])

    curv_knn = np.array(
        [curv_lookup.get((int(s), int(d)), 0.0) for s, d in zip(src, dst)],
        dtype=np.float64,
    )
    weights = 1.0 / (1.0 + np.exp(-alpha * curv_knn))
    affinity_w = sp.coo_matrix((weights, (src, dst)), shape=affinity.shape).tocsr()
    affinity_w = affinity_w.maximum(affinity_w.T)
    return _njw(affinity_w, n_classes, seed)


def _njw(affinity: sp.spmatrix, n_classes: int, seed: int) -> np.ndarray:
    """Ng-Jordan-Weiss on a symmetric affinity matrix."""
    deg = np.asarray(affinity.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    d_inv = sp.diags(d_inv_sqrt)
    laplacian = (sp.eye(affinity.shape[0]) - d_inv @ affinity @ d_inv).tocsr()
    try:
        eigvals, eigvecs = spla.eigsh(
            laplacian, k=n_classes + 1, which="SM", tol=1e-3
        )
    except Exception:
        eigvals, eigvecs = np.linalg.eigh(laplacian.toarray())
        eigvals, eigvecs = eigvals[: n_classes + 1], eigvecs[:, : n_classes + 1]
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order[1 : n_classes + 1]]
    eigvecs = eigvecs / (np.linalg.norm(eigvecs, axis=1, keepdims=True) + 1e-9)
    return KMeans(n_clusters=n_classes, random_state=seed, n_init=10).fit_predict(
        eigvecs
    )


CLUSTER_STRATEGIES = {
    "kmeans": cluster_kmeans,
    "spectral": cluster_spectral,
    "cspec": cluster_cspec,
}
