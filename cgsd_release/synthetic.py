"""Stochastic Block Model generator with closed-form heterophily targeting.

The generator supports a target heterophily ratio `h` (fraction of edges
whose endpoints are in different classes) by fixing `p_in=0.3` and solving
analytically for `p_out`. Class-conditional features are Gaussians with
mean `(1-h) * mu_y` and variance `h / d`, so features are simultaneously
informative at low h and simultaneously uninformative at high h.

See paper §5.3 for the variable-isolation rationale.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# Closed-form for c=5, n=800, p_in=0.3 (verified by derivation in spec §4).
# For other (n, c, p_in), recompute via _p_out_for_h().
_P_IN_DEFAULT = 0.3


def _p_out_for_h(h: float, n: int, c: int, p_in: float) -> float:
    """Solve for p_out such that realized heterophily == h under Bernoulli SBM.

    For balanced classes (n_i = n // c), the realized heterophily is:
        h ≈ (C(c,2)*(n/c)^2 * p_out) /
            (C(c,2)*(n/c)^2 * p_out + c * C(n/c,2) * p_in)

    Solve for p_out:
        p_out = (c * C(n/c,2) * p_in * h) / (C(c,2) * (n/c)^2 * (1 - h))
    """
    n_per = n / c
    intra = c * (n_per * (n_per - 1) / 2) * p_in
    inter = (c * (c - 1) / 2) * (n_per ** 2)
    if inter == 0:
        raise ValueError(f"n={n}, c={c}: cannot solve p_out (inter denom 0)")
    p_out = intra * h / (inter * (1.0 - h))
    return float(p_out)


def realized_heterophily_with_labels(A: sp.csr_matrix, y: np.ndarray) -> float:
    """Empirical heterophily: fraction of edges whose endpoints differ in `y`."""
    coo = A.tocoo()
    src = coo.row
    dst = coo.col
    diff = (y[src] != y[dst]).sum()
    n_edges = src.shape[0]
    if n_edges == 0:
        return 0.0
    return float(diff) / float(n_edges)


def make_sbm(
    n: int = 800,
    c: int = 5,
    h: float = 0.5,
    d: int = 64,
    p_in: float = _P_IN_DEFAULT,
    seed: int = 0,
) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """Generate a synthetic SBM realization with target heterophily `h`.

    Nodes are ordered by class (contiguous blocks). Features are class-
    conditional Gaussians x_v = (1-h) * mu_{y_v} + sqrt(h) * eps_v,
    eps_v ~ N(0, I_d/d).

    Args:
        n  : total number of nodes (must be >= c).
        c  : number of communities.
        h  : target heterophily ratio in [0, 0.9].
        d  : feature dimension (must be >= c).
        p_in: fixed intra-class edge probability (default 0.3).
        seed: numpy RNG seed.

    Returns:
        (x, A, y) where
            x: (n, d) float32 features.
            A: (n, n) scipy CSR adjacency (symmetric, no self-loops).
            y: (n,) int64 class labels in {0, ..., c-1}.

    Raises:
        ValueError: if `n < c`, `h` is out of range, `d < c`, or the
            closed-form solution requires `p_out > 1`.
    """
    if n < c:
        raise ValueError(f"n={n} must be >= c={c}")
    if h < 0 or h >= 1:
        raise ValueError(f"h={h} must be in [0, 1)")
    if d < c:
        raise ValueError(f"d={d} must be >= c={c}")

    p_out = _p_out_for_h(h, n, c, p_in)
    if p_out > 1.0:
        raise ValueError(
            f"h={h}: required p_out > 1 ({p_out:.4f}); "
            "raise p_in or pick a smaller h."
        )

    rng = np.random.default_rng(seed)

    # Labels: contiguous blocks, distributed as evenly as possible.
    # n_per = n // c is the base size; the first (n % c) classes get +1.
    n_per_base = n // c
    rem = n % c
    sizes = np.array([n_per_base + (1 if i < rem else 0) for i in range(c)],
                     dtype=np.int64)
    y = np.repeat(np.arange(c, dtype=np.int64), sizes)

    # Cumulative offsets: class i occupies node IDs [offsets[i], offsets[i+1]).
    offsets = np.concatenate([[0], np.cumsum(sizes)])

    # Edges: sample per class pair
    src_list = []
    dst_list = []
    for i in range(c):
        # Intra-class edges: pairs within class i
        block_i = np.arange(offsets[i], offsets[i + 1])
        ii, jj = np.meshgrid(block_i, block_i, indexing="ij")
        mask = ii < jj  # upper triangle only, no self-loops
        u = ii[mask]
        v = jj[mask]
        # Bernoulli per pair
        keep = rng.random(u.shape) < p_in
        if keep.any():
            src_list.append(u[keep])
            dst_list.append(v[keep])
        for j in range(i + 1, c):
            block_j = np.arange(offsets[j], offsets[j + 1])
            ii, jj = np.meshgrid(block_i, block_j, indexing="ij")
            u = ii.ravel()
            v = jj.ravel()
            keep = rng.random(u.shape) < p_out
            if keep.any():
                src_list.append(u[keep])
                dst_list.append(v[keep])

    if not src_list:
        # Empty graph (very small h or tiny n): still return valid csr
        A = sp.csr_matrix((n, n), dtype=np.float32)
    else:
        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)
        # Symmetrize
        src_full = np.concatenate([src, dst])
        dst_full = np.concatenate([dst, src])
        A = sp.csr_matrix(
            (np.ones(src_full.shape[0], dtype=np.float32),
             (src_full, dst_full)),
            shape=(n, n),
        )
        # Remove duplicate edges (keep one)
        A.sum_duplicates()
        # Drop self-loops defensively
        A.setdiag(0)
        A.eliminate_zeros()

    # Features: x_v = (1-h) * mu_{y_v} + sqrt(h) * eps_v
    mu = np.zeros((c, d), dtype=np.float64)
    for i in range(c):
        mu[i, i] = 1.0  # one-hot, orthogonal unit vectors
    eps = rng.standard_normal(size=(n, d)) / np.sqrt(d)
    x = np.empty((n, d), dtype=np.float32)
    for i in range(c):
        block = (y == i)
        x[block] = ((1.0 - h) * mu[i] + np.sqrt(h) * eps[block]).astype(np.float32)
    return x, A, y