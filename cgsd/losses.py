"""Three label-free structural losses for CGSD training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree


def modularity_loss(
    soft_assign: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Negative modularity of a soft cluster assignment.

    Args:
        soft_assign: Soft assignment ``Z``, shape ``[N, K]``.
        edge_index: Edge index, shape ``[2, E]``.
        num_nodes: Number of nodes.

    Returns:
        Scalar loss ``-Q``.
    """
    row, col = edge_index
    soft_assign = torch.clamp(soft_assign, min=1e-10, max=1.0)
    m = row.numel() / 2.0 + 1e-9
    deg = degree(row, num_nodes, dtype=soft_assign.dtype)
    d_c = soft_assign.t() @ deg / (2 * m)
    s_u, s_v = soft_assign[row], soft_assign[col]
    e_c = (s_u * s_v).sum(dim=0) / m
    return -(e_c - d_c ** 2).sum()


def collapse_loss(
    soft_assign: torch.Tensor,
    mode: str = "balance",
) -> torch.Tensor:
    """Anti-collapse penalty on soft cluster assignments.

    Args:
        soft_assign: Soft assignment ``Z``, shape ``[N, K]``.
        mode: One of ``balance``, ``uniform``, ``entropy``, ``minmax``.

    Returns:
        Scalar anti-collapse loss.
    """
    n_clusters = soft_assign.shape[1]
    gram = soft_assign.t() @ soft_assign
    n_nodes = soft_assign.shape[0]

    if mode == "balance":
        target = (n_nodes / n_clusters) * torch.eye(
            n_clusters, device=soft_assign.device, dtype=soft_assign.dtype
        )
        return ((gram - target) ** 2).sum() / (n_nodes ** 2)
    if mode == "uniform":
        cluster_size = soft_assign.sum(0)
        target_size = n_nodes / n_clusters
        return ((cluster_size - target_size) ** 2).mean() / (
            n_nodes ** 2 / n_clusters ** 2
        )
    if mode == "entropy":
        ent = -(soft_assign * torch.log(soft_assign + 1e-10)).sum(dim=1).mean()
        return -ent
    if mode == "minmax":
        cluster_frac = soft_assign.sum(0) / n_nodes
        return (cluster_frac.std() / (cluster_frac.mean() + 1e-9)) ** 2
    raise ValueError(f"Unknown mode: {mode}")


def curvature_recon_loss(
    soft_assign: torch.Tensor,
    curvature: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Curvature-weighted reconstruction loss.

    Args:
        soft_assign: Soft assignment ``Z``, shape ``[N, K]``.
        curvature: Per-edge Forman-Ricci curvature, shape ``[E]``.
        edge_index: Edge index, shape ``[2, E]``.

    Returns:
        MSE between predicted edge agreement and ``σ(κ_e)``.
    """
    row, col = edge_index
    curv = (
        curvature
        if isinstance(curvature, torch.Tensor)
        else torch.tensor(curvature, dtype=soft_assign.dtype, device=soft_assign.device)
    )
    target = torch.sigmoid(curv)
    pred = (soft_assign[row] * soft_assign[col]).sum(dim=-1)
    return F.mse_loss(pred, target)
