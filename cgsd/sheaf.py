"""Curvature-gated sheaf-diffusion encoder."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.utils import degree


def compute_forman_ricci_curvature(adj) -> np.ndarray:
    """Compute Forman-Ricci curvature per edge.

    Args:
        adj: Sparse or dense adjacency matrix.

    Returns:
        Array of shape ``[E]`` with ``κ_e = 4 - deg(u) - deg(v)``.
    """
    if sp.issparse(adj):
        deg = np.array(adj.sum(axis=1)).flatten()
        coo = adj.tocoo()
    else:
        deg = adj.sum(axis=1)
        coo = sp.coo_matrix(adj).tocoo()
    return 4.0 - deg[coo.row] - deg[coo.col]


class SheafDiffusion(nn.Module):
    """Curvature-gated sheaf diffusion with residual connection.

    Edge messages are weighted by
    ``w_e = deg(u)^{-1/2} deg(v)^{-1/2} · σ(κ_e)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_projections = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(n_heads)
        ])
        self.residual = nn.Linear(in_features, out_features * n_heads, bias=False)
        self._init_weights()

    def _init_weights(self):
        for proj in self.head_projections:
            nn.init.xavier_uniform_(proj.weight, gain=1.414)
        nn.init.xavier_uniform_(self.residual.weight, gain=1.414)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        curvature: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Apply one sheaf-diffusion layer.

        Args:
            x: Node features, shape ``[N, in_features]``.
            edge_index: Edge index, shape ``[2, E]``.
            curvature: Per-edge Forman-Ricci curvature, shape ``[E]``.

        Returns:
            Layer output, shape ``[N, out_features * n_heads]``.
        """
        n_nodes = x.shape[0]
        row, col = edge_index

        deg = degree(row, n_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if curvature is not None:
            curv = (
                curvature
                if isinstance(curvature, torch.Tensor)
                else torch.tensor(curvature, dtype=x.dtype, device=x.device)
            )
            edge_w = norm * torch.sigmoid(curv)
        else:
            edge_w = norm

        head_outputs = []
        for head in self.head_projections:
            h_x = head(x)
            msg = h_x[col] * edge_w.unsqueeze(1)
            out_h = torch.zeros_like(h_x)
            out_h.index_add_(0, row, msg)
            head_outputs.append(out_h)

        out = torch.cat(head_outputs, dim=1)
        return 0.5 * out + 0.5 * self.residual(x)
