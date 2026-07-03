"""Label-free CGSD trainer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cgsd.sheaf import SheafDiffusion, compute_forman_ricci_curvature
from cgsd.losses import modularity_loss, collapse_loss, curvature_recon_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CGSD(nn.Module):
    """Two-layer sheaf encoder with a soft cluster head."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        n_clusters: int,
        n_heads: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.sheaf1 = SheafDiffusion(in_features, hidden_dim, n_heads, dropout)
        self.sheaf2 = SheafDiffusion(
            hidden_dim * n_heads, hidden_dim, n_heads, dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.cluster_head = nn.Linear(hidden_dim * n_heads, n_clusters, bias=False)
        nn.init.xavier_uniform_(self.cluster_head.weight)

    def encode(self, x, edge_index, curvature=None):
        """Encode nodes into the sheaf embedding."""
        x = F.relu(self.sheaf1(x, edge_index, curvature))
        x = self.dropout(x)
        return F.relu(self.sheaf2(x, edge_index, curvature))

    def forward(self, x, edge_index, curvature):
        """Return soft assignment and embedding."""
        emb = self.encode(x, edge_index, curvature)
        soft_assign = F.softmax(self.cluster_head(emb), dim=-1)
        return soft_assign, emb


def train_cgsd(name: str, cfg: dict, seed: int = 0) -> np.ndarray:
    """Train CGSD and return the final embedding.

    Args:
        name: Dataset name (one of the five Table-1 datasets).
        cfg: Training config with keys ``hidden``, ``n_heads``, ``dropout``,
            ``lr``, ``wd``, ``struct_epochs``, ``w_mod``, ``w_col``, ``w_rec``,
            ``collapse_mode``.
        seed: RNG seed.

    Returns:
        Embedding array of shape ``[N, hidden * n_heads]``.
    """
    from cgsd.data import load_dataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    adj, features, _, n_nodes, n_classes = load_dataset(name)
    coo = adj.tocoo()
    edge_index = torch.LongTensor(np.stack([coo.row, coo.col])).to(DEVICE)
    x = torch.FloatTensor(features).to(DEVICE)
    curvature = torch.tensor(
        compute_forman_ricci_curvature(adj),
        dtype=torch.float32,
        device=DEVICE,
    )

    model = CGSD(
        features.shape[1],
        cfg["hidden"],
        n_classes,
        n_heads=cfg.get("n_heads", 2),
        dropout=cfg.get("dropout", 0.5),
    ).to(DEVICE)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 0.005),
        weight_decay=cfg.get("wd", 5e-4),
    )

    n_struct = cfg.get("struct_epochs", 30)
    w_mod = cfg.get("w_mod", 1.0)
    w_col = cfg.get("w_col", 5.0)
    w_rec = cfg.get("w_rec", 1.0)
    collapse_mode = cfg.get("collapse_mode", "balance")

    best_loss = float("inf")
    best_emb = None

    for _ in range(n_struct):
        model.train()
        opt.zero_grad()
        soft_assign, emb = model(x, edge_index, curvature)
        loss = (
            w_mod * modularity_loss(soft_assign, edge_index, n_nodes)
            + w_col * collapse_loss(soft_assign, mode=collapse_mode)
            + w_rec * curvature_recon_loss(soft_assign, curvature, edge_index)
        )
        loss.backward()
        opt.step()

        with torch.no_grad():
            soft_assign, emb = model(x, edge_index, curvature)
            cur_loss = (
                w_mod * modularity_loss(soft_assign, edge_index, n_nodes)
                + w_col * collapse_loss(soft_assign, mode=collapse_mode)
            ).item()
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_emb = emb.detach().cpu().numpy()

    if best_emb is None:
        with torch.no_grad():
            _, emb = model(x, edge_index, curvature)
            best_emb = emb.detach().cpu().numpy()
    return best_emb
