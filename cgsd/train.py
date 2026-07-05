"""Single-embedding CGSD-pure trainer (label-free).

Distilled from experiments/sheaf_curvature_001/experiments/iter11_kmeans_init_seed.py.
The trainer returns the final embedding h (not NMI) — the ensemble step
uses these embeddings to fuse.

Key features (all label-free):
    - 3 structural losses: modularity, anti-collapse, curvature-recon
    - K-means init for cluster head after pretraining (gives +1% NMI
      "for free" — iter 11 finding)
    - Optional self-supervised pretraining (DGI / feature reconstruction)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from cgsd.sheaf import SheafDiffusion, compute_forman_ricci_curvature
from cgsd.losses import (
    _modularity_loss, _collapse_loss_v2, _curvature_recon_loss,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CGSDPure(nn.Module):
    """2-layer sheaf encoder + K-head cluster assignment.

    Optional self-supervised heads (DGI discriminator, feature decoder)
    for the pretrain phase.
    """

    def __init__(self, in_features: int, hidden_dim: int, n_clusters: int,
                 n_heads: int = 2, dropout: float = 0.5,
                 use_dgi: bool = False, use_recon: bool = False):
        super().__init__()
        self.sheaf1 = SheafDiffusion(in_features, hidden_dim, n_heads, dropout)
        self.sheaf2 = SheafDiffusion(hidden_dim * n_heads, hidden_dim, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.cluster_head = nn.Linear(hidden_dim * n_heads, n_clusters, bias=False)
        nn.init.xavier_uniform_(self.cluster_head.weight)
        self.use_dgi = use_dgi
        self.use_recon = use_recon
        if use_dgi:
            self.discriminator = nn.Bilinear(hidden_dim * n_heads, hidden_dim * n_heads, 1)
        if use_recon:
            self.decoder = nn.Linear(hidden_dim * n_heads, in_features, bias=True)

    def encode(self, x, edge_index, curvature=None):
        x = F.relu(self.sheaf1(x, edge_index, curvature))
        x = self.dropout(x)
        x = F.relu(self.sheaf2(x, edge_index, curvature))
        return x

    def cluster_forward(self, x, edge_index, curvature):
        h = self.encode(x, edge_index, curvature)
        S = F.softmax(self.cluster_head(h), dim=-1)
        return S, h


def _train_cgsd_pure_inner(
    adj,
    features: np.ndarray,
    n_classes: int,
    cfg: dict,
    seed: int,
) -> np.ndarray:
    """In-memory trainer: takes scipy adjacency + numpy features.

    Mirrors what train_cgsd_pure does after `load_dataset`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_nodes = features.shape[0]
    coo = adj.tocoo()
    edge_index = torch.LongTensor(np.stack([coo.row, coo.col])).to(DEVICE)
    x = torch.FloatTensor(features).to(DEVICE)
    curvature = torch.tensor(
        compute_forman_ricci_curvature(adj),
        dtype=torch.float32, device=DEVICE,
    )

    in_dim = features.shape[1]
    model = CGSDPure(
        in_dim, cfg["hidden"], n_classes,
        n_heads=cfg.get("n_heads", 2),
        dropout=cfg.get("dropout", 0.5),
        use_dgi=cfg.get("use_dgi", False),
        use_recon=cfg.get("use_recon", False),
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 0.005),
                            weight_decay=cfg.get("wd", 5e-4))

    n_pretrain = cfg.get("pretrain_epochs", 0)
    n_struct = cfg.get("struct_epochs", 200)
    w_mod = cfg.get("w_mod", 1.0)
    w_col = cfg.get("w_col", 0.5)
    w_rec = cfg.get("w_rec", 0.1)
    w_recon = cfg.get("w_recon", 1.0)
    collapse_mode = cfg.get("collapse_mode", "balance")

    best_loss = float("inf")
    best_emb = None
    total_epochs = n_pretrain + n_struct

    for epoch in range(total_epochs):
        model.train()
        opt.zero_grad()
        S, h = model.cluster_forward(x, edge_index, curvature)
        loss = torch.tensor(0.0, device=DEVICE)

        if epoch < n_pretrain:
            if cfg.get("use_recon", False):
                mask = torch.bernoulli(torch.full_like(x, 0.5))
                x_corrupted = x * mask
                h_corrupted = model.encode(x_corrupted, edge_index, curvature)
                x_hat = model.decoder(h_corrupted)
                loss = loss + w_recon * F.mse_loss(x_hat, x)
        else:
            loss_mod = _modularity_loss(S, edge_index, n_nodes)
            loss_col = _collapse_loss_v2(S, mode=collapse_mode)
            loss_rec = _curvature_recon_loss(S, curvature, edge_index)
            loss = (w_mod * loss_mod + w_col * loss_col + w_rec * loss_rec)

        loss.backward()
        opt.step()

        # K-means init for cluster head right after pretraining
        if epoch == n_pretrain - 1 and cfg.get("kmeans_init", False) and n_pretrain > 0:
            with torch.no_grad():
                h_init = model.encode(x, edge_index, curvature).detach().cpu().numpy()
            km = KMeans(n_clusters=n_classes, random_state=0, n_init=10)
            km.fit(h_init)
            with torch.no_grad():
                model.cluster_head.weight.data = torch.tensor(
                    km.cluster_centers_, dtype=torch.float32, device=DEVICE,
                )

        if epoch >= n_pretrain:
            with torch.no_grad():
                S_new, h_new = model.cluster_forward(x, edge_index, curvature)
                cur_loss = (w_mod * _modularity_loss(S_new, edge_index, n_nodes)
                            + w_col * _collapse_loss_v2(S_new, mode=collapse_mode)).item()
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_emb = h_new.detach().cpu().numpy()

    if best_emb is None:
        # All epochs were pretrain; do one struct pass.
        with torch.no_grad():
            _, h_new = model.cluster_forward(x, edge_index, curvature)
            best_emb = h_new.detach().cpu().numpy()
    return best_emb


def train_cgsd_pure_from_data(
    adj,
    features: np.ndarray,
    n_classes: int,
    cfg: dict,
    seed: int = 0,
) -> np.ndarray:
    """Public wrapper for in-memory training. Returns (n, hidden*n_heads) embedding."""
    return _train_cgsd_pure_inner(adj, features, n_classes, cfg, seed)


def train_cgsd_pure(name: str, cfg: dict, seed: int = 0) -> np.ndarray:
    """Train one CGSD-pure run on a named dataset. Returns final embedding."""
    from cgsd.data import load_dataset
    adj, features, labels, n_nodes, n_classes = load_dataset(name)
    return _train_cgsd_pure_inner(adj, features, n_classes, cfg, seed)
