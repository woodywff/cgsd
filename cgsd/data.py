"""5-dataset loader for the CGSD release.

Supports exactly the 5 datasets in the paper's Table 1:
    Cora, Cornell, Texas, Wisconsin, Chameleon.

The parent project (experiments/dmon_mincut.py) supports more datasets
(PubMed, Squirrel, ogbn-arxiv, etc.) but those are NOT in the canonical
Table 1 and are deliberately omitted from the release.

Returns labels only for downstream evaluation (KMeans+NMI). Training
in cgsd.train NEVER uses labels.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix

# Datasets in the canonical Table 1.
DATASETS = ("Cora", "Cornell", "Texas", "Wisconsin", "Chameleon")


def load_dataset(name: str, root: str = "/tmp/cgsd_data") -> tuple:
    """Load one of the 5 Table-1 datasets.

    Args:
        name: one of DATASETS.
        root: cache directory for PyG raw downloads.

    Returns:
        (adj, features, labels, n_nodes, n_classes) where:
            adj      : scipy.sparse.csr_matrix, shape (N, N)
            features : np.ndarray, shape (N, d), float32
            labels   : np.ndarray, shape (N,), int64 — for EVAL ONLY
            n_nodes  : int
            n_classes: int
    """
    warnings.filterwarnings("ignore")
    name_l = name.lower()
    if name_l in ("cornell", "texas", "wisconsin"):
        from torch_geometric.datasets import WebKB
        data = WebKB(root=root, name=name_l)[0]
    elif name_l == "chameleon":
        from torch_geometric.datasets import WikipediaNetwork
        data = WikipediaNetwork(root=root, name=name_l)[0]
    elif name_l == "cora":
        from torch_geometric.datasets import Planetoid
        data = Planetoid(root=root + "_planetoid", name=name_l)[0]
    else:
        raise ValueError(
            f"Unknown dataset: {name!r}. Supported: {DATASETS}"
        )
    # Some datasets (Chameleon) have edge indices that exceed data.num_nodes
    # by a few; use the actual max + 1 to be safe.
    n_nodes = max(int(data.num_nodes), int(data.edge_index.max()) + 1)
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=n_nodes)
    features = data.x.numpy().astype(np.float32)
    labels = data.y.numpy()
    return adj, features, labels, n_nodes, len(np.unique(labels))
