"""Dataset loaders for the five Geom-GCN benchmarks (Table 1)."""

from __future__ import annotations

import warnings

import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix

DATASETS = ("Cora", "Cornell", "Texas", "Wisconsin", "Chameleon")


def load_dataset(name: str, root: str = "/tmp/cgsd_data") -> tuple:
    """Load one of the five Table-1 datasets.

    Args:
        name: Dataset name in ``DATASETS``.
        root: Cache directory for PyG downloads.

    Returns:
        Tuple ``(adj, features, labels, n_nodes, n_classes)`` where labels
        are for evaluation only and are never used in training.
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
        raise ValueError(f"Unknown dataset: {name!r}. Supported: {DATASETS}")

    n_nodes = max(int(data.num_nodes), int(data.edge_index.max()) + 1)
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=n_nodes)
    features = data.x.numpy().astype(np.float32)
    labels = data.y.numpy()
    return adj, features, labels, n_nodes, len(np.unique(labels))
