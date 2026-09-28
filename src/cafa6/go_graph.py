"""Pilot: learned GAT over the GO DAG (vocab-restricted), refining hierarchical GO-term queries.

Literature: BioReason-Pro's GO-graph encoder (GAT over go-basic.obo) and POSA-GO/TRGOA's
partial-order / topological attention over the ontology motivate a learned, message-passing
representation of GO structure instead of a fixed ancestor-mean operator alone.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GOGraphEncoder(nn.Module):
    """Small GAT stack over the vocab-restricted GO DAG (child<->parent, undirected + self-loops).

    Refines term query embeddings q (T x d) -> q' (T x d) via message passing, additively on top
    of the existing fixed ancestor-mean operator (see HiGO.queries()).
    """

    def __init__(self, n_terms: int, d: int, edges: np.ndarray, n_layers: int = 2, heads: int = 4):
        super().__init__()
        assert d % heads == 0, f"d={d} must be divisible by heads={heads} for GATConv(concat=True)"
        child, parent = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
        self_loops = np.arange(n_terms, dtype=np.int64)
        src = np.concatenate([child, parent, self_loops])
        dst = np.concatenate([parent, child, self_loops])
        edge_index = torch.as_tensor(np.stack([src, dst]), dtype=torch.long)
        self.register_buffer("edge_index", edge_index)
        self.convs = nn.ModuleList(
            [GATConv(d, d // heads, heads=heads, concat=True, add_self_loops=False) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        x = q
        for conv, norm in zip(self.convs, self.norms):
            x = F.gelu(norm(conv(x, self.edge_index) + x))
        return x
