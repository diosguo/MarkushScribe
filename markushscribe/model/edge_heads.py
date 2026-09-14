"""Factorised pairwise edge head (``M1方案.md`` §6.5/§7).

Symmetric properties (adjacency, order, aromaticity, depiction, variability) are
predicted from features that are invariant under swapping the pair, so the head
is symmetric by construction. Wedge direction is the one genuinely asymmetric
property and gets its own branch.
"""

from __future__ import annotations

import torch
from torch import nn

from .. import constants


class EdgeHeads(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        *,
        hidden: int = 256,
        dropout: float = 0.1,
        bond_len_est: float = 0.12,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.bond_len_est = bond_len_est
        self.sym_mlp = nn.Sequential(
            nn.Linear(dim * 3 + 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.wedge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.adjacency = nn.Linear(hidden, 2)
        self.order = nn.Linear(hidden, len(constants.ORDER_CLASSES))
        self.aromatic = nn.Linear(hidden, 2)
        self.depiction = nn.Linear(hidden, len(constants.DEPICTED_TYPES))
        self.variable = nn.Linear(hidden, 2)
        self.wedge = nn.Linear(hidden, len(constants.WEDGE_DIRECTIONS))

    def forward(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        left = node_features.unsqueeze(2)
        right = node_features.unsqueeze(1)
        x = coords[..., 0]
        y = coords[..., 1]
        dx = x.unsqueeze(2) - x.unsqueeze(1)
        dy = y.unsqueeze(2) - y.unsqueeze(1)
        dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
        geometry = torch.stack(
            [dx.abs(), dy.abs(), dist, 1.0 / (1.0 + dist / self.bond_len_est)], dim=-1
        )
        symmetric = torch.cat([left + right, left * right, (left - right).abs(), geometry], dim=-1)
        sym_emb = self.sym_mlp(symmetric)
        nodes = node_features.shape[1]
        left_full = left.expand(-1, -1, nodes, -1)
        right_full = right.expand(-1, nodes, -1, -1)
        asymmetric = torch.cat([left_full, right_full, dx.unsqueeze(-1), dy.unsqueeze(-1)], dim=-1)
        wedge_emb = self.wedge_mlp(asymmetric)

        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        eye = torch.eye(coords.shape[1], dtype=torch.bool, device=coords.device)
        pair_mask = pair_mask & ~eye.unsqueeze(0)
        return {
            "adjacency_logits": self.adjacency(sym_emb),
            "order_logits": self.order(sym_emb),
            "aromatic_logits": self.aromatic(sym_emb),
            "depiction_logits": self.depiction(sym_emb),
            "variable_logits": self.variable(sym_emb),
            "wedge_logits": self.wedge(wedge_emb),
            "pair_mask": pair_mask,
        }


__all__ = ["EdgeHeads"]
