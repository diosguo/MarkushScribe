"""Per-node closed-set heads (``M1方案.md`` §6.3).

These sit on the gathered NODE AR embedding and predict the node class, variable
subtype, visibility and quantised coordinates. Coordinates are predicted two
ways -- as 64-way bins (shared with the token stream) and as a continuous
in-bin offset regression -- so the model can be decoded either way.
"""

from __future__ import annotations

import torch
from torch import nn

from .. import constants


class NodeHeads(nn.Module):
    def __init__(self, dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden = dim * 2
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.GELU(),
        )
        self.kind = nn.Linear(dim, len(constants.KIND_CLASSES))
        self.subtype = nn.Linear(dim, len(constants.SUBTYPE_TOKENS))
        self.visible = nn.Linear(dim, 2)
        self.x_bin = nn.Linear(dim, constants.COORD_BINS)
        self.y_bin = nn.Linear(dim, constants.COORD_BINS)
        self.offset = nn.Linear(dim, 2)
        self.charge = nn.Linear(dim, len(constants.CHARGE_CLASSES))
        self.h_count = nn.Linear(dim, len(constants.H_CLASSES))
        self.isotope = nn.Linear(dim, 2)
        self.placeholder_shape = nn.Linear(dim, len(constants.PLACEHOLDER_SHAPES) + 1)

    def forward(self, node_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(self.norm(node_hidden))
        return {
            "kind_logits": self.kind(features),
            "subtype_logits": self.subtype(features),
            "visible_logits": self.visible(features),
            "x_bin_logits": self.x_bin(features),
            "y_bin_logits": self.y_bin(features),
            "coord_offset": torch.tanh(self.offset(features)) * 0.5,
            "charge_logits": self.charge(features),
            "h_logits": self.h_count(features),
            "isotope_logits": self.isotope(features),
            "placeholder_shape_logits": self.placeholder_shape(features),
        }


__all__ = ["NodeHeads"]
