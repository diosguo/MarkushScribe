"""Aggregate the M1 loss terms into one weighted objective (``M1方案.md`` §7/§12)."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .edge import edge_losses
from .label import label_loss
from .node import node_losses

DEFAULT_WEIGHTS: dict[str, float] = {
    "token": 1.0,
    "kind": 1.0,
    "subtype": 1.0,
    "visible": 1.0,
    "x_bin": 1.0,
    "y_bin": 1.0,
    "offset": 1.0,
    "coord_l1": 1.0,
    "charge": 1.0,
    "h_count": 1.0,
    "isotope": 1.0,
    "placeholder_shape": 1.0,
    "label": 1.0,
    "adjacency": 1.0,
    "order": 1.0,
    "aromatic": 1.0,
    "depiction": 1.0,
    "variable": 1.0,
    "wedge": 1.0,
    "wedge_sym": 0.5,
}


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``(total, parts)``; ``parts`` holds the unweighted per-term values."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    parts = node_losses(outputs, batch)
    parts["label"] = label_loss(outputs, batch)
    parts.update(edge_losses(outputs, batch))
    total = outputs["kind_logits"].sum() * 0.0
    for name, value in parts.items():
        total = total + weights.get(name, 1.0) * value
    return total, parts


__all__ = ["DEFAULT_WEIGHTS", "compute_losses"]
