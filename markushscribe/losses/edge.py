"""Edge head losses plus the wedge-direction symmetry regulariser (``M1方案.md`` §7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import masked_cross_entropy


def _wedge_symmetry(logits: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    if not bool(upper.any()):
        return logits.sum() * 0.0
    forward = logits[upper]
    backward = logits.transpose(1, 2)[upper]
    fake_target = F.softmax(forward, dim=-1).detach()
    return -(fake_target * F.log_softmax(backward, dim=-1)).sum(dim=-1).mean()


def edge_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict:
    parts: dict[str, torch.Tensor] = {}
    pair_mask = outputs["pair_mask"]
    adjacency = batch["edge_adjacency"]
    parts["adjacency"] = masked_cross_entropy(
        outputs["adjacency_logits"], adjacency.long(), pair_mask
    )

    edge_mask = pair_mask & (adjacency > 0.5)
    parts["order"] = masked_cross_entropy(
        outputs["order_logits"], batch["edge_order"] - 1, edge_mask
    )
    parts["aromatic"] = masked_cross_entropy(
        outputs["aromatic_logits"], batch["edge_aromatic"], edge_mask
    )
    parts["depiction"] = masked_cross_entropy(
        outputs["depiction_logits"], batch["edge_depiction"], edge_mask
    )
    parts["variable"] = masked_cross_entropy(
        outputs["variable_logits"], batch["edge_variable"], edge_mask
    )

    upper = (
        torch.triu(torch.ones_like(pair_mask[0], dtype=torch.bool), diagonal=1).unsqueeze(0)
        & edge_mask
    )
    parts["wedge"] = masked_cross_entropy(
        outputs["wedge_logits"], batch["edge_wedge_direction"], upper
    )
    parts["wedge_sym"] = _wedge_symmetry(outputs["wedge_logits"], upper)
    return parts


__all__ = ["edge_losses"]
