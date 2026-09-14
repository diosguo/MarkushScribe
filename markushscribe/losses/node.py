"""Node, coordinate and closed-set attribute losses (``M1方案.md`` §7/§12)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import constants
from .common import masked_cross_entropy, masked_l1, masked_smooth_l1


def _expected_bin(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    indices = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return (probabilities * indices).sum(dim=-1)


def node_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict:
    parts: dict[str, torch.Tensor] = {}
    parts["token"] = masked_cross_entropy(
        outputs["token_logits"], batch["tokens"][:, 1:], batch["token_mask"][:, 1:]
    )
    mask = batch["node_mask"]
    parts["kind"] = masked_cross_entropy(outputs["kind_logits"], batch["node_kind"], mask)
    parts["subtype"] = masked_cross_entropy(outputs["subtype_logits"], batch["node_subtype"], mask)
    parts["visible"] = masked_cross_entropy(outputs["visible_logits"], batch["node_visible"], mask)
    parts["x_bin"] = masked_cross_entropy(
        outputs["x_bin_logits"], batch["coord_bin"][:, :, 0], mask
    )
    parts["y_bin"] = masked_cross_entropy(
        outputs["y_bin_logits"], batch["coord_bin"][:, :, 1], mask
    )
    parts["offset"] = masked_smooth_l1(outputs["coord_offset"], batch["coord_offset"], mask)

    bins = constants.COORD_BINS - 1
    predicted_x = (_expected_bin(outputs["x_bin_logits"]) + outputs["coord_offset"][..., 0]) / bins
    predicted_y = (_expected_bin(outputs["y_bin_logits"]) + outputs["coord_offset"][..., 1]) / bins
    target_x = (batch["coord_bin"][:, :, 0].float() + batch["coord_offset"][..., 0]) / bins
    target_y = (batch["coord_bin"][:, :, 1].float() + batch["coord_offset"][..., 1]) / bins
    parts["coord_l1"] = masked_l1(
        torch.stack([predicted_x, predicted_y], dim=-1),
        torch.stack([target_x, target_y], dim=-1),
        mask,
    )

    field_mask = batch["node_field_mask"]
    parts["charge"] = masked_cross_entropy(
        outputs["charge_logits"], batch["node_charge"], field_mask["charge"]
    )
    parts["h_count"] = masked_cross_entropy(
        outputs["h_logits"], batch["node_h_count"], field_mask["h_count"]
    )
    parts["isotope"] = masked_cross_entropy(
        outputs["isotope_logits"], batch["node_isotope"], field_mask["isotope"]
    )
    parts["placeholder_shape"] = masked_cross_entropy(
        outputs["placeholder_shape_logits"],
        batch["node_placeholder_shape"],
        field_mask["placeholder_shape"],
    )
    return parts


__all__ = ["node_losses"]
