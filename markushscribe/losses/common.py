"""Masked loss primitives shared by every M1 loss term."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean cross entropy over the masked positions (0 with correct grad if empty)."""
    mask = mask.bool()
    classes = logits.shape[-1]
    flat_logits = logits.reshape(-1, classes)
    flat_target = target.reshape(-1).long()
    flat_mask = mask.reshape(-1)
    if flat_mask.numel() == 0 or not bool(flat_mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(flat_logits[flat_mask], flat_target[flat_mask])


def masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.bool()
    if mask.numel() == 0 or not bool(mask.any()):
        return prediction.sum() * 0.0
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    expanded = mask
    while expanded.dim() < loss.dim():
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(loss)
    return loss[expanded].mean()


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.numel() == 0 or not bool(mask.any()):
        return prediction.sum() * 0.0
    loss = (prediction - target).abs()
    expanded = mask
    while expanded.dim() < loss.dim():
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(loss)
    return loss[expanded].mean()


__all__ = ["masked_cross_entropy", "masked_l1", "masked_smooth_l1"]
