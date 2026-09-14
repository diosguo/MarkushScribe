"""Visual-label token loss (``M1方案.md`` §7)."""

from __future__ import annotations

import torch

from .common import masked_cross_entropy


def label_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return masked_cross_entropy(
        outputs["label_logits"],
        batch["label_tokens"][:, :, 1:],
        batch["label_mask"][:, :, 1:],
    )


__all__ = ["label_loss"]
