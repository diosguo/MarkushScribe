"""Simple confidence estimates for decoded tokens and edges."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_confidence(token_logits: torch.Tensor, token_ids: torch.Tensor) -> list[float]:
    """Per-node confidence: the probability assigned to the chosen next token."""
    if token_logits.dim() == 3:
        token_logits = token_logits[0]
    if token_ids.dim() == 2:
        token_ids = token_ids[0]
    targets = token_ids[1:].reshape(-1)
    logits = token_logits.reshape(-1, token_logits.shape[-1])
    probabilities = F.softmax(logits, dim=-1)
    chosen = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return [float(value) for value in chosen.tolist()]


def edge_confidence(adjacency_logits: torch.Tensor) -> torch.Tensor:
    if adjacency_logits.dim() == 4:
        adjacency_logits = adjacency_logits[0]
    return F.softmax(adjacency_logits, dim=-1)[..., 1]


__all__ = ["edge_confidence", "token_confidence"]
