"""Checkpoint save/load including vocabulary and configuration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..model import ModelConfig
from ..tokenizer import Vocab
from .config import TrainConfig


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    vocab: Vocab,
    model_config: ModelConfig,
    train_config: TrainConfig,
    history: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "epoch": epoch,
            "vocab": vocab.tokens,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "history": history or [],
            **extra,
        },
        target,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and payload.get("model_state") is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


__all__ = ["load_checkpoint", "save_checkpoint"]
