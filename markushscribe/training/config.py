"""YAML configuration loading for training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..model import ModelConfig

_DEFAULT_WEIGHTS = {
    "wedge_sym": 0.5,
}


@dataclass
class TrainConfig:
    train_shards: list[str] = field(default_factory=list)
    valid_shards: list[str] = field(default_factory=list)
    image_size: int = 384
    q_capacity: int = 128
    t_capacity: int = 32
    l_capacity: int = 2048
    batch_size: int = 4
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 5.0
    seed: int = 0
    num_workers: int = 0
    limit: int | None = None
    stage: int = 1
    freeze_encoder: bool = False
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    teacher_forcing_epochs: int = 0
    output_dir: str = "runs/node_ar"
    log_every: int = 10
    checkpoint_every: int = 1
    device: str = "cpu"


def _model_config(payload: dict[str, Any]) -> ModelConfig:
    allowed = {
        "encoder_name",
        "out_indices",
        "dim",
        "nhead",
        "ar_layers",
        "label_layers",
        "dropout",
        "max_len",
        "label_max_len",
        "roi",
        "roi_radius",
        "pretrained",
        "freeze_encoder",
        "coord_bins",
        "bond_len_est",
    }
    values = {key: value for key, value in payload.items() if key in allowed}
    if "out_indices" in values and values["out_indices"] is not None:
        values["out_indices"] = tuple(int(index) for index in values["out_indices"])
    return ModelConfig(**values)


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    model = _model_config(dict(payload.get("model") or {}))
    train_payload = dict(payload.get("train") or {})
    data_payload = dict(payload.get("data") or {})
    weights = {**_DEFAULT_WEIGHTS, **(payload.get("loss_weights") or {})}
    explicit = {
        "train_shards",
        "valid_shards",
        "image_size",
        "q_capacity",
        "t_capacity",
        "loss_weights",
    }
    extra = {
        key: value
        for key, value in train_payload.items()
        if key in TrainConfig.__dataclass_fields__ and key not in explicit
    }
    train = TrainConfig(
        train_shards=list(data_payload.get("train_shards") or []),
        valid_shards=list(data_payload.get("valid_shards") or []),
        image_size=int(data_payload.get("image_size", 384)),
        q_capacity=int(data_payload.get("q_capacity", 128)),
        t_capacity=int(data_payload.get("t_capacity", 32)),
        loss_weights={key: float(value) for key, value in weights.items()},
        **extra,
    )
    return model, train


__all__ = ["TrainConfig", "load_config"]
