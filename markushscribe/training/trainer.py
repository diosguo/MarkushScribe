"""Training loop for the NODE AR baseline (``M1方案.md`` §6/§8)."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import schema
from ..losses import compute_losses
from ..model import ModelConfig, NodeARModel
from ..tokenizer import Vocab, build_vocab
from ..torch_data import TargetDataset, TorchDataConfig, collate_fn
from .checkpoint import save_checkpoint
from .config import TrainConfig
from .curriculum import TeacherForcingSchedule, predicted_node_indices

EXTRA_CHARS = "0123456789+-"


def expand_shards(paths: list[str]) -> list[str]:
    """Expand any glob pattern in ``paths`` (sorted), leaving literals untouched."""
    expanded: list[str] = []
    for pattern in paths:
        matches = sorted(glob(str(pattern)))
        expanded.extend(matches or [str(pattern)])
    return expanded


def collect_vocab(shard_paths: list[str], limit: int | None = None) -> Vocab:
    dataset = TargetDataset(expand_shards(shard_paths), None, TorchDataConfig(), limit=limit)
    graphs = [schema.parse_target(payload) for payload in dataset.iter_targets()]
    if not graphs:
        raise ValueError("no targets found in the provided shards")
    return build_vocab(graphs, extra_chars=EXTRA_CHARS)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


class Trainer:
    def __init__(
        self,
        model: NodeARModel,
        vocab: Vocab,
        model_config: ModelConfig,
        train_config: TrainConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.model_config = model_config
        self.train_config = train_config
        self.device = torch.device(device or train_config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        self.history: list[dict[str, Any]] = []
        self.node_start_id = vocab.to_id("<node>")

    # -- data -------------------------------------------------------------
    def _data_config(self) -> TorchDataConfig:
        config = self.train_config
        return TorchDataConfig(
            image_size=config.image_size,
            q_capacity=config.q_capacity,
            t_capacity=config.t_capacity,
            l_capacity=config.l_capacity,
        )

    def _loader(self, shards: list[str], *, shuffle: bool, limit: int | None) -> DataLoader:
        dataset = TargetDataset(expand_shards(shards), self.vocab, self._data_config(), limit=limit)
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=shuffle,
            num_workers=self.train_config.num_workers,
            collate_fn=partial(collate_fn, pad_id=self.vocab.pad_id),
        )

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            elif key == "node_field_mask":
                batch[key] = {k: v.to(self.device) for k, v in value.items()}
        return batch

    def _filter_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        flags = batch["capacity_exceeded"]
        if not bool(flags.any()):
            return batch
        keep = (~flags).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            return None
        filtered: dict[str, Any] = {}
        for key, value in batch.items():
            if key == "node_field_mask":
                filtered[key] = {k: v[keep] for k, v in value.items()}
            elif key == "meta":
                filtered[key] = [value[index] for index in keep.tolist()]
            elif (
                isinstance(value, torch.Tensor)
                and value.dim() >= 1
                and value.shape[0] == flags.shape[0]
            ):
                filtered[key] = value[keep]
            else:
                filtered[key] = value
        return filtered

    # -- forward ----------------------------------------------------------
    def _forward(self, batch: dict[str, Any], *, use_predicted_nodes: bool):
        if not use_predicted_nodes:
            return self.model(batch)
        with torch.no_grad():
            preliminary = self.model(batch)
        predicted = preliminary["token_logits"].argmax(dim=-1)
        full = torch.cat([batch["tokens"][:, :1], predicted], dim=1)
        indices = predicted_node_indices(full, self.node_start_id)
        return self.model(batch, node_token_index=indices)

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        schedule = TeacherForcingSchedule(
            start_epoch=0,
            end_epoch=max(1, self.train_config.teacher_forcing_epochs),
        )
        probability = schedule.probability(epoch)
        totals: dict[str, float] = {}
        steps = 0
        for raw_batch in loader:
            batch = self._filter_batch(self._move(raw_batch))
            if batch is None:
                continue
            use_predicted = probability > 0 and random.random() < probability
            outputs = self._forward(batch, use_predicted_nodes=use_predicted)
            total, parts = compute_losses(outputs, batch, self.train_config.loss_weights)
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
            self.optimizer.step()
            totals["total"] = totals.get("total", 0.0) + float(total.detach())
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            steps += 1
        return {name: value / max(steps, 1) for name, value in totals.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        steps = 0
        for raw_batch in loader:
            batch = self._filter_batch(self._move(raw_batch))
            if batch is None:
                continue
            outputs = self.model(batch)
            total, parts = compute_losses(outputs, batch, self.train_config.loss_weights)
            totals["total"] = totals.get("total", 0.0) + float(total)
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            steps += 1
        return {name: value / max(steps, 1) for name, value in totals.items()}

    def fit(self, epochs: int | None = None) -> list[dict[str, Any]]:
        config = self.train_config
        epochs = epochs or config.epochs
        train_loader = self._loader(config.train_shards, shuffle=True, limit=config.limit)
        valid_loader = (
            self._loader(config.valid_shards, shuffle=False, limit=config.limit)
            if config.valid_shards
            else None
        )
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for epoch in range(epochs):
            start = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            record: dict[str, Any] = {"epoch": epoch, "seconds": time.time() - start}
            record["train"] = train_metrics
            if valid_loader is not None:
                record["valid"] = self.validate(valid_loader)
            self.history.append(record)
            print(json.dumps(record, ensure_ascii=False))
            if (epoch + 1) % max(config.checkpoint_every, 1) == 0:
                save_checkpoint(
                    output / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    vocab=self.vocab,
                    model_config=self.model_config,
                    train_config=config,
                    history=self.history,
                )
        (output / "history.json").write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.history

    def overfit(
        self,
        *,
        steps: int = 300,
        tolerance: float = 1e-2,
        limit: int | None = None,
    ) -> list[float]:
        """Repeatedly train on one batch until the loss collapses (debug/acceptance)."""
        loader = self._loader(
            self.train_config.train_shards,
            shuffle=False,
            limit=limit or self.train_config.batch_size,
        )
        batch = next(iter(loader))
        batch = self._filter_batch(self._move(batch))
        if batch is None:
            raise ValueError("the first batch was empty after capacity filtering")
        self.model.train()
        losses: list[float] = []
        for _step in range(steps):
            outputs = self.model(batch)
            total, _parts = compute_losses(outputs, batch, self.train_config.loss_weights)
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
            self.optimizer.step()
            losses.append(float(total.detach()))
            if losses[-1] <= tolerance:
                break
        return losses


def dump_config(model_config: ModelConfig, train_config: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"model": asdict(model_config), "train": asdict(train_config)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = ["Trainer", "collect_vocab", "dump_config", "seed_everything"]
