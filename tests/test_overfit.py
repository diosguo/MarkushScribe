"""Single-batch overfit acceptance test (slow)."""

from __future__ import annotations

import pytest
import torch

from markushscribe.model import ModelConfig, NodeARModel
from markushscribe.training.config import TrainConfig
from markushscribe.training.trainer import Trainer, collect_vocab, seed_everything


@pytest.mark.slow
def test_single_batch_overfits(tiny_shard):
    seed_everything(0)
    shards = [str(tiny_shard)]
    vocab = collect_vocab(shards, limit=2)
    config = ModelConfig(dim=32, nhead=4, ar_layers=1, label_layers=1, dropout=0.0)
    model = NodeARModel(len(vocab), config, pad_id=vocab.pad_id)
    train = TrainConfig(
        train_shards=shards,
        image_size=64,
        q_capacity=32,
        t_capacity=16,
        batch_size=2,
        lr=3e-3,
        weight_decay=0.0,
        limit=2,
        device="cpu",
    )
    trainer = Trainer(model, vocab, config, train, device="cpu")
    losses = trainer.overfit(steps=250, tolerance=1e-3, limit=2)
    assert losses[-1] < losses[0], losses[:3] + losses[-3:]
    assert losses[-1] < losses[0] * 0.5, (
        f"did not converge enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
    assert torch.isfinite(torch.tensor(losses[-1]))
