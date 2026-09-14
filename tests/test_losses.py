"""Loss finiteness, masking and gradient tests."""

from __future__ import annotations

import torch

from markushscribe.losses import DEFAULT_WEIGHTS, compute_losses
from markushscribe.model import ModelConfig, NodeARModel


def _model(vocab):
    config = ModelConfig(dim=32, nhead=4, ar_layers=1, label_layers=1, dropout=0.0)
    return NodeARModel(len(vocab), config, pad_id=vocab.pad_id)


def test_all_losses_finite_and_backward(tiny_vocab, tiny_batch):
    model = _model(tiny_vocab)
    outputs = model(tiny_batch)
    total, parts = compute_losses(outputs, tiny_batch)
    assert torch.isfinite(total)
    assert set(parts) == set(DEFAULT_WEIGHTS)
    for name, value in parts.items():
        assert torch.isfinite(value), name
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_fully_masked_edges_do_not_nan(tiny_vocab, tiny_batch):
    model = _model(tiny_vocab)
    batch = dict(tiny_batch)
    for key in ("edge_adjacency", "edge_mask"):
        batch[key] = torch.zeros_like(batch[key])
    for key in (
        "edge_order",
        "edge_aromatic",
        "edge_depiction",
        "edge_variable",
        "edge_wedge_direction",
    ):
        batch[key] = torch.zeros_like(batch[key])
    batch["node_field_mask"] = {
        key: torch.zeros_like(value) for key, value in batch["node_field_mask"].items()
    }
    outputs = model(batch)
    total, parts = compute_losses(outputs, batch)
    assert torch.isfinite(total)
    for name, value in parts.items():
        assert torch.isfinite(value), name


def test_weights_scale_total(tiny_vocab, tiny_batch):
    model = _model(tiny_vocab)
    outputs = model(tiny_batch)
    base, _ = compute_losses(outputs, tiny_batch)
    doubled, _ = compute_losses(
        outputs, tiny_batch, {key: value * 2 for key, value in DEFAULT_WEIGHTS.items()}
    )
    assert torch.isclose(doubled, base * 2, atol=1e-4)
