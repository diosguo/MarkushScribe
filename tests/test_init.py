"""Tests for the pretrained-weight loader."""

from __future__ import annotations

import pytest
import torch

from markushscribe.model import (
    ModelConfig,
    NodeARModel,
    assert_loaded,
    load_molscribe_encoder,
    load_weights,
    strip_prefixes,
)


def _model(vocab):
    config = ModelConfig(dim=32, nhead=4, ar_layers=1, label_layers=1, dropout=0.0)
    return NodeARModel(len(vocab), config, pad_id=vocab.pad_id)


def test_strip_prefixes():
    assert strip_prefixes("module.encoder.backbone.conv1.weight") == "backbone.conv1.weight"
    assert strip_prefixes("node_ar.embedding.weight") == "node_ar.embedding.weight"


def test_load_all_matching_weights(tiny_vocab, tmp_path):
    source = _model(tiny_vocab).state_dict()
    prefixed = {f"module.encoder.{key}": value for key, value in source.items()}
    path = tmp_path / "molscribe.pt"
    torch.save({"state_dict": prefixed}, path)
    target = _model(tiny_vocab)
    report = load_molscribe_encoder(target, path)
    assert report.loaded_fraction == 1.0


def test_missing_and_unexpected_reported(tiny_vocab):
    source = _model(tiny_vocab).state_dict()
    dropped = next(iter(source))
    partial = {key: value for key, value in source.items() if key != dropped}
    partial["totally.unrelated.weight"] = torch.zeros(3)
    report = load_weights(_model(tiny_vocab), partial)
    assert dropped in report.missing
    assert "totally.unrelated.weight" in report.unexpected
    assert 0.0 < report.loaded_fraction < 1.0


def test_assert_loaded_threshold():
    from markushscribe.model import LoadReport

    assert_loaded(LoadReport(loaded=["a"] * 10, missing=[]))
    with pytest.raises(ValueError):
        assert_loaded(LoadReport(loaded=[], missing=["a"] * 10), minimum=0.5)
