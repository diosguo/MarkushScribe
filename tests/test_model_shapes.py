"""Shape and mask-propagation tests for the NODE AR model."""

from __future__ import annotations

import pytest
import torch

from markushscribe import constants
from markushscribe.model import ModelConfig, NodeARModel


def _model(vocab, **overrides):
    config = ModelConfig(
        dim=32,
        nhead=4,
        ar_layers=1,
        label_layers=1,
        dropout=0.0,
        **overrides,
    )
    return NodeARModel(len(vocab), config, pad_id=vocab.pad_id)


@pytest.fixture()
def model(tiny_vocab):
    return _model(tiny_vocab)


def test_forward_shapes(model, tiny_batch):
    outputs = model(tiny_batch)
    batch, nodes = tiny_batch["node_mask"].shape
    labels = tiny_batch["label_tokens"].shape[2] - 1
    vocab = model.vocab_size
    assert outputs["token_logits"].shape[:2] == tiny_batch["token_mask"][:, :-1].shape
    assert outputs["token_logits"].shape[-1] == vocab
    assert outputs["kind_logits"].shape == (batch, nodes, len(constants.KIND_CLASSES))
    assert outputs["subtype_logits"].shape == (batch, nodes, len(constants.SUBTYPE_TOKENS))
    assert outputs["x_bin_logits"].shape == (batch, nodes, constants.COORD_BINS)
    assert outputs["coord_offset"].shape == (batch, nodes, 2)
    assert outputs["label_logits"].shape == (batch, nodes, labels, vocab)
    assert outputs["adjacency_logits"].shape == (batch, nodes, nodes, 2)
    assert outputs["order_logits"].shape == (batch, nodes, nodes, len(constants.ORDER_CLASSES))
    assert outputs["depiction_logits"].shape[-1] == len(constants.DEPICTED_TYPES)
    assert outputs["wedge_logits"].shape == (batch, nodes, nodes, len(constants.WEDGE_DIRECTIONS))


def test_symmetric_heads_are_symmetric(tiny_vocab, tiny_batch):
    model = _model(tiny_vocab)
    outputs = model(tiny_batch)
    for key in (
        "adjacency_logits",
        "order_logits",
        "aromatic_logits",
        "depiction_logits",
        "variable_logits",
    ):
        logits = outputs[key]
        assert torch.allclose(logits, logits.transpose(1, 2), atol=1e-5), key


def test_padded_nodes_are_zeroed(model, tiny_batch):
    outputs = model(tiny_batch)
    padded = ~tiny_batch["node_mask"]
    assert padded.any()
    assert outputs["node_features"][padded].abs().max().item() == 0.0


def test_pair_mask_excludes_self_and_padding(model, tiny_batch):
    outputs = model(tiny_batch)
    mask = outputs["pair_mask"]
    assert mask.shape == outputs["adjacency_logits"].shape[:3]
    assert not mask.diagonal(dim1=1, dim2=2).any()
    counts = tiny_batch["node_mask"].sum(dim=1)
    expected = int((counts * counts - counts).sum().item())
    assert mask.sum().item() == expected


def test_backward_populates_all_gradients(model, tiny_batch):
    outputs = model(tiny_batch)
    loss = sum(
        value.sum()
        for value in outputs.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.requires_grad
    )
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, missing
