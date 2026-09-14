"""Inference decode/assemble smoke tests."""

from __future__ import annotations

import torch

from markushscribe import schema
from markushscribe.inference import assemble_graph, decode_nodes, greedy_decode_tokens
from markushscribe.inference.confidence import token_confidence
from markushscribe.model import ModelConfig, NodeARModel
from markushscribe.predict import predict_graph
from markushscribe.tokenizer import DecodedNode
from markushscribe.torch_data import TargetDataset, TorchDataConfig


def _model(vocab):
    config = ModelConfig(dim=32, nhead=4, ar_layers=1, label_layers=1, dropout=0.0)
    return NodeARModel(len(vocab), config, pad_id=vocab.pad_id)


def test_greedy_decode_and_assemble(tiny_vocab, tiny_batch):
    model = _model(tiny_vocab)
    image = tiny_batch["image"][:1]
    tokens = greedy_decode_tokens(model, image, tiny_vocab, max_len=48)
    assert int(tokens[0, 0]) == tiny_vocab.bos_id
    assert tokens.shape[1] >= 2
    nodes = decode_nodes(tokens, tiny_vocab)
    graph = assemble_graph(nodes)
    assert isinstance(graph, schema.MarkushGraph)
    assert len(graph.nodes) == len(nodes)
    confidence = token_confidence(model(tiny_batch)["token_logits"], tiny_batch["tokens"])
    assert all(0.0 <= value <= 1.0 for value in confidence)


def test_predict_graph_runs(tiny_shard, tiny_vocab):
    dataset = TargetDataset([tiny_shard], tiny_vocab, TorchDataConfig(image_size=64))
    image, _payload, _uid = dataset.raw(0)
    model = _model(tiny_vocab)
    graph, tokens = predict_graph(
        model, image, tiny_vocab, TorchDataConfig(image_size=64), max_len=24
    )
    assert isinstance(graph, schema.MarkushGraph)
    assert tokens.shape[0] == 1


def test_assemble_graph_accepts_edge_logits_and_aromatic():
    nodes = [
        DecodedNode(
            kind="atom", subtype=None, x_bin=i, y_bin=0, runs=[{"text": "C", "script": "base"}]
        )
        for i in range(3)
    ]
    adjacency = torch.zeros(3, 3, 2)
    adjacency[..., 0] = 10.0
    adjacency[0, 2, 0] = adjacency[2, 0, 0] = 0.0
    adjacency[0, 2, 1] = adjacency[2, 0, 1] = 10.0
    aromatic = torch.zeros(3, 3, dtype=torch.long)
    aromatic[0, 2] = aromatic[2, 0] = 1
    graph = assemble_graph(
        nodes,
        adjacency=adjacency,
        order=torch.zeros(3, 3, dtype=torch.long),
        aromatic=aromatic,
        depiction=torch.zeros(3, 3, dtype=torch.long),
        variable=torch.zeros(3, 3, dtype=torch.long),
        wedge=torch.zeros(3, 3, dtype=torch.long),
    )
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert (edge.source, edge.target) == (0, 2)
    assert edge.normalized.aromatic and edge.normalized.order == 1

    probabilities = torch.softmax(adjacency, dim=-1)[..., 1]
    assert len(assemble_graph(nodes, adjacency=probabilities).edges) == 1
