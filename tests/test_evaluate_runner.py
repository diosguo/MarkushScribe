"""Evaluation runner smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from markushscribe.evaluate import evaluate_shards, main
from markushscribe.model import ModelConfig, NodeARModel
from markushscribe.torch_data import TorchDataConfig
from markushscribe.training.checkpoint import save_checkpoint
from markushscribe.training.config import TrainConfig


def _model(vocab):
    config = ModelConfig(dim=32, nhead=4, ar_layers=1, label_layers=1, dropout=0.0)
    return NodeARModel(len(vocab), config, pad_id=vocab.pad_id)


def test_evaluate_shards_returns_summary_and_records(tiny_shard, tiny_vocab):
    config = TorchDataConfig(image_size=64, q_capacity=32, t_capacity=16)
    summary, records = evaluate_shards(
        _model(tiny_vocab), tiny_vocab, [str(tiny_shard)], config, limit=2, max_len=24
    )
    assert len(records) == 2
    assert {"node_f1", "edge_f1", "graph_exact"} <= set(summary)
    assert all(0.0 <= record["graph_exact"] <= 1.0 for record in records)
    assert all(Path(record["uid"]).name for record in records)


def test_evaluate_cli_writes_reports(tiny_shard, tiny_vocab, tmp_path):
    model_config = ModelConfig(dim=32, nhead=4, ar_layers=1, label_layers=1, dropout=0.0)
    model = NodeARModel(len(tiny_vocab), model_config, pad_id=tiny_vocab.pad_id)
    train_config = TrainConfig(image_size=64, q_capacity=32, t_capacity=16)
    ckpt = tmp_path / "last.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=None,
        epoch=0,
        vocab=tiny_vocab,
        model_config=model_config,
        train_config=train_config,
    )
    out = tmp_path / "eval"
    code = main(
        [
            "--ckpt",
            str(ckpt),
            "--shard",
            str(tiny_shard),
            "--limit",
            "2",
            "--max-len",
            "24",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["samples"] == 2
    lines = (out / "per_sample.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
