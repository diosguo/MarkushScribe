#!/usr/bin/env python3
"""Evaluate a trained NODE AR checkpoint on rendered shards (``M1方案.md`` §18).

Aggregates the graph-level metrics from :mod:`markushscribe.evaluation` over a
split and optionally writes per-sample records for error analysis.

Example::

    python -m markushscribe.evaluate --ckpt runs/node_ar_debug/last.pt \\
        --shard data/rendered/valid/shard_*.tar --limit 50 --out data/eval_debug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import schema
from .evaluation import evaluate_dataset, evaluate_graph
from .model import ModelConfig, NodeARModel
from .predict import predict_graph
from .tokenizer import Vocab
from .torch_data import TargetDataset, TorchDataConfig
from .training.checkpoint import load_checkpoint
from .training.trainer import expand_shards


def data_config_from_train(train_config: dict) -> TorchDataConfig:
    """Rebuild the inference tensor contract from a checkpoint's train config."""
    return TorchDataConfig(
        image_size=int(train_config.get("image_size", 384)),
        q_capacity=int(train_config.get("q_capacity", 128)),
        t_capacity=int(train_config.get("t_capacity", 32)),
    )


def evaluate_shards(
    model: NodeARModel,
    vocab: Vocab,
    shards: list[str],
    config: TorchDataConfig,
    *,
    limit: int | None = None,
    threshold: float = 0.5,
    strict_exact: bool = False,
    max_len: int | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Run prediction + evaluation over one or more shards."""
    dataset = TargetDataset(expand_shards(shards), vocab, config, limit=limit)
    decode_limit = max_len or (2 + 8 * config.q_capacity)
    samples: list[tuple[schema.MarkushGraph, schema.MarkushGraph]] = []
    records: list[dict] = []
    for index in range(len(dataset)):
        image, payload, uid = dataset.raw(index)
        gold = schema.parse_target(payload)
        pred, _tokens = predict_graph(
            model, image, vocab, config, threshold=threshold, max_len=decode_limit
        )
        report = evaluate_graph(gold, pred, strict_exact=strict_exact)
        samples.append((gold, pred))
        records.append(
            {
                "uid": uid,
                "gold_nodes": len(gold.nodes),
                "pred_nodes": len(pred.nodes),
                **{key: float(value) for key, value in report.items()},
            }
        )
    return evaluate_dataset(samples, strict_exact=strict_exact), records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--shard", nargs="+", required=True, help="rendered shard path(s)/glob(s)")
    parser.add_argument(
        "--out", default=None, help="optional dir for summary.json + per_sample.jsonl"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-len", type=int, default=None, help="greedy decode cap (default 2+8*q)"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--strict-exact",
        action="store_true",
        help="require visual label + wedge equality for exact",
    )
    args = parser.parse_args(argv)

    payload = load_checkpoint(args.ckpt, map_location=args.device)
    vocab = Vocab(payload["vocab"])
    model_config = ModelConfig(**payload["model_config"])
    model = NodeARModel(len(vocab), model_config, pad_id=vocab.pad_id)
    load_checkpoint(args.ckpt, model=model, map_location=args.device)
    model.to(args.device).eval()

    config = data_config_from_train(payload.get("train_config", {}))
    summary, records = evaluate_shards(
        model,
        vocab,
        args.shard,
        config,
        limit=args.limit,
        threshold=args.threshold,
        strict_exact=args.strict_exact,
        max_len=args.max_len,
    )
    result = {
        "ckpt": args.ckpt,
        "shards": args.shard,
        "samples": len(records),
        "strict_exact": args.strict_exact,
        "threshold": args.threshold,
        "metrics": summary,
    }
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (out / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
