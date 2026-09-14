#!/usr/bin/env python3
"""Train the NODE AR baseline.

Examples::

    python -m markushscribe.train --config configs/train_node_ar_debug.yaml --overfit
    python -m markushscribe.train --config configs/train_node_ar.yaml
"""

from __future__ import annotations

import argparse

from .model import NodeARModel
from .training.config import load_config
from .training.trainer import Trainer, collect_vocab, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training YAML config")
    parser.add_argument("--out", default=None, help="override output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap samples per shard")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug", action="store_true", help="tiny CPU run")
    parser.add_argument("--overfit", action="store_true", help="overfit a single batch")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_config, train_config = load_config(args.config)
    if args.out:
        train_config.output_dir = args.out
    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.device is not None:
        train_config.device = args.device
    if args.limit is not None:
        train_config.limit = args.limit
    if args.debug:
        train_config.limit = train_config.limit or 2
        train_config.epochs = 1
        train_config.batch_size = min(train_config.batch_size, 2)
        train_config.teacher_forcing_epochs = 0
    seed_everything(train_config.seed)
    vocab = collect_vocab(train_config.train_shards, limit=train_config.limit)
    model = NodeARModel(len(vocab), model_config, pad_id=vocab.pad_id)
    trainer = Trainer(model, vocab, model_config, train_config)
    if args.overfit:
        losses = trainer.overfit(steps=args.steps, tolerance=args.tolerance)
        print(f"overfit steps={len(losses)} first={losses[0]:.4f} last={losses[-1]:.4f}")
        return 0
    trainer.fit()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
