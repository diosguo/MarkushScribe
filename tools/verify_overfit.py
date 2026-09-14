"""Overfit-convergence check on a small real shard (CPU-friendly).

Usage:
    python tools/verify_overfit.py --limit 10 --steps 800 --tolerance 1e-2 --eval
"""

from __future__ import annotations

import argparse
import glob
import time

from markushscribe import schema
from markushscribe.evaluation import evaluate_graph
from markushscribe.model import ModelConfig, NodeARModel
from markushscribe.predict import predict_graph
from markushscribe.torch_data import TargetDataset, TorchDataConfig
from markushscribe.training.config import TrainConfig
from markushscribe.training.trainer import Trainer, collect_vocab, seed_everything


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default="data/rendered/train/shard_*.tar")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args(argv)

    shards = sorted(glob.glob(args.shard))[:1]
    if not shards:
        raise SystemExit(f"no shard matched: {args.shard}")
    seed_everything(0)
    vocab = collect_vocab(shards, limit=args.limit)
    model_config = ModelConfig(
        encoder_name="resnet18",
        out_indices=(3, 4),
        dim=args.dim,
        nhead=4,
        ar_layers=2,
        label_layers=2,
        dropout=0.0,
    )
    model = NodeARModel(len(vocab), model_config, pad_id=vocab.pad_id)
    train_config = TrainConfig(
        train_shards=shards,
        image_size=args.image_size,
        q_capacity=64,
        t_capacity=24,
        l_capacity=1024,
        batch_size=args.limit,
        lr=args.lr,
        weight_decay=0.0,
        limit=args.limit,
        device="cpu",
    )
    data_config = TorchDataConfig(image_size=args.image_size, q_capacity=64, t_capacity=24)
    dataset = TargetDataset(shards, vocab, data_config, limit=args.limit)
    print(f"samples in dataset: {len(dataset)} (requested {args.limit})")

    trainer = Trainer(model, vocab, model_config, train_config, device="cpu")
    start = time.time()
    losses = trainer.overfit(steps=args.steps, tolerance=args.tolerance, limit=args.limit)
    elapsed = time.time() - start
    for index, loss in enumerate(losses):
        if index % args.every == 0 or index == len(losses) - 1:
            print(f"step {index:5d}  loss={loss:.4f}")
    print(f"overfit: {losses[0]:.4f} -> {losses[-1]:.4f} (steps={len(losses)}, {elapsed:.1f}s)")

    if args.eval:
        exact = 0
        for index in range(len(dataset)):
            image, payload, uid = dataset.raw(index)
            gold = schema.parse_target(payload)
            pred, _ = predict_graph(model, image, vocab, data_config)
            result = evaluate_graph(gold, pred)
            exact += int(result["graph_exact"])
            print(
                f"{uid}: gold={len(gold.nodes):2d} pred={len(pred.nodes):2d} "
                f"node_f1={result['node_f1']:.2f} edge_f1={result['edge_f1']:.2f} "
                f"order={result['edge_order_accuracy']:.2f} "
                f"arom={result['edge_aromatic_accuracy']:.2f} "
                f"dep={result['edge_depiction_accuracy']:.2f} "
                f"exact={result['graph_exact']:.0f}"
            )
        print(f"graph_exact: {exact}/{len(dataset)}")


if __name__ == "__main__":
    main()
