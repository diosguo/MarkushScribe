#!/usr/bin/env python3
"""Overlay model predictions on rendered shard images for eyeballing.

Example::

    python tools/inspect_predictions.py --ckpt ckpts/last.pt \\
        --shard data/rendered/valid/shard_00000.tar --out /tmp/pred --limit 8
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from markushscribe import schema  # noqa: E402
from markushscribe.model import ModelConfig, NodeARModel  # noqa: E402
from markushscribe.predict import predict_graph  # noqa: E402
from markushscribe.tokenizer import Vocab  # noqa: E402
from markushscribe.torch_data import TorchDataConfig  # noqa: E402
from markushscribe.training.checkpoint import load_checkpoint  # noqa: E402
from markushscribe.visualization.overlay import draw_overlay  # noqa: E402


def _iter_samples(path: Path):
    with tarfile.open(path) as archive:
        names = set(archive.getnames())
        stems = sorted({name.rsplit(".", 1)[0] for name in names if name.endswith(".json")})
        for stem in stems:
            png = archive.extractfile(f"{stem}.png")
            js = archive.extractfile(f"{stem}.json")
            if png is None or js is None:
                continue
            with Image.open(io.BytesIO(png.read())) as handle:
                yield stem, handle.convert("RGB"), json.loads(js.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    payload = load_checkpoint(args.ckpt, map_location="cpu")
    vocab = Vocab(payload["vocab"])
    model_config = ModelConfig(**payload["model_config"])
    train_config = payload["train_config"]
    model = NodeARModel(len(vocab), model_config, pad_id=vocab.pad_id)
    load_checkpoint(args.ckpt, model=model, map_location="cpu")
    model.eval()
    config = TorchDataConfig(
        image_size=int(train_config.get("image_size", 384)),
        q_capacity=int(train_config.get("q_capacity", 128)),
        t_capacity=int(train_config.get("t_capacity", 32)),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for stem, image, gold_payload in _iter_samples(Path(args.shard)):
        if count >= args.limit:
            break
        graph, _tokens = predict_graph(model, image, vocab, config, threshold=args.threshold)
        gold = schema.to_dict(schema.parse_target(gold_payload))
        pred = schema.to_dict(graph)
        gold_overlay = draw_overlay(image, gold)
        pred_overlay = draw_overlay(image, pred)
        side = Image.new("RGB", (image.width * 2 + 8, image.height), (255, 255, 255))
        side.paste(gold_overlay, (0, 0))
        side.paste(pred_overlay, (image.width + 8, 0))
        side.save(out / f"{stem}_compare.png")
        count += 1
    print(json.dumps({"written": count, "out": str(out)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
