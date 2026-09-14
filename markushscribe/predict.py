#!/usr/bin/env python3
"""Predict a Markush graph for one image with a trained NODE AR checkpoint.

Example::

    python -m markushscribe.predict --ckpt ckpts/last.pt --image assets/example.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from . import schema
from .inference import assemble_graph, greedy_decode_tokens
from .inference.decode import decode_nodes
from .model import ModelConfig, NodeARModel
from .tokenizer import Vocab
from .torch_data import TorchDataConfig, build_sample, collate_fn, normalize_image
from .training.checkpoint import load_checkpoint


def predict_graph(
    model: NodeARModel,
    image: Image.Image,
    vocab: Vocab,
    config: TorchDataConfig,
    *,
    threshold: float = 0.5,
    max_len: int | None = None,
) -> tuple[schema.MarkushGraph, torch.Tensor]:
    """Greedily decode tokens, then use the edge head for the bond graph."""
    pixel = normalize_image(image, config.image_size)
    tokens = greedy_decode_tokens(model, pixel.unsqueeze(0), vocab, max_len=max_len)
    nodes = decode_nodes(tokens[0], vocab)
    provisional = assemble_graph(nodes, image_size=image.size)
    if not nodes:
        return provisional, tokens
    sample = build_sample(provisional, image, vocab, config, uid="pred")
    batch = collate_fn([sample], pad_id=vocab.pad_id)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    graph = assemble_graph(
        nodes,
        image_size=image.size,
        adjacency=outputs["adjacency_logits"][0],
        order=outputs["order_logits"][0].argmax(dim=-1),
        aromatic=outputs["aromatic_logits"][0].argmax(dim=-1),
        depiction=outputs["depiction_logits"][0].argmax(dim=-1),
        variable=outputs["variable_logits"][0].argmax(dim=-1),
        wedge=outputs["wedge_logits"][0].argmax(dim=-1),
        threshold=threshold,
    )
    return graph, tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    payload = load_checkpoint(args.ckpt, map_location=args.device)
    vocab = Vocab(payload["vocab"])
    model_config = ModelConfig(**payload["model_config"])
    train_config = payload["train_config"]
    model = NodeARModel(len(vocab), model_config, pad_id=vocab.pad_id)
    load_checkpoint(args.ckpt, model=model, map_location=args.device)
    model.to(args.device).eval()

    with Image.open(args.image) as handle:
        image = handle.convert("RGB")
    config = TorchDataConfig(
        image_size=int(train_config.get("image_size", 384)),
        q_capacity=int(train_config.get("q_capacity", 128)),
        t_capacity=int(train_config.get("t_capacity", 32)),
    )
    graph, _tokens = predict_graph(model, image, vocab, config, threshold=args.threshold)
    result = schema.to_dict(graph)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
