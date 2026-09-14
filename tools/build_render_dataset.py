#!/usr/bin/env python3
"""Build a rendered Markush dataset from generated IR shards.

Example::

    python tools/build_render_dataset.py \
        --ir data/markush_ir/shard_00000.jsonl \
        --out data/rendered --preset-choices patent_bw acs chemdraw \
        --shard-size 500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markushscribe.dataset import RenderConfig, build_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", nargs="+", required=True, help="IR .jsonl/.json files")
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--base", default="patent_bw", help="base style preset")
    parser.add_argument(
        "--preset-choices",
        nargs="*",
        default=[],
        help="style presets to sample across (empty keeps the base family)",
    )
    parser.add_argument(
        "--split-key", default="auto", help="auto|patent_family|canonical|scaffold|source"
    )
    parser.add_argument("--train", type=float, default=0.98)
    parser.add_argument("--valid", type=float, default=0.01)
    parser.add_argument("--test", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="stop after N records")
    parser.add_argument(
        "--no-degrade", action="store_true", help="skip pixel/geometry augmentation"
    )
    parser.add_argument("--no-qa", action="store_true", help="skip per-variant overlap checks")
    parser.add_argument(
        "--drop-overlaps", action="store_true", help="skip variants whose labels overlap"
    )
    parser.add_argument(
        "--min-ink",
        type=float,
        default=0.001,
        help="drop near-blank renders below this ink fraction",
    )
    parser.add_argument(
        "--no-style", action="store_true", help="disable intra-family style sampling"
    )
    parser.add_argument("--no-lowres", action="store_true", help="disable low-resolution raster")
    parser.add_argument("--no-rotate", action="store_true", help="disable layout rotation")
    parser.add_argument("--no-mirror", action="store_true", help="disable mirroring")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = RenderConfig(
        base=args.base,
        preset_choices=tuple(args.preset_choices),
        style=not args.no_style,
        lowres=not args.no_lowres,
        rotate=not args.no_rotate,
        mirror=not args.no_mirror,
    )
    summary = build_dataset(
        args.ir,
        args.out,
        config=config,
        degrade=not args.no_degrade,
        split_key=args.split_key,
        ratios=(args.train, args.valid, args.test),
        seed=args.seed,
        shard_size=args.shard_size,
        limit=args.limit,
        qa=not args.no_qa,
        drop_overlaps=args.drop_overlaps,
        min_ink_fraction=args.min_ink,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
