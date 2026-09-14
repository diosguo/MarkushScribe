#!/usr/bin/env python3
"""Overlay prediction targets onto their images for manual QA.

Example::

    python tools/inspect_dataset.py data/rendered --out data/inspect \
        --split train --per-shard 6 --cols 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markushscribe.visualization import inspect_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", help="dataset directory with *.tar shards")
    parser.add_argument("--out", required=True, help="output directory for sheets")
    parser.add_argument("--split", default=None, help="restrict to one split subdirectory")
    parser.add_argument("--limit", type=int, default=None, help="max shards to inspect")
    parser.add_argument("--per-shard", type=int, default=6, help="samples per shard")
    parser.add_argument("--cols", type=int, default=3, help="contact sheet columns")
    parser.add_argument(
        "--save-overlays", action="store_true", help="also write full-size overlays"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    summary = inspect_dataset(
        args.shard_dir,
        args.out,
        split=args.split,
        limit=args.limit,
        per_shard=args.per_shard,
        cols=args.cols,
        save_overlays=args.save_overlays,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
