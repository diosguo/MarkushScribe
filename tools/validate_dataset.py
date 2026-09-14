#!/usr/bin/env python3
"""Validate every prediction target in a rendered shard tree.

Example::

    python tools/validate_dataset.py data/rendered --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markushscribe.dataset import validate_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", help="dataset directory with *.tar shards")
    parser.add_argument("--strict", action="store_true", help="exit 1 on invalid/unpaired samples")
    parser.add_argument("--errors", type=int, default=20, help="max errors to print")
    parser.add_argument("--min-ink", type=float, default=0.001, help="blank-image ink threshold")
    parser.add_argument("--no-ink", action="store_true", help="skip blank-image detection")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    stats = validate_dataset(args.shard_dir, min_ink_fraction=0.0 if args.no_ink else args.min_ink)
    errors = stats["errors"]
    printable = {**stats, "errors": errors[: args.errors]}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if args.strict and (stats["invalid"] or stats["unpaired"] or stats["blank"]):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
