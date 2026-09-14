#!/usr/bin/env python3
"""Check that no group (canonical/scaffold/family/source) spans two splits.

Example::

    python tools/check_split_leakage.py \
        --split train=data/ir/train.jsonl --split valid=data/ir/valid.jsonl \
        --split test=data/ir/test.jsonl --key scaffold
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markushscribe.dataset import read_jsonl  # noqa: E402
from markushscribe.split import SPLITS, group_key  # noqa: E402

_KEYS = ("auto", "patent_family", "canonical", "scaffold", "source")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="one or more split assignments, e.g. --split train=train.jsonl",
    )
    parser.add_argument("--key", default="auto", choices=_KEYS)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def parse_split_specs(specs: Iterable[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for spec in specs:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"expected NAME=PATH, got {spec!r}")
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
        assignments[name] = path
    return assignments


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    assignments = parse_split_specs(args.split)
    seen: dict[str, set[str]] = {}
    for name, path in assignments.items():
        for record in read_jsonl(path):
            seen.setdefault(group_key(record, args.key), set()).add(name)
    leaks = {group: sorted(splits) for group, splits in seen.items() if len(splits) > 1}
    report = {
        "key": args.key,
        "splits": {name: path for name, path in assignments.items()},
        "groups": len(seen),
        "leaks": len(leaks),
        "examples": {group: leaks[group] for group in sorted(leaks)[: args.max_examples]},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if leaks else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
