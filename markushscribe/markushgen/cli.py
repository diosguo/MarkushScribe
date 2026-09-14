"""Batch generator: a ``.smi`` file -> Markush IR JSONL shards."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from .abstract import abstract_molecule
from .config import MarkushConfig
from .ir import MarkushIR
from .molecule import load_molecule

_SEED_STRIDE = 1_000_003


def _derive_seed(base_seed: int, record_index: int, variant: int) -> int:
    return base_seed * _SEED_STRIDE + record_index * 101 + variant


def read_smi(path: Path) -> Iterator[tuple[int, str, str]]:
    """Yield ``(index, smiles, source_id)`` for every non-empty line."""
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            source_id = parts[1] if len(parts) > 1 else f"rec{index}"
            yield index, smiles, source_id


def generate_for_record(
    smiles: str,
    source_id: str,
    config: MarkushConfig,
    count: int,
    record_index: int,
    base_seed: int,
) -> Iterator[MarkushIR]:
    mol = load_molecule(smiles)
    if mol is None:
        return
    for variant in range(count):
        seed = _derive_seed(base_seed, record_index, variant)
        rng = random.Random(seed)
        ir = abstract_molecule(
            mol,
            config.replace(seed=seed),
            rng,
            source_smiles=smiles,
            source_id=source_id,
        )
        if ir is not None:
            yield ir


def _generate_payload(args: argparse.Namespace) -> Iterator[tuple[str, dict]]:
    config = MarkushConfig(
        seed=args.seed,
        core_policy=args.core_policy,
        core_retention=args.core_retention,
    )
    records = read_smi(Path(args.smi))
    if args.start:
        records = (item for item in records if item[0] >= args.start)
    for record_index, smiles, source_id in records:
        if args.limit is not None and record_index >= args.limit:
            break
        for ir in generate_for_record(
            smiles, source_id, config, args.count_per_mol, record_index, args.seed
        ):
            yield source_id, ir.to_dict()


def _shard_writer(out_dir: Path, shard_size: int) -> Iterator:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_index = 0
    count = 0
    handle = None
    try:
        while True:
            item = yield
            if item is None:
                break
            if handle is None:
                handle = (out_dir / f"shard_{shard_index:05d}.jsonl").open("w", encoding="utf-8")
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if count % shard_size == 0:
                handle.close()
                handle = None
                shard_index += 1
    finally:
        if handle is not None:
            handle.close()


def build_dataset(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = _shard_writer(out_dir, args.shard_size)
    next(writer)
    total = 0
    single_dir = out_dir / "ir" if args.emit_single else None
    if single_dir is not None:
        single_dir.mkdir(parents=True, exist_ok=True)
    for source_id, payload in _generate_payload(args):
        writer.send(payload)
        if single_dir is not None:
            path = single_dir / f"{source_id}_{payload['_meta']['seed']}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        total += 1
    writer.close()
    return {"generated": total, "out": str(out_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smi", required=True, help="input .smi file (SMILES [id] per line)")
    parser.add_argument("--out", default="data/markush_ir", help="output directory")
    parser.add_argument("--count-per-mol", type=int, default=4, help="variants per molecule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="stop after record index N")
    parser.add_argument("--start", type=int, default=0, help="start at record index N")
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--core-policy", default="random_bfs", choices=("murcko", "random_bfs"))
    parser.add_argument("--core-retention", type=float, default=0.6)
    parser.add_argument("--emit-single", action="store_true", help="also write one JSON per IR")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    summary = build_dataset(args)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
