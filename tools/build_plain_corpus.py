#!/usr/bin/env python3
"""Build plain-molecule Markush IR (Stage 1 OCSR data) from a ``.smi`` file.

Example::

    python tools/build_plain_corpus.py --smi data/chembl_sample.smi \\
        --out data/plain_ir --shard-size 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markushscribe.chemistry import mol_to_ir  # noqa: E402
from markushscribe.markushgen.cli import read_smi  # noqa: E402

_SEED_STRIDE = 1_000_003


def _derive_seed(base_seed: int, record_index: int) -> int:
    return base_seed * _SEED_STRIDE + record_index


def _load(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


def build(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_size = max(1, args.shard_size)
    shard_index = 0
    written = 0
    failed = 0
    handle = (out_dir / f"shard_{shard_index:05d}.jsonl").open("w", encoding="utf-8")
    try:
        for record_index, smiles, source_id in read_smi(Path(args.smi)):
            if args.limit is not None and written >= args.limit:
                break
            mol = _load(smiles)
            if mol is None:
                failed += 1
                continue
            try:
                ir = mol_to_ir(
                    mol,
                    source_smiles=smiles,
                    source_id=source_id,
                    keep_stereo=args.keep_stereo,
                )
            except Exception:  # noqa: BLE001 - one bad molecule must not stop the batch
                failed += 1
                continue
            payload = ir.to_dict()
            payload["_meta"]["seed"] = _derive_seed(args.seed, record_index)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            if written % shard_size == 0:
                handle.close()
                shard_index += 1
                handle = (out_dir / f"shard_{shard_index:05d}.jsonl").open("w", encoding="utf-8")
    finally:
        handle.close()
    return {"written": written, "failed": failed, "out": str(out_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smi", required=True, help="input .smi file")
    parser.add_argument("--out", default="data/plain_ir", help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="stop after N written records")
    parser.add_argument("--shard-size", type=int, default=20000)
    parser.add_argument("--keep-stereo", action="store_true", help="keep wedge/hatch bonds")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    print(json.dumps(build(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
