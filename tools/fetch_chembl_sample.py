#!/usr/bin/env python3
"""Fetch a small ChEMBL SMILES sample for dataset construction.

Example::

    python tools/fetch_chembl_sample.py --out data/chembl_sample.smi --count 2000

The output is a ``.smi`` file with one ``canonical_smiles molecule_chembl_id``
line per molecule, ready for ``markushscribe.markushgen.cli``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_API = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
_FIELDS = "molecule_chembl_id,molecule_structures"


def fetch_page(offset: int, page_size: int, timeout: float, retries: int = 3) -> list[dict]:
    url = f"{_API}?limit={page_size}&offset={offset}&only={_FIELDS}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload.get("molecules", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"failed to fetch ChEMBL offset {offset}: {last_error}")


def smiles_of(molecule: dict) -> str | None:
    structures = molecule.get("molecule_structures") or {}
    smiles = structures.get("canonical_smiles")
    return smiles if smiles else None


def fetch_sample(
    count: int,
    *,
    page_size: int = 1000,
    offset: int = 0,
    timeout: float = 30.0,
) -> list[tuple[str, str]]:
    """Return ``(smiles, chembl_id)`` pairs, de-duplicated by SMILES."""
    page_size = max(1, min(page_size, 1000))
    collected: list[tuple[str, str]] = []
    seen: set[str] = set()
    cursor = offset
    while len(collected) < count:
        molecules = fetch_page(cursor, page_size, timeout)
        if not molecules:
            break
        for molecule in molecules:
            smiles = smiles_of(molecule)
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            collected.append((smiles, str(molecule.get("molecule_chembl_id") or "?")))
            if len(collected) >= count:
                break
        cursor += page_size
    return collected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output .smi path")
    parser.add_argument("--count", type=int, default=2000, help="number of molecules")
    parser.add_argument("--page-size", type=int, default=1000, help="ChEMBL page size (max 1000)")
    parser.add_argument("--offset", type=int, default=0, help="ChEMBL result offset")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout (s)")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    samples = fetch_sample(
        args.count, page_size=args.page_size, offset=args.offset, timeout=args.timeout
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{smiles} {chembl_id}\n" for smiles, chembl_id in samples), "utf-8")
    print(json.dumps({"out": str(out), "molecules": len(samples)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
