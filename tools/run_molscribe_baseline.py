#!/usr/bin/env python3
"""Run the pretrained MolScribe model as a SMILES baseline on rendered shards.

MolScribe is the "SMILES directly" reference (``M1方案.md`` §16.3 exit criterion
3): it decodes a flat character stream, so it cannot express R-groups, variables
or ring placeholders. This runner scores its predictions against the v2 gold
graph at the SMILES level and writes per-sample records for analysis.

The bundled ``MolScribe/`` copy targets old dependency versions; we stay on the
project environment and shim the few removed timm helpers / the onmt package
init at import time instead of downgrading anything.

Example::

    python tools/run_molscribe_baseline.py \
        --shard data/plain_rendered/valid/shard_*.tar --limit 30 --out data/baseline_plain
    python tools/run_molscribe_baseline.py \
        --shard data/rendered/valid/shard_*.tar --limit 30 --out data/baseline_markush
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import DataStructs, rdFingerprintGenerator  # noqa: E402

from markushscribe import schema  # noqa: E402
from markushscribe.chemistry import graph_to_smiles  # noqa: E402

_DEFAULT_CKPT = "MolScribe/ckpts/swin_base_char_aux_1m680k.pth"


def load_molscribe(model_path: str, device: str = "cpu"):
    """Import and instantiate MolScribe with the compatibility shims applied."""
    root = Path(__file__).resolve().parents[1]
    site = next((root / ".venv" / "lib").glob("python*/site-packages"), None)
    if site is not None:
        onmt = types.ModuleType("onmt")
        onmt.__path__ = [str(site / "onmt")]
        sys.modules.setdefault("onmt", onmt)

    import timm.models.helpers as helpers
    from timm.models import vision_transformer

    original = helpers.build_model_with_cfg

    def build_model_with_cfg(model_cls, variant, pretrained=False, **kwargs):
        for key in ("default_cfg", "pretrained_cfg", "pretrained_strict"):
            kwargs.pop(key, None)
        return original(model_cls, variant, pretrained, **kwargs)

    helpers.build_model_with_cfg = build_model_with_cfg
    helpers.overlay_external_default_cfg = lambda default_cfg, kwargs: None
    if not hasattr(vision_transformer, "_init_vit_weights"):
        vision_transformer._init_vit_weights = lambda *args, **kwargs: None

    molscribe_root = str(root / "MolScribe")
    if molscribe_root not in sys.path:
        sys.path.insert(0, molscribe_root)
    from molscribe.interface import MolScribe

    return MolScribe(model_path, device=device, num_workers=0)


def iter_samples(shard: str):
    with tarfile.open(shard) as archive:
        names = archive.getnames()
        stems = sorted({name.rsplit(".", 1)[0] for name in names if name.endswith(".json")})
        for stem in stems:
            png = archive.extractfile(f"{stem}.png")
            payload = archive.extractfile(f"{stem}.json")
            if png is None or payload is None:
                continue
            from PIL import Image

            with Image.open(io.BytesIO(png.read())) as handle:
                image = handle.convert("RGB")
            yield stem, image, json.loads(payload.read())


def canonical(smiles: str | None) -> str | None:
    """Canonical, stereo-insensitive SMILES (v2 graphs drop wedge direction)."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None


def tanimoto(left, right) -> float:
    if left is None or right is None:
        return 0.0
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(left), generator.GetFingerprint(right)
        )
    )


def run(model, shards: list[str], *, limit: int | None) -> tuple[dict, list[dict]]:
    import numpy as np

    records: list[dict] = []
    for shard in shards:
        for uid, image, gold_payload in iter_samples(shard):
            if limit is not None and len(records) >= limit:
                break
            gold = schema.parse_target(gold_payload)
            gold_canonical = canonical(graph_to_smiles(gold))
            prediction = model.predict_image(np.asarray(image))
            pred_smiles = prediction.get("smiles") or ""
            pred_canonical = canonical(pred_smiles)
            exact = gold_canonical is not None and pred_canonical == gold_canonical
            records.append(
                {
                    "uid": uid,
                    "shard": shard,
                    "gold_nodes": len(gold.nodes),
                    "gold_smiles": gold_canonical,
                    "pred_smiles": pred_smiles,
                    "valid": pred_canonical is not None,
                    "exact_match": bool(exact),
                    "tanimoto": tanimoto(
                        Chem.MolFromSmiles(gold_canonical) if gold_canonical else None,
                        Chem.MolFromSmiles(pred_canonical) if pred_canonical else None,
                    ),
                }
            )
        if limit is not None and len(records) >= limit:
            break
    summary = {
        "samples": len(records),
        "valid_rate": float(np.mean([r["valid"] for r in records])) if records else 0.0,
        "exact_match_rate": float(np.mean([r["exact_match"] for r in records])) if records else 0.0,
        "mean_tanimoto": float(np.mean([r["tanimoto"] for r in records])) if records else 0.0,
    }
    return summary, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--ckpt", default=_DEFAULT_CKPT)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    import glob

    shards: list[str] = []
    for pattern in args.shard:
        shards.extend(sorted(glob.glob(pattern)) or [pattern])

    model = load_molscribe(args.ckpt, device=args.device)
    summary, records = run(model, shards, limit=args.limit)
    result = {"ckpt": args.ckpt, "shards": shards, "metrics": summary}
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
        with (out / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
