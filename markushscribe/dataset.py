"""Dataset assembly: IR records -> render -> target -> degrade -> shards.

The pipeline owns everything between a generated Markush IR and a training
shard. Rendering and target construction come from MarkushRender, pixel and
geometry augmentation from :mod:`markushscribe.transforms`, and the split is
computed here before any rendering happens. Shards are uncompressed tar files
with paired ``<uid>.png`` / ``<uid>.json`` members.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import schema, transforms
from .split import SPLITS, split_records

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class RenderConfig:
    """How one source IR becomes one training variant."""

    base: str = "patent_bw"
    preset_choices: tuple[str, ...] = ()
    style: bool = True
    lowres: bool = True
    rotate: bool = True
    mirror: bool = True
    order: str = schema.DEFAULT_ORDER
    coord_bins: int = schema.COORD_BINS

    def augment(self):
        """Build the MarkushRender ``AugmentConfig`` for this recipe."""
        from markushrender import AugmentConfig

        return AugmentConfig(
            seed=0,
            rotate=self.rotate,
            mirror=self.mirror,
            style=self.style,
            lowres=self.lowres,
            preset_choices=self.preset_choices,
        )


@dataclass
class ShardStats:
    samples: int = 0
    rendered: int = 0
    failed: int = 0
    degraded: int = 0
    dropped: int = 0
    blank: int = 0
    nodes: int = 0
    edges: int = 0
    overlaps: int = 0
    overlap_samples: int = 0
    overlap_ids: list[str] = field(default_factory=list)
    blank_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every record from a ``.jsonl`` shard or a single ``.json`` file."""
    source = Path(path)
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        yield from records
        return
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_uid(meta: dict[str, Any], fallback: int) -> str:
    """Filesystem-safe unique id for one rendered variant."""
    source_id = str(meta.get("source_id") or meta.get("source_smiles") or fallback)
    seed = int(meta.get("seed", fallback))
    return f"{_SAFE.sub('_', source_id)[:80]}_{seed}"


def render_sample(
    record: dict[str, Any],
    variant_seed: int,
    config: RenderConfig | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Render one IR record into ``(RenderResult, PIL.Image, validated target)``."""
    import markushrender
    from PIL import Image

    cfg = config or RenderConfig()
    result = markushrender.render_variant(
        record,
        seed=variant_seed,
        base=cfg.base,
        config=cfg.augment(),
        rasterize=True,
    )
    target = markushrender.prediction_target(result, coord_bins=cfg.coord_bins, order=cfg.order)
    schema.validate_target(target)
    if result.png is None:
        raise RuntimeError("renderer returned no raster image")
    with Image.open(io.BytesIO(result.png)) as handle:
        image = handle.convert("RGB")
    return result, image, target


def render_record(
    record: dict[str, Any],
    variant_seed: int,
    config: RenderConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render one IR record into ``(PIL.Image, validated target)``."""
    _result, image, target = render_sample(record, variant_seed, config)
    return image, target


def _write_shard(samples: list[tuple[str, bytes, bytes]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w") as archive:
        for stem, png_bytes, json_bytes in samples:
            for name, payload in ((f"{stem}.png", png_bytes), (f"{stem}.json", json_bytes)):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def read_shard(path: str | Path) -> Iterator[tuple[str, bytes]]:
    """Yield ``(member_name, bytes)`` for every file member of a tar shard."""
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                yield member.name, handle.read()


def build_split_shards(
    records: list[dict[str, Any]],
    split_name: str,
    out_dir: str | Path,
    *,
    config: RenderConfig | None = None,
    degrade: bool = True,
    degrade_config: transforms.DegradeConfig | None = None,
    base_seed: int = 0,
    shard_size: int = 1000,
    qa: bool = True,
    drop_overlaps: bool = False,
    min_ink_fraction: float = 0.001,
) -> ShardStats:
    """Render one split's records into consecutive ``.tar`` shards."""
    import markushrender

    stats = ShardStats()
    cfg = config or RenderConfig()
    buffer: list[tuple[str, bytes, bytes]] = []

    def flush(index: int) -> None:
        if buffer:
            _write_shard(buffer, Path(out_dir) / f"shard_{index:05d}.tar")
            buffer.clear()

    shard_index = 0
    for index, record in enumerate(records):
        meta = record.get("_meta") if isinstance(record.get("_meta"), dict) else {}
        variant_seed = int(meta.get("seed", base_seed + index))
        stats.samples += 1
        try:
            result, image, target = render_sample(record, variant_seed, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad record must not kill a shard
            stats.failed += 1
            stats.errors.append(f"{split_name}:{index}: {type(exc).__name__}: {exc}")
            continue
        stats.rendered += 1
        stem = safe_uid(meta, index)
        if qa:
            overlaps = int(markushrender.variant_report(result, seed=variant_seed)["overlaps"])
            if overlaps:
                stats.overlaps += overlaps
                stats.overlap_samples += 1
                stats.overlap_ids.append(stem)
                if drop_overlaps:
                    stats.dropped += 1
                    continue
        if degrade:
            seed = transforms.derive_seed(variant_seed, 1)
            image, target = transforms.degrade_sample(image, target, degrade_config, seed)
            schema.validate_target(target)
            stats.degraded += 1
        if transforms.ink_fraction(image) < min_ink_fraction:
            stats.blank += 1
            stats.blank_ids.append(stem)
            continue
        stats.nodes += len(target["nodes"])
        stats.edges += len(target["edges"])

        png_buffer = io.BytesIO()
        image.save(png_buffer, "PNG")
        json_bytes = json.dumps(target, ensure_ascii=False).encode("utf-8")
        buffer.append((stem, png_buffer.getvalue(), json_bytes))
        if len(buffer) >= max(1, shard_size):
            flush(shard_index)
            shard_index += 1
    flush(shard_index)
    return stats


def build_dataset(
    ir_paths: Iterable[str | Path],
    out_dir: str | Path,
    *,
    config: RenderConfig | None = None,
    degrade: bool = True,
    degrade_config: transforms.DegradeConfig | None = None,
    split_key: str = "auto",
    ratios: tuple[float, float, float] = (0.98, 0.01, 0.01),
    seed: int = 0,
    shard_size: int = 1000,
    limit: int | None = None,
    qa: bool = True,
    drop_overlaps: bool = False,
    min_ink_fraction: float = 0.001,
) -> dict[str, Any]:
    """Run the whole pipeline: read IR, split, render, degrade, write shards."""
    records: list[dict[str, Any]] = []
    for path in ir_paths:
        for record in read_jsonl(path):
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError("no IR records found")

    split_result = split_records(records, ratios=ratios, key=split_key, seed=seed)
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for record in records:
        buckets[split_result.require(record)].append(record)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name in SPLITS:
        stats = build_split_shards(
            buckets[name],
            name,
            out / name,
            config=config,
            degrade=degrade,
            degrade_config=degrade_config,
            base_seed=seed,
            shard_size=shard_size,
            qa=qa,
            drop_overlaps=drop_overlaps,
            min_ink_fraction=min_ink_fraction,
        )
        summary[name] = asdict(stats)

    manifest = {
        "split_key": split_key,
        "ratios": list(ratios),
        "seed": seed,
        "counts": split_result.counts,
        "group_to_split": split_result.group_to_split,
    }
    (out / "splits.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    return {"out": str(out), "splits": summary}


def validate_dataset(
    shard_dir: str | Path,
    *,
    raise_on_error: bool = False,
    min_ink_fraction: float = 0.001,
) -> dict[str, Any]:
    """Validate every target inside a shard tree and summarise the dataset."""
    from PIL import Image

    stats: dict[str, Any] = {
        "shards": 0,
        "samples": 0,
        "valid": 0,
        "invalid": 0,
        "images": 0,
        "blank": 0,
        "nodes": 0,
        "edges": 0,
        "unpaired": 0,
        "by_split": {name: 0 for name in SPLITS},
        "errors": [],
    }
    root = Path(shard_dir)
    for shard in sorted(root.rglob("*.tar")):
        stats["shards"] += 1
        relative = shard.relative_to(root)
        split_name = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        json_stems: set[str] = set()
        image_stems: set[str] = set()
        for name, payload in read_shard(shard):
            stem, _, suffix = name.rpartition(".")
            if suffix == "png":
                stats["images"] += 1
                image_stems.add(stem)
                try:
                    with Image.open(io.BytesIO(payload)) as handle:
                        blank = transforms.ink_fraction(handle) < min_ink_fraction
                except Exception as exc:  # noqa: BLE001 - corrupt members are QA findings
                    stats["errors"].append(f"{shard.name}:{name}: unreadable image: {exc}")
                    stats["blank"] += 1
                    continue
                if blank:
                    stats["blank"] += 1
                    stats["errors"].append(f"{shard.name}:{name}: near-blank image")
            elif suffix == "json":
                stats["samples"] += 1
                json_stems.add(stem)
                try:
                    graph = schema.parse_target(json.loads(payload.decode("utf-8")))
                except Exception as exc:  # noqa: BLE001 - collect, do not stop
                    stats["invalid"] += 1
                    stats["errors"].append(f"{shard.name}:{name}: {type(exc).__name__}: {exc}")
                    continue
                stats["valid"] += 1
                stats["nodes"] += len(graph.nodes)
                stats["edges"] += len(graph.edges)
                if split_name in stats["by_split"]:
                    stats["by_split"][split_name] += 1
        missing = json_stems.symmetric_difference(image_stems)
        if missing:
            stats["unpaired"] += len(missing)
            stats["errors"].append(f"{shard.name}: unpaired members {sorted(missing)[:5]}")
    if raise_on_error and (stats["invalid"] or stats["unpaired"] or stats["blank"]):
        raise ValueError(f"dataset at {root} has invalid, unpaired or blank samples")
    return stats


__all__ = [
    "RenderConfig",
    "ShardStats",
    "build_dataset",
    "build_split_shards",
    "read_jsonl",
    "read_shard",
    "render_record",
    "render_sample",
    "safe_uid",
    "validate_dataset",
]
