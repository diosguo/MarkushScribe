"""Tests for the end-to-end dataset assembly pipeline."""

from __future__ import annotations

import io
import json
import random
import tarfile

import pytest

from markushscribe import dataset, schema

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


def _ir_records() -> list[dict]:
    from markushscribe.markushgen import MarkushConfig, abstract_molecule
    from markushscribe.markushgen.molecule import load_molecule

    records = []
    for smiles in SMILES:
        mol = load_molecule(smiles)
        assert mol is not None
        for seed in range(2):
            ir = abstract_molecule(
                mol, MarkushConfig(seed=seed), random.Random(seed), source_smiles=smiles
            )
            if ir is not None:
                records.append(ir.to_dict())
    return records


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_read_jsonl_and_json(tmp_path) -> None:
    records = _ir_records()
    jsonl = tmp_path / "in.jsonl"
    _write_jsonl(jsonl, records)
    assert len(list(dataset.read_jsonl(jsonl))) == len(records)

    single = tmp_path / "one.json"
    single.write_text(json.dumps(records[0]), encoding="utf-8")
    assert len(list(dataset.read_jsonl(single))) == 1


def test_safe_uid_sanitizes() -> None:
    meta = {"source_id": "CHEMBL:123/x", "seed": 9}
    uid = dataset.safe_uid(meta, fallback=0)
    assert uid == "CHEMBL_123_x_9"
    assert "/" not in uid and ":" not in uid


def test_render_record_is_valid_and_rasterized() -> None:
    pytest.importorskip("markushrender")
    record = _ir_records()[0]
    image, target = dataset.render_record(record, variant_seed=1)
    assert image.mode == "RGB"
    schema.validate_target(target)
    assert target["image_size"] == [image.width, image.height]


def test_build_and_validate_dataset(tmp_path) -> None:
    pytest.importorskip("markushrender")
    records = _ir_records()
    source = tmp_path / "ir.jsonl"
    _write_jsonl(source, records)

    out = tmp_path / "rendered"
    summary = dataset.build_dataset(
        [source],
        out,
        degrade=True,
        split_key="canonical",
        ratios=(0.6, 0.2, 0.2),
        seed=0,
        shard_size=1,
    )
    assert (out / "splits.json").exists()
    manifest = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    assert sum(manifest["counts"].values()) == len(records)

    total = sum(stats["samples"] for stats in summary["splits"].values())
    assert total == len(records)
    assert sum(stats["failed"] for stats in summary["splits"].values()) == 0

    stats = dataset.validate_dataset(out)
    assert stats["valid"] == len(records)
    assert stats["invalid"] == 0
    assert stats["unpaired"] == 0
    assert stats["images"] == len(records)


def test_build_dataset_without_degrade(tmp_path) -> None:
    pytest.importorskip("markushrender")
    records = _ir_records()[:2]
    source = tmp_path / "ir.jsonl"
    _write_jsonl(source, records)
    out = tmp_path / "rendered"
    dataset.build_dataset([source], out, degrade=False, shard_size=10)
    stats = dataset.validate_dataset(out)
    assert stats["valid"] == len(records)


def test_validate_dataset_flags_invalid_target(tmp_path) -> None:
    shard_dir = tmp_path / "data" / "train"
    shard_dir.mkdir(parents=True)
    payload = b'{"format": "markush_graph_v1"}'
    with tarfile.open(shard_dir / "shard_00000.tar", "w") as archive:
        for name, blob in (("x.json", payload), ("x.png", b"\x89PNG")):
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
    stats = dataset.validate_dataset(tmp_path / "data")
    assert stats["samples"] == 1 and stats["invalid"] == 1
    with pytest.raises(ValueError, match="blank"):
        dataset.validate_dataset(tmp_path / "data", raise_on_error=True)


def test_validate_dataset_flags_unpaired_member(tmp_path) -> None:
    shard_dir = tmp_path / "data" / "train"
    shard_dir.mkdir(parents=True)
    with tarfile.open(shard_dir / "shard_00000.tar", "w") as archive:
        info = tarfile.TarInfo("only.png")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"\x89PNG"))
    stats = dataset.validate_dataset(tmp_path / "data")
    assert stats["unpaired"] == 1
    assert stats["images"] == 1 and stats["samples"] == 0


def test_build_dataset_rejects_empty_input(tmp_path) -> None:
    source = tmp_path / "ir.jsonl"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no IR records"):
        dataset.build_dataset([source], tmp_path / "out")


def test_read_shard_roundtrip(tmp_path) -> None:
    samples = [("a", b"png", b"{}")]
    path = tmp_path / "shard.tar"
    dataset._write_shard(samples, path)
    assert dict(dataset.read_shard(path)) == {"a.png": b"png", "a.json": b"{}"}


def test_overlap_ids_are_recorded(tmp_path, monkeypatch) -> None:
    markushrender = pytest.importorskip("markushrender")
    monkeypatch.setattr(markushrender, "variant_report", lambda result, seed=None: {"overlaps": 3})
    source = tmp_path / "ir.jsonl"
    _write_jsonl(source, _ir_records()[:1])
    out = tmp_path / "rendered"
    summary = dataset.build_dataset([source], out, degrade=False, shard_size=10)
    train = summary["splits"]["train"]
    assert train["overlap_samples"] == 1
    assert train["overlaps"] == 3
    assert len(train["overlap_ids"]) == 1
    assert dataset.validate_dataset(out)["samples"] == 1


def test_drop_overlaps_filters_bad_renders(tmp_path, monkeypatch) -> None:
    markushrender = pytest.importorskip("markushrender")
    monkeypatch.setattr(markushrender, "variant_report", lambda result, seed=None: {"overlaps": 1})
    source = tmp_path / "ir.jsonl"
    _write_jsonl(source, _ir_records()[:1])
    out = tmp_path / "rendered"
    summary = dataset.build_dataset([source], out, degrade=False, shard_size=10, drop_overlaps=True)
    train = summary["splits"]["train"]
    assert train["dropped"] == 1
    assert dataset.validate_dataset(out)["samples"] == 0


def test_blank_render_is_dropped(tmp_path, monkeypatch) -> None:
    pytest.importorskip("markushrender")
    from PIL import Image

    record = _ir_records()[0]
    _image, target = dataset.render_record(record, record["_meta"]["seed"])
    blank = Image.new("RGB", (64, 64), "white")
    monkeypatch.setattr(
        dataset, "render_sample", lambda record, seed, config=None: (None, blank, target)
    )
    source = tmp_path / "ir.jsonl"
    _write_jsonl(source, [record])
    out = tmp_path / "rendered"
    summary = dataset.build_dataset([source], out, degrade=False, qa=False, shard_size=10)
    assert summary["splits"]["train"]["blank"] == 1
    assert dataset.validate_dataset(out)["samples"] == 0


def test_validate_dataset_flags_blank_image(tmp_path) -> None:
    pytest.importorskip("markushrender")
    import tarfile

    from PIL import Image

    record = _ir_records()[0]
    _image, target = dataset.render_record(record, record["_meta"]["seed"])
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, "PNG")
    shard_dir = tmp_path / "data" / "train"
    shard_dir.mkdir(parents=True)
    with tarfile.open(shard_dir / "shard_00000.tar", "w") as archive:
        for name, blob in (
            ("x.png", buffer.getvalue()),
            ("x.json", json.dumps(target).encode("utf-8")),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
    stats = dataset.validate_dataset(tmp_path / "data")
    assert stats["blank"] == 1
    with pytest.raises(ValueError, match="blank"):
        dataset.validate_dataset(tmp_path / "data", raise_on_error=True)
