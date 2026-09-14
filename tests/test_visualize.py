"""Tests for the overlay viewer and contact sheets."""

from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image

from markushscribe import dataset
from markushscribe.visualization import (
    contact_sheet,
    draw_overlay,
    inspect_dataset,
    inspect_shard,
    node_text,
)

BINS = 64


def _coord(value: float) -> tuple[int, float]:
    scaled = value * (BINS - 1)
    index = round(scaled)
    return index, round(scaled - index, 6)


def _node(node_id: int, kind: str, x: float, y: float, text: str) -> dict:
    x_bin, x_offset = _coord(x)
    y_bin, y_offset = _coord(y)
    entry = {
        "id": node_id,
        "node_class": kind,
        "variable_subtype": "linker" if kind == "variable" else None,
        "anchor": {
            "x": round(x, 6),
            "y": round(y, 6),
            "rule": "placeholder_center" if kind == "ring_placeholder" else "base_core",
        },
        "coord_target": {
            "bins": BINS,
            "x_bin": x_bin,
            "x_offset": x_offset,
            "y_bin": y_bin,
            "y_offset": y_offset,
        },
        "visual_label": {
            "visible": True,
            "plain_text": text,
            "runs": [{"text": text, "script": "base"}],
        },
        "semantic_label": {},
        "loss_mask": {},
    }
    if kind == "atom":
        entry["semantic_label"] = {"element": text}
    elif kind == "rgroup":
        entry["semantic_label"] = {"family": "R", "index": "1", "multiplicity": None}
    elif kind == "variable":
        entry["semantic_label"] = {"family": "L", "index": "1", "multiplicity": None}
    elif kind == "ring_placeholder":
        entry["semantic_label"] = {"placeholder_id": text}
        entry["placeholder"] = {"shape": "circle", "radius": 0.89}
    return entry


def _target() -> dict:
    return {
        "format": "markush_graph_v2",
        "coordinate_system": "normalized [0,1], origin top-left, y increases downward",
        "image_size": [200, 100],
        "coord_bins": BINS,
        "order": "reading",
        "nodes": [
            _node(0, "atom", 0.2, 0.5, "C"),
            _node(1, "rgroup", 0.5, 0.5, "R1"),
            _node(2, "ring_placeholder", 0.8, 0.5, "A"),
        ],
        "edges": [
            {
                "source": 0,
                "target": 1,
                "adjacent": True,
                "normalized": {"order": 1, "aromatic": False, "kekule_order": None},
                "depicted": {"type": "double", "visible_order": 2, "ring_mode": None},
                "variable": False,
                "wedge_direction": "none",
            },
            {
                "source": 1,
                "target": 2,
                "adjacent": True,
                "normalized": {"order": 1, "aromatic": False, "kekule_order": None},
                "depicted": {"type": "plain", "visible_order": 1, "ring_mode": "cross"},
                "variable": False,
                "wedge_direction": "none",
            },
        ],
        "graph_annotations": [],
        "masks": {"cxsmiles": False, "normalized_chemistry_complete": True},
    }


def _blank() -> Image.Image:
    return Image.new("RGB", (200, 100), "white")


def test_node_text_combines_id_and_class() -> None:
    node = _node(3, "rgroup", 0.5, 0.5, "R1")
    assert node_text(node) == "#3 R1 [rgroup]"


def test_draw_overlay_preserves_size_and_changes_pixels() -> None:
    image = _blank()
    target = _target()
    overlay = draw_overlay(image, target)
    assert overlay.size == image.size
    assert overlay.mode == "RGB"
    assert not np.array_equal(np.asarray(overlay), np.asarray(image))


def test_draw_overlay_marks_annotation_region() -> None:
    target = _target()
    target["graph_annotations"] = [
        {"kind": "aromatic_ring", "nodes": [0, 1, 2], "depiction": "circle"}
    ]
    overlay = draw_overlay(_blank(), target)
    assert not np.array_equal(np.asarray(overlay), np.asarray(_blank()))


def test_contact_sheet_grid_dimensions() -> None:
    image = _blank()
    one = contact_sheet([("only", image)], cols=1)
    multi = contact_sheet([("a", image), ("b", image), ("c", image)], cols=2)
    assert multi.width == 2 * one.width
    assert multi.height == 2 * one.height


def test_inspect_shard_writes_sheet_and_overlays(tmp_path) -> None:
    target = _target()
    buffer = io.BytesIO()
    _blank().save(buffer, "PNG")
    shard = tmp_path / "shard_00000.tar"
    dataset._write_shard([("uid_1", buffer.getvalue(), json.dumps(target).encode("utf-8"))], shard)

    result = inspect_shard(shard, tmp_path / "inspect", save_overlays=True)
    assert result["overlays"] == 1
    sheets = list((tmp_path / "inspect").glob("*_sheet.png"))
    assert len(sheets) == 1 and sheets[0].stat().st_size > 0
    assert (tmp_path / "inspect" / "overlays" / "uid_1.png").exists()


def test_inspect_dataset_limits_shards(tmp_path) -> None:
    target = _target()
    buffer = io.BytesIO()
    _blank().save(buffer, "PNG")
    split_dir = tmp_path / "rendered" / "train"
    for index in range(3):
        dataset._write_shard(
            [(f"uid_{index}", buffer.getvalue(), json.dumps(target).encode("utf-8"))],
            split_dir / f"shard_{index:05d}.tar",
        )
    summary = inspect_dataset(tmp_path / "rendered", tmp_path / "inspect", split="train", limit=2)
    assert summary["shards"] == 2
    assert len(list((tmp_path / "inspect").glob("*_sheet.png"))) == 2


def test_inspect_shard_skips_unpaired_members(tmp_path) -> None:
    import tarfile

    buffer = io.BytesIO()
    _blank().save(buffer, "PNG")
    shard = tmp_path / "shard_00000.tar"
    with tarfile.open(shard, "w") as archive:
        info = tarfile.TarInfo("only.png")
        info.size = len(buffer.getvalue())
        archive.addfile(info, io.BytesIO(buffer.getvalue()))
    result = inspect_shard(shard, tmp_path / "inspect")
    assert result["overlays"] == 0
    assert list((tmp_path / "inspect").glob("*_sheet.png"))


def test_overlay_is_repeatable() -> None:
    first = np.asarray(draw_overlay(_blank(), _target()))
    second = np.asarray(draw_overlay(_blank(), _target()))
    assert np.array_equal(first, second)
