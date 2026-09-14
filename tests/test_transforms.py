"""Tests for pixel degradation and target-synchronized geometry warps."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from markushscribe import schema
from markushscribe.transforms import (
    DegradeConfig,
    GeometryConfig,
    ImageDegradeConfig,
    degrade_image,
    degrade_sample,
    derive_seed,
    ink_fraction,
    sample_geometry_matrix,
    warp_image,
    warp_target,
)

BINS = 64


def _image() -> Image.Image:
    image = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((10, 10, 150, 110), fill="black", width=3)
    draw.ellipse((60, 40, 100, 80), outline="black", width=2)
    return image


def _coord(value: float) -> tuple[int, float]:
    scaled = value * (BINS - 1)
    index = round(scaled)
    return index, round(scaled - index, 6)


def _node(node_id: int, x: float, y: float) -> dict:
    x_bin, x_offset = _coord(x)
    y_bin, y_offset = _coord(y)
    return {
        "id": node_id,
        "node_class": "atom",
        "variable_subtype": None,
        "anchor": {"x": round(x, 6), "y": round(y, 6), "rule": "base_core"},
        "coord_target": {
            "bins": BINS,
            "x_bin": x_bin,
            "x_offset": x_offset,
            "y_bin": y_bin,
            "y_offset": y_offset,
        },
        "visual_label": {
            "visible": True,
            "plain_text": "C",
            "runs": [{"text": "C", "script": "base"}],
        },
        "semantic_label": {"element": "C", "h_count": None, "charge": 0, "isotope": None},
        "loss_mask": {"h_count": False},
    }


def _target() -> dict:
    return {
        "format": schema.TARGET_FORMAT,
        "coordinate_system": schema.COORDINATE_SYSTEM,
        "image_size": [160, 120],
        "coord_bins": BINS,
        "order": schema.DEFAULT_ORDER,
        "nodes": [_node(0, 0.3, 0.5), _node(1, 0.9, 0.5)],
        "edges": [
            {
                "source": 0,
                "target": 1,
                "adjacent": True,
                "normalized": {"order": 1, "aromatic": False, "kekule_order": None},
                "depicted": {"type": "plain", "visible_order": 1, "ring_mode": None},
                "variable": False,
                "wedge_direction": "none",
            }
        ],
        "graph_annotations": [],
        "masks": {"cxsmiles": False, "normalized_chemistry_complete": True},
    }


def test_derive_seed_is_deterministic() -> None:
    assert derive_seed(1, 2, 3) == derive_seed(1, 2, 3)
    assert derive_seed(1, 2, 3) != derive_seed(1, 2, 4)
    assert 0 <= derive_seed(999, 123) < 2**31


def test_ink_fraction_separates_blank_from_drawn() -> None:
    assert ink_fraction(Image.new("RGB", (32, 32), "white")) == 0.0
    assert ink_fraction(_image()) > 0.0


def test_no_op_config_preserves_pixels() -> None:
    image = _image()
    quiet = ImageDegradeConfig(
        blur_prob=0.0,
        noise_prob=0.0,
        jpeg_prob=0.0,
        binarize_prob=0.0,
        illumination_prob=0.0,
        ink_dropout_prob=0.0,
        invert_prob=0.0,
    )
    result = degrade_image(image, quiet, seed=0)
    assert np.array_equal(np.asarray(result), np.asarray(image))


def test_degrade_image_is_deterministic_and_varies() -> None:
    image = _image()
    config = ImageDegradeConfig(
        blur_prob=1.0,
        noise_prob=1.0,
        jpeg_prob=1.0,
        binarize_prob=0.0,
        illumination_prob=0.0,
        ink_dropout_prob=0.0,
    )
    first = degrade_image(image, config, seed=42)
    second = degrade_image(image, config, seed=42)
    third = degrade_image(image, config, seed=43)
    assert first.size == image.size and first.mode == "RGB"
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(third))


def test_binarize_and_invert_produce_binary_image() -> None:
    config = ImageDegradeConfig(
        blur_prob=0.0,
        noise_prob=0.0,
        jpeg_prob=0.0,
        illumination_prob=0.0,
        ink_dropout_prob=0.0,
        binarize_prob=1.0,
    )
    result = degrade_image(_image(), config, seed=5)
    values = set(np.unique(np.asarray(result.convert("L"))).tolist())
    assert values <= {0, 255}


def test_geometry_disabled_returns_none() -> None:
    assert (
        sample_geometry_matrix(GeometryConfig(perspective_prob=0, affine_prob=0), (100, 80), 0)
        is None
    )


def test_geometry_is_deterministic() -> None:
    config = GeometryConfig(perspective_prob=1.0, affine_prob=0.0)
    first = sample_geometry_matrix(config, (200, 100), seed=3)
    second = sample_geometry_matrix(config, (200, 100), seed=3)
    third = sample_geometry_matrix(config, (200, 100), seed=4)
    assert first is not None and second is not None and third is not None
    assert np.allclose(first, second)
    assert not np.allclose(first, third)


def test_warp_target_identity_keeps_structure() -> None:
    target = _target()
    warped = warp_target(target, np.eye(3), (160, 120), (160, 120))
    assert len(warped["nodes"]) == 2
    assert len(warped["edges"]) == 1
    for before, after in zip(target["nodes"], warped["nodes"], strict=True):
        assert after["anchor"]["x"] == pytest.approx(before["anchor"]["x"])
        assert after["anchor"]["y"] == pytest.approx(before["anchor"]["y"])
    schema.validate_target(warped)


def test_warp_target_drops_off_canvas_nodes_and_edges() -> None:
    target = _target()
    matrix = np.array([[1.0, 0.0, 0.2 * 160], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    warped = warp_target(target, matrix, (160, 120), (160, 120))
    assert [node["id"] for node in warped["nodes"]] == [0]
    assert warped["edges"] == []
    schema.validate_target(warped)


def test_warp_image_and_sample_roundtrip() -> None:
    image = _image()
    target = _target()
    config = DegradeConfig(
        pixel=ImageDegradeConfig(
            blur_prob=0.0,
            noise_prob=0.0,
            jpeg_prob=0.0,
            binarize_prob=0.0,
            illumination_prob=0.0,
            ink_dropout_prob=0.0,
        ),
        geometry=GeometryConfig(perspective_prob=1.0, affine_prob=0.0),
    )
    warped_image, warped_target = degrade_sample(image, target, config, seed=11)
    assert warped_image.size == image.size
    assert warped_target["image_size"] == list(warped_image.size)
    schema.validate_target(warped_target)


def test_warp_image_identity_preserves_size() -> None:
    image = _image()
    assert warp_image(image, np.eye(3)).size == image.size
