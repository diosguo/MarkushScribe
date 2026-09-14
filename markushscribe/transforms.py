"""Image-level augmentation with target synchronization.

Pixel degradation (blur, noise, JPEG, illumination, ink dropout, binarization)
changes only the PNG and leaves the prediction target intact. Geometric warps
(affine tilt, mild perspective) move the pixels, so the target anchors and
``coord_target`` bins are transformed with the *same* homography and nodes that
leave the canvas are dropped together with their edges and annotations.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

COORD_BINS = 64


@dataclass(frozen=True)
class ImageDegradeConfig:
    """Probabilities and ranges for pixel-only corruptions."""

    blur_prob: float = 0.3
    noise_prob: float = 0.4
    jpeg_prob: float = 0.4
    binarize_prob: float = 0.2
    illumination_prob: float = 0.2
    ink_dropout_prob: float = 0.15
    invert_prob: float = 0.0
    blur_radius: tuple[float, float] = (0.4, 1.6)
    noise_std: tuple[float, float] = (2.0, 14.0)
    jpeg_quality: tuple[int, int] = (35, 90)
    illumination_strength: tuple[float, float] = (0.25, 0.7)
    dropout_fraction: tuple[float, float] = (0.01, 0.05)


@dataclass(frozen=True)
class GeometryConfig:
    """Probabilities and bounds for canvas-space warps."""

    perspective_prob: float = 0.2
    affine_prob: float = 0.3
    max_perspective: float = 0.05
    max_rotate_deg: float = 6.0
    max_translate: float = 0.03
    scale_range: tuple[float, float] = (0.96, 1.04)
    fill: int = 255


@dataclass(frozen=True)
class DegradeConfig:
    """Full augmentation recipe: pixel corruption plus optional geometry."""

    pixel: ImageDegradeConfig = field(default_factory=ImageDegradeConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)


def derive_seed(*parts: int) -> int:
    """Deterministic 31-bit seed from a tuple of integers (epoch, index, ...)."""
    value = 0
    for part in parts:
        value = (value * 1_000_003 + int(part)) & 0x7FFFFFFF
    return value


def ink_fraction(image: Image.Image, threshold: int = 180) -> float:
    """Fraction of pixels darker than ``threshold``; detects near-blank renders."""
    array = np.asarray(image.convert("L"))
    if array.size == 0:
        return 0.0
    return float((array < threshold).mean())


def _quantize(value: float, bins: int) -> tuple[int, float]:
    scaled = min(max(value, 0.0), 1.0) * (bins - 1)
    index = int(round(scaled))
    return index, round(scaled - index, 6)


def _otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 127
    levels = np.arange(256)
    weight_back = np.cumsum(histogram)
    weight_fore = total - weight_back
    sum_total = float((histogram * levels).sum())
    sum_back = np.cumsum(histogram * levels)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
    variance = weight_back * weight_fore * (mean_back - mean_fore) ** 2
    variance = np.nan_to_num(variance, nan=-1.0)
    return int(np.argmax(variance))


def _add_noise(image: Image.Image, rng: np.random.Generator, config: ImageDegradeConfig):
    array = np.asarray(image).astype(np.float32)
    sigma = float(rng.uniform(*config.noise_std))
    array = array + rng.normal(0.0, sigma, array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode=image.mode)


def _illuminate(image: Image.Image, rng: np.random.Generator, config: ImageDegradeConfig):
    array = np.asarray(image).astype(np.float32)
    height, width = array.shape[:2]
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij")
    ramp = np.cos(angle) * xx + np.sin(angle) * yy
    span = float(ramp.max() - ramp.min()) or 1.0
    ramp = (ramp - ramp.min()) / span
    strength = float(rng.uniform(*config.illumination_strength))
    factor = 1.0 - strength * (1.0 - ramp)
    if array.ndim == 3:
        factor = factor[..., None]
    array = array * factor
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode=image.mode)


def _drop_ink(
    image: Image.Image,
    rng: np.random.Generator,
    py_rng: random.Random,
    config: ImageDegradeConfig,
):
    array = np.asarray(image).copy()
    height, width = array.shape[:2]
    fraction = float(rng.uniform(*config.dropout_fraction))
    count = max(1, int(fraction * height * width / 400))
    for _ in range(count):
        box_w = py_rng.randint(1, max(2, width // 25))
        box_h = py_rng.randint(1, max(2, height // 25))
        x0 = py_rng.randint(0, width - 1)
        y0 = py_rng.randint(0, height - 1)
        array[y0 : y0 + box_h, x0 : x0 + box_w] = 255
    return Image.fromarray(array, mode=image.mode)


def _jpeg(image: Image.Image, rng: np.random.Generator, config: ImageDegradeConfig):
    quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as reloaded:
        return reloaded.convert(image.mode).copy()


def _binarize(image: Image.Image):
    gray = image.convert("L")
    array = np.asarray(gray)
    threshold = _otsu_threshold(array)
    binary = ((array > threshold) * 255).astype(np.uint8)
    return Image.fromarray(binary, mode="L").convert(image.mode)


def degrade_image(
    image: Image.Image, config: ImageDegradeConfig | None = None, seed: int = 0
) -> Image.Image:
    """Apply seeded pixel corruptions; geometry and labels are untouched."""
    cfg = config or ImageDegradeConfig()
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    result = image
    if cfg.blur_prob > 0 and float(rng.random()) < cfg.blur_prob:
        radius = float(rng.uniform(*cfg.blur_radius))
        result = result.filter(ImageFilter.GaussianBlur(radius))
    if cfg.noise_prob > 0 and float(rng.random()) < cfg.noise_prob:
        result = _add_noise(result, rng, cfg)
    if cfg.illumination_prob > 0 and float(rng.random()) < cfg.illumination_prob:
        result = _illuminate(result, rng, cfg)
    if cfg.ink_dropout_prob > 0 and float(rng.random()) < cfg.ink_dropout_prob:
        result = _drop_ink(result, rng, py_rng, cfg)
    if cfg.jpeg_prob > 0 and float(rng.random()) < cfg.jpeg_prob:
        result = _jpeg(result, rng, cfg)
    if cfg.binarize_prob > 0 and float(rng.random()) < cfg.binarize_prob:
        result = _binarize(result)
    if cfg.invert_prob > 0 and float(rng.random()) < cfg.invert_prob:
        result = ImageOps.invert(result.convert("RGB"))
    return result


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    transformed = homogeneous @ matrix.T
    return transformed[:, :2] / transformed[:, 2:3]


def _affine_matrix(
    config: GeometryConfig, rng: np.random.Generator, size: tuple[int, int]
) -> np.ndarray:
    width, height = float(size[0]), float(size[1])
    center_x, center_y = width / 2.0, height / 2.0
    angle = np.deg2rad(float(rng.uniform(-config.max_rotate_deg, config.max_rotate_deg)))
    scale = float(rng.uniform(*config.scale_range))
    cos, sin = float(np.cos(angle)) * scale, float(np.sin(angle)) * scale
    linear = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array(
        [
            [
                1.0,
                0.0,
                center_x + float(rng.uniform(-config.max_translate, config.max_translate)) * width,
            ],
            [
                0.0,
                1.0,
                center_y + float(rng.uniform(-config.max_translate, config.max_translate)) * height,
            ],
            [0.0, 0.0, 1.0],
        ]
    )
    undo_center = np.array([[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]])
    return shift @ linear @ undo_center


def _find_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    return vt[-1].reshape(3, 3)


def sample_geometry_matrix(
    config: GeometryConfig | None, size: tuple[int, int], seed: int
) -> np.ndarray | None:
    """Forward pixel homography (source -> destination) or ``None`` when idle."""
    cfg = config or GeometryConfig()
    rng = np.random.default_rng(seed ^ 0x9E3779B9)
    draw = float(rng.random())
    perspective = max(0.0, cfg.perspective_prob)
    affine = max(0.0, cfg.affine_prob)
    total = perspective + affine
    if total <= 0.0 or draw >= total:
        return None
    width, height = float(size[0]), float(size[1])
    source = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    if draw < perspective:
        jitter = rng.normal(0.0, cfg.max_perspective, size=(4, 2))
        target = source + jitter * np.array([width, height])
    else:
        target = _apply_affine(_affine_matrix(cfg, rng, size), source)
    return _find_homography(source, target)


def warp_image(
    image: Image.Image,
    matrix: np.ndarray,
    *,
    fill: int = 255,
    resample: int = Image.Resampling.BICUBIC,
) -> Image.Image:
    """Resample ``image`` through the forward pixel homography ``matrix``."""
    inverse = np.linalg.inv(matrix)
    inverse = inverse / inverse[2, 2]
    coefficients = inverse.flatten()[:8].tolist()
    fill_color = fill if image.mode in ("L", "1") else (fill, fill, fill)
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=resample,
        fillcolor=fill_color,
    )


def warp_target(
    target: dict[str, Any],
    matrix: np.ndarray,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> dict[str, Any]:
    """Move anchors of ``target`` by the same homography used for the pixels."""
    old_width, old_height = float(old_size[0]), float(old_size[1])
    new_width, new_height = float(new_size[0]), float(new_size[1])
    bins = int(target.get("coord_bins", COORD_BINS))
    kept: list[dict[str, Any]] = []
    for node in target.get("nodes", []):
        anchor = node.get("anchor") or {}
        x, y = anchor.get("x"), anchor.get("y")
        if x is None or y is None:
            continue
        points = np.array([[float(x) * old_width, float(y) * old_height]])
        moved = _apply_affine(matrix, points)[0]
        norm_x = float(moved[0]) / new_width
        norm_y = float(moved[1]) / new_height
        if not (-1e-6 <= norm_x <= 1.0 + 1e-6 and -1e-6 <= norm_y <= 1.0 + 1e-6):
            continue
        norm_x = min(max(norm_x, 0.0), 1.0)
        norm_y = min(max(norm_y, 0.0), 1.0)
        x_bin, x_offset = _quantize(norm_x, bins)
        y_bin, y_offset = _quantize(norm_y, bins)
        updated = dict(node)
        updated["anchor"] = {**anchor, "x": round(norm_x, 6), "y": round(norm_y, 6)}
        updated["coord_target"] = {
            "bins": bins,
            "x_bin": x_bin,
            "x_offset": x_offset,
            "y_bin": y_bin,
            "y_offset": y_offset,
        }
        kept.append(updated)

    kept.sort(key=lambda node: (node["coord_target"]["y_bin"], node["anchor"]["x"]))
    remap: dict[int, int] = {}
    for new_id, node in enumerate(kept):
        remap[int(node["id"])] = new_id
        node["id"] = new_id

    edges = []
    for edge in target.get("edges", []):
        source, destination = int(edge["source"]), int(edge["target"])
        if source not in remap or destination not in remap:
            continue
        updated_edge = dict(edge)
        updated_edge["source"] = remap[source]
        updated_edge["target"] = remap[destination]
        if updated_edge["source"] > updated_edge["target"]:
            updated_edge["source"], updated_edge["target"] = (
                updated_edge["target"],
                updated_edge["source"],
            )
        edges.append(updated_edge)
    edges.sort(key=lambda edge: (edge["source"], edge["target"]))

    annotations = []
    for annotation in target.get("graph_annotations", []):
        members = [int(node_id) for node_id in annotation["nodes"]]
        if all(node_id in remap for node_id in members):
            annotations.append({**annotation, "nodes": [remap[node_id] for node_id in members]})

    rebuilt = dict(target)
    rebuilt["nodes"] = kept
    rebuilt["edges"] = edges
    rebuilt["graph_annotations"] = annotations
    rebuilt["image_size"] = [int(new_width), int(new_height)]
    return rebuilt


def degrade_sample(
    image: Image.Image,
    target: dict[str, Any],
    config: DegradeConfig | None = None,
    seed: int = 0,
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply pixel corruption and (optionally) a synchronized geometric warp."""
    cfg = config or DegradeConfig()
    degraded = degrade_image(image, cfg.pixel, seed)
    matrix = sample_geometry_matrix(cfg.geometry, degraded.size, seed)
    if matrix is None:
        return degraded, target
    warped = warp_image(degraded, matrix, fill=cfg.geometry.fill)
    return warped, warp_target(target, matrix, degraded.size, warped.size)


__all__ = [
    "COORD_BINS",
    "DegradeConfig",
    "GeometryConfig",
    "ImageDegradeConfig",
    "degrade_image",
    "degrade_sample",
    "derive_seed",
    "ink_fraction",
    "sample_geometry_matrix",
    "warp_image",
    "warp_target",
]
