"""Draw prediction targets back onto their rendered image for QA.

The recogniser is trained to predict exactly this target, so a visual overlay is
the fastest way to catch a label that does not match the pixels (a shifted
anchor, a dropped node, a wrong edge depiction). :func:`draw_overlay` annotates
anchors, node classes, depicted edges and ring-level annotations, while
:func:`inspect_shard` and :func:`inspect_dataset` build contact sheets for human
review.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..dataset import read_shard

NODE_COLORS = {
    "atom": (0, 150, 60),
    "rgroup": (30, 90, 220),
    "variable": (230, 120, 0),
    "ring_placeholder": (150, 40, 180),
    "abbreviation": (0, 150, 160),
}
_EDGE_STYLES = {
    "plain": ((90, 90, 90), 1),
    "double": ((0, 0, 0), 2),
    "triple": ((0, 0, 0), 3),
    "companion": ((200, 0, 190), 2),
    "circle": ((200, 0, 190), 1),
    "solid_wedge": ((0, 90, 210), 4),
    "dashed_wedge": ((0, 165, 210), 4),
    "wavy": ((235, 120, 0), 2),
    "dotted": ((120, 120, 0), 2),
    "variable": ((235, 120, 0), 2),
}
_ANCHOR_COLOR = (230, 30, 30)
_CAPTION_BG = (245, 245, 245)


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:  # pragma: no cover - font availability varies
        return ImageFont.load_default(size=size)


def node_text(node: dict[str, Any]) -> str:
    """Short human-readable label combining the drawn text and node class."""
    visual = node.get("visual_label") or {}
    text = str(visual.get("plain_text") or "")
    if not text:
        text = str((node.get("semantic_label") or {}).get("element") or "?")
    return f"#{node['id']} {text} [{node['node_class']}]"


def _anchor_px(target: dict[str, Any], node: dict[str, Any]) -> tuple[float, float]:
    width, _height = target["image_size"]
    height = target["image_size"][1]
    anchor = node["anchor"]
    return float(anchor["x"]) * width, float(anchor["y"]) * height


def _edge_color(edge: dict[str, Any]) -> tuple[tuple[int, int, int], int]:
    depicted = edge.get("depicted") or {}
    kind = str(depicted.get("type") or "plain")
    if edge.get("variable") and kind == "plain":
        kind = "variable"
    return _EDGE_STYLES.get(kind, _EDGE_STYLES["plain"])


def draw_overlay(
    image: Image.Image,
    target: dict[str, Any],
    *,
    show_ids: bool = True,
    show_edges: bool = True,
    show_edge_types: bool = True,
    show_annotations: bool = True,
) -> Image.Image:
    """Return ``image`` with anchors, nodes, edges and annotations drawn on top."""
    base = image.convert("RGBA")
    canvas = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    by_id = {int(node["id"]): node for node in target.get("nodes", [])}
    anchor_font = _font(11)
    edge_font = _font(10)

    if show_annotations:
        for annotation in target.get("graph_annotations", []):
            members = [by_id[node_id] for node_id in annotation["nodes"] if node_id in by_id]
            if len(members) < 3:
                continue
            points = [_anchor_px(target, node) for node in members]
            draw.polygon(points, fill=(200, 0, 190, 45), outline=(200, 0, 190, 180))

    if show_edges:
        for edge in target.get("edges", []):
            source, destination = int(edge["source"]), int(edge["target"])
            if source not in by_id or destination not in by_id:
                continue
            start = _anchor_px(target, by_id[source])
            end = _anchor_px(target, by_id[destination])
            color, width = _edge_color(edge)
            draw.line((*start, *end), fill=color, width=width)
            if show_edge_types:
                midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                label = (edge.get("depicted") or {}).get("type") or "?"
                draw.text(
                    (midpoint[0] + 3, midpoint[1] + 3), str(label), fill=color, font=edge_font
                )

    for node in target.get("nodes", []):
        x, y = _anchor_px(target, node)
        radius = 9 if node["node_class"] == "ring_placeholder" else 6
        color = NODE_COLORS.get(node["node_class"], (0, 0, 0))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.line((x - 10, y, x + 10, y), fill=_ANCHOR_COLOR, width=2)
        draw.line((x, y - 10, x, y + 10), fill=_ANCHOR_COLOR, width=2)
        if show_ids:
            text = node_text(node)
            box = draw.textbbox((x + 12, y - 8), text, font=anchor_font)
            draw.rectangle(
                (box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(255, 255, 255, 210)
            )
            draw.text((x + 12, y - 8), text, fill=color, font=anchor_font)

    return Image.alpha_composite(base, canvas).convert("RGB")


def _load_shard_samples(path: str | Path) -> Iterator[tuple[str, Image.Image, dict[str, Any]]]:
    members = dict(read_shard(path))
    stems = sorted({name.rsplit(".", 1)[0] for name in members})
    for stem in stems:
        png = members.get(f"{stem}.png")
        payload = members.get(f"{stem}.json")
        if png is None or payload is None:
            continue
        with Image.open(io.BytesIO(png)) as handle:
            image = handle.convert("RGB")
        yield stem, image, json.loads(payload.decode("utf-8"))


def _thumbnail(image: Image.Image, max_width: int, caption: str) -> Image.Image:
    scale = min(1.0, max_width / max(image.width, 1))
    thumb = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    strip = Image.new("RGB", (thumb.width, thumb.height + 18), _CAPTION_BG)
    strip.paste(thumb, (0, 18))
    ImageDraw.Draw(strip).text((4, 3), caption, fill=(20, 20, 20), font=_font(11))
    return strip


def contact_sheet(
    samples: Iterable[tuple[str, Image.Image]], *, cols: int = 3, max_width: int = 420
) -> Image.Image:
    """Compose labeled thumbnails into one grid image."""
    cells = [_thumbnail(image, max_width, caption) for caption, image in samples]
    if not cells:
        return Image.new("RGB", (max_width, 40), _CAPTION_BG)
    cols = max(1, cols)
    rows = (len(cells) + cols - 1) // cols
    cell_w = max(cell.width for cell in cells)
    cell_h = max(cell.height for cell in cells)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    for index, cell in enumerate(cells):
        row, col = divmod(index, cols)
        sheet.paste(cell, (col * cell_w, row * cell_h))
    return sheet


def inspect_shard(
    shard_path: str | Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    cols: int = 3,
    save_overlays: bool = False,
) -> dict[str, Any]:
    """Overlay up to ``limit`` samples of one shard and write a contact sheet."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet_samples: list[tuple[str, Image.Image]] = []
    count = 0
    for stem, image, target in _load_shard_samples(shard_path):
        if limit is not None and count >= limit:
            break
        overlay = draw_overlay(image, target)
        caption = f"{stem}  {image.width}x{image.height}  n={len(target.get('nodes', []))}"
        sheet_samples.append((caption, overlay))
        if save_overlays:
            overlays_dir = out / "overlays"
            overlays_dir.mkdir(parents=True, exist_ok=True)
            overlay.save(overlays_dir / f"{stem}.png")
        count += 1
    sheet = contact_sheet(sheet_samples, cols=cols)
    split_name = Path(shard_path).parent.name
    sheet_path = out / f"{split_name}_{Path(shard_path).stem}_sheet.png"
    sheet.save(sheet_path)
    return {"shard": str(shard_path), "overlays": count, "sheet": str(sheet_path)}


def inspect_dataset(
    shard_dir: str | Path,
    out_dir: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
    per_shard: int | None = None,
    cols: int = 3,
    save_overlays: bool = False,
) -> dict[str, Any]:
    """Overlay samples across a shard tree and write one sheet per shard."""
    root = Path(shard_dir)
    search_root = root / split if split else root
    shards = sorted(search_root.rglob("*.tar"))
    if limit is not None:
        shards = shards[:limit]
    results = []
    for shard in shards:
        results.append(
            inspect_shard(
                shard,
                out_dir,
                limit=per_shard,
                cols=cols,
                save_overlays=save_overlays,
            )
        )
    return {"shards": len(results), "results": results}


__all__ = [
    "contact_sheet",
    "draw_overlay",
    "inspect_dataset",
    "inspect_shard",
    "node_text",
]
