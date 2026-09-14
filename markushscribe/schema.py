"""The ``markush_graph_v2`` prediction-target contract.

This module is the single source of truth for the target format consumed by the
recogniser: it mirrors what MarkushRender emits in ``prediction_target`` and
validates every field before a sample enters a shard. It depends on nothing but
the standard library so the schema stays stable when the renderer changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TARGET_FORMAT = "markush_graph_v2"
COORDINATE_SYSTEM = "normalized [0,1], origin top-left, y increases downward"
COORD_BINS = 64
DEFAULT_ORDER = "reading"
ORDERS = frozenset({DEFAULT_ORDER})

SCRIPTS = frozenset({"base", "sub", "sup"})
NODE_CLASSES = frozenset({"atom", "rgroup", "variable", "abbreviation", "ring_placeholder"})
VARIABLE_SUBTYPES = frozenset({"ring_atom", "linker", "bond", "generic"})
ANCHOR_RULES = frozenset({"bond_vertex", "base_core", "placeholder_center"})
PLACEHOLDER_SHAPES = frozenset({"circle", "polygon", "bracket"})
WEDGE_DIRECTIONS = frozenset({"none", "source_to_target", "target_to_source"})
ANNOTATION_KINDS = frozenset({"aromatic_ring"})
ANNOTATION_DEPICTIONS = frozenset({"circle", "companion"})
MASK_KEYS = frozenset({"cxsmiles", "normalized_chemistry_complete"})
DEPICTED_TYPES = frozenset(
    {
        "plain",
        "double",
        "triple",
        "companion",
        "circle",
        "solid_wedge",
        "dashed_wedge",
        "wavy",
        "dotted",
        "variable",
    }
)
AROMATIC_DEPICTIONS = frozenset({"plain", "double", "companion", "circle"})
SPECIAL_DEPICTIONS = frozenset({"solid_wedge", "dashed_wedge", "wavy", "dotted", "variable"})
_ORDER_FOR_DEPICTION = {1: "plain", 2: "double", 3: "triple"}
_VISIBLE_ORDER = {"plain": 1, "double": 2, "triple": 3}


@dataclass(frozen=True)
class Anchor:
    """Continuous semantic anchor in normalized image coordinates."""

    x: float
    y: float
    rule: str


@dataclass(frozen=True)
class CoordTarget:
    """Coarse bin plus in-bin offset for a node anchor."""

    bins: int
    x_bin: int
    x_offset: float
    y_bin: int
    y_offset: float


@dataclass(frozen=True)
class VisualLabel:
    """What is actually drawn for a node, run by run."""

    visible: bool
    plain_text: str
    runs: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class Placeholder:
    """Auxiliary geometry of a circle/polygon ring placeholder."""

    shape: str
    radius: float


@dataclass(frozen=True)
class TargetNode:
    id: int
    node_class: str
    variable_subtype: str | None
    anchor: Anchor
    coord_target: CoordTarget
    visual_label: VisualLabel
    semantic_label: dict[str, Any]
    loss_mask: dict[str, bool]
    placeholder: Placeholder | None = None


@dataclass(frozen=True)
class NormalizedEdge:
    order: int
    aromatic: bool
    kekule_order: int | None


@dataclass(frozen=True)
class DepictedEdge:
    type: str
    visible_order: int | None
    ring_mode: str | None


@dataclass(frozen=True)
class TargetEdge:
    source: int
    target: int
    adjacent: bool
    normalized: NormalizedEdge
    depicted: DepictedEdge
    variable: bool
    wedge_direction: str


@dataclass(frozen=True)
class GraphAnnotation:
    kind: str
    nodes: tuple[int, ...]
    depiction: str


@dataclass(frozen=True)
class MarkushGraph:
    image_size: tuple[int, int]
    nodes: tuple[TargetNode, ...] = ()
    edges: tuple[TargetEdge, ...] = ()
    graph_annotations: tuple[GraphAnnotation, ...] = ()
    masks: dict[str, bool] = field(default_factory=dict)
    coord_bins: int = COORD_BINS
    order: str = DEFAULT_ORDER
    format: str = TARGET_FORMAT
    coordinate_system: str = COORDINATE_SYSTEM

    def by_id(self) -> dict[int, TargetNode]:
        return {node.id: node for node in self.nodes}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_int(value: Any, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(message)
    return int(value)


def _as_float(value: Any, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    return float(value)


def decode_coord(entry: dict[str, Any], axis: str, bins: int) -> float:
    """Decode one axis from ``anchor``/``coord_target`` and cross-check them."""
    anchor = entry.get("anchor") or {}
    coord = entry.get("coord_target") or {}
    if not isinstance(anchor, dict) or not isinstance(coord, dict):
        raise ValueError("anchor and coord_target must be objects")
    if coord.get("bins") is not None and int(coord["bins"]) != bins:
        raise ValueError("node coord_target.bins does not match target coord_bins")
    bin_key, offset_key = f"{axis}_bin", f"{axis}_offset"
    decoded: float | None = None
    if coord.get(bin_key) is not None:
        index = _as_int(coord[bin_key], f"{bin_key} must be an integer")
        offset = _as_float(coord.get(offset_key, 0.0), f"{offset_key} must be a number")
        if not 0 <= index < bins:
            raise ValueError(f"target coordinate {bin_key}={index} is outside its bins")
        if not -0.5 <= offset <= 0.5:
            raise ValueError(f"target coordinate {offset_key}={offset} is outside [-0.5, 0.5]")
        decoded = (index + offset) / (bins - 1)
    if anchor.get(axis) is not None:
        value = _as_float(anchor[axis], f"anchor.{axis} must be a number")
        if decoded is not None and abs(value - decoded) > 1e-5:
            raise ValueError(f"anchor.{axis} disagrees with its quantized coordinate")
    elif decoded is None:
        raise ValueError(f"target node is missing anchor.{axis} and {bin_key}")
    else:
        value = decoded
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"target coordinate {axis}={value} is outside [0, 1]")
    return value


def _parse_visual_label(payload: Any) -> VisualLabel:
    _require(isinstance(payload, dict), "visual_label must be an object")
    runs: list[dict[str, str]] = []
    raw_runs = payload.get("runs") or []
    _require(isinstance(raw_runs, list), "visual_label.runs must be a list")
    for run in raw_runs:
        _require(isinstance(run, dict), "visual_label runs must be objects")
        text = run.get("text")
        script = run.get("script")
        _require(isinstance(text, str) and bool(text), "visual label runs require non-empty text")
        _require(script in SCRIPTS, f"unknown label script {script!r}")
        runs.append({"text": text, "script": script})
    plain = "".join(run["text"] for run in runs)
    _require(plain == payload.get("plain_text", ""), "visual_label.plain_text != its runs")
    _require(
        bool(runs) == bool(payload.get("visible", False)),
        "visual_label.visible does not match its runs",
    )
    return VisualLabel(visible=bool(runs), plain_text=plain, runs=tuple(runs))


def _parse_loss_mask(payload: Any) -> dict[str, bool]:
    _require(isinstance(payload, dict), "loss_mask must be an object")
    mask: dict[str, bool] = {}
    for key, value in payload.items():
        _require(isinstance(key, str), "loss_mask keys must be strings")
        _require(isinstance(value, bool), "loss_mask values must be booleans")
        mask[key] = value
    return mask


def _parse_placeholder(payload: Any) -> Placeholder:
    _require(isinstance(payload, dict), "placeholder must be an object")
    shape = payload.get("shape")
    _require(shape in PLACEHOLDER_SHAPES, f"unknown placeholder shape {shape!r}")
    radius = _as_float(payload.get("radius"), "placeholder.radius must be a number")
    _require(radius > 0.0, "placeholder.radius must be positive")
    return Placeholder(shape=shape, radius=radius)


def _parse_node(entry: Any, bins: int) -> TargetNode:
    _require(isinstance(entry, dict), "nodes entries must be objects")
    node_id = _as_int(entry.get("id"), "node id must be an integer")
    node_class = entry.get("node_class")
    _require(node_class in NODE_CLASSES, f"unknown node_class {node_class!r}")
    subtype = entry.get("variable_subtype")
    if node_class == "variable":
        _require(subtype in VARIABLE_SUBTYPES, "variable nodes require a valid variable_subtype")
    else:
        _require(subtype is None, "variable_subtype is only valid for variable nodes")

    anchor_payload = entry.get("anchor")
    _require(isinstance(anchor_payload, dict), "nodes require an anchor object")
    rule = anchor_payload.get("rule")
    _require(rule in ANCHOR_RULES, f"unknown anchor rule {rule!r}")
    x = decode_coord(entry, "x", bins)
    y = decode_coord(entry, "y", bins)
    coord = entry.get("coord_target") or {}
    anchor = Anchor(x=round(x, 6), y=round(y, 6), rule=rule)
    coord_target = CoordTarget(
        bins=bins,
        x_bin=_as_int(coord.get("x_bin"), "coord_target.x_bin must be an integer"),
        x_offset=_as_float(coord.get("x_offset", 0.0), "coord_target.x_offset must be a number"),
        y_bin=_as_int(coord.get("y_bin"), "coord_target.y_bin must be an integer"),
        y_offset=_as_float(coord.get("y_offset", 0.0), "coord_target.y_offset must be a number"),
    )

    visual = _parse_visual_label(entry.get("visual_label"))
    semantic = entry.get("semantic_label") or {}
    _require(isinstance(semantic, dict), "semantic_label must be an object")
    loss_mask = _parse_loss_mask(entry.get("loss_mask") or {})

    if node_class == "ring_placeholder":
        _require(
            bool(str(semantic.get("placeholder_id") or "").strip()),
            "ring placeholders require semantic_label.placeholder_id",
        )
        _require(rule == "placeholder_center", "ring placeholders require a placeholder_center")
    elif node_class == "atom":
        _require(
            bool(str(semantic.get("element") or "").strip()),
            "atom nodes require semantic_label.element",
        )
    elif node_class in ("rgroup", "variable"):
        _require(
            bool(str(semantic.get("family") or "").strip()),
            f"{node_class} nodes require semantic_label.family",
        )
    else:
        _require(
            bool(str(semantic.get("canonical_text") or "").strip()),
            "abbreviation nodes require semantic_label.canonical_text",
        )

    placeholder = None
    if node_class == "ring_placeholder":
        _require("placeholder" in entry, "ring placeholders require a placeholder object")
        placeholder = _parse_placeholder(entry["placeholder"])
    else:
        _require("placeholder" not in entry, "placeholder is only valid on ring_placeholder nodes")

    return TargetNode(
        id=node_id,
        node_class=node_class,
        variable_subtype=subtype,
        anchor=anchor,
        coord_target=coord_target,
        visual_label=visual,
        semantic_label=dict(semantic),
        loss_mask=loss_mask,
        placeholder=placeholder,
    )


def _parse_edge(entry: Any, nodes: dict[int, TargetNode]) -> TargetEdge:
    _require(isinstance(entry, dict), "edges entries must be objects")
    source = _as_int(entry.get("source"), "edge source must be an integer")
    target = _as_int(entry.get("target"), "edge target must be an integer")
    _require(source != target, "prediction target cannot contain self edges")
    _require(source in nodes and target in nodes, f"edge {source}-{target} references unknown node")
    _require(entry.get("adjacent", True) is True, "non-adjacent pairs must not be serialized")

    normalized_payload = entry.get("normalized") or {}
    depicted_payload = entry.get("depicted") or {}
    _require(isinstance(normalized_payload, dict), "normalized must be an object")
    _require(isinstance(depicted_payload, dict), "depicted must be an object")
    _require(
        isinstance(normalized_payload.get("aromatic", False), bool),
        "normalized.aromatic must be a boolean",
    )
    depiction = depicted_payload.get("type")
    _require(depiction in DEPICTED_TYPES, f"unknown depicted edge type {depiction!r}")
    order = _as_int(normalized_payload.get("order", 1), "normalized.order must be an integer")
    _require(order in (1, 2, 3), f"invalid normalized bond order {order}")
    aromatic = bool(normalized_payload.get("aromatic", False))
    if aromatic:
        _require(
            order == 1 and depiction in AROMATIC_DEPICTIONS,
            "aromatic edges have an incompatible normalized/depicted target",
        )
    elif depiction in SPECIAL_DEPICTIONS:
        _require(order == 1, "special edges require normalized order 1")
    elif depiction != _ORDER_FOR_DEPICTION[order]:
        raise ValueError("edge has an incompatible normalized/depicted target")

    kekule = normalized_payload.get("kekule_order")
    if kekule is not None:
        _require(
            aromatic and _as_int(kekule, "kekule_order must be an integer") in (1, 2),
            "kekule_order is only valid on aromatic edges and must be 1 or 2",
        )

    variable = bool(entry.get("variable", False))
    _require(
        not (depiction == "variable" and not variable),
        "depicted variable bonds require variable=true",
    )
    expected_visible = _VISIBLE_ORDER.get(depiction)
    _require(
        depicted_payload.get("visible_order") == expected_visible,
        "depicted.visible_order is inconsistent with depicted.type",
    )
    ring_mode = depicted_payload.get("ring_mode")
    touches_placeholder = (
        nodes[source].node_class == "ring_placeholder"
        or nodes[target].node_class == "ring_placeholder"
    )
    if ring_mode is not None and not touches_placeholder:
        raise ValueError("ring_mode is only valid on an edge touching a ring placeholder")

    direction = entry.get("wedge_direction", "none")
    _require(direction in WEDGE_DIRECTIONS, f"unknown wedge_direction {direction!r}")
    if depiction in ("solid_wedge", "dashed_wedge"):
        _require(direction != "none", "wedge edges require a direction")
    else:
        _require(direction == "none", "wedge_direction is only valid for wedge edges")

    return TargetEdge(
        source=source,
        target=target,
        adjacent=True,
        normalized=NormalizedEdge(
            order=order,
            aromatic=aromatic,
            kekule_order=None if kekule is None else int(kekule),
        ),
        depicted=DepictedEdge(type=depiction, visible_order=expected_visible, ring_mode=ring_mode),
        variable=variable,
        wedge_direction=direction,
    )


def _parse_annotation(entry: Any, nodes: dict[int, TargetNode]) -> GraphAnnotation:
    _require(isinstance(entry, dict), "graph_annotations entries must be objects")
    kind = entry.get("kind")
    _require(kind in ANNOTATION_KINDS, f"unknown annotation kind {kind!r}")
    depiction = entry.get("depiction")
    _require(depiction in ANNOTATION_DEPICTIONS, f"unknown annotation depiction {depiction!r}")
    raw_nodes = entry.get("nodes") or []
    _require(isinstance(raw_nodes, list) and bool(raw_nodes), "annotations require member nodes")
    members: list[int] = []
    for node_id in raw_nodes:
        value = _as_int(node_id, "annotation node ids must be integers")
        _require(value in nodes, f"annotation references unknown node {value}")
        members.append(value)
    _require(len(set(members)) == len(members), "annotation member nodes must be unique")
    return GraphAnnotation(kind=kind, nodes=tuple(members), depiction=depiction)


def parse_target(payload: Any) -> MarkushGraph:
    """Validate ``payload`` and return its typed form, raising on any violation."""
    _require(isinstance(payload, dict), "target must be an object")
    fmt = payload.get("format")
    _require(fmt == TARGET_FORMAT, f"unsupported target format {fmt!r}")
    system = payload.get("coordinate_system")
    _require(system == COORDINATE_SYSTEM, f"unsupported coordinate system {system!r}")
    order = payload.get("order", DEFAULT_ORDER)
    _require(order in ORDERS, f"unsupported target order {order!r}")
    bins = _as_int(payload.get("coord_bins", COORD_BINS), "coord_bins must be an integer")
    _require(bins >= 2, "coord_bins must be at least 2")

    raw_image_size = payload.get("image_size")
    _require(
        isinstance(raw_image_size, list) and len(raw_image_size) == 2,
        "image_size must be a two-element list",
    )
    width = _as_int(raw_image_size[0], "image_size values must be positive integers")
    height = _as_int(raw_image_size[1], "image_size values must be positive integers")
    _require(width > 0 and height > 0, "image_size values must be positive")

    nodes = tuple(_parse_node(entry, bins) for entry in payload.get("nodes", []))
    by_id = {node.id: node for node in nodes}
    _require(len(by_id) == len(nodes), "target node ids must be unique")
    edges = tuple(_parse_edge(entry, by_id) for entry in payload.get("edges", []))
    pairs = [frozenset((edge.source, edge.target)) for edge in edges]
    _require(len(set(pairs)) == len(pairs), "target contains duplicate edges")

    annotations = tuple(
        _parse_annotation(entry, by_id) for entry in payload.get("graph_annotations", [])
    )
    masks_raw = payload.get("masks") or {}
    _require(isinstance(masks_raw, dict), "masks must be an object")
    for key in MASK_KEYS:
        _require(key in masks_raw, f"masks is missing {key!r}")
    masks: dict[str, bool] = {}
    for key, value in masks_raw.items():
        _require(isinstance(value, bool), "mask values must be booleans")
        masks[str(key)] = value

    return MarkushGraph(
        image_size=(width, height),
        nodes=nodes,
        edges=edges,
        graph_annotations=annotations,
        masks=masks,
        coord_bins=bins,
        order=order,
        format=fmt,
        coordinate_system=system,
    )


def validate_target(payload: Any) -> None:
    """Raise :class:`ValueError` when ``payload`` is not a valid target."""
    parse_target(payload)


def to_dict(graph: MarkushGraph) -> dict[str, Any]:
    """Serialise a :class:`MarkushGraph` back to the wire format."""
    nodes: list[dict[str, Any]] = []
    for node in graph.nodes:
        entry: dict[str, Any] = {
            "id": node.id,
            "node_class": node.node_class,
            "variable_subtype": node.variable_subtype,
            "anchor": {"x": node.anchor.x, "y": node.anchor.y, "rule": node.anchor.rule},
            "coord_target": {
                "bins": node.coord_target.bins,
                "x_bin": node.coord_target.x_bin,
                "x_offset": node.coord_target.x_offset,
                "y_bin": node.coord_target.y_bin,
                "y_offset": node.coord_target.y_offset,
            },
            "visual_label": {
                "visible": node.visual_label.visible,
                "plain_text": node.visual_label.plain_text,
                "runs": [dict(run) for run in node.visual_label.runs],
            },
            "semantic_label": dict(node.semantic_label),
            "loss_mask": dict(node.loss_mask),
        }
        if node.placeholder is not None:
            entry["placeholder"] = {
                "shape": node.placeholder.shape,
                "radius": node.placeholder.radius,
            }
        nodes.append(entry)

    edges = [
        {
            "source": edge.source,
            "target": edge.target,
            "adjacent": edge.adjacent,
            "normalized": {
                "order": edge.normalized.order,
                "aromatic": edge.normalized.aromatic,
                "kekule_order": edge.normalized.kekule_order,
            },
            "depicted": {
                "type": edge.depicted.type,
                "visible_order": edge.depicted.visible_order,
                "ring_mode": edge.depicted.ring_mode,
            },
            "variable": edge.variable,
            "wedge_direction": edge.wedge_direction,
        }
        for edge in graph.edges
    ]

    return {
        "format": graph.format,
        "coordinate_system": graph.coordinate_system,
        "image_size": [graph.image_size[0], graph.image_size[1]],
        "coord_bins": graph.coord_bins,
        "order": graph.order,
        "nodes": nodes,
        "edges": edges,
        "graph_annotations": [
            {
                "kind": annotation.kind,
                "nodes": list(annotation.nodes),
                "depiction": annotation.depiction,
            }
            for annotation in graph.graph_annotations
        ],
        "masks": dict(graph.masks),
    }


__all__ = [
    "ANCHOR_RULES",
    "ANNOTATION_DEPICTIONS",
    "ANNOTATION_KINDS",
    "COORDINATE_SYSTEM",
    "COORD_BINS",
    "DEFAULT_ORDER",
    "DEPICTED_TYPES",
    "NODE_CLASSES",
    "PLACEHOLDER_SHAPES",
    "SCRIPTS",
    "TARGET_FORMAT",
    "VARIABLE_SUBTYPES",
    "WEDGE_DIRECTIONS",
    "Anchor",
    "CoordTarget",
    "DepictedEdge",
    "GraphAnnotation",
    "MarkushGraph",
    "NormalizedEdge",
    "Placeholder",
    "TargetEdge",
    "TargetNode",
    "VisualLabel",
    "decode_coord",
    "parse_target",
    "to_dict",
    "validate_target",
]
