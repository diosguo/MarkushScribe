"""Tests for the ``markush_graph_v2`` schema and validator."""

from __future__ import annotations

import copy
import re

import pytest

from markushscribe import schema

BINS = 64


def _coord(value: float, bins: int = BINS) -> tuple[int, float]:
    scaled = value * (bins - 1)
    index = round(scaled)
    return index, round(scaled - index, 6)


def _atom_node(node_id: int, x: float, y: float, element: str = "C") -> dict:
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
            "plain_text": element,
            "runs": [{"text": element, "script": "base"}],
        },
        "semantic_label": {"element": element, "h_count": None, "charge": 0, "isotope": None},
        "loss_mask": {"h_count": False},
    }


def _placeholder_node(node_id: int, x: float, y: float, symbol: str = "A") -> dict:
    node = _atom_node(node_id, x, y, symbol)
    node["node_class"] = "ring_placeholder"
    node["anchor"]["rule"] = "placeholder_center"
    node["semantic_label"] = {"placeholder_id": symbol}
    node["placeholder"] = {"shape": "circle", "radius": 0.89}
    return node


def _edge(source: int, target: int, **overrides) -> dict:
    edge = {
        "source": source,
        "target": target,
        "adjacent": True,
        "normalized": {"order": 1, "aromatic": False, "kekule_order": None},
        "depicted": {"type": "plain", "visible_order": 1, "ring_mode": None},
        "variable": False,
        "wedge_direction": "none",
    }
    edge.update(overrides)
    return edge


def valid_target() -> dict:
    return {
        "format": schema.TARGET_FORMAT,
        "coordinate_system": schema.COORDINATE_SYSTEM,
        "image_size": [320, 200],
        "coord_bins": BINS,
        "order": schema.DEFAULT_ORDER,
        "nodes": [_atom_node(0, 0.25, 0.5, "N"), _placeholder_node(1, 0.75, 0.5)],
        "edges": [_edge(0, 1)],
        "graph_annotations": [],
        "masks": {"cxsmiles": False, "normalized_chemistry_complete": True},
    }


def test_parse_and_roundtrip() -> None:
    payload = valid_target()
    graph = schema.parse_target(payload)
    assert [node.id for node in graph.nodes] == [0, 1]
    assert graph.nodes[1].placeholder is not None
    assert graph.nodes[1].placeholder.shape == "circle"
    assert schema.to_dict(graph) == payload
    schema.validate_target(schema.to_dict(graph))


def test_empty_graph_is_valid() -> None:
    payload = valid_target()
    payload["nodes"] = []
    payload["edges"] = []
    graph = schema.parse_target(payload)
    assert graph.nodes == ()
    assert graph.edges == ()


def test_single_node_is_valid() -> None:
    payload = valid_target()
    payload["nodes"] = [_atom_node(0, 0.5, 0.5, "O")]
    payload["edges"] = []
    assert len(schema.parse_target(payload).nodes) == 1


def test_default_order_and_bins_are_applied() -> None:
    payload = valid_target()
    del payload["order"]
    del payload["coord_bins"]
    payload["nodes"][0]["coord_target"].pop("bins")
    graph = schema.parse_target(payload)
    assert graph.order == schema.DEFAULT_ORDER
    assert graph.coord_bins == schema.COORD_BINS


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p.update(format="v1"), "unsupported target format"),
        (lambda p: p.update(coordinate_system="pixels"), "unsupported coordinate system"),
        (lambda p: p.update(order="bfs"), "unsupported target order"),
        (lambda p: p.update(coord_bins=1), "coord_bins must be at least 2"),
        (lambda p: p.update(image_size=[320]), "image_size must be a two-element list"),
        (lambda p: p.update(image_size=[0, 200]), "image_size values must be positive"),
        (lambda p: p["nodes"].append(_atom_node(0, 0.1, 0.1)), "node ids must be unique"),
        (lambda p: p["nodes"][0].update(node_class="mystery"), "unknown node_class"),
        (lambda p: p["nodes"][0].update(variable_subtype="linker"), "only valid for variable"),
        (lambda p: p["nodes"][0]["anchor"].update(rule="mystery"), "unknown anchor rule"),
        (lambda p: p["nodes"][0]["coord_target"].update(x_offset=0.9), "outside [-0.5, 0.5]"),
        (lambda p: p["nodes"][0]["coord_target"].update(x_bin=999), "outside its bins"),
        (lambda p: p["nodes"][0]["anchor"].update(x=0.9), "disagrees with its quantized"),
        (lambda p: p["nodes"][0]["visual_label"].update(plain_text="X"), "plain_text != its runs"),
        (
            lambda p: p["nodes"][0]["visual_label"]["runs"].append({"text": "2", "script": "x"}),
            "unknown label script",
        ),
        (lambda p: p["nodes"][0]["loss_mask"].update(h_count=1), "must be booleans"),
        (
            lambda p: p["nodes"][1].pop("placeholder"),
            "ring placeholders require a placeholder",
        ),
        (
            lambda p: p["nodes"][0].update(placeholder={"shape": "circle", "radius": 1.0}),
            "only valid on ring_placeholder",
        ),
        (
            lambda p: p["nodes"][0]["semantic_label"].update(element=""),
            "require semantic_label.element",
        ),
        (lambda p: p["edges"][0].update(source=99), "references unknown node"),
        (lambda p: p["edges"][0].update(source=1, target=1), "self edges"),
        (lambda p: p["edges"].append(_edge(0, 1)), "duplicate edges"),
        (
            lambda p: p["edges"][0]["normalized"].update(aromatic=True, order=2),
            "incompatible normalized/depicted",
        ),
        (
            lambda p: p["edges"][0]["normalized"].update(kekule_order=2),
            "kekule_order is only valid",
        ),
        (
            lambda p: p["edges"][0].update(
                depicted={"type": "solid_wedge", "visible_order": None, "ring_mode": None}
            ),
            "wedge edges require a direction",
        ),
        (
            lambda p: (
                p["nodes"].append(_atom_node(2, 0.4, 0.2, "C")),
                p["edges"].append(
                    _edge(
                        0,
                        2,
                        depicted={"type": "plain", "visible_order": 1, "ring_mode": "cross"},
                    )
                ),
            ),
            "ring_mode is only valid",
        ),
        (lambda p: p.update(masks={"cxsmiles": False}), "masks is missing"),
        (
            lambda p: p.update(
                graph_annotations=[{"kind": "x", "nodes": [0, 1], "depiction": "circle"}]
            ),
            "unknown annotation kind",
        ),
        (
            lambda p: p.update(
                graph_annotations=[{"kind": "aromatic_ring", "nodes": [7], "depiction": "circle"}]
            ),
            "unknown node",
        ),
    ],
)
def test_invalid_targets_raise(mutate, message: str) -> None:
    payload = valid_target()
    mutate(payload)
    with pytest.raises(ValueError, match=re.escape(message)):
        schema.parse_target(payload)


def test_aromatic_kekule_edge_parses() -> None:
    payload = valid_target()
    payload["edges"][0]["normalized"] = {"order": 1, "aromatic": True, "kekule_order": 2}
    payload["edges"][0]["depicted"] = {"type": "double", "visible_order": 2, "ring_mode": None}
    graph = schema.parse_target(payload)
    assert graph.edges[0].normalized.kekule_order == 2


def test_annotations_roundtrip() -> None:
    payload = valid_target()
    payload["graph_annotations"] = [
        {"kind": "aromatic_ring", "nodes": [0, 1], "depiction": "companion"}
    ]
    graph = schema.parse_target(payload)
    assert graph.graph_annotations[0].nodes == (0, 1)
    assert schema.to_dict(graph)["graph_annotations"] == payload["graph_annotations"]


def test_real_renderer_target_roundtrips() -> None:
    markushrender = pytest.importorskip("markushrender")
    import random

    from markushscribe.markushgen import MarkushConfig, abstract_molecule
    from markushscribe.markushgen.molecule import load_molecule

    mol = load_molecule("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None
    ir = abstract_molecule(mol, MarkushConfig(seed=3), random.Random(3), source_smiles="x")
    assert ir is not None
    result = markushrender.render(ir.to_dict(), rasterize=False)
    target = markushrender.prediction_target(result)
    graph = schema.parse_target(target)
    assert schema.to_dict(graph) == target


def test_deepcopy_isolation() -> None:
    first = valid_target()
    second = copy.deepcopy(first)
    second["nodes"][0]["semantic_label"]["element"] = "O"
    assert first["nodes"][0]["semantic_label"]["element"] == "N"
