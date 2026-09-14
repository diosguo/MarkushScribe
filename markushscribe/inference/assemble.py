"""Assemble decoded nodes (and optional edge predictions) into a Markush graph."""

from __future__ import annotations

import re
from typing import Any

import torch

from .. import constants, schema
from ..tokenizer import DecodedNode

_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")
_CHARGE_RE = re.compile(r"^(\d*)([+-])$")


def _runs_text(node: DecodedNode, script: str) -> str:
    return "".join(run["text"] for run in node.runs if run["script"] == script)


def _parse_charge(sup: str) -> int:
    match = _CHARGE_RE.match(sup.strip())
    if not match:
        return 0
    magnitude = int(match.group(1) or 1)
    return magnitude if match.group(2) == "+" else -magnitude


def parse_semantic(kind: str, runs: list[dict[str, str]]) -> dict[str, Any]:
    """Best-effort semantic label parsed from the visual runs."""
    node = DecodedNode(kind=kind, subtype=None, x_bin=0, y_bin=0, runs=runs)
    base = _runs_text(node, "base")
    sub = _runs_text(node, "sub")
    sup = _runs_text(node, "sup")
    plain = "".join(run["text"] for run in runs)
    if kind == "ring_placeholder":
        return {"placeholder_id": plain.strip()}
    if kind == "abbreviation":
        return {"canonical_text": plain.strip()}
    if kind in ("rgroup", "variable"):
        match = re.match(r"^\(?([A-Za-z]+)(\d*)\)?(?:(\d+))?$", plain.strip())
        if match:
            family, index, multiplicity = match.group(1), match.group(2), match.group(3)
        else:
            family, index, multiplicity = base, sub or None, None
        return {
            "family": family,
            "index": index or None,
            "multiplicity": multiplicity or None,
        }
    element_match = _ELEMENT_RE.match(base)
    element = element_match.group(1) if element_match else base or "C"
    h_match = re.search(r"H(\d*)", base)
    h_count = None
    if h_match:
        h_count = int(h_match.group(1) or 1)
    isotope = None
    if sup:
        digits = re.match(r"^(\d+)", sup)
        if digits:
            isotope = int(digits.group(1))
    return {
        "element": element,
        "h_count": h_count,
        "charge": _parse_charge(sup),
        "isotope": isotope,
        "aromatic": False,
    }


def _anchor_rule(kind: str, visible: bool) -> str:
    if kind == "ring_placeholder":
        return "placeholder_center"
    if not visible:
        return "bond_vertex"
    return "base_core"


def assemble_graph(
    nodes: list[DecodedNode],
    *,
    image_size: tuple[int, int] = (1, 1),
    adjacency: torch.Tensor | None = None,
    order: torch.Tensor | None = None,
    aromatic: torch.Tensor | None = None,
    depiction: torch.Tensor | None = None,
    variable: torch.Tensor | None = None,
    wedge: torch.Tensor | None = None,
    threshold: float = 0.5,
    bins: int = constants.COORD_BINS,
) -> schema.MarkushGraph:
    """Build a :class:`schema.MarkushGraph` from decoded nodes and edge logits."""
    target_nodes: list[schema.TargetNode] = []
    for index, node in enumerate(nodes):
        x_bin = node.x_bin if 0 <= node.x_bin < bins else 0
        y_bin = node.y_bin if 0 <= node.y_bin < bins else 0
        runs = tuple(dict(run) for run in node.runs)
        visual = schema.VisualLabel(
            visible=bool(runs), plain_text="".join(run["text"] for run in runs), runs=runs
        )
        placeholder = None
        if node.kind == "ring_placeholder":
            placeholder = schema.Placeholder(shape="circle", radius=0.1)
        target_nodes.append(
            schema.TargetNode(
                id=index,
                node_class=node.kind,
                variable_subtype=node.subtype if node.kind == "variable" else None,
                anchor=schema.Anchor(
                    x=x_bin / (bins - 1),
                    y=y_bin / (bins - 1),
                    rule=_anchor_rule(node.kind, visual.visible),
                ),
                coord_target=schema.CoordTarget(
                    bins=bins, x_bin=x_bin, x_offset=0.0, y_bin=y_bin, y_offset=0.0
                ),
                visual_label=visual,
                semantic_label=parse_semantic(node.kind, list(runs)),
                loss_mask={},
                placeholder=placeholder,
            )
        )

    target_edges: list[schema.TargetEdge] = []
    if adjacency is not None:
        target_edges = _assemble_edges(
            adjacency, order, aromatic, depiction, variable, wedge, threshold
        )
    return schema.MarkushGraph(
        image_size=image_size,
        nodes=tuple(target_nodes),
        edges=tuple(target_edges),
        masks={"cxsmiles": False, "normalized_chemistry_complete": False},
    )


def _class_index(tensor: torch.Tensor, i: int, j: int) -> int:
    if tensor.dim() == 3:
        return int(tensor[i, j].argmax().item())
    return int(tensor[i, j])


def _assemble_edges(
    adjacency: torch.Tensor,
    order: torch.Tensor | None,
    aromatic: torch.Tensor | None,
    depiction: torch.Tensor | None,
    variable: torch.Tensor | None,
    wedge: torch.Tensor | None,
    threshold: float,
) -> list[schema.TargetEdge]:
    if adjacency.dim() == 4 and adjacency.shape[0] == 1:
        adjacency = adjacency[0]
    if adjacency.dim() == 3 and adjacency.shape[-1] == 2:
        probabilities = torch.softmax(adjacency, dim=-1)[..., 1]
    elif adjacency.dim() == 2:
        probabilities = adjacency
    else:
        raise ValueError(f"unsupported adjacency shape {tuple(adjacency.shape)}")
    count = probabilities.shape[0]
    edges: list[schema.TargetEdge] = []
    for i in range(count):
        for j in range(i + 1, count):
            if float(probabilities[i, j]) < threshold:
                continue
            is_aromatic = aromatic is not None and _class_index(aromatic, i, j) == 1
            depiction_type = "plain"
            if depiction is not None:
                index = max(
                    0, min(_class_index(depiction, i, j), len(constants.DEPICTED_TYPES) - 1)
                )
                depiction_type = constants.DEPICTED_TYPES[index]
            if is_aromatic:
                edge_order = 1
                if depiction_type not in schema.AROMATIC_DEPICTIONS:
                    depiction_type = "plain"
            else:
                edge_order = _class_index(order, i, j) + 1 if order is not None else 1
                if depiction_type not in schema.SPECIAL_DEPICTIONS:
                    depiction_type = {1: "plain", 2: "double", 3: "triple"}.get(edge_order, "plain")
            direction = "none"
            if wedge is not None and depiction_type in ("solid_wedge", "dashed_wedge"):
                index = _class_index(wedge, i, j)
                if index == 1:
                    direction = "source_to_target"
                elif index == 2:
                    direction = "target_to_source"
            is_variable = variable is not None and _class_index(variable, i, j) == 1
            edges.append(
                schema.TargetEdge(
                    source=i,
                    target=j,
                    adjacent=True,
                    normalized=schema.NormalizedEdge(
                        order=edge_order, aromatic=is_aromatic, kekule_order=None
                    ),
                    depicted=schema.DepictedEdge(
                        type=depiction_type,
                        visible_order={"plain": 1, "double": 2, "triple": 3}.get(depiction_type),
                        ring_mode=None,
                    ),
                    variable=is_variable,
                    wedge_direction=direction,
                )
            )
    return edges


__all__ = ["assemble_graph", "parse_semantic"]
