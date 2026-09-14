"""IR dataclasses and JSON serialisation for ``markush_graph_v2`` sources.

This mirrors the wire contract consumed by MarkushRender (``markushrender.ir``).
The generator owns *what is drawn*, so it only needs to emit the structural
subset: nodes, edges and the ``_meta`` audit block. Variable definitions and
rgroup definitions are intentionally out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NodeKind = Literal["atom", "rgroup", "variable", "abbreviation"]
VarType = Literal["ring_atom", "linker", "ring", "bond", "generic"]
BondType = Literal[
    "single",
    "double",
    "triple",
    "aromatic",
    "solid wedge",
    "dashed wedge",
    "wavy",
    "dotted",
    "variable",
]

IR_VERSION = 1


@dataclass
class IRNode:
    id: int
    kind: NodeKind
    symbol: str
    element: str | None = None
    x: float | None = None
    y: float | None = None
    variable_type: VarType | None = None
    variable_def: dict[str, Any] | None = None
    multiplicity: str | None = None
    charge: int = 0
    h_count: int | None = None
    isotope: int | None = None
    aromatic: bool = False

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "symbol": self.symbol,
        }
        optional = {
            "element": self.element,
            "x": None if self.x is None else round(self.x, 6),
            "y": None if self.y is None else round(self.y, 6),
            "variable_type": self.variable_type,
            "variable_def": self.variable_def,
            "multiplicity": self.multiplicity,
            "charge": self.charge or None,
            "h_count": self.h_count,
            "isotope": self.isotope,
            "aromatic": True if self.aromatic else None,
        }
        raw.update({key: value for key, value in optional.items() if value is not None})
        return raw


@dataclass
class IREdge:
    source: int
    target: int
    bond_type: BondType = "single"
    variable: bool = False
    ring_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "bond_type": self.bond_type,
        }
        if self.variable:
            raw["variable"] = True
        if self.ring_mode is not None:
            raw["ring_mode"] = self.ring_mode
        return raw


@dataclass
class MarkushIR:
    nodes: list[IRNode]
    edges: list[IREdge]
    rgroup_definitions: dict[str, list[str]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def by_id(self) -> dict[int, IRNode]:
        return {node.id: node for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        meta = {**self.meta, "ir_version": IR_VERSION}
        return {
            "_meta": meta,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "rgroup_definitions": self.rgroup_definitions,
        }

    def validate(self) -> None:
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node ids in generated IR")
        known = set(ids)
        for edge in self.edges:
            if edge.source == edge.target:
                raise ValueError("self edge in generated IR")
            if edge.source not in known or edge.target not in known:
                raise ValueError("edge references an unknown node in generated IR")


def atom_symbol(element: str, h_count: int | None, charge: int, isotope: int | None) -> str:
    """Build a label that MarkushRender can parse even without the element field."""
    text = f"{isotope or ''}{element}"
    if h_count:
        text += "H" if h_count == 1 else f"H{h_count}"
    if charge:
        magnitude = abs(charge)
        sign = "+" if charge > 0 else "-"
        text += sign if magnitude == 1 else f"{magnitude}{sign}"
    return text
