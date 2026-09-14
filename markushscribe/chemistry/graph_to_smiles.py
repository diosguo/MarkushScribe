"""Convert a validated ``markush_graph_v2`` target into a SMILES string.

Atoms map onto their element with charge/isotope/H-count preserved. R-groups,
variables, abbreviations and ring placeholders become dummy atoms ``*`` that
carry the numeric part of their family as an atom map (``[*:1]`` for ``R1``) so
a round-trip through :func:`markushscribe.chemistry.mol_to_ir` can recover the
label. Bonds keep their normalized order/aromatic flag; depiction and wedge
direction are dropped because SMILES cannot express them.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

from .. import schema

_BOND_ORDERS = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def _family_index(semantic: dict[str, Any]) -> int:
    family = str(semantic.get("family") or semantic.get("label") or "").strip()
    digits = "".join(char for char in family if char.isdigit())
    return int(digits) if digits else 0


def _dummy_atom(rw_mol: Chem.RWMol, semantic: dict[str, Any]) -> int:
    atom = Chem.Atom(0)
    index = _family_index(semantic)
    if index:
        atom.SetAtomMapNum(index)
    return rw_mol.AddAtom(atom)


def _real_atom(rw_mol: Chem.RWMol, node: schema.TargetNode) -> int:
    semantic = node.semantic_label
    element = str(semantic.get("element") or "").strip()
    if element == "*":
        return _dummy_atom(rw_mol, semantic)
    atom = Chem.Atom(element)
    if semantic.get("charge"):
        atom.SetFormalCharge(int(semantic["charge"]))
    if semantic.get("isotope"):
        atom.SetIsotope(int(semantic["isotope"]))
    if semantic.get("aromatic"):
        atom.SetIsAromatic(True)
    h_count = semantic.get("h_count")
    if h_count is not None:
        atom.SetNumExplicitHs(int(h_count))
        atom.SetNoImplicit(True)
    return rw_mol.AddAtom(atom)


def graph_to_mol(graph: schema.MarkushGraph) -> Chem.Mol | None:
    """Build an un-sanitized RWMol from a target graph (``None`` if incomplete)."""
    rw_mol = Chem.RWMol()
    index_of: dict[int, int] = {}
    for node in graph.nodes:
        if node.node_class == "atom":
            index_of[node.id] = _real_atom(rw_mol, node)
        else:
            index_of[node.id] = _dummy_atom(rw_mol, node.semantic_label)
    for edge in graph.edges:
        source = index_of.get(edge.source)
        target = index_of.get(edge.target)
        if source is None or target is None:
            return None
        if edge.normalized.aromatic:
            bond_type = Chem.BondType.AROMATIC
        else:
            bond_type = _BOND_ORDERS.get(int(edge.normalized.order))
        if bond_type is None:
            return None
        rw_mol.AddBond(source, target, bond_type)
    return rw_mol.GetMol()


def graph_to_smiles(graph: schema.MarkushGraph) -> str | None:
    """Canonicalize a target graph to SMILES, or ``None`` when chemistry fails."""
    from rdkit import RDLogger

    mol = graph_to_mol(graph)
    if mol is None:
        return None
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001 - unparsable graphs simply yield no SMILES
        return None
    finally:
        logger.setLevel(RDLogger.ERROR)


__all__ = ["graph_to_mol", "graph_to_smiles"]
