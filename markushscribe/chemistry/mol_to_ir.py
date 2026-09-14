"""Turn a plain RDKit molecule into MarkushRender IR.

Stage 1 of M1 trains the recogniser on ordinary OCSR (all atoms explicit, no
R-groups or variables), so the generator needs a straight mol -> IR projection.
Coordinates are deliberately omitted: the renderer lays the structure out with
CoordGen, which is exactly the diversity the model must learn to read.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

from ..markushgen import molecule as molutil
from ..markushgen.ir import IREdge, IRNode, MarkushIR, atom_symbol

_WEDGE_DIRS = {
    Chem.BondDir.BEGINWEDGE: "solid wedge",
    Chem.BondDir.BEGINDASH: "dashed wedge",
}


def _safe_sanitize(mol: Chem.Mol) -> None:
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - keep the molecule as RDKit parsed it
        pass


def mol_to_ir(
    mol: Chem.Mol,
    *,
    source_smiles: str = "",
    source_id: str = "",
    keep_stereo: bool = False,
) -> MarkushIR:
    """Project a plain molecule onto the structural Markush IR subset.

    Carbon and hydrogen counts are captured for every atom, mirroring the
    abstraction generator, so the renderer can decide which hydrogens to draw.
    With ``keep_stereo`` false any wedge/hatch direction is dropped and recorded
    in ``_meta.stereo_dropped`` rather than silently mislabelled.
    """
    fragment = molutil.largest_fragment(mol)
    if fragment is None or fragment.GetNumAtoms() == 0:
        raise ValueError("molecule has no heavy atoms")
    fragment = Chem.Mol(fragment)
    _safe_sanitize(fragment)

    nodes: list[IRNode] = []
    for atom in fragment.GetAtoms():
        h_count = max(0, int(atom.GetTotalNumHs()))
        nodes.append(
            IRNode(
                id=atom.GetIdx(),
                kind="atom",
                symbol=atom_symbol(
                    atom.GetSymbol(),
                    h_count or None,
                    atom.GetFormalCharge(),
                    atom.GetIsotope() or None,
                ),
                element=atom.GetSymbol(),
                h_count=h_count or None,
                charge=atom.GetFormalCharge(),
                isotope=atom.GetIsotope() or None,
                aromatic=atom.GetIsAromatic(),
            )
        )

    stereo_dropped = False
    edges: list[IREdge] = []
    for bond in fragment.GetBonds():
        bond_type = molutil.bond_type(bond)
        if keep_stereo and bond.GetBondDir() in _WEDGE_DIRS:
            bond_type = _WEDGE_DIRS[bond.GetBondDir()]  # type: ignore[assignment]
        elif not keep_stereo and bond.GetBondDir() in _WEDGE_DIRS:
            stereo_dropped = True
        edges.append(
            IREdge(source=bond.GetBeginAtomIdx(), target=bond.GetEndAtomIdx(), bond_type=bond_type)
        )

    canonical = molutil.canonical_smiles(fragment)
    scaffold = molutil.scaffold_smiles(fragment)
    meta: dict[str, Any] = {
        "generator": "markushscribe.chemistry.mol_to_ir",
        "source_smiles": source_smiles or canonical,
        "source_id": source_id,
        "canonical_smiles": canonical,
        "scaffold_smiles": scaffold,
        "node_kind_counts": {"atom": len(nodes)},
        "stereo_dropped": stereo_dropped,
    }
    ir = MarkushIR(nodes=nodes, edges=edges, meta=meta)
    ir.validate()
    return ir


__all__ = ["mol_to_ir"]
