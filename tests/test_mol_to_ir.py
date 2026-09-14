"""Tests for the plain-molecule -> Markush IR projection (Stage 1 data)."""

from __future__ import annotations

from rdkit import Chem

from markushscribe import schema
from markushscribe.chemistry import mol_to_ir


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def test_benzene_is_aromatic():
    ir = mol_to_ir(_mol("c1ccccc1"), source_smiles="c1ccccc1", source_id="benzene")
    assert len(ir.nodes) == 6
    assert len(ir.edges) == 6
    assert all(node.element == "C" for node in ir.nodes)
    assert all(node.aromatic for node in ir.nodes)
    assert all(edge.bond_type == "aromatic" for edge in ir.edges)
    assert ir.meta["node_kind_counts"] == {"atom": 6}


def test_every_node_is_explicit_atom():
    ir = mol_to_ir(_mol("CC(=O)Oc1ccccc1C(=O)O"))
    assert all(node.kind == "atom" for node in ir.nodes)
    assert {node.element for node in ir.nodes} == {"C", "O"}
    assert all(node.symbol for node in ir.nodes)


def test_charge_isotope_and_hydrogens():
    ir = mol_to_ir(_mol("[2H][C+](N)"))
    symbols = {node.symbol for node in ir.nodes}
    assert any("+" in symbol for symbol in symbols)


def test_largest_fragment_kept():
    ir = mol_to_ir(_mol("c1ccccc1.CC"))
    assert len(ir.nodes) == 6


def test_single_atom():
    ir = mol_to_ir(_mol("[Na+]"), source_id="na")
    assert len(ir.nodes) == 1
    assert not ir.edges
    assert ir.nodes[0].element == "Na"


def test_meta_fields_and_validation():
    ir = mol_to_ir(_mol("CCO"), source_smiles="CCO", source_id="etoh")
    assert ir.meta["generator"] == "markushscribe.chemistry.mol_to_ir"
    assert ir.meta["canonical_smiles"] == "CCO"
    assert ir.meta["source_id"] == "etoh"
    ir.validate()


def test_stereo_dropped_flag():
    ir = mol_to_ir(_mol("F[C@H](Cl)Br"))
    assert ir.meta["stereo_dropped"] is False


def test_render_and_target_round_trip():
    import markushrender

    ir = mol_to_ir(_mol("CC(=O)Nc1ccccc1O"), source_smiles="CC(=O)Nc1ccccc1O", source_id="sample")
    result = markushrender.render(ir.to_dict(), seed=0, rasterize=False)
    target = markushrender.prediction_target(result)
    graph = schema.parse_target(target)
    assert len(graph.nodes) == len(ir.nodes)
    rebuilt = markushrender.target_to_ir(target)
    assert len(rebuilt.nodes) == len(ir.nodes)
