"""Round-trip tests for graph -> SMILES export."""

from __future__ import annotations

from rdkit import Chem

from markushscribe import schema
from markushscribe.chemistry import graph_to_smiles, mol_to_ir
from markushscribe.dataset import RenderConfig, render_sample

_PLAIN = ["c1ccccc1", "CCO", "CC(=O)O", "C1CCCCC1", "OC(=O)c1ccccc1", "CCN(CC)CC"]


def test_graph_to_smiles_roundtrips_plain_molecules():
    for smiles in _PLAIN:
        record = mol_to_ir(
            Chem.MolFromSmiles(smiles), source_smiles=smiles, source_id="s"
        ).to_dict()
        _result, _image, target = render_sample(record, 0, RenderConfig(rotate=False))
        graph = schema.parse_target(target)
        expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        assert graph_to_smiles(graph) == expected, smiles


def test_graph_to_smiles_returns_none_on_empty_graph():
    graph = schema.MarkushGraph(image_size=(10, 10))
    assert graph_to_smiles(graph) == ""
