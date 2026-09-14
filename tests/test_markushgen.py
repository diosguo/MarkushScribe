"""Tests for the SMILES -> Markush IR generator."""

from __future__ import annotations

import json
import random

import pytest

from markushscribe.markushgen import MarkushConfig, abstract_molecule
from markushscribe.markushgen.cli import main
from markushscribe.markushgen.ir import MarkushIR, atom_symbol
from markushscribe.markushgen.molecule import load_molecule

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Clc1ccc(cc1)C(=O)N1CCN(CC1)c1ncccn1",
    "C1CCCCC1",
    "O=C(O)c1ccccc1O",
]


def _generate(smiles: str, seed: int) -> MarkushIR | None:
    mol = load_molecule(smiles)
    assert mol is not None
    return abstract_molecule(
        mol, MarkushConfig(seed=seed), random.Random(seed), source_smiles=smiles
    )


def test_atom_symbol_roundtrip_shapes() -> None:
    assert atom_symbol("N", 1, 0, None) == "NH"
    assert atom_symbol("C", 3, 0, None) == "CH3"
    assert atom_symbol("C", None, 0, 13) == "13C"
    assert atom_symbol("N", 0, 1, None) == "N+"


@pytest.mark.parametrize("smiles", SMILES)
def test_generates_valid_ir(smiles: str) -> None:
    generated = [_generate(smiles, seed) for seed in range(6)]
    generated = [ir for ir in generated if ir is not None]
    assert generated, f"no IR produced for {smiles}"
    for ir in generated:
        ir.validate()
        kinds = {node.kind for node in ir.nodes}
        assert kinds
        assert all(edge.source != edge.target for edge in ir.edges)
        # every node id must be resolvable and every edge directed at a real node
        ids = {node.id for node in ir.nodes}
        assert {node.id for node in ir.nodes} == set(range(len(ir.nodes)))
        for edge in ir.edges:
            assert edge.source in ids and edge.target in ids


def test_deterministic_for_same_seed() -> None:
    first = _generate(SMILES[0], 7)
    second = _generate(SMILES[0], 7)
    assert first is not None and second is not None
    assert first.to_dict() == second.to_dict()


def test_seed_changes_output() -> None:
    payloads = {
        json.dumps(_generate(SMILES[0], seed).to_dict(), sort_keys=True)
        for seed in range(8)
        if _generate(SMILES[0], seed) is not None
    }
    assert len(payloads) > 1


def test_multiplicity_and_placeholders_can_appear() -> None:
    found_multiplicity = False
    found_placeholder = False
    found_variable = False
    for smiles in SMILES:
        for seed in range(40):
            ir = _generate(smiles, seed)
            if ir is None:
                continue
            found_multiplicity |= any(node.multiplicity for node in ir.nodes)
            found_placeholder |= any(node.variable_type == "ring" for node in ir.nodes)
            found_variable |= any(
                node.variable_type in ("ring_atom", "linker") for node in ir.nodes
            )
    assert found_multiplicity
    assert found_placeholder
    assert found_variable


def test_ir_roundtrip_through_renderer() -> None:
    markushrender = pytest.importorskip("markushrender")
    for smiles in SMILES:
        for seed in range(3):
            ir = _generate(smiles, seed)
            if ir is None:
                continue
            result = markushrender.render(ir.to_dict(), rasterize=False)
            target = markushrender.prediction_target(result)
            assert target["format"] == "markush_graph_v2"
            rebuilt = markushrender.target_to_ir(target)
            assert len(rebuilt.nodes) == len(target["nodes"])


def test_cli_writes_shards(tmp_path) -> None:
    smi = tmp_path / "in.smi"
    smi.write_text(
        "CC(=O)Oc1ccccc1C(=O)O aspirin\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C caffeine\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    code = main(
        [
            "--smi",
            str(smi),
            "--out",
            str(out),
            "--count-per-mol",
            "2",
            "--shard-size",
            "3",
        ]
    )
    assert code == 0
    shards = sorted(out.glob("shard_*.jsonl"))
    lines = [line for shard in shards for line in shard.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4
    for line in lines:
        payload = json.loads(line)
        assert payload["_meta"]["generator"] == "markushscribe.markushgen"
