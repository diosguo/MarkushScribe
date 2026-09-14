"""Tests for the command-line tools that wrap the library pipeline."""

from __future__ import annotations

import json

import pytest

from tools.check_split_leakage import main as leakage_main
from tools.check_split_leakage import parse_split_specs
from tools.fetch_chembl_sample import smiles_of


def _record(source: str, scaffold: str) -> dict:
    return {"_meta": {"source_id": source, "scaffold_smiles": scaffold}}


def _write(path, records) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_parse_split_specs_accepts_named_paths() -> None:
    assert parse_split_specs(["train=a.jsonl", "test=b.jsonl"]) == {
        "train": "a.jsonl",
        "test": "b.jsonl",
    }


@pytest.mark.parametrize("spec", ["train", "=x.jsonl", "holdout=x.jsonl"])
def test_parse_split_specs_rejects_bad_input(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_split_specs([spec])


def test_leakage_cli_clean_and_leaky(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    clean = tmp_path / "clean.jsonl"
    leaky = tmp_path / "leaky.jsonl"
    _write(train, [_record("a", "S1"), _record("b", "S1")])
    _write(clean, [_record("c", "S2")])
    _write(leaky, [_record("d", "S1")])

    assert leakage_main([f"--split=train={train}", f"--split=test={clean}", "--key=scaffold"]) == 0
    assert leakage_main([f"--split=train={train}", f"--split=test={leaky}", "--key=scaffold"]) == 1


def test_smiles_of_handles_missing_structures() -> None:
    molecule = {"molecule_structures": {"canonical_smiles": "CCO"}}
    assert smiles_of(molecule) == "CCO"
    assert smiles_of({"molecule_structures": None}) is None
    assert smiles_of({"molecule_structures": {}}) is None
    assert smiles_of({}) is None
