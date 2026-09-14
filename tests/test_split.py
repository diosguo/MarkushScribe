"""Tests for grouped, leakage-free dataset splitting."""

from __future__ import annotations

import pytest

from markushscribe.split import (
    SPLITS,
    assert_no_leakage,
    assign_splits,
    group_key,
    leakage_report,
    split_records,
)


def _record(source: str, canonical: str, scaffold: str, variant: int = 0) -> dict:
    return {
        "_meta": {
            "source_id": source,
            "canonical_smiles": canonical,
            "scaffold_smiles": scaffold,
            "seed": variant,
        }
    }


def test_group_key_precedence() -> None:
    record = {
        "_meta": {
            "source_id": "s1",
            "canonical_smiles": "CCC",
            "scaffold_smiles": "c1ccccc1",
            "patent_family_id": "fam7",
        }
    }
    assert group_key(record, "auto") == "patent_family:fam7"
    assert group_key(record, "canonical") == "canonical:CCC"
    assert group_key(record, "scaffold") == "scaffold:c1ccccc1"
    assert group_key(record, "source") == "source:s1"


def test_group_key_falls_back_when_meta_missing() -> None:
    assert group_key({"_meta": {"source_id": "s9"}}) == "source:s9"
    assert group_key({"_meta": {"source_smiles": "CCO"}}) == "source:CCO"


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown split key"):
        group_key({"_meta": {"source_id": "s"}}, "nope")


def test_variants_of_one_group_stay_together() -> None:
    records = []
    for molecule in range(20):
        for variant in range(4):
            records.append(_record(f"m{molecule}", f"SMILES{molecule}", f"SCAF{molecule}", variant))
    result = split_records(records, ratios=(0.6, 0.2, 0.2), key="canonical", seed=1)
    seen: dict[str, set[str]] = {}
    for record in records:
        seen.setdefault(group_key(record, "canonical"), set()).add(result.require(record))
    assert all(len(splits) == 1 for splits in seen.values())
    assert leakage_report(records, result) == {}
    assert_no_leakage(records, result)
    assert sum(result.counts.values()) == len(records)


def test_split_is_deterministic_for_seed() -> None:
    records = [_record(f"m{i}", f"SMILES{i}", f"SCAF{i}") for i in range(50)]
    first = split_records(records, key="scaffold", seed=7)
    second = split_records(records, key="scaffold", seed=7)
    assert first.group_to_split == second.group_to_split
    different = split_records(records, key="scaffold", seed=8)
    assert different.group_to_split != first.group_to_split


def test_ratios_are_approximately_honoured() -> None:
    sizes = {f"g{i}": 1 for i in range(400)}
    assignment = assign_splits(sizes, ratios=(0.8, 0.1, 0.1), seed=0)
    counts = {name: 0 for name in SPLITS}
    for split_name in assignment.values():
        counts[split_name] += 1
    assert counts["train"] == pytest.approx(320, abs=10)
    assert counts["valid"] == pytest.approx(40, abs=10)
    assert counts["test"] == pytest.approx(40, abs=10)


def test_unseen_group_returns_none_and_require_raises() -> None:
    records = [_record("m1", "A", "S1")]
    result = split_records(records, key="scaffold", seed=0)
    stranger = _record("m2", "B", "S2")
    assert result.split_for(stranger) is None
    with pytest.raises(ValueError, match="not seen"):
        result.require(stranger)


def test_leakage_report_detects_foreign_assignment() -> None:
    records = [_record(f"m{i}", f"SMILES{i}", "SHARED") for i in range(4)]
    result = split_records(records, key="scaffold", seed=0)
    assert_no_leakage(records, result)

    def leaky(record: dict) -> str:
        return "test" if record["_meta"]["source_id"] == "m0" else "train"

    assert leakage_report(records, leaky, key="scaffold") == {"scaffold:SHARED": ["test", "train"]}
    with pytest.raises(ValueError, match="split leakage"):
        assert_no_leakage(records, leaky, key="scaffold")


@pytest.mark.parametrize("ratios", [(1.0, 0.0), (0.5, 0.4, 0.2), (0.5, 0.6, -0.1)])
def test_bad_ratios_raise(ratios) -> None:
    with pytest.raises(ValueError):
        split_records([_record("m", "A", "S")], ratios=ratios, seed=0)


def test_empty_records_raise() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        split_records([], seed=0)
