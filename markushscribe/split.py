"""Group-aware train/valid/test splitting for Markush source records.

Splitting must happen *before* rendering: every abstraction variant of one
molecule, and ideally every molecule sharing a scaffold, has to land in a single
split or the evaluation set leaks. This module assigns whole groups to splits
with a deterministic greedy fill, so the result depends only on the records and
the seed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SPLITS: tuple[str, str, str] = ("train", "valid", "test")

_GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "patent_family": ("patent_family_id", "family_id"),
    "canonical": ("canonical_structure_id", "canonical_smiles"),
    "scaffold": ("scaffold_id", "scaffold_smiles"),
    "source": ("source_id", "source_record_id"),
}
_GROUP_PRECEDENCE: tuple[str, ...] = ("patent_family", "canonical", "scaffold", "source")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def meta_of(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``_meta`` block of an IR record (or the record itself)."""
    meta = record.get("_meta")
    return meta if isinstance(meta, dict) else dict(record)


def group_key(record: Mapping[str, Any], key: str = "auto") -> str:
    """Stable group identifier for ``record`` under the requested key policy."""
    meta = meta_of(record)
    if key == "auto":
        for candidate in _GROUP_PRECEDENCE:
            value = _first_value(meta, _GROUP_FIELDS[candidate])
            if value:
                return f"{candidate}:{value}"
        return _fallback_key(meta)
    _require(key in _GROUP_FIELDS, f"unknown split key {key!r}")
    value = _first_value(meta, _GROUP_FIELDS[key])
    if value:
        return f"{key}:{value}"
    return _fallback_key(meta)


def _first_value(meta: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for name in fields:
        value = meta.get(name)
        if value:
            return str(value)
    return None


def _fallback_key(meta: Mapping[str, Any]) -> str:
    for name in ("source_smiles", "source_id"):
        value = meta.get(name)
        if value:
            return f"source:{value}"
    return "source:unknown"


@dataclass(frozen=True)
class SplitResult:
    """A frozen group -> split assignment plus its lookup helpers."""

    group_to_split: dict[str, str]
    key: str = "auto"
    ratios: tuple[float, float, float] = (0.98, 0.01, 0.01)
    counts: dict[str, int] = field(default_factory=dict)

    def split_for(self, record: Mapping[str, Any]) -> str | None:
        return self.group_to_split.get(group_key(record, self.key))

    def require(self, record: Mapping[str, Any]) -> str:
        split = self.split_for(record)
        _require(split is not None, "record group was not seen during splitting")
        return split  # type: ignore[return-value]


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    _require(len(ratios) == 3, "ratios must contain train, valid and test")
    values = tuple(float(value) for value in ratios)
    _require(all(value >= 0.0 for value in values), "ratios must be non-negative")
    _require(abs(sum(values) - 1.0) < 1e-6, "ratios must sum to 1")
    return values  # type: ignore[return-value]


def assign_splits(
    group_sizes: Mapping[str, int],
    *,
    ratios: Sequence[float] = (0.98, 0.01, 0.01),
    seed: int = 0,
) -> dict[str, str]:
    """Assign each group to a split, filling the largest deficit first.

    Groups are shuffled for seed reproducibility and then considered largest
    first, so a split's population tracks ``ratios`` while no group is ever cut.
    """
    values = _validate_ratios(ratios)
    _require(bool(group_sizes), "group_sizes must not be empty")
    rng = random.Random(seed)
    groups = list(group_sizes)
    rng.shuffle(groups)
    groups.sort(key=lambda name: -int(group_sizes[name]))

    total = sum(int(size) for size in group_sizes.values())
    targets = {name: ratio * total for name, ratio in zip(SPLITS, values, strict=True)}
    assigned = dict.fromkeys(SPLITS, 0)
    result: dict[str, str] = {}
    for name in groups:
        split = min(
            SPLITS, key=lambda candidate: (assigned[candidate] - targets[candidate], candidate)
        )
        result[name] = split
        assigned[split] += int(group_sizes[name])
    return result


def split_records(
    records: Iterable[Mapping[str, Any]],
    *,
    ratios: Sequence[float] = (0.98, 0.01, 0.01),
    key: str = "auto",
    seed: int = 0,
) -> SplitResult:
    """Split an iterable of IR records (or ``_meta`` dicts) into three sets."""
    materialized = [meta_of(record) for record in records]
    _require(bool(materialized), "records must not be empty")
    sizes: dict[str, int] = {}
    for meta in materialized:
        group = group_key(meta, key)
        sizes[group] = sizes.get(group, 0) + 1
    group_to_split = assign_splits(sizes, ratios=ratios, seed=seed)
    counts = {name: 0 for name in SPLITS}
    for group, size in sizes.items():
        counts[group_to_split[group]] += size
    return SplitResult(
        group_to_split=group_to_split,
        key=key,
        ratios=_validate_ratios(ratios),
        counts=counts,
    )


def leakage_report(
    records: Iterable[Mapping[str, Any]],
    split_of: SplitResult | Mapping[str, str] | Callable[[Mapping[str, Any]], str | None],
    key: str = "auto",
) -> dict[str, list[str]]:
    """Groups found in more than one split; empty means leakage-free.

    ``split_of`` may be a :class:`SplitResult`, a ``group -> split`` mapping, or
    a ``record -> split`` callable. Passing an explicit assignment is the point:
    a record-level splitter that contradicts itself is exactly what a leak is.
    """
    if isinstance(split_of, SplitResult):
        mapping: Mapping[str, str] | None = split_of.group_to_split
        key = split_of.key
        lookup = None
    elif callable(split_of):
        mapping, lookup = None, split_of
    else:
        mapping, lookup = split_of, None

    seen: dict[str, set[str]] = {}
    for record in records:
        group = group_key(record, key)
        if lookup is not None:
            split = lookup(record)
        else:
            split = (mapping or {}).get(group)
        if split is None:
            continue
        seen.setdefault(group, set()).add(split)
    return {group: sorted(splits) for group, splits in seen.items() if len(splits) > 1}


def assert_no_leakage(
    records: Iterable[Mapping[str, Any]],
    split_of: SplitResult | Mapping[str, str] | Callable[[Mapping[str, Any]], str | None],
    key: str = "auto",
) -> None:
    """Raise when any group spans more than one split."""
    leaks = leakage_report(records, split_of, key)
    _require(not leaks, f"split leakage detected for {len(leaks)} group(s): {sorted(leaks)[:5]}")


__all__ = [
    "SPLITS",
    "SplitResult",
    "assert_no_leakage",
    "assign_splits",
    "group_key",
    "leakage_report",
    "meta_of",
    "split_records",
]
