"""Load pretrained weights (e.g. MolScribe) into the M1 model.

The MolScribe vocabulary is incompatible with ours, so only shape-matching
tensors are copied; everything else is reported instead of silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_COMMON_PREFIXES = ("module.", "model.", "encoder.")


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def loaded_fraction(self) -> float:
        total = len(self.loaded) + len(self.missing)
        return len(self.loaded) / total if total else 0.0

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded)} missing={len(self.missing)} "
            f"unexpected={len(self.unexpected)} fraction={self.loaded_fraction:.3f}"
        )


def strip_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _COMMON_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _candidate_keys(key: str, own: dict[str, torch.Tensor]) -> list[str]:
    candidates = [key]
    stripped = strip_prefixes(key)
    if stripped != key:
        candidates.append(stripped)
    candidates.extend(name for name in own if name.endswith("." + stripped))
    seen: list[str] = []
    for name in candidates:
        if name not in seen:
            seen.append(name)
    return seen


def load_weights(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> LoadReport:
    """Copy every shape-matching tensor in ``state`` into ``model``."""
    own = model.state_dict()
    report = LoadReport()
    claimed: set[str] = set()
    for key, value in state.items():
        target_name = None
        for candidate in _candidate_keys(key, own):
            if candidate in claimed:
                continue
            target = own.get(candidate)
            if target is not None and tuple(target.shape) == tuple(value.shape):
                target_name = candidate
                break
        if target_name is None:
            report.unexpected.append(key)
            continue
        own[target_name] = value
        claimed.add(target_name)
        report.loaded.append(target_name)
    report.loaded = sorted(set(report.loaded))
    report.missing = sorted(set(own) - claimed)
    report.unexpected = sorted(report.unexpected)
    model.load_state_dict(own)
    return report


def _extract_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state", "model"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return nested
    if isinstance(payload, dict):
        return payload
    raise ValueError("checkpoint does not contain a state dict")


def load_molscribe_encoder(
    model: torch.nn.Module, path: str | Path, map_location: str = "cpu"
) -> LoadReport:
    """Load a MolScribe checkpoint's matching encoder tensors into ``model``."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return load_weights(model, _extract_state(payload))


def assert_loaded(report: LoadReport, minimum: float = 0.05) -> None:
    if report.loaded_fraction < minimum:
        raise ValueError(f"pretrained load ratio too low: {report.summary()}")


__all__ = [
    "LoadReport",
    "assert_loaded",
    "load_molscribe_encoder",
    "load_weights",
    "strip_prefixes",
]
