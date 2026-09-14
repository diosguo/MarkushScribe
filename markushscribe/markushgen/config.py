"""Configuration for stochastic Markush abstraction.

Every probability is a knob on a single well-defined transformation, so a
dataset run can be described by one immutable object plus a seed. The defaults
favour a balanced mix of terminal R groups, whole-branch R groups, ring-atom
(M) variables, linker (L) variables, circle placeholder rings (A/B) and
repeated `(R)n` groups.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

VALID_CORE_POLICIES = ("murcko", "random_bfs")


@dataclass(frozen=True)
class MarkushConfig:
    seed: int = 0

    # Core selection.
    core_policy: str = "random_bfs"
    core_retention: float = 0.6
    min_core_atoms: int = 3

    # Branch abstraction. `keep_length_choices` is the number of branch atoms
    # (counted from the attachment) left in the scaffold before an R cap is
    # placed; 0 replaces the whole branch, >0 keeps a terminal stub.
    whole_branch_prob: float = 0.5
    keep_length_choices: tuple[int, ...] = (0, 0, 1, 2)

    # Variable abstraction.
    p_ring_atom_m: float = 0.25
    p_linker_l: float = 0.20
    p_ring_placeholder: float = 0.15
    max_m_vars: int = 2
    max_l_vars: int = 2
    max_placeholders: int = 2

    # Repeating groups and variable bonds.
    p_multiplicity: float = 0.20
    p_repeat_chain: float = 0.15
    p_variable_bond: float = 0.15

    max_rgroups: int = 12
    max_abstraction_ratio: float = 0.80

    r_families: tuple[str, ...] = ("R",)
    m_families: tuple[str, ...] = ("M",)
    l_families: tuple[str, ...] = ("L",)
    placeholder_ids: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    multiplicity_choices: tuple[str, ...] = ("m", "n", "x", "y", "p", "q")
    repeat_families: tuple[str, ...] = ("R", "L")

    def replace(self, **changes: object) -> MarkushConfig:
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]
