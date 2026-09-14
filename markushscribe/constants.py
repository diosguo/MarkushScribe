"""Shared vocabulary, class tables and bucket mappings for the M1 baseline.

Everything the tokenizer, dataset and model heads must agree on lives here so
the wire contract of :mod:`markushscribe.schema` stays the single source of
truth for what a target *is*, while this module owns how it is *parameterised*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
NODE_START = "<node>"
NODE_END = "</node>"
LABEL_START = "<label>"
LABEL_END = "</label>"

SPECIAL_TOKENS: tuple[str, ...] = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    NODE_START,
    NODE_END,
    LABEL_START,
    LABEL_END,
)

SCRIPT_TOKENS: tuple[str, ...] = ("<base>", "<sub>", "<sup>")
SCRIPT_NAMES: tuple[str, ...] = ("base", "sub", "sup")

KIND_CLASSES: tuple[str, ...] = (
    "atom",
    "rgroup",
    "variable",
    "abbreviation",
    "ring_placeholder",
)
KIND_TOKENS: tuple[str, ...] = tuple(f"<kind_{name}>" for name in KIND_CLASSES)

VARIABLE_SUBTYPES: tuple[str, ...] = ("ring_atom", "linker", "bond", "generic")
SUBTYPE_NONE = "<subtype_none>"
SUBTYPE_TOKENS: tuple[str, ...] = (SUBTYPE_NONE,) + tuple(
    f"<subtype_{name}>" for name in VARIABLE_SUBTYPES
)

COORD_BINS = 64
XBIN_TOKENS: tuple[str, ...] = tuple(f"<xbin_{index}>" for index in range(COORD_BINS))
YBIN_TOKENS: tuple[str, ...] = tuple(f"<ybin_{index}>" for index in range(COORD_BINS))

BASE_TOKENS: tuple[str, ...] = (
    SPECIAL_TOKENS + SCRIPT_TOKENS + KIND_TOKENS + SUBTYPE_TOKENS + XBIN_TOKENS + YBIN_TOKENS
)

# Closed-set attributes predicted by the node heads (not part of the token stream).
DEPICTED_TYPES: tuple[str, ...] = (
    "plain",
    "double",
    "triple",
    "companion",
    "circle",
    "solid_wedge",
    "dashed_wedge",
    "wavy",
    "dotted",
    "variable",
)
WEDGE_DIRECTIONS: tuple[str, ...] = ("none", "source_to_target", "target_to_source")
ORDER_CLASSES: tuple[int, ...] = (1, 2, 3)

CHARGE_CLASSES: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
H_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 4)
PLACEHOLDER_SHAPES: tuple[str, ...] = ("circle", "polygon", "bracket")

# Defaults mirrored in the model/eval configs.
DEFAULT_VOCAB_PATH = "vocab.json"


def kind_id(name: str) -> int:
    return KIND_CLASSES.index(name)


def subtype_id(name: str | None) -> int:
    if name is None:
        return 0
    return VARIABLE_SUBTYPES.index(name) + 1


def charge_bucket(charge: int) -> int:
    """Map an integer charge onto ``CHARGE_CLASSES`` (nearest-clamped bucket)."""
    value = int(charge)
    return min(max(value - CHARGE_CLASSES[0], 0), len(CHARGE_CLASSES) - 1)


def h_bucket(h_count: int | None) -> int:
    if h_count is None:
        return 0
    return min(max(int(h_count), 0), len(H_CLASSES) - 1)


def placeholder_shape_id(name: str | None) -> int:
    if name is None:
        return 0
    return PLACEHOLDER_SHAPES.index(name) + 1


def save_vocab(tokens: list[str], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"tokens": tokens}, ensure_ascii=False), encoding="utf-8")


def load_vocab_tokens(path: str | Path) -> list[str]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(token) for token in payload["tokens"]]


__all__ = [
    "BASE_TOKENS",
    "BOS_TOKEN",
    "CHARGE_CLASSES",
    "COORD_BINS",
    "DEFAULT_VOCAB_PATH",
    "DEPICTED_TYPES",
    "EOS_TOKEN",
    "H_CLASSES",
    "KIND_CLASSES",
    "KIND_TOKENS",
    "LABEL_END",
    "LABEL_START",
    "NODE_END",
    "NODE_START",
    "ORDER_CLASSES",
    "PAD_TOKEN",
    "PLACEHOLDER_SHAPES",
    "SCRIPT_NAMES",
    "SCRIPT_TOKENS",
    "SPECIAL_TOKENS",
    "SUBTYPE_NONE",
    "SUBTYPE_TOKENS",
    "UNK_TOKEN",
    "VARIABLE_SUBTYPES",
    "WEDGE_DIRECTIONS",
    "XBIN_TOKENS",
    "YBIN_TOKENS",
    "charge_bucket",
    "h_bucket",
    "kind_id",
    "load_vocab_tokens",
    "placeholder_shape_id",
    "save_vocab",
    "subtype_id",
]
