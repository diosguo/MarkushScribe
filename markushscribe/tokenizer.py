"""Tokenise ``markush_graph_v2`` targets into the NODE AR token stream.

The stream is deliberately close to the schema: each node emits a fixed header
(``<node> kind subtype xbin ybin``) followed by its visual label runs, so a
decoded sequence recovers the *drawn* structure without any learned detokeniser.
See ``M1方案.md`` §4 for the grammar.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import constants, schema


@dataclass
class Vocab:
    """Ordered token table; characters are appended after the fixed groups."""

    tokens: list[str]

    def __post_init__(self) -> None:
        self._index: dict[str, int] = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self._index

    def to_id(self, token: str) -> int:
        return self._index.get(token, self.unk_id)

    def from_id(self, token_id: int) -> str:
        if 0 <= token_id < len(self.tokens):
            return self.tokens[token_id]
        return constants.UNK_TOKEN

    @property
    def unk_id(self) -> int:
        return self._index[constants.UNK_TOKEN]

    @property
    def pad_id(self) -> int:
        return self._index[constants.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._index[constants.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._index[constants.EOS_TOKEN]

    @property
    def char_tokens(self) -> list[str]:
        return self.tokens[len(constants.BASE_TOKENS) :]

    def save(self, path: str | Path) -> None:
        constants.save_vocab(self.tokens, path)

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        return cls(constants.load_vocab_tokens(path))

    @classmethod
    def base(cls, char_tokens: list[str] | None = None) -> Vocab:
        return cls(list(constants.BASE_TOKENS) + list(char_tokens or []))


@dataclass
class EncodedSample:
    """Token ids plus the aligned node targets used by the model heads."""

    tokens: list[int]
    node_token_index: list[int]
    node_kind: list[int]
    node_subtype: list[int]
    node_visible: list[int]
    coord_bin: list[tuple[int, int]]
    coord_offset: list[tuple[float, float]]
    label_tokens: list[list[int]]
    node_ids: list[int]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)


def _iter_label_chars(graph: schema.MarkushGraph) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node in graph.nodes:
        for run in node.visual_label.runs:
            counts.update(run["text"])
    return counts


def build_vocab(
    graphs: list[schema.MarkushGraph],
    *,
    min_char_freq: int = 1,
    extra_chars: str = "",
) -> Vocab:
    """Collect label characters from ``graphs`` into a deterministic vocabulary."""
    counts: Counter[str] = Counter()
    for graph in graphs:
        counts.update(_iter_label_chars(graph))
    counts.update(extra_chars)
    chars = [
        char
        for char, freq in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if freq >= min_char_freq
    ]
    return Vocab.base(chars)


def _graph(payload: schema.MarkushGraph | dict[str, Any]) -> schema.MarkushGraph:
    if isinstance(payload, schema.MarkushGraph):
        return payload
    return schema.parse_target(payload)


def _run_tokens(runs: tuple[dict[str, str], ...], vocab: Vocab) -> list[int]:
    tokens: list[int] = []
    for run in runs:
        tokens.append(vocab.to_id(f"<{run['script']}>"))
        tokens.extend(vocab.to_id(char) for char in run["text"])
    return tokens


def encode_target(payload: schema.MarkushGraph | dict[str, Any], vocab: Vocab) -> EncodedSample:
    """Encode one target into the NODE AR token stream and aligned head targets."""
    graph = _graph(payload)
    tokens: list[int] = [vocab.bos_id]
    node_token_index: list[int] = []
    node_kind: list[int] = []
    node_subtype: list[int] = []
    node_visible: list[int] = []
    coord_bin: list[tuple[int, int]] = []
    coord_offset: list[tuple[float, float]] = []
    label_tokens: list[list[int]] = []
    node_ids: list[int] = []

    for node in graph.nodes:
        node_token_index.append(len(tokens))
        tokens.append(vocab.to_id(constants.NODE_START))
        tokens.append(vocab.to_id(f"<kind_{node.node_class}>"))
        subtype = node.variable_subtype if node.node_class == "variable" else None
        tokens.append(vocab.to_id(_subtype_token(subtype)))
        coord = node.coord_target
        tokens.append(vocab.to_id(constants.XBIN_TOKENS[coord.x_bin]))
        tokens.append(vocab.to_id(constants.YBIN_TOKENS[coord.y_bin]))
        tokens.append(vocab.to_id(constants.LABEL_START))
        body = _run_tokens(node.visual_label.runs, vocab)
        tokens.extend(body)
        tokens.append(vocab.to_id(constants.LABEL_END))
        tokens.append(vocab.to_id(constants.NODE_END))

        node_kind.append(constants.kind_id(node.node_class))
        node_subtype.append(constants.subtype_id(subtype))
        node_visible.append(1 if node.visual_label.visible else 0)
        coord_bin.append((coord.x_bin, coord.y_bin))
        coord_offset.append((coord.x_offset, coord.y_offset))
        label_tokens.append(
            [vocab.to_id(constants.LABEL_START), *body, vocab.to_id(constants.LABEL_END)]
        )
        node_ids.append(node.id)

    tokens.append(vocab.eos_id)
    return EncodedSample(
        tokens=tokens,
        node_token_index=node_token_index,
        node_kind=node_kind,
        node_subtype=node_subtype,
        node_visible=node_visible,
        coord_bin=coord_bin,
        coord_offset=coord_offset,
        label_tokens=label_tokens,
        node_ids=node_ids,
        meta={"uid": None},
    )


def _subtype_token(subtype: str | None) -> str:
    if subtype is None:
        return constants.SUBTYPE_NONE
    return f"<subtype_{subtype}>"


@dataclass
class DecodedNode:
    kind: str
    subtype: str | None
    x_bin: int
    y_bin: int
    runs: list[dict[str, str]]


def decode_tokens(token_ids: list[int], vocab: Vocab) -> list[DecodedNode]:
    """Parse a token stream back into nodes (best effort; ``<unk>`` chars are kept)."""
    nodes: list[DecodedNode] = []
    index = 0
    length = len(token_ids)
    while index < length:
        token = vocab.from_id(token_ids[index])
        if token in (constants.BOS_TOKEN,):
            index += 1
            continue
        if token in (constants.EOS_TOKEN, constants.PAD_TOKEN):
            break
        if token != constants.NODE_START:
            index += 1
            continue
        index += 1
        if index + 3 >= length:  # kind, subtype, xbin, ybin must all be present
            break
        kind = _strip(vocab.from_id(token_ids[index]), "<kind_", ">")
        index += 1
        subtype = _strip(vocab.from_id(token_ids[index]), "<subtype_", ">")
        index += 1
        x_token = vocab.from_id(token_ids[index])
        y_token = vocab.from_id(token_ids[index + 1])
        index += 2
        runs: list[dict[str, str]] = []
        script = "base"
        text = ""
        saw_label = False
        while index < length:
            token = vocab.from_id(token_ids[index])
            index += 1
            if token == constants.LABEL_START:
                saw_label = True
                continue
            if token == constants.LABEL_END:
                break
            if token in constants.SCRIPT_TOKENS:
                if text:
                    runs.append({"text": text, "script": script})
                script = token[1:-1]
                text = ""
                continue
            text += token
        if text:
            runs.append({"text": text, "script": script})
        if not saw_label:
            continue
        nodes.append(
            DecodedNode(
                kind=kind,
                subtype=None if subtype == "none" else subtype,
                x_bin=_bin_index(x_token),
                y_bin=_bin_index(y_token),
                runs=runs,
            )
        )
    return nodes


def _strip(token: str, prefix: str, suffix: str) -> str:
    if token.startswith(prefix) and token.endswith(suffix):
        return token[len(prefix) : -len(suffix)]
    return token


def _bin_index(token: str) -> int:
    if token.startswith("<xbin_") or token.startswith("<ybin_"):
        return int(token.split("_")[-1].rstrip(">"))
    return -1


__all__ = [
    "DecodedNode",
    "EncodedSample",
    "Vocab",
    "build_vocab",
    "decode_tokens",
    "encode_target",
]
