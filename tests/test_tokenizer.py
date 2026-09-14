"""Tokenizer round-trip and vocabulary tests."""

from __future__ import annotations

import pytest

from markushscribe import constants, schema
from markushscribe.tokenizer import Vocab, build_vocab, decode_tokens, encode_target


def _runs(*items: tuple[str, str]) -> list[dict[str, str]]:
    return [{"text": text, "script": script} for text, script in items]


def _node(
    node_id: int,
    node_class: str,
    x_bin: int,
    y_bin: int,
    runs: list[dict[str, str]],
    semantic: dict,
    *,
    subtype: str | None = None,
    rule: str = "base_core",
    placeholder: dict | None = None,
) -> dict:
    entry = {
        "id": node_id,
        "node_class": node_class,
        "variable_subtype": subtype,
        "anchor": {"x": x_bin / 63, "y": y_bin / 63, "rule": rule},
        "coord_target": {
            "bins": 64,
            "x_bin": x_bin,
            "x_offset": 0.0,
            "y_bin": y_bin,
            "y_offset": 0.0,
        },
        "visual_label": {
            "visible": bool(runs),
            "plain_text": "".join(run["text"] for run in runs),
            "runs": runs,
        },
        "semantic_label": semantic,
        "loss_mask": {},
    }
    if placeholder is not None:
        entry["placeholder"] = placeholder
    return entry


def _target() -> dict:
    return {
        "format": schema.TARGET_FORMAT,
        "coordinate_system": schema.COORDINATE_SYSTEM,
        "image_size": [128, 128],
        "coord_bins": 64,
        "order": "reading",
        "nodes": [
            _node(0, "atom", 5, 5, [], {"element": "C"}),
            _node(1, "atom", 10, 6, _runs(("N", "base")), {"element": "N"}),
            _node(
                2,
                "rgroup",
                20,
                20,
                _runs(("(", "base"), ("R", "base"), ("1", "sub"), (")", "base"), ("2", "sub")),
                {"family": "R", "index": "1", "multiplicity": "2"},
            ),
            _node(
                3,
                "atom",
                30,
                4,
                _runs(("2", "sup"), ("C", "base"), ("H", "base"), ("3", "sub"), ("+", "sup")),
                {"element": "C", "isotope": 2, "h_count": 3, "charge": 1},
            ),
            _node(
                4,
                "ring_placeholder",
                40,
                40,
                _runs(("R", "base"), ("7", "sub")),
                {"placeholder_id": "R7"},
                rule="placeholder_center",
                placeholder={"shape": "circle", "radius": 0.1},
            ),
        ],
        "edges": [],
        "graph_annotations": [],
        "masks": {"cxsmiles": False, "normalized_chemistry_complete": True},
    }


def test_encode_decode_round_trip():
    graph = schema.parse_target(_target())
    vocab = build_vocab([graph])
    encoded = encode_target(graph, vocab)
    decoded = decode_tokens(encoded.tokens, vocab)

    assert len(decoded) == len(graph.nodes)
    for node, plain in zip(decoded, graph.nodes, strict=True):
        assert node.kind == plain.node_class
        assert node.subtype == plain.variable_subtype
        assert node.x_bin == plain.coord_target.x_bin
        assert node.y_bin == plain.coord_target.y_bin
        assert node.runs == [dict(run) for run in plain.visual_label.runs]


def test_scripts_and_multiplicity_preserved():
    graph = schema.parse_target(_target())
    vocab = build_vocab([graph])
    decoded = decode_tokens(encode_target(graph, vocab).tokens, vocab)
    rgroup = decoded[2]
    assert [run["script"] for run in rgroup.runs] == [
        "base",
        "base",
        "sub",
        "base",
        "sub",
    ]
    assert "".join(run["text"] for run in rgroup.runs) == "(R1)2"
    assert decoded[3].runs[0] == {"text": "2", "script": "sup"}


def test_header_fields_align_with_nodes():
    graph = schema.parse_target(_target())
    vocab = build_vocab([graph])
    encoded = encode_target(graph, vocab)
    assert encoded.num_nodes == 5
    assert encoded.node_kind[0] == constants.kind_id("atom")
    assert encoded.node_subtype[2] == constants.subtype_id(None)
    assert encoded.node_visible[0] == 0
    assert encoded.node_visible[1] == 1
    assert encoded.node_token_index == [
        index for index, token in enumerate(encoded.tokens) if vocab.from_id(token) == "<node>"
    ]


def test_vocab_contains_no_duplicate_tokens():
    vocab = build_vocab([schema.parse_target(_target())])
    assert len(vocab.tokens) == len(set(vocab.tokens))
    assert "<unk>" in vocab and "<xbin_63>" in vocab


def test_vocab_save_load(tmp_path):
    vocab = build_vocab([schema.parse_target(_target())])
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocab.load(path)
    assert loaded.tokens == vocab.tokens
    assert loaded.unk_id == vocab.unk_id


def test_unseen_character_maps_to_unk():
    graph = schema.parse_target(_target())
    vocab = build_vocab([graph])
    assert "Q" not in vocab
    single = _target()
    single["nodes"] = [_node(0, "atom", 3, 3, _runs(("Q", "base")), {"element": "C"})]
    single["edges"] = []
    encoded = encode_target(schema.parse_target(single), vocab)
    assert vocab.unk_id in encoded.tokens


def test_min_char_freq_filters_rare_chars():
    graph = schema.parse_target(_target())
    vocab = build_vocab([graph], min_char_freq=2)
    assert "N" not in vocab
    assert "R" in vocab


def test_single_node_and_invisible_label():
    target = _target()
    target["nodes"] = [_node(0, "atom", 3, 3, [], {"element": "C"})]
    target["edges"] = []
    graph = schema.parse_target(target)
    vocab = build_vocab([graph])
    encoded = encode_target(graph, vocab)
    decoded = decode_tokens(encoded.tokens, vocab)
    assert len(decoded) == 1
    assert decoded[0].runs == []
    assert decoded[0].kind == "atom"


@pytest.mark.parametrize("node_class", ["atom", "rgroup", "variable", "abbreviation"])
def test_kind_round_trip(node_class):
    semantic = {
        "atom": {"element": "C"},
        "rgroup": {"family": "R"},
        "variable": {"family": "X"},
        "abbreviation": {"canonical_text": "Ac"},
    }[node_class]
    subtype = "generic" if node_class == "variable" else None
    target = _target()
    target["nodes"] = [_node(0, node_class, 1, 1, _runs(("C", "base")), semantic, subtype=subtype)]
    target["edges"] = []
    graph = schema.parse_target(target)
    vocab = build_vocab([graph])
    decoded = decode_tokens(encode_target(graph, vocab).tokens, vocab)
    assert decoded[0].kind == node_class
    assert decoded[0].subtype == subtype
