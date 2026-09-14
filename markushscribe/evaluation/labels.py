"""Label metrics: visual transcription, script-aware distance and R/M/L fields."""

from __future__ import annotations

import numpy as np

from .. import schema
from ..matching.costs import edit_distance, run_sequence


def label_scores(
    gold: schema.MarkushGraph,
    pred: schema.MarkushGraph,
    pairs: list[tuple[int, int]],
) -> dict[str, float]:
    if not pairs:
        return {
            "label_plain_exact": 0.0,
            "label_script_exact": 0.0,
            "label_edit": 0.0,
            "label_family_exact": 0.0,
            "label_index_exact": 0.0,
            "label_multiplicity_exact": 0.0,
            "label_charge_exact": 0.0,
            "label_isotope_exact": 0.0,
        }
    plain = script = family = index = multiplicity = charge = isotope = 0
    edit = 0.0
    for gold_index, pred_index in pairs:
        gold_node = gold.nodes[gold_index]
        pred_node = pred.nodes[pred_index]
        if gold_node.visual_label.plain_text == pred_node.visual_label.plain_text:
            plain += 1
        if tuple(run["text"] for run in gold_node.visual_label.runs) == tuple(
            run["text"] for run in pred_node.visual_label.runs
        ):
            script += 1
        left = run_sequence(gold_node.visual_label.runs)
        right = run_sequence(pred_node.visual_label.runs)
        edit += edit_distance(left, right) / max(len(left), len(right), 1)
        gold_semantic = gold_node.semantic_label
        pred_semantic = pred_node.semantic_label
        if (
            gold_node.node_class in ("rgroup", "variable")
            and pred_node.node_class == gold_node.node_class
        ):
            if gold_semantic.get("family") == pred_semantic.get("family"):
                family += 1
            if gold_semantic.get("index") == pred_semantic.get("index"):
                index += 1
            if gold_semantic.get("multiplicity") == pred_semantic.get("multiplicity"):
                multiplicity += 1
        if (gold_semantic.get("charge") or 0) == (pred_semantic.get("charge") or 0):
            charge += 1
        if (gold_semantic.get("isotope") is None) == (pred_semantic.get("isotope") is None):
            isotope += 1
    total = len(pairs)
    return {
        "label_plain_exact": plain / total,
        "label_script_exact": script / total,
        "label_edit": edit / total,
        "label_family_exact": family / total,
        "label_index_exact": index / total,
        "label_multiplicity_exact": multiplicity / total,
        "label_charge_exact": charge / total,
        "label_isotope_exact": isotope / total,
    }


def anchor_errors(
    gold: schema.MarkushGraph,
    pred: schema.MarkushGraph,
    pairs: list[tuple[int, int]],
) -> list[float]:
    return [
        float(
            np.hypot(
                gold.nodes[i].anchor.x - pred.nodes[j].anchor.x,
                gold.nodes[i].anchor.y - pred.nodes[j].anchor.y,
            )
        )
        for i, j in pairs
    ]
