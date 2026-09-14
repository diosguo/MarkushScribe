"""Combine node, label and edge metrics into one report per graph."""

from __future__ import annotations

import numpy as np

from .. import schema
from ..matching.costs import node_cost_matrix
from ..matching.hungarian import match
from .edges import _edge_map, edge_scores
from .labels import anchor_errors, label_scores


def _node_scores(
    gold: schema.MarkushGraph, pred: schema.MarkushGraph, pairs: list[tuple[int, int]]
) -> dict[str, float]:
    correct = 0
    kind_correct = 0
    for gold_index, pred_index in pairs:
        gold_node = gold.nodes[gold_index]
        pred_node = pred.nodes[pred_index]
        if gold_node.node_class == pred_node.node_class:
            kind_correct += 1
            if gold_node.visual_label.plain_text == pred_node.visual_label.plain_text:
                correct += 1
    precision = correct / len(pred.nodes) if pred.nodes else 0.0
    recall = correct / len(gold.nodes) if gold.nodes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "node_kind_accuracy": kind_correct / len(pairs) if pairs else 0.0,
        "node_matched": float(len(pairs)),
    }


def evaluate_graph(
    gold: schema.MarkushGraph,
    pred: schema.MarkushGraph,
    *,
    strict_exact: bool = False,
) -> dict[str, float]:
    cost = node_cost_matrix(gold, pred)
    pairs = match(cost)
    mapping = {pred_index: gold_index for gold_index, pred_index in pairs}
    errors = anchor_errors(gold, pred, pairs)
    report: dict[str, float] = {}
    report.update(_node_scores(gold, pred, pairs))
    report.update(label_scores(gold, pred, pairs))
    if errors:
        report["anchor_mean"] = float(np.mean(errors))
        report["anchor_median"] = float(np.median(errors))
        report["anchor_p90"] = float(np.percentile(errors, 90))
    else:
        report["anchor_mean"] = report["anchor_median"] = report["anchor_p90"] = 0.0
    report.update(edge_scores(gold, pred, mapping))
    report["graph_exact"] = float(graph_exact(gold, pred, strict=strict_exact))
    return report


def graph_exact(
    gold: schema.MarkushGraph, pred: schema.MarkushGraph, *, strict: bool = False
) -> bool:
    """Whether the two graphs are identical up to an attribute-aware node bijection."""
    if len(gold.nodes) != len(pred.nodes) or len(gold.nodes) == 0:
        return len(gold.nodes) == len(pred.nodes)
    pairs = match(node_cost_matrix(gold, pred))
    if len(pairs) != len(gold.nodes):
        return False
    mapping: dict[int, int] = {}
    for gold_index, pred_index in pairs:
        gold_node = gold.nodes[gold_index]
        pred_node = pred.nodes[pred_index]
        if gold_node.node_class != pred_node.node_class:
            return False
        if strict:
            if [dict(run) for run in gold_node.visual_label.runs] != [
                dict(run) for run in pred_node.visual_label.runs
            ]:
                return False
        mapping[pred_index] = gold_index
    gold_edges = _edge_map(gold, None)
    pred_edges = _edge_map(pred, mapping)
    if set(gold_edges) != set(pred_edges):
        return False
    for key, gold_edge in gold_edges.items():
        pred_edge = pred_edges[key]
        if (
            gold_edge.normalized.order != pred_edge.normalized.order
            or gold_edge.normalized.aromatic != pred_edge.normalized.aromatic
        ):
            return False
        if strict and (
            gold_edge.depicted.type != pred_edge.depicted.type
            or gold_edge.wedge_direction != pred_edge.wedge_direction
            or bool(gold_edge.variable) != bool(pred_edge.variable)
        ):
            return False
    return True


def evaluate_dataset(
    samples: list[tuple[schema.MarkushGraph, schema.MarkushGraph]],
    *,
    strict_exact: bool = False,
) -> dict[str, float]:
    reports = [evaluate_graph(gold, pred, strict_exact=strict_exact) for gold, pred in samples]
    if not reports:
        return {}
    return {key: float(np.mean([report[key] for report in reports])) for key in reports[0]}


__all__ = ["evaluate_dataset", "evaluate_graph", "graph_exact"]
