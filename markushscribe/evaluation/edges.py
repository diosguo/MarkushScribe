"""Edge metrics, oracle and end-to-end (``M1方案.md`` §18.5)."""

from __future__ import annotations

from .. import schema


def _edge_map(
    graph: schema.MarkushGraph, mapping: dict[int, int] | None
) -> dict[tuple[int, int], schema.TargetEdge]:
    index_of = {node.id: index for index, node in enumerate(graph.nodes)}
    edges: dict[tuple[int, int], schema.TargetEdge] = {}
    for edge in graph.edges:
        source = index_of[edge.source]
        target = index_of[edge.target]
        if mapping is not None:
            if source not in mapping or target not in mapping:
                continue
            source, target = mapping[source], mapping[target]
        if source == target:
            continue
        key = (source, target) if source < target else (target, source)
        edges.setdefault(key, edge)
    return edges


def edge_scores(
    gold: schema.MarkushGraph,
    pred: schema.MarkushGraph,
    mapping: dict[int, int] | None = None,
) -> dict[str, float]:
    """Score edges; ``mapping`` maps predicted node indices to gold indices.

    With ``mapping=None`` the predicted graph is assumed indexed like the gold
    graph (the oracle setting); pass the Hungarian result for end-to-end scores.
    """
    gold_edges = _edge_map(gold, None)
    pred_edges = _edge_map(pred, mapping)
    gold_keys = set(gold_edges)
    pred_keys = set(pred_edges)
    true_positive = gold_keys & pred_keys
    precision = len(true_positive) / len(pred_keys) if pred_keys else 0.0
    recall = len(true_positive) / len(gold_keys) if gold_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def accuracy(attribute: str) -> float:
        if not true_positive:
            return 0.0
        correct = 0
        for key in true_positive:
            if attribute == "order":
                ok = gold_edges[key].normalized.order == pred_edges[key].normalized.order
            elif attribute == "aromatic":
                ok = gold_edges[key].normalized.aromatic == pred_edges[key].normalized.aromatic
            elif attribute == "depiction":
                ok = gold_edges[key].depicted.type == pred_edges[key].depicted.type
            elif attribute == "variable":
                ok = bool(gold_edges[key].variable) == bool(pred_edges[key].variable)
            elif attribute == "wedge":
                ok = gold_edges[key].wedge_direction == pred_edges[key].wedge_direction
            else:  # pragma: no cover - guarded by callers
                raise ValueError(attribute)
            correct += int(ok)
        return correct / len(true_positive)

    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_order_accuracy": accuracy("order"),
        "edge_aromatic_accuracy": accuracy("aromatic"),
        "edge_depiction_accuracy": accuracy("depiction"),
        "edge_variable_accuracy": accuracy("variable"),
        "edge_wedge_accuracy": accuracy("wedge"),
    }


__all__ = ["edge_scores"]
