"""Node-pair costs for matching a predicted graph to gold.

The cost blends structural class, the visual label (a script-aware edit
distance over the run sequence) and the anchor distance, so an assignment can
trade a small coordinate error against a label mismatch.
"""

from __future__ import annotations

import numpy as np

from .. import schema


def run_sequence(runs: tuple[dict[str, str], ...]) -> list[str]:
    """Flatten runs into ``script:char`` tokens so base/sub/sup are distinguished."""
    tokens: list[str] = []
    for run in runs:
        tokens.extend(f"{run['script']}:{char}" for char in run["text"])
    return tokens


def edit_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            cost = 0 if token == other else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def label_distance(gold: schema.TargetNode, pred: schema.TargetNode) -> float:
    left = run_sequence(gold.visual_label.runs)
    right = run_sequence(pred.visual_label.runs)
    length = max(len(left), len(right), 1)
    return edit_distance(left, right) / length


def node_cost_matrix(
    gold: schema.MarkushGraph,
    pred: schema.MarkushGraph,
    *,
    class_weight: float = 2.0,
    label_weight: float = 1.0,
    coord_weight: float = 1.0,
) -> np.ndarray:
    cost = np.zeros((len(gold.nodes), len(pred.nodes)), dtype=np.float64)
    for i, gold_node in enumerate(gold.nodes):
        for j, pred_node in enumerate(pred.nodes):
            class_penalty = 0.0 if gold_node.node_class == pred_node.node_class else class_weight
            coordinate = np.hypot(
                gold_node.anchor.x - pred_node.anchor.x, gold_node.anchor.y - pred_node.anchor.y
            )
            cost[i, j] = (
                class_penalty
                + label_weight * label_distance(gold_node, pred_node)
                + coord_weight * coordinate
            )
    return cost


__all__ = ["edit_distance", "label_distance", "node_cost_matrix", "run_sequence"]
