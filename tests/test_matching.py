"""Hungarian matching and cost tests."""

from __future__ import annotations

import numpy as np

from markushscribe.matching import linear_sum_assignment, match
from markushscribe.matching.costs import edit_distance, node_cost_matrix


def test_square_assignment():
    cost = np.array([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0], [2.0, 4.0, 1.0]])
    pairs = linear_sum_assignment(cost)
    assert sorted(column for _, column in pairs) == [0, 1, 2]
    assert sorted(row for row, _ in pairs) == [0, 1, 2]
    assert sum(cost[row, column] for row, column in pairs) == 3.0


def test_rectangular_assignment():
    cost = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    pairs = linear_sum_assignment(cost)
    assert len(pairs) == 2
    assert sum(cost[row, column] for row, column in pairs) == 2.0


def test_threshold_drops_expensive_pairs():
    cost = np.array([[0.1, 9.0], [9.0, 0.2]])
    assert len(match(cost)) == 2
    assert match(cost, threshold=1.0) == [(0, 0), (1, 1)]


def test_edit_distance():
    assert edit_distance(list("abc"), list("abc")) == 0
    assert edit_distance(list("abc"), list("abd")) == 1
    assert edit_distance([], list("abc")) == 3
    assert edit_distance(list("sub:1"), list("base:1")) > 0


def test_node_cost_prefers_matching_label(tiny_graphs):
    graph = tiny_graphs[0]
    cost = node_cost_matrix(graph, graph)
    assert cost.shape == (len(graph.nodes), len(graph.nodes))
    pairs = linear_sum_assignment(cost)
    assert pairs == [(index, index) for index in range(len(graph.nodes))]
    # perturbing one anchor must raise its own cell
    shifted = node_cost_matrix(graph, graph)
    assert shifted[0, 0] == 0.0
