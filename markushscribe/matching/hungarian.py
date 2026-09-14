"""A dependency-free O(n^3) Hungarian assignment (rectangular supported).

Implemented from the standard shortest-augmenting-path formulation so the
evaluation does not need SciPy on the CPU-only M1 environment.
"""

from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return the minimum-cost assignment as ``(row, column)`` pairs.

    Handles rectangular matrices by transposing when there are more rows than
    columns; every row is assigned to a distinct column.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    rows, cols = matrix.shape
    if rows == 0:
        return []
    if cols == 0:
        return []
    inf = float("inf")
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    p = np.zeros(cols + 1, dtype=int)
    way = np.zeros(cols + 1, dtype=int)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        minv = np.full(cols + 1, inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = inf
            column1 = 0
            for column in range(1, cols + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minv[column]:
                    minv[column] = current
                    way[column] = column0
                if minv[column] < delta:
                    delta = minv[column]
                    column1 = column
            for column in range(cols + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minv[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment: list[tuple[int, int]] = []
    for column in range(1, cols + 1):
        if p[column] > 0:
            pair = (p[column] - 1, column - 1)
            assignment.append((pair[1], pair[0]) if transposed else pair)
    assignment.sort()
    return assignment


def match(cost: np.ndarray, threshold: float | None = None) -> list[tuple[int, int]]:
    """Optimal assignment, optionally dropping pairs whose cost exceeds ``threshold``."""
    pairs = linear_sum_assignment(cost)
    if threshold is None:
        return pairs
    return [(row, column) for row, column in pairs if cost[row, column] <= threshold]


__all__ = ["linear_sum_assignment", "match"]
