"""Graph-level evaluation for the M1 baseline (``M1方案.md`` §9/§18).

Predictions are compared to gold after an attribute-aware optimal node matching;
edge metrics can be reported *oracle* (gold node identities) or end-to-end
(after matching). The functions return plain dicts so a runner can aggregate
them by summing and averaging.
"""

from .metrics import evaluate_dataset, evaluate_graph

__all__ = ["evaluate_dataset", "evaluate_graph"]
