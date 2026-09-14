"""Teacher-forcing curriculum: ramp from gold to predicted node positions.

``M1方案.md`` §8: early training gathers node embeddings at the ground-truth
``<node>`` tokens; later it uses the positions the model actually decodes so the
edge/label heads do not only ever see clean input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TeacherForcingSchedule:
    start_epoch: int = 0
    end_epoch: int = 5
    max_probability: float = 1.0

    def probability(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return 0.0
        if self.end_epoch <= self.start_epoch:
            return self.max_probability
        fraction = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return float(min(max(fraction, 0.0), 1.0) * self.max_probability)


def predicted_node_indices(tokens: torch.Tensor, node_start_id: int) -> torch.Tensor:
    """Positions of ``<node>`` tokens in ``[B, L]``, padded with ``-1``."""
    batch = tokens.shape[0]
    found = [
        (tokens[row] == node_start_id).nonzero(as_tuple=True)[0].tolist() for row in range(batch)
    ]
    width = max((len(row) for row in found), default=0)
    out = torch.full((batch, width), -1, dtype=torch.long, device=tokens.device)
    for row, positions in enumerate(found):
        if positions:
            out[row, : len(positions)] = torch.tensor(positions, device=tokens.device)
    return out


__all__ = ["TeacherForcingSchedule", "predicted_node_indices"]
