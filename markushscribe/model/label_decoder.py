"""Per-node label decoder with an ROI crop from the encoder pyramid.

Node embeddings alone cannot describe an abbreviation, so each node also reads a
small patch of the image around its anchor and decodes its visual label runs as
an autoregressive token sequence (``M1方案.md`` §6.4). All nodes of a batch are
processed in parallel by flattening ``[B, N]`` into the sequence dimension.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def sample_roi(
    feature: torch.Tensor,
    coords: torch.Tensor,
    *,
    roi: int = 7,
    radius: float = 0.08,
) -> torch.Tensor:
    """Bilinear-crop an ``roi x roi`` patch per node from one pyramid level.

    ``coords`` are normalised image coordinates (origin top-left, y down); the
    result is ``[B, N, C * roi * roi]``.
    """
    batch, channels, height, width = feature.shape
    nodes = coords.shape[1]
    offsets = torch.linspace(-radius, radius, roi, device=coords.device, dtype=coords.dtype)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    gx = coords[..., 0].unsqueeze(-1).unsqueeze(-1) + grid_x
    gy = coords[..., 1].unsqueeze(-1).unsqueeze(-1) + grid_y
    grid = torch.stack([gx, gy], dim=-1).reshape(batch, nodes * roi, roi, 2) * 2.0 - 1.0
    sampled = F.grid_sample(feature, grid, align_corners=False, padding_mode="border")
    sampled = sampled.reshape(batch, channels, nodes, roi, roi).permute(0, 2, 1, 3, 4)
    return sampled.reshape(batch, nodes, channels * roi * roi)


class LabelDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        *,
        nhead: int = 8,
        layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
        roi: int = 7,
        roi_radius: float = 0.08,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.roi = roi
        self.roi_radius = roi_radius
        self.pad_id = pad_id
        self.roi_proj = nn.Linear(dim * roi * roi, dim)
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.positions = nn.Embedding(max_len, dim)
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        label_in: torch.Tensor,
        node_hidden: torch.Tensor,
        pyramid: list[torch.Tensor],
        coords: torch.Tensor,
        *,
        label_padding: torch.Tensor | None = None,
        roi_level: int = -2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(label_logits [B,N,T,V], label_summary [B,N,D])``."""
        batch, nodes, length = label_in.shape
        if batch * nodes * length == 0:
            logits = node_hidden.new_zeros(batch, nodes, length, self.head.out_features)
            return logits, node_hidden
        level = pyramid[roi_level] if len(pyramid) >= abs(roi_level) else pyramid[-1]
        roi = sample_roi(level, coords, roi=self.roi, radius=self.roi_radius)
        condition = node_hidden + self.roi_proj(roi)

        flat = label_in.reshape(batch * nodes, length)
        positions = torch.arange(length, device=label_in.device)
        embedded = self.embedding(flat) + self.positions(positions).unsqueeze(0)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=label_in.device), diagonal=1
        )
        padding = None if label_padding is None else label_padding.reshape(batch * nodes, length)
        hidden = self.decoder(
            embedded,
            condition.reshape(batch * nodes, 1, self.dim),
            tgt_mask=causal,
            tgt_key_padding_mask=padding,
        )
        hidden = self.norm(hidden)
        logits = self.head(hidden).reshape(batch, nodes, length, -1)
        summary = hidden.mean(dim=1).reshape(batch, nodes, self.dim)
        return logits, summary


__all__ = ["LabelDecoder", "sample_roi"]
