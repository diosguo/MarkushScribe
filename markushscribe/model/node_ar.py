"""NODE AR decoder: autoregressive token stream + node embedding gathering.

A standard causal Transformer decoder reads the token stream while attending to
the encoder memory. The hidden state at each ``<node>`` token is gathered into a
per-node embedding that the node heads and the edge head consume (``M1方案.md``
§6.2/§15.2).
"""

from __future__ import annotations

import torch
from torch import nn


class NodeAR(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        dim: int = 256,
        nhead: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 4096,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.positions = nn.Embedding(max_len, dim)
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.token_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        tokens_in: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(hidden, token_logits)`` for teacher forcing.

        ``hidden[:, p]`` predicts ``tokens_in[:, p + 1]``; ``tokens_in`` is the
        ground-truth stream shifted right, i.e. ``tokens[:, :-1]``.
        """
        length = tokens_in.shape[1]
        if positions is None:
            positions = torch.arange(length, device=tokens_in.device)
        embedded = self.embedding(tokens_in) + self.positions(positions).unsqueeze(0)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens_in.device), diagonal=1
        )
        hidden = self.decoder(
            embedded,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        hidden = self.norm(hidden)
        return hidden, self.token_head(hidden)

    @staticmethod
    def gather(hidden: torch.Tensor, node_token_index: torch.Tensor) -> torch.Tensor:
        """Gather per-node embeddings at ``node_token_index`` (``-1`` -> zeros)."""
        padded = node_token_index.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        gathered = torch.gather(hidden, 1, padded)
        mask = (node_token_index >= 0).unsqueeze(-1)
        return gathered.masked_fill(~mask, 0.0)


__all__ = ["NodeAR"]
