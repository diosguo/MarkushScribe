"""The NODE AR baseline model: encoder + token decoder + heads (``M1方案.md`` §6)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .. import constants
from .edge_heads import EdgeHeads
from .encoder import Encoder
from .label_decoder import LabelDecoder
from .node_ar import NodeAR
from .node_heads import NodeHeads


@dataclass
class ModelConfig:
    encoder_name: str = "resnet18"
    out_indices: tuple[int, ...] = (2, 3, 4)
    dim: int = 256
    nhead: int = 8
    ar_layers: int = 4
    label_layers: int = 2
    dropout: float = 0.1
    max_len: int = 4096
    label_max_len: int = 64
    roi: int = 7
    roi_radius: float = 0.08
    pretrained: bool = False
    freeze_encoder: bool = False
    coord_bins: int = constants.COORD_BINS
    bond_len_est: float = 0.12
    tags: dict[str, str] = field(default_factory=dict)


class NodeARModel(nn.Module):
    def __init__(
        self, vocab_size: int, config: ModelConfig | None = None, *, pad_id: int = 0
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.pad_id = pad_id
        self.encoder = Encoder(
            self.config.encoder_name,
            self.config.dim,
            self.config.out_indices,
            pretrained=self.config.pretrained,
            freeze=self.config.freeze_encoder,
        )
        self.node_ar = NodeAR(
            vocab_size,
            dim=self.config.dim,
            nhead=self.config.nhead,
            layers=self.config.ar_layers,
            dropout=self.config.dropout,
            max_len=self.config.max_len,
            pad_id=pad_id,
        )
        self.node_heads = NodeHeads(self.config.dim, dropout=self.config.dropout)
        self.label_decoder = LabelDecoder(
            vocab_size,
            self.config.dim,
            nhead=self.config.nhead,
            layers=self.config.label_layers,
            dropout=self.config.dropout,
            max_len=self.config.label_max_len,
            roi=self.config.roi,
            roi_radius=self.config.roi_radius,
            pad_id=pad_id,
        )
        self.label_proj = nn.Linear(self.config.dim, self.config.dim)
        self.edge_heads = EdgeHeads(
            self.config.dim,
            dropout=self.config.dropout,
            bond_len_est=self.config.bond_len_est,
        )

    @property
    def vocab_size(self) -> int:
        return self.node_ar.embedding.num_embeddings

    def encode_image(self, image: torch.Tensor):
        return self.encoder(image)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        token_ids: torch.Tensor | None = None,
        node_token_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image = batch["image"]
        tokens = batch["tokens"] if token_ids is None else token_ids
        token_mask = batch.get("token_mask")
        if token_mask is None:
            token_mask = torch.ones_like(tokens, dtype=torch.bool)

        encoder_out = self.encoder(image)
        hidden, token_logits = self.node_ar(
            tokens[:, :-1],
            encoder_out.memory,
            tgt_key_padding_mask=~token_mask[:, :-1],
        )
        gather_index = batch["node_token_index"] if node_token_index is None else node_token_index
        node_hidden = self.node_ar.gather(hidden, gather_index)
        node_outputs = self.node_heads(node_hidden)

        coords = self._coords(batch)
        label_in = batch["label_tokens"][:, :, :-1]
        label_padding = ~batch["label_mask"][:, :, :-1]
        label_logits, label_summary = self.label_decoder(
            label_in,
            node_hidden,
            encoder_out.pyramid,
            coords,
            label_padding=label_padding,
        )
        node_features = node_hidden + self.label_proj(label_summary)
        node_features = node_features * batch["node_mask"].unsqueeze(-1)
        edge_outputs = self.edge_heads(node_features, coords, batch["node_mask"])

        outputs: dict[str, torch.Tensor] = {
            "token_logits": token_logits,
            "label_logits": label_logits,
            "coords": coords,
            "node_features": node_features,
        }
        outputs.update(node_outputs)
        outputs.update(edge_outputs)
        return outputs

    def _coords(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bins = self.config.coord_bins - 1
        return (batch["coord_bin"].float() + batch["coord_offset"]) / bins


__all__ = ["ModelConfig", "NodeARModel"]
