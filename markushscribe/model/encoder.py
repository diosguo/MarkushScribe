"""Image backbone with a projected multi-scale pyramid.

The encoder turns a 384x384 image into (a) a flat ``memory`` token sequence for
the NODE AR decoder to attend to and (b) the projected feature maps the label
decoder crops and the edge head samples along bonds. The backbone name and the
stage indices are configuration, so a CPU debug run can swap Swin for a small
ResNet without touching the rest of the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class EncoderOutput:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    pyramid: list[torch.Tensor]
    strides: list[int]


class Positional2D(nn.Module):
    """Fixed sinusoidal positional map added to each pyramid level."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        half = self.dim // 2
        omega = torch.arange(half, device=feature.device, dtype=feature.dtype)
        omega = torch.exp(-math.log(10000.0) * omega / half)
        y = torch.arange(height, device=feature.device, dtype=feature.dtype)
        x = torch.arange(width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        pe_y = grid_y.unsqueeze(-1) * omega.view(1, 1, -1)
        pe_x = grid_x.unsqueeze(-1) * omega.view(1, 1, -1)
        pe = torch.cat([pe_y.sin(), pe_y.cos(), pe_x.sin(), pe_x.cos()], dim=-1)
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        if pe.shape[1] > channels:
            pe = pe[:, :channels]
        elif pe.shape[1] < channels:
            pad = channels - pe.shape[1]
            pe = F.pad(pe, (0, 0, 0, 0, 0, pad))
        return feature + pe


class Encoder(nn.Module):
    """timm backbone -> per-level projection -> flat memory + pyramid."""

    def __init__(
        self,
        name: str = "resnet18",
        dim: int = 256,
        out_indices: tuple[int, ...] = (2, 3, 4),
        *,
        pretrained: bool = False,
        freeze: bool = False,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        import timm

        self.name = name
        self.out_indices = tuple(out_indices)
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )
        channels = list(self.backbone.feature_info.channels())
        self.projections = nn.ModuleList([nn.Conv2d(ch, dim, kernel_size=1) for ch in channels])
        self.positional = Positional2D(dim)
        self.norms = nn.ModuleList([nn.GroupNorm(8, dim) for _ in channels])
        self.strides = (
            list(strides) if strides is not None else list(self.backbone.feature_info.reduction())
        )
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> Encoder:  # keep a frozen backbone in eval
        super().train(mode)
        if not self.backbone.training:
            return self
        if all(not parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> EncoderOutput:
        features = self.backbone(image)
        pyramid: list[torch.Tensor] = []
        memory: list[torch.Tensor] = []
        for feature, projection, norm in zip(features, self.projections, self.norms, strict=True):
            projected = norm(projection(feature))
            projected = self.positional(projected)
            pyramid.append(projected)
            memory.append(projected.flatten(2).transpose(1, 2))
        memory_tensor = torch.cat(memory, dim=1)
        mask = torch.ones(memory_tensor.shape[:2], dtype=torch.bool, device=image.device)
        return EncoderOutput(
            memory=memory_tensor,
            memory_mask=mask,
            pyramid=pyramid,
            strides=list(self.strides),
        )


__all__ = ["Encoder", "EncoderOutput", "Positional2D"]
