"""Torch dataset and collate for the NODE AR baseline.

This module owns the tensor contract of M1 (``M1方案.md`` §5): one shard member
becomes the image, the token stream and every aligned node/edge target the model
heads consume. It is import-optional for the rest of the package: only the
trainer, the model and their tests import it.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import constants, schema
from .tokenizer import Vocab, encode_target

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TorchDataConfig:
    """Shapes and capacities of one training sample."""

    image_size: int = 384
    q_capacity: int = 128
    t_capacity: int = 32
    l_capacity: int = 2048


def _normalize(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def normalize_image(image: Image.Image, image_size: int) -> torch.Tensor:
    """Public wrapper around the training-time image normalisation."""
    return _normalize(image, image_size)


def _edge_targets(graph: schema.MarkushGraph) -> dict[str, torch.Tensor]:
    count = len(graph.nodes)
    index_of = {node.id: index for index, node in enumerate(graph.nodes)}
    adjacency = np.zeros((count, count), dtype=np.float32)
    order = np.zeros((count, count), dtype=np.int64)
    aromatic = np.zeros((count, count), dtype=np.int64)
    depiction = np.zeros((count, count), dtype=np.int64)
    variable = np.zeros((count, count), dtype=np.int64)
    wedge = np.zeros((count, count), dtype=np.int64)

    for edge in graph.edges:
        low, high = index_of[edge.source], index_of[edge.target]
        if low > high:
            low, high = high, low
        adjacency[low, high] = adjacency[high, low] = 1.0
        order[low, high] = order[high, low] = edge.normalized.order
        aromatic[low, high] = aromatic[high, low] = int(edge.normalized.aromatic)
        depiction[low, high] = depiction[high, low] = constants.DEPICTED_TYPES.index(
            edge.depicted.type
        )
        variable[low, high] = variable[high, low] = int(edge.variable)
        if edge.wedge_direction == "source_to_target":
            wedge[low, high], wedge[high, low] = 1, 2
        elif edge.wedge_direction == "target_to_source":
            wedge[low, high], wedge[high, low] = 2, 1

    mask = adjacency > 0.5
    return {
        "edge_adjacency": torch.from_numpy(adjacency),
        "edge_order": torch.from_numpy(order),
        "edge_aromatic": torch.from_numpy(aromatic),
        "edge_depiction": torch.from_numpy(depiction),
        "edge_variable": torch.from_numpy(variable),
        "edge_wedge_direction": torch.from_numpy(wedge),
        "edge_mask": torch.from_numpy(mask),
    }


def _node_attributes(graph: schema.MarkushGraph) -> dict[str, torch.Tensor]:
    count = len(graph.nodes)
    charge = np.zeros(count, dtype=np.int64)
    h_count = np.zeros(count, dtype=np.int64)
    isotope = np.zeros(count, dtype=np.int64)
    shape = np.zeros(count, dtype=np.int64)
    charge_mask = np.zeros(count, dtype=bool)
    h_mask = np.zeros(count, dtype=bool)
    isotope_mask = np.zeros(count, dtype=bool)
    shape_mask = np.zeros(count, dtype=bool)

    for index, node in enumerate(graph.nodes):
        semantic = node.semantic_label
        if node.node_class == "atom":
            charge_mask[index] = True
            charge[index] = constants.charge_bucket(int(semantic.get("charge") or 0))
            isotope_mask[index] = True
            isotope[index] = 1 if semantic.get("isotope") is not None else 0
            h_mask[index] = bool(node.loss_mask.get("h_count", False))
            raw_h = semantic.get("h_count")
            h_count[index] = constants.h_bucket(None if raw_h is None else int(raw_h))
        if node.node_class == "ring_placeholder" and node.placeholder is not None:
            shape_mask[index] = True
            shape[index] = constants.placeholder_shape_id(node.placeholder.shape)

    return {
        "node_charge": torch.from_numpy(charge),
        "node_h_count": torch.from_numpy(h_count),
        "node_isotope": torch.from_numpy(isotope),
        "node_placeholder_shape": torch.from_numpy(shape),
        "node_field_mask": {
            "charge": torch.from_numpy(charge_mask),
            "h_count": torch.from_numpy(h_mask),
            "isotope": torch.from_numpy(isotope_mask),
            "placeholder_shape": torch.from_numpy(shape_mask),
        },
    }


def build_sample(
    graph: schema.MarkushGraph,
    image: Image.Image,
    vocab: Vocab,
    config: TorchDataConfig,
    *,
    uid: str = "",
    source_smiles: str = "",
    to_tensor: bool = True,
) -> dict[str, Any]:
    """Encode one in-memory ``(graph, image)`` pair into the sample contract.

    Shared by :class:`TargetDataset` and inference, so the model sees exactly the
    same tensors whether the graph came from a shard or from the decoder.
    """
    encoded = encode_target(graph, vocab)
    num_nodes = encoded.num_nodes
    label_width = max((len(row) for row in encoded.label_tokens), default=1)
    label_matrix = [row + [vocab.pad_id] * (label_width - len(row)) for row in encoded.label_tokens]
    capacity_exceeded = (
        num_nodes > config.q_capacity
        or label_width > config.t_capacity
        or len(encoded.tokens) > config.l_capacity
    )
    sample: dict[str, Any] = {
        "uid": uid,
        "tokens": torch.tensor(encoded.tokens, dtype=torch.long),
        "node_token_index": torch.tensor(encoded.node_token_index, dtype=torch.long),
        "node_kind": torch.tensor(encoded.node_kind, dtype=torch.long),
        "node_subtype": torch.tensor(encoded.node_subtype, dtype=torch.long),
        "node_visible": torch.tensor(encoded.node_visible, dtype=torch.long),
        "coord_bin": torch.tensor(encoded.coord_bin, dtype=torch.long).reshape(num_nodes, 2),
        "coord_offset": torch.tensor(encoded.coord_offset, dtype=torch.float32).reshape(
            num_nodes, 2
        ),
        "label_tokens": torch.tensor(label_matrix, dtype=torch.long).reshape(
            num_nodes, label_width
        ),
        "capacity_exceeded": capacity_exceeded,
        "meta": {
            "uid": uid,
            "image_size": list(graph.image_size),
            "source_smiles": source_smiles,
        },
    }
    sample.update(_edge_targets(graph))
    sample.update(_node_attributes(graph))
    sample["image"] = _normalize(image, config.image_size) if to_tensor else image
    return sample


class TargetDataset(Dataset):
    """One shard member -> the tensors consumed by :class:`NodeARModel`."""

    def __init__(
        self,
        shard_paths: list[str | Path],
        vocab: Vocab | None = None,
        config: TorchDataConfig | None = None,
        *,
        limit: int | None = None,
        to_tensor: bool = True,
    ) -> None:
        self.vocab = vocab
        self.config = config or TorchDataConfig()
        self.to_tensor = to_tensor
        self.index: list[tuple[Path, str]] = []
        for shard in shard_paths:
            path = Path(shard)
            with tarfile.open(path, "r") as archive:
                names = set(archive.getnames())
            pngs = {name.rsplit(".", 1)[0] for name in names if name.endswith(".png")}
            jsons = {name.rsplit(".", 1)[0] for name in names if name.endswith(".json")}
            for stem in sorted(pngs & jsons):
                self.index.append((path, stem))
                if limit is not None and len(self.index) >= limit:
                    break
            if limit is not None and len(self.index) >= limit:
                break

    def __len__(self) -> int:
        return len(self.index)

    def _read_members(self, shard: Path, stem: str) -> tuple[bytes, bytes]:
        with tarfile.open(shard, "r") as archive:
            png = archive.extractfile(f"{stem}.png")
            js = archive.extractfile(f"{stem}.json")
            if png is None or js is None:
                raise FileNotFoundError(f"{stem} is missing from {shard}")
            return png.read(), js.read()

    def raw(self, index: int) -> tuple[Image.Image, dict[str, Any], str]:
        """Return ``(image, target dict, uid)`` before tensorisation."""
        shard, stem = self.index[index]
        png_bytes, json_bytes = self._read_members(shard, stem)
        with Image.open(io.BytesIO(png_bytes)) as handle:
            image = handle.convert("RGB")
        return image, json.loads(json_bytes.decode("utf-8")), stem

    def iter_targets(self):
        """Yield every raw target dict without loading its image."""
        for index in range(len(self.index)):
            shard, stem = self.index[index]
            with tarfile.open(shard, "r") as archive:
                handle = archive.extractfile(f"{stem}.json")
                if handle is not None:
                    yield json.loads(handle.read().decode("utf-8"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.vocab is None:
            raise ValueError("TargetDataset requires a vocabulary to encode samples")
        image, payload, uid = self.raw(index)
        graph = schema.parse_target(payload)
        source_smiles = payload.get("_meta", {}).get("source_smiles", "")
        return build_sample(
            graph,
            image,
            self.vocab,
            self.config,
            uid=uid,
            source_smiles=source_smiles,
            to_tensor=self.to_tensor,
        )


def _pad_stack(tensors: list[torch.Tensor], value: float | int | bool = 0) -> torch.Tensor:
    sizes = {tuple(tensor.shape[1:]) for tensor in tensors}
    if len(sizes) > 1:
        raise ValueError(f"cannot stack tensors with different trailing shapes: {sizes}")
    width = max(tensor.shape[0] for tensor in tensors)
    padded = []
    for tensor in tensors:
        pad = width - tensor.shape[0]
        if pad:
            fill = torch.full((pad, *tensor.shape[1:]), value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, fill], dim=0)
        padded.append(tensor)
    return torch.stack(padded, dim=0)


def _pad_edges(tensors: list[torch.Tensor], value: float | int | bool = 0) -> torch.Tensor:
    width = max(tensor.shape[0] for tensor in tensors)
    padded = []
    for tensor in tensors:
        size = tensor.shape[0]
        canvas = torch.full((width, width), value, dtype=tensor.dtype)
        canvas[:size, :size] = tensor
        padded.append(canvas)
    return torch.stack(padded, dim=0)


def collate_fn(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, Any]:
    """Pad a list of samples into a batch; masks mark every padded slot."""
    token_lengths = [sample["tokens"].shape[0] for sample in batch]
    node_lengths = [sample["node_kind"].shape[0] for sample in batch]
    label_lengths = [sample["label_tokens"].shape[1] for sample in batch]
    max_tokens = max(token_lengths)
    max_nodes = max(node_lengths)
    max_labels = max(label_lengths)

    def pad_tokens(sample: dict[str, Any]) -> torch.Tensor:
        tokens = sample["tokens"]
        if tokens.shape[0] == max_tokens:
            return tokens
        return torch.cat(
            [tokens, torch.full((max_tokens - tokens.shape[0],), pad_id, dtype=torch.long)]
        )

    tokens = torch.stack([pad_tokens(sample) for sample in batch], dim=0)
    token_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for row, length in enumerate(token_lengths):
        token_mask[row, :length] = True

    def pad_nodes(sample: dict[str, Any], key: str, value: float | int | bool = 0) -> torch.Tensor:
        tensor = sample[key]
        pad = max_nodes - tensor.shape[0]
        if pad:
            fill = torch.full((pad, *tensor.shape[1:]), value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, fill], dim=0)
        return tensor

    def pad_labels(sample: dict[str, Any]) -> torch.Tensor:
        labels = sample["label_tokens"]
        node_pad = max_nodes - labels.shape[0]
        col_pad = max_labels - labels.shape[1]
        if col_pad:
            fill = torch.full((labels.shape[0], col_pad), pad_id, dtype=torch.long)
            labels = torch.cat([labels, fill], dim=1)
        if node_pad:
            fill = torch.full((node_pad, max_labels), pad_id, dtype=torch.long)
            labels = torch.cat([labels, fill], dim=0)
        return labels

    node_mask = torch.zeros((len(batch), max_nodes), dtype=torch.bool)
    for row, length in enumerate(node_lengths):
        node_mask[row, :length] = True

    label_mask = torch.zeros((len(batch), max_nodes, max_labels), dtype=torch.bool)
    for row, length in enumerate(node_lengths):
        for node_index in range(length):
            width = batch[row]["label_tokens"].shape[1]
            label_mask[row, node_index, :width] = True

    out: dict[str, Any] = {
        "tokens": tokens,
        "token_mask": token_mask,
        "node_token_index": _pad_stack([s["node_token_index"] for s in batch], -1),
        "node_kind": _pad_stack([s["node_kind"] for s in batch]),
        "node_subtype": _pad_stack([s["node_subtype"] for s in batch]),
        "node_visible": _pad_stack([s["node_visible"] for s in batch]),
        "coord_bin": _pad_stack([s["coord_bin"] for s in batch]),
        "coord_offset": _pad_stack([s["coord_offset"] for s in batch]),
        "label_tokens": torch.stack([pad_labels(s) for s in batch], dim=0),
        "label_mask": label_mask,
        "node_mask": node_mask,
        "edge_adjacency": _pad_edges([s["edge_adjacency"] for s in batch]),
        "edge_order": _pad_edges([s["edge_order"] for s in batch]),
        "edge_aromatic": _pad_edges([s["edge_aromatic"] for s in batch]),
        "edge_depiction": _pad_edges([s["edge_depiction"] for s in batch]),
        "edge_variable": _pad_edges([s["edge_variable"] for s in batch]),
        "edge_wedge_direction": _pad_edges([s["edge_wedge_direction"] for s in batch]),
        "edge_mask": _pad_edges([s["edge_mask"] for s in batch]),
        "node_charge": _pad_stack([s["node_charge"] for s in batch]),
        "node_h_count": _pad_stack([s["node_h_count"] for s in batch]),
        "node_isotope": _pad_stack([s["node_isotope"] for s in batch]),
        "node_placeholder_shape": _pad_stack([s["node_placeholder_shape"] for s in batch]),
        "capacity_exceeded": torch.tensor(
            [bool(sample.get("capacity_exceeded", False)) for sample in batch], dtype=torch.bool
        ),
        "meta": [sample["meta"] for sample in batch],
    }
    attr_keys = ("charge", "h_count", "isotope", "placeholder_shape")
    out["node_field_mask"] = {
        key: _pad_stack([sample["node_field_mask"][key] for sample in batch]) for key in attr_keys
    }
    if "image" in batch[0]:
        out["image"] = torch.stack([sample["image"] for sample in batch], dim=0)
    return out


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "TargetDataset",
    "TorchDataConfig",
    "build_sample",
    "collate_fn",
    "normalize_image",
]
