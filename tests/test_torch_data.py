"""Tests for the torch dataset, tensor contract and collate."""

from __future__ import annotations

import io
import json
import tarfile

import pytest
import torch
from rdkit import Chem

from markushscribe import schema
from markushscribe.chemistry import mol_to_ir
from markushscribe.dataset import RenderConfig, render_sample
from markushscribe.tokenizer import build_vocab, decode_tokens
from markushscribe.torch_data import TargetDataset, TorchDataConfig, collate_fn


def _write_shard(path, records):
    with tarfile.open(path, "w") as archive:
        for stem, record, seed in records:
            _result, image, target = render_sample(record, seed, RenderConfig(rotate=False))
            schema.validate_target(target)
            png = io.BytesIO()
            image.save(png, "PNG")
            for name, payload in (
                (f"{stem}.png", png.getvalue()),
                (f"{stem}.json", json.dumps(target).encode("utf-8")),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def _record(smiles, source_id):
    ir = mol_to_ir(Chem.MolFromSmiles(smiles), source_smiles=smiles, source_id=source_id)
    return ir.to_dict()


@pytest.fixture()
def shard(tmp_path):
    path = tmp_path / "shard_00000.tar"
    _write_shard(
        path,
        [
            ("benzene_0", _record("c1ccccc1", "benzene"), 0),
            ("ethanol_0", _record("CCO", "ethanol"), 1),
            ("na_0", _record("[Na+]", "na"), 2),
        ],
    )
    return path


@pytest.fixture()
def dataset(shard):
    graphs = []
    import tarfile as _tar

    with _tar.open(shard) as archive:
        for name in archive.getnames():
            if name.endswith(".json"):
                payload = json.loads(archive.extractfile(name).read())
                graphs.append(schema.parse_target(payload))
    vocab = build_vocab(graphs, extra_chars="0123456789+-")
    return TargetDataset([shard], vocab, TorchDataConfig(image_size=64))


def test_dataset_length_and_keys(dataset):
    assert len(dataset) == 3
    sample = dataset[0]
    expected = {
        "image",
        "tokens",
        "node_token_index",
        "node_kind",
        "node_subtype",
        "node_visible",
        "coord_bin",
        "coord_offset",
        "label_tokens",
        "edge_adjacency",
        "edge_order",
        "edge_aromatic",
        "edge_depiction",
        "edge_variable",
        "edge_wedge_direction",
        "edge_mask",
        "node_charge",
        "node_h_count",
        "node_isotope",
        "node_placeholder_shape",
        "node_field_mask",
        "capacity_exceeded",
        "meta",
        "uid",
    }
    assert expected <= set(sample)
    assert sample["image"].shape == (3, 64, 64)


def test_token_stream_matches_schema(dataset):
    sample = dataset[0]
    _image, payload, _uid = dataset.raw(0)
    graph = schema.parse_target(payload)
    decoded = decode_tokens(sample["tokens"].tolist(), dataset.vocab)
    assert len(decoded) == len(graph.nodes) == len(sample["node_kind"])


def test_edge_targets_are_symmetric_and_masked(dataset):
    sample = dataset[1]  # ethanol: 3 nodes, 2 bonds
    adjacency = sample["edge_adjacency"]
    assert torch.equal(adjacency, adjacency.T)
    assert sample["edge_mask"].sum().item() == 2 * 2  # two undirected bonds
    assert (sample["edge_order"] > 0).sum().item() == 4
    non_edges = ~sample["edge_mask"]
    assert (sample["edge_order"][non_edges] == 0).all()


def test_single_node_has_no_edges(dataset):
    sample = dataset[2]  # [Na+]
    assert sample["node_kind"].shape[0] == 1
    assert not sample["edge_mask"].any()


def test_collate_shapes_and_masks(dataset):
    batch = collate_fn([dataset[0], dataset[2]], pad_id=dataset.vocab.pad_id)
    assert batch["tokens"].shape[0] == 2
    assert batch["tokens"].shape[1] == max(s["tokens"].numel() for s in (dataset[0], dataset[2]))
    assert batch["image"].shape[0] == 2
    assert batch["node_mask"].shape == batch["node_kind"].shape
    assert batch["edge_adjacency"].shape == (2, 6, 6)
    assert batch["edge_adjacency"].shape[-1] == batch["node_mask"].shape[1]
    assert batch["token_mask"].dtype == torch.bool
    assert batch["label_mask"].shape[:2] == batch["node_mask"].shape
    assert len(batch["meta"]) == 2
    # padded node slots are masked out
    padded = ~batch["node_mask"]
    assert (batch["node_kind"][padded] == 0).all()


def test_capacity_exceeded_flag(shard):
    import tarfile as _tar

    graphs = []
    with _tar.open(shard) as archive:
        for name in archive.getnames():
            if name.endswith(".json"):
                graphs.append(schema.parse_target(json.loads(archive.extractfile(name).read())))
    vocab = build_vocab(graphs, extra_chars="0123456789+-")
    small = TargetDataset([shard], vocab, TorchDataConfig(image_size=32, q_capacity=1))
    flags = {
        sample["uid"]: sample["capacity_exceeded"] for sample in map(small.__getitem__, range(3))
    }
    assert flags["benzene_0"] is True
    assert flags["na_0"] is False


def test_collate_capacity_flags(dataset):
    batch = collate_fn([dataset[0], dataset[2]])
    assert batch["capacity_exceeded"].dtype == torch.bool
    assert batch["capacity_exceeded"].shape == (2,)
