"""Shared fixtures for the M1 test-suite."""

from __future__ import annotations

import io
import json
import tarfile

import pytest
from rdkit import Chem

from markushscribe import schema
from markushscribe.chemistry import mol_to_ir
from markushscribe.dataset import RenderConfig, render_sample
from markushscribe.tokenizer import build_vocab
from markushscribe.torch_data import TargetDataset, TorchDataConfig, collate_fn

TINY_SMILES = ("c1ccccc1", "CCO", "[Na+]")


def build_shard(path, smiles_list=TINY_SMILES, start_seed=0):
    """Render a few plain molecules into one tar shard."""
    with tarfile.open(path, "w") as archive:
        for offset, smiles in enumerate(smiles_list):
            record = mol_to_ir(
                Chem.MolFromSmiles(smiles), source_smiles=smiles, source_id=f"mol{offset}"
            ).to_dict()
            _result, image, target = render_sample(
                record, start_seed + offset, RenderConfig(rotate=False)
            )
            schema.validate_target(target)
            png = io.BytesIO()
            image.save(png, "PNG")
            stem = f"mol{offset}_{start_seed + offset}"
            for name, payload in (
                (f"{stem}.png", png.getvalue()),
                (f"{stem}.json", json.dumps(target).encode("utf-8")),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def read_graphs(shard):
    graphs = []
    with tarfile.open(shard) as archive:
        for name in sorted(archive.getnames()):
            if name.endswith(".json"):
                graphs.append(schema.parse_target(json.loads(archive.extractfile(name).read())))
    return graphs


def build_vocab_from_shard(shard):
    return build_vocab(read_graphs(shard), extra_chars="0123456789+-")


@pytest.fixture()
def tiny_shard(tmp_path):
    path = tmp_path / "shard_00000.tar"
    build_shard(path)
    return path


@pytest.fixture()
def tiny_graphs(tiny_shard):
    return read_graphs(tiny_shard)


@pytest.fixture()
def tiny_vocab(tiny_shard):
    return build_vocab_from_shard(tiny_shard)


@pytest.fixture()
def tiny_batch(tiny_shard, tiny_vocab):
    dataset = TargetDataset(
        [tiny_shard],
        tiny_vocab,
        TorchDataConfig(image_size=64, q_capacity=32, t_capacity=16),
    )
    return collate_fn([dataset[i] for i in range(len(dataset))], pad_id=tiny_vocab.pad_id)
