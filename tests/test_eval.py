"""Evaluation metric tests on constructed graphs."""

from __future__ import annotations

from dataclasses import replace

from markushscribe import schema
from markushscribe.evaluation import evaluate_dataset, evaluate_graph
from markushscribe.evaluation.metrics import graph_exact


def test_identical_graph_is_perfect(tiny_graphs):
    graph = tiny_graphs[0]
    report = evaluate_graph(graph, graph)
    assert report["node_f1"] == 1.0
    assert report["node_kind_accuracy"] == 1.0
    assert report["label_plain_exact"] == 1.0
    assert report["edge_f1"] == 1.0
    assert report["graph_exact"] == 1.0
    assert graph_exact(graph, graph, strict=True)


def test_changed_label_breaks_exact(tiny_graphs):
    graph = tiny_graphs[0]
    node = graph.nodes[0]
    label = schema.VisualLabel(
        visible=True, plain_text="Zz", runs=({"text": "Zz", "script": "base"},)
    )
    nodes = list(graph.nodes)
    nodes[0] = replace(node, visual_label=label)
    altered = schema.MarkushGraph(
        image_size=graph.image_size,
        nodes=tuple(nodes),
        edges=graph.edges,
        graph_annotations=graph.graph_annotations,
        masks=dict(graph.masks),
        coord_bins=graph.coord_bins,
    )
    report = evaluate_graph(graph, altered, strict_exact=True)
    assert report["graph_exact"] == 0.0
    assert report["label_plain_exact"] < 1.0


def test_evaluate_dataset_averages(tiny_graphs):
    samples = [(graph, graph) for graph in tiny_graphs]
    report = evaluate_dataset(samples)
    assert report["node_f1"] == 1.0
    assert report["graph_exact"] == 1.0
