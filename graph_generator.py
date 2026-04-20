from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np


@dataclass
class GraphConfig:
    min_nodes: int = 4
    max_nodes: int = 8
    edge_prob_low: float = 0.2
    edge_prob_high: float = 0.55
    bidirected_prob_low: float = 0.05
    bidirected_prob_high: float = 0.25


@dataclass
class ADMGGraph:
    directed_graph: nx.DiGraph
    bidirected_edges: set[tuple[int, int]]

    @property
    def nodes(self) -> list[int]:
        return list(self.directed_graph.nodes)

    def number_of_nodes(self) -> int:
        return self.directed_graph.number_of_nodes()

    def number_of_directed_edges(self) -> int:
        return self.directed_graph.number_of_edges()

    def number_of_bidirected_edges(self) -> int:
        return len(self.bidirected_edges)


def generate_random_admg(seed: int, config: GraphConfig) -> ADMGGraph:
    rng = np.random.default_rng(seed)

    for _ in range(100):
        n_nodes = int(rng.integers(config.min_nodes, config.max_nodes + 1))
        order = rng.permutation(n_nodes).tolist()
        edge_prob = float(rng.uniform(config.edge_prob_low, config.edge_prob_high))
        bidirected_prob = float(rng.uniform(config.bidirected_prob_low, config.bidirected_prob_high))

        dag = nx.DiGraph()
        dag.add_nodes_from(range(n_nodes))

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < edge_prob:
                    dag.add_edge(order[i], order[j])

        if dag.number_of_edges() == 0:
            continue

        isolates = list(nx.isolates(dag))
        dag.remove_nodes_from(isolates)
        if dag.number_of_nodes() < config.min_nodes or dag.number_of_edges() == 0:
            continue

        relabel = {old: new for new, old in enumerate(sorted(dag.nodes))}
        dag = nx.relabel_nodes(dag, relabel, copy=True)

        bidirected_edges: set[tuple[int, int]] = set()
        nodes = list(dag.nodes)
        for a_index, a in enumerate(nodes):
            for b in nodes[a_index + 1 :]:
                if rng.random() < bidirected_prob:
                    bidirected_edges.add((min(a, b), max(a, b)))

        return ADMGGraph(directed_graph=dag, bidirected_edges=bidirected_edges)

    raise RuntimeError(f"Could not generate a valid ADMG from seed {seed}.")


def node_names(graph: ADMGGraph) -> list[str]:
    return [f"V{i}" for i in range(graph.number_of_nodes())]


def named_directed_edges(graph: ADMGGraph, names: list[str]) -> list[list[str]]:
    return [[names[src], names[dst]] for src, dst in sorted(graph.directed_graph.edges())]


def named_bidirected_edges(graph: ADMGGraph, names: list[str]) -> list[list[str]]:
    return [[names[src], names[dst]] for src, dst in sorted(graph.bidirected_edges)]
