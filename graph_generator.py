from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class GraphConfig:
    min_nodes: int = 4
    max_nodes: int = 8
    edge_prob_low: float = 0.2
    edge_prob_high: float = 0.55


def generate_random_dag(seed: int, config: GraphConfig) -> nx.DiGraph:
    rng = np.random.default_rng(seed)

    for _ in range(100):
        n_nodes = int(rng.integers(config.min_nodes, config.max_nodes + 1))
        order = rng.permutation(n_nodes).tolist()
        edge_prob = float(rng.uniform(config.edge_prob_low, config.edge_prob_high))

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
        return nx.relabel_nodes(dag, relabel, copy=True)

    raise RuntimeError(f"Could not generate a valid DAG from seed {seed}.")


def node_names(dag: nx.DiGraph) -> list[str]:
    return [f"V{i}" for i in range(dag.number_of_nodes())]


def named_edges(dag: nx.DiGraph, names: list[str]) -> list[list[str]]:
    return [[names[src], names[dst]] for src, dst in sorted(dag.edges())]
