from __future__ import annotations

from collections import deque
from typing import Iterable

import networkx as nx

from graph_generator import ADMGGraph


def ancestors_of_set(dag: nx.DiGraph, nodes: set[int]) -> set[int]:
    ancestors = set(nodes)
    stack = list(nodes)
    while stack:
        node = stack.pop()
        for parent in dag.predecessors(node):
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    return ancestors


def is_d_separated(dag: nx.DiGraph, x: int, y: int, conditioned: Iterable[int]) -> bool:
    observed = set(conditioned)
    ancestors = ancestors_of_set(dag, observed)
    queue = deque([(x, "up"), (x, "down")])
    visited: set[tuple[int, str]] = set()
    reachable: set[int] = set()

    while queue:
        node, direction = queue.popleft()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))

        if node not in observed:
            reachable.add(node)

        if direction == "up":
            if node not in observed:
                for parent in dag.predecessors(node):
                    queue.append((parent, "up"))
                for child in dag.successors(node):
                    queue.append((child, "down"))
        else:
            if node not in observed:
                for child in dag.successors(node):
                    queue.append((child, "down"))
            if node in ancestors:
                for parent in dag.predecessors(node):
                    queue.append((parent, "up"))

    return y not in reachable


def is_d_separated_sets(dag: nx.DiGraph, x_set: set[int], y_set: set[int], conditioned: set[int]) -> bool:
    for x in x_set:
        for y in y_set:
            if not is_d_separated(dag, x, y, conditioned):
                return False
    return True


def canonical_dag_from_admg(graph: ADMGGraph) -> tuple[nx.DiGraph, set[int]]:
    canonical = graph.directed_graph.copy()
    latent_nodes: set[int] = set()
    next_node = canonical.number_of_nodes()

    for a, b in sorted(graph.bidirected_edges):
        latent = next_node
        next_node += 1
        latent_nodes.add(latent)
        canonical.add_node(latent)
        canonical.add_edge(latent, a)
        canonical.add_edge(latent, b)

    return canonical, latent_nodes


def is_m_separated_sets(graph: ADMGGraph, x_set: set[int], y_set: set[int], conditioned: set[int]) -> bool:
    canonical, latent_nodes = canonical_dag_from_admg(graph)
    observed_conditioned = {node for node in conditioned if node not in latent_nodes}
    return is_d_separated_sets(canonical, x_set, y_set, observed_conditioned)
