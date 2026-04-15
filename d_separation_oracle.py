from __future__ import annotations

from collections import deque
from typing import Iterable

import networkx as nx


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
