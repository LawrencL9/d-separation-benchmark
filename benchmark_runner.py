from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from d_separation_oracle import is_m_separated_sets
from graph_generator import GraphConfig, generate_random_admg, named_bidirected_edges, named_directed_edges, node_names
from prompt_templates import build_prompt


def sample_set_query(
    n_nodes: int,
    rng: np.random.Generator,
    max_xy_size: int = 2,
    max_z_size: int = 3,
) -> tuple[set[int], set[int], set[int]] | None:
    nodes = list(range(n_nodes))
    rng.shuffle(nodes)

    x_size = int(rng.integers(1, max_xy_size + 1))
    y_size = int(rng.integers(1, max_xy_size + 1))
    z_size = int(rng.integers(0, max_z_size + 1))

    if x_size + y_size + z_size > n_nodes:
        return None

    x_set = set(nodes[:x_size])
    y_set = set(nodes[x_size : x_size + y_size])
    z_set = set(nodes[x_size + y_size : x_size + y_size + z_size])
    return x_set, y_set, z_set


def sample_queries(
    graph,
    names: list[str],
    rng: np.random.Generator,
    queries_per_graph: int,
    max_condition_size: int,
    max_xy_size: int,
) -> list[dict[str, object]]:
    n_nodes = graph.number_of_nodes()
    target_per_class = queries_per_graph // 2
    positives: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]] = []
    negatives: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = set()

    attempts = 0
    max_attempts = max(queries_per_graph * 200, 2000)

    # 一直采样，直到正负各凑够 target_per_class 个（或达到最大尝试次数）
    while attempts < max_attempts:
        if len(positives) >= target_per_class and len(negatives) >= target_per_class:
            break
        attempts += 1
        sample = sample_set_query(n_nodes, rng, max_xy_size=max_xy_size, max_z_size=max_condition_size)
        if sample is None:
            continue
        x_set, y_set, z_set = sample
        key = (tuple(sorted(x_set)), tuple(sorted(y_set)), tuple(sorted(z_set)))
        if key in seen:
            continue
        seen.add(key)
        label = 1 if is_m_separated_sets(graph, x_set, y_set, z_set) else 0
        item = (key[0], key[1], key[2], label)
        if label == 1 and len(positives) < target_per_class:
            positives.append(item)
        elif label == 0 and len(negatives) < target_per_class:
            negatives.append(item)

    # 组合正负样本
    selected: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]] = []
    selected.extend(positives)
    selected.extend(negatives)

    # 如果某一类不够，就用另一类补齐
    if len(selected) < queries_per_graph:
        remaining = queries_per_graph - len(selected)
        extra_attempts = 0
        while remaining > 0 and extra_attempts < max_attempts:
            extra_attempts += 1
            sample = sample_set_query(n_nodes, rng, max_xy_size=max_xy_size, max_z_size=max_condition_size)
            if sample is None:
                continue
            x_set, y_set, z_set = sample
            key = (tuple(sorted(x_set)), tuple(sorted(y_set)), tuple(sorted(z_set)))
            if key in seen:
                continue
            seen.add(key)
            label = 1 if is_m_separated_sets(graph, x_set, y_set, z_set) else 0
            selected.append((key[0], key[1], key[2], label))
            remaining -= 1

    rng.shuffle(selected)

    examples: list[dict[str, object]] = []
    for query_id, (x_set, y_set, cond_set, label) in enumerate(selected):
        examples.append(
            {
                "query_id": query_id,
                "x": [names[node] for node in x_set],
                "y": [names[node] for node in y_set],
                "z": [names[node] for node in cond_set],
                "label": label,
            }
        )

    return examples


def build_benchmark(
    num_graphs: int,
    queries_per_graph: int,
    max_condition_size: int,
    max_xy_size: int,
    graph_config: GraphConfig,
) -> dict[str, object]:
    seeds = list(range(num_graphs))
    graphs: list[dict[str, object]] = []
    flat_examples: list[dict[str, object]] = []

    total_positive = 0
    total_negative = 0
    node_counts: list[int] = []
    directed_edge_counts: list[int] = []
    bidirected_edge_counts: list[int] = []
    condition_sizes: list[int] = []
    x_sizes: list[int] = []
    y_sizes: list[int] = []
    per_graph_stats: list[dict[str, object]] = []

    for graph_id, seed in enumerate(seeds):
        graph = generate_random_admg(seed, graph_config)
        names = node_names(graph)
        directed_edges = named_directed_edges(graph, names)
        bidirected_edges = named_bidirected_edges(graph, names)
        rng = np.random.default_rng(seed + 50_000)
        queries = sample_queries(graph, names, rng, queries_per_graph, max_condition_size, max_xy_size)
        graph_positive = 0
        graph_negative = 0

        for item in queries:
            item["prompt"] = build_prompt(graph_id, names, directed_edges, bidirected_edges, item["x"], item["y"], item["z"])
            flat_examples.append(
                {
                    "graph_id": graph_id,
                    "seed": seed,
                    "vertices": names,
                    "directed_edges": directed_edges,
                    "bidirected_edges": bidirected_edges,
                    **item,
                }
            )
            condition_sizes.append(len(item["z"]))
            x_sizes.append(len(item["x"]))
            y_sizes.append(len(item["y"]))
            if item["label"] == 1:
                total_positive += 1
                graph_positive += 1
            else:
                total_negative += 1
                graph_negative += 1

        node_counts.append(graph.number_of_nodes())
        directed_edge_counts.append(graph.number_of_directed_edges())
        bidirected_edge_counts.append(graph.number_of_bidirected_edges())

        graphs.append(
            {
                "graph_id": graph_id,
                "seed": seed,
                "vertices": names,
                "directed_edges": directed_edges,
                "bidirected_edges": bidirected_edges,
                "num_nodes": graph.number_of_nodes(),
                "num_directed_edges": graph.number_of_directed_edges(),
                "num_bidirected_edges": graph.number_of_bidirected_edges(),
                "queries": queries,
            }
        )
        per_graph_stats.append(
            {
                "graph_id": graph_id,
                "seed": seed,
                "num_nodes": graph.number_of_nodes(),
                "num_directed_edges": graph.number_of_directed_edges(),
                "num_bidirected_edges": graph.number_of_bidirected_edges(),
                "num_queries": len(queries),
                "positive_queries": graph_positive,
                "negative_queries": graph_negative,
            }
        )

    return {
        "metadata": {
            "task": "m_separation",
            "num_graphs": num_graphs,
            "queries_per_graph": queries_per_graph,
            "max_condition_size": max_condition_size,
            "max_xy_size": max_xy_size,
            "label_meaning": {
                "1": "m-separated",
                "0": "not m-separated",
            },
        },
        "seeds": seeds,
        "graphs": graphs,
        "examples": flat_examples,
        "summary": {
            "total_examples": len(flat_examples),
            "positive_examples": total_positive,
            "negative_examples": total_negative,
            "positive_ratio": round(total_positive / max(len(flat_examples), 1), 4),
            "negative_ratio": round(total_negative / max(len(flat_examples), 1), 4),
            "avg_nodes_per_graph": round(sum(node_counts) / max(len(node_counts), 1), 4),
            "avg_directed_edges_per_graph": round(sum(directed_edge_counts) / max(len(directed_edge_counts), 1), 4),
            "avg_bidirected_edges_per_graph": round(sum(bidirected_edge_counts) / max(len(bidirected_edge_counts), 1), 4),
            "avg_condition_size": round(sum(condition_sizes) / max(len(condition_sizes), 1), 4),
            "avg_x_set_size": round(sum(x_sizes) / max(len(x_sizes), 1), 4),
            "avg_y_set_size": round(sum(y_sizes) / max(len(y_sizes), 1), 4),
            "min_nodes": min(node_counts) if node_counts else 0,
            "max_nodes": max(node_counts) if node_counts else 0,
            "min_directed_edges": min(directed_edge_counts) if directed_edge_counts else 0,
            "max_directed_edges": max(directed_edge_counts) if directed_edge_counts else 0,
            "min_bidirected_edges": min(bidirected_edge_counts) if bidirected_edge_counts else 0,
            "max_bidirected_edges": max(bidirected_edge_counts) if bidirected_edge_counts else 0,
            "per_graph_stats": per_graph_stats,
        },
    }


def save_examples_csv(examples: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "graph_id",
        "seed",
        "vertices",
        "directed_edges",
        "bidirected_edges",
        "query_id",
        "x",
        "y",
        "z",
        "label",
        "prompt",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in examples:
            writer.writerow(
                {
                    "graph_id": row["graph_id"],
                    "seed": row["seed"],
                    "vertices": json.dumps(row["vertices"], ensure_ascii=False),
                    "directed_edges": json.dumps(row["directed_edges"], ensure_ascii=False),
                    "bidirected_edges": json.dumps(row["bidirected_edges"], ensure_ascii=False),
                    "query_id": row["query_id"],
                    "x": json.dumps(row["x"], ensure_ascii=False),
                    "y": json.dumps(row["y"], ensure_ascii=False),
                    "z": json.dumps(row["z"], ensure_ascii=False),
                    "label": row["label"],
                    "prompt": row["prompt"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 100-graph benchmark for m-separation reasoning in ADMGs.")
    parser.add_argument("--num-graphs", type=int, default=100, help="Number of ADMGs to generate.")
    parser.add_argument("--queries-per-graph", type=int, default=12, help="Number of m-separation queries per graph.")
    parser.add_argument("--max-condition-size", type=int, default=3, help="Maximum conditioning set size for Z.")
    parser.add_argument("--max-xy-size", type=int, default=2, help="Maximum size of the X and Y node sets.")
    parser.add_argument("--min-nodes", type=int, default=4, help="Minimum number of observed nodes in each ADMG.")
    parser.add_argument("--max-nodes", type=int, default=8, help="Maximum number of observed nodes in each ADMG.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("m_separation_100_graph_benchmark.json"),
        help="Path to the output JSON benchmark file.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("m_separation_100_graph_benchmark.csv"),
        help="Path to the flat CSV benchmark file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_config = GraphConfig(min_nodes=args.min_nodes, max_nodes=args.max_nodes)
    benchmark = build_benchmark(
        num_graphs=args.num_graphs,
        queries_per_graph=args.queries_per_graph,
        max_condition_size=args.max_condition_size,
        max_xy_size=args.max_xy_size,
        graph_config=graph_config,
    )
    args.output.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    save_examples_csv(benchmark["examples"], args.csv_output)
    print(f"Saved benchmark to: {args.output.resolve()}")
    print(f"Saved CSV to: {args.csv_output.resolve()}")
    print(f"Graphs: {benchmark['metadata']['num_graphs']}")
    print(f"Examples: {benchmark['summary']['total_examples']}")
    print(f"Positive labels: {benchmark['summary']['positive_examples']}")
    print(f"Negative labels: {benchmark['summary']['negative_examples']}")


if __name__ == "__main__":
    main()
