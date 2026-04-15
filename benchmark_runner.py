from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from d_separation_oracle import is_d_separated
from graph_generator import GraphConfig, generate_random_dag, named_edges, node_names
from prompt_templates import build_prompt


def sample_queries(
    dag,
    names: list[str],
    rng: np.random.Generator,
    queries_per_graph: int,
    max_condition_size: int,
) -> list[dict[str, object]]:
    n_nodes = dag.number_of_nodes()
    candidates: list[tuple[int, int, tuple[int, ...], int]] = []

    for x, y in combinations(range(n_nodes), 2):
        others = [node for node in range(n_nodes) if node not in (x, y)]
        max_size = min(max_condition_size, len(others))
        for size in range(max_size + 1):
            for cond_set in combinations(others, size):
                label = 1 if is_d_separated(dag, x, y, cond_set) else 0
                candidates.append((x, y, tuple(cond_set), label))

    rng.shuffle(candidates)

    positives = [item for item in candidates if item[3] == 1]
    negatives = [item for item in candidates if item[3] == 0]

    selected: list[tuple[int, int, tuple[int, ...], int]] = []
    half = queries_per_graph // 2
    selected.extend(positives[:half])
    selected.extend(negatives[:half])

    remaining = queries_per_graph - len(selected)
    if remaining > 0:
        used = set(selected)
        leftovers = [item for item in candidates if item not in used]
        selected.extend(leftovers[:remaining])

    rng.shuffle(selected)

    examples: list[dict[str, object]] = []
    for query_id, (x, y, cond_set, label) in enumerate(selected):
        z_names = [names[node] for node in cond_set]
        examples.append(
            {
                "query_id": query_id,
                "x": names[x],
                "y": names[y],
                "z": z_names,
                "label": label,
            }
        )

    return examples


def build_benchmark(
    num_graphs: int,
    queries_per_graph: int,
    max_condition_size: int,
    graph_config: GraphConfig,
) -> dict[str, object]:
    seeds = list(range(num_graphs))
    graphs: list[dict[str, object]] = []
    flat_examples: list[dict[str, object]] = []

    total_positive = 0
    total_negative = 0
    node_counts: list[int] = []
    edge_counts: list[int] = []
    condition_sizes: list[int] = []
    per_graph_stats: list[dict[str, object]] = []

    for graph_id, seed in enumerate(seeds):
        dag = generate_random_dag(seed, graph_config)
        names = node_names(dag)
        edges = named_edges(dag, names)
        rng = np.random.default_rng(seed + 50_000)
        queries = sample_queries(dag, names, rng, queries_per_graph, max_condition_size)
        graph_positive = 0
        graph_negative = 0

        for item in queries:
            item["prompt"] = build_prompt(graph_id, names, edges, item["x"], item["y"], item["z"])
            flat_examples.append(
                {
                    "graph_id": graph_id,
                    "seed": seed,
                    "vertices": names,
                    "edges": edges,
                    **item,
                }
            )
            condition_sizes.append(len(item["z"]))
            if item["label"] == 1:
                total_positive += 1
                graph_positive += 1
            else:
                total_negative += 1
                graph_negative += 1

        node_counts.append(dag.number_of_nodes())
        edge_counts.append(dag.number_of_edges())

        graphs.append(
            {
                "graph_id": graph_id,
                "seed": seed,
                "vertices": names,
                "edges": edges,
                "num_nodes": dag.number_of_nodes(),
                "num_edges": dag.number_of_edges(),
                "queries": queries,
            }
        )
        per_graph_stats.append(
            {
                "graph_id": graph_id,
                "seed": seed,
                "num_nodes": dag.number_of_nodes(),
                "num_edges": dag.number_of_edges(),
                "num_queries": len(queries),
                "positive_queries": graph_positive,
                "negative_queries": graph_negative,
            }
        )

    return {
        "metadata": {
            "task": "d_separation",
            "num_graphs": num_graphs,
            "queries_per_graph": queries_per_graph,
            "max_condition_size": max_condition_size,
            "label_meaning": {
                "1": "d-separated",
                "0": "not d-separated",
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
            "avg_edges_per_graph": round(sum(edge_counts) / max(len(edge_counts), 1), 4),
            "avg_condition_size": round(sum(condition_sizes) / max(len(condition_sizes), 1), 4),
            "min_nodes": min(node_counts) if node_counts else 0,
            "max_nodes": max(node_counts) if node_counts else 0,
            "min_edges": min(edge_counts) if edge_counts else 0,
            "max_edges": max(edge_counts) if edge_counts else 0,
            "per_graph_stats": per_graph_stats,
        },
    }


def save_examples_csv(examples: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "graph_id",
        "seed",
        "vertices",
        "edges",
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
                    "edges": json.dumps(row["edges"], ensure_ascii=False),
                    "query_id": row["query_id"],
                    "x": row["x"],
                    "y": row["y"],
                    "z": json.dumps(row["z"], ensure_ascii=False),
                    "label": row["label"],
                    "prompt": row["prompt"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 100-graph benchmark for d-separation reasoning.")
    parser.add_argument("--num-graphs", type=int, default=100, help="Number of DAGs to generate.")
    parser.add_argument("--queries-per-graph", type=int, default=12, help="Number of d-separation queries per graph.")
    parser.add_argument("--max-condition-size", type=int, default=3, help="Maximum conditioning set size.")
    parser.add_argument("--min-nodes", type=int, default=4, help="Minimum number of nodes in each DAG.")
    parser.add_argument("--max-nodes", type=int, default=8, help="Maximum number of nodes in each DAG.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("d_separation_100_graph_benchmark.json"),
        help="Path to the output JSON benchmark file.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("d_separation_100_graph_benchmark.csv"),
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
