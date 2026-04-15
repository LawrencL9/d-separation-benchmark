# d-Separation Benchmark

This project is a benchmark for d-separation reasoning over directed acyclic graphs (DAGs).

The goal is to automatically generate DAGs, sample multiple `(X, Y, Z)` queries from each graph, and assign ground-truth labels indicating whether `X` and `Y` are d-separated by `Z`.

## Project Goal

This benchmark is designed to evaluate reasoning over graph structure.

For each generated DAG, the project:
- creates multiple d-separation queries
- checks whether `X` and `Y` are d-separated given conditioning set `Z`
- saves the results in reusable output formats

Each example is a binary classification task:
- `1` = d-separated
- `0` = not d-separated

## Files

- `graph_generator.py`  
  Generates random DAGs for the benchmark.

- `d_separation_oracle.py`  
  Computes whether two nodes are d-separated given a conditioning set.

- `prompt_templates.py`  
  Builds a natural-language benchmark prompt for each query.

- `benchmark_runner.py`  
  Main entry point. Generates the benchmark and saves the output.

- `d_separation_100_graph_benchmark.json`  
  Structured benchmark output.

- `d_separation_100_graph_benchmark.csv`  
  Flat table version of the benchmark, easier to inspect in Excel.

## Run

```bash
python benchmark_runner.py
