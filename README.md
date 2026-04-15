# d-Separation Benchmark

This is a separate version of the project focused only on `d-separation`.

The goal is to build a benchmark dataset for reasoning over DAGs:
- generate 100 DAGs
- sample multiple `(X, Y, Z)` queries from each DAG
- automatically label whether `X` and `Y` are d-separated by `Z`
- save the benchmark in a standard JSON format

In this benchmark, each example is a binary classification task:
- `1` means `X` and `Y` are d-separated by `Z`
- `0` means `X` and `Y` are not d-separated by `Z`

## Files

- `graph_generator.py`
  Generates random DAGs for the benchmark.

- `d_separation_oracle.py`
  Computes whether two nodes are d-separated given a conditioning set.

- `prompt_templates.py`
  Builds a simple benchmark prompt for each query.

- `benchmark_runner.py`
  Main entry point. Generates the benchmark and saves the output file.

- `d_separation_100_graph_benchmark.json`
  Example benchmark output after running the script.

- `d_separation_100_graph_benchmark.csv`
  A flat table version of the benchmark that is easier to inspect and analyze.

## Run

```bash
python benchmark_runner.py
```

This will generate:
- 100 graphs
- 12 queries per graph by default
- a JSON benchmark file
- a CSV benchmark file

You can also change the benchmark size:

```bash
python benchmark_runner.py --num-graphs 50 --queries-per-graph 20 --max-condition-size 2
```

## Output Format

The output JSON contains:
- `metadata`
- `seeds`
- `graphs`
- `examples`
- `summary`

Each example contains:
- graph id
- node set
- directed edges
- `X`
- `Y`
- conditioning set `Z`
- binary label
- benchmark prompt

## What Makes This A Benchmark

This project is designed as a reusable benchmark rather than just a one-off script:
- it generates a fixed-size collection of DAGs
- it samples multiple reasoning tasks from each DAG
- it provides ground-truth binary labels automatically
- it exports both structured JSON and flat CSV formats
- it reports summary statistics over the whole dataset

## Summary Statistics

The JSON summary includes:
- total number of examples
- positive and negative label counts
- positive and negative ratios
- average number of nodes per graph
- average number of edges per graph
- average conditioning set size
- per-graph statistics
