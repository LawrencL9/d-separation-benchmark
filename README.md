# m-Separation Benchmark

This is a separate benchmark version of the project focused on `m-separation` in ADMGs.

Compared with the earlier DAG-only benchmark, this version supports:
- node sets for `X`, `Y`, and `Z`
- bi-directed edges such as `V1<->V2`
- latent-confounder semantics through ADMG to canonical DAG conversion

The goal is to build a reusable benchmark dataset for reasoning over mixed causal graphs:
- generate 100 random ADMGs
- sample multiple set-valued `(X, Y, Z)` queries from each graph
- automatically label whether `X` and `Y` are m-separated by `Z`
- save the benchmark in both JSON and CSV formats

In this benchmark, each example is a binary classification task:
- `1` means `X` and `Y` are m-separated by `Z`
- `0` means `X` and `Y` are not m-separated by `Z`

## Files

- `graph_generator.py`
  Generates random ADMGs with directed edges and optional bi-directed edges.

- `d_separation_oracle.py`
  Computes d-separation on DAGs and m-separation on ADMGs.
  It converts each bi-directed edge into a latent parent in a canonical DAG and then reuses the d-separation oracle.

- `prompt_templates.py`
  Builds a benchmark prompt for each query, including directed edges, bi-directed edges, and set-valued `X`, `Y`, `Z`.

- `benchmark_runner.py`
  Main entry point. Generates the benchmark, samples set-valued queries, labels them, and saves the output files.

- `m_separation_100_graph_benchmark.json`
  Example benchmark output after running the script.

- `m_separation_100_graph_benchmark.csv`
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
python benchmark_runner.py --num-graphs 50 --queries-per-graph 20 --max-condition-size 2 --max-xy-size 2
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
- bi-directed edges
- `X`
- `Y`
- conditioning set `Z`
- binary label
- benchmark prompt

`X`, `Y`, and `Z` are all stored as arrays of node names. For example:

```json
{
  "x": ["V0", "V1"],
  "y": ["V3"],
  "z": ["V2"]
}
```

## What Makes This A Benchmark

This project is designed as a reusable benchmark rather than just a one-off script:
- it generates a fixed-size collection of ADMGs
- it samples multiple reasoning tasks from each graph
- it provides ground-truth binary labels automatically
- it exports both structured JSON and flat CSV formats
- it reports summary statistics over the whole dataset

## Summary Statistics

The JSON summary includes:
- total number of examples
- positive and negative label counts
- positive and negative ratios
- average number of nodes per graph
- average number of directed edges per graph
- average number of bi-directed edges per graph
- average conditioning set size
- average `X` set size
- average `Y` set size
- per-graph statistics
