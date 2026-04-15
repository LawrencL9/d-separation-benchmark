from __future__ import annotations


def build_prompt(graph_id: int, vertices: list[str], edges: list[list[str]], x: str, y: str, z: list[str]) -> str:
    edge_text = ", ".join(f"{src}->{dst}" for src, dst in edges)
    z_text = "{" + ", ".join(z) + "}" if z else "{}"
    return (
        f"Graph {graph_id}: The DAG has vertices {vertices} and directed edges {edge_text}. "
        f"Given X={x}, Y={y}, and conditioning set Z={z_text}, determine whether X and Y are d-separated by Z. "
        f"Answer with 1 if d-separated and 0 if not d-separated."
    )
