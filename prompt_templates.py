from __future__ import annotations


def format_set(nodes: list[str]) -> str:
    return "{" + ", ".join(nodes) + "}" if nodes else "{}"


def build_prompt(
    graph_id: int,
    vertices: list[str],
    directed_edges: list[list[str]],
    bidirected_edges: list[list[str]],
    x: list[str],
    y: list[str],
    z: list[str],
) -> str:
    directed_text = ", ".join(f"{src}->{dst}" for src, dst in directed_edges) if directed_edges else "none"
    bidirected_text = ", ".join(f"{src}<->{dst}" for src, dst in bidirected_edges) if bidirected_edges else "none"
    return (
        f"Graph {graph_id}: The ADMG has vertices {vertices}, directed edges {directed_text}, "
        f"and bi-directed edges {bidirected_text}. Given X={format_set(x)}, Y={format_set(y)}, "
        f"and conditioning set Z={format_set(z)}, determine whether X and Y are m-separated by Z. "
        f"Answer with 1 if m-separated and 0 if not m-separated."
    )
