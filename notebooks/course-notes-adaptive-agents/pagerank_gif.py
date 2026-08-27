"""Renders a looping GIF of personalized PageRank propagating outward from an
anchor node, iteration by iteration, over the real code graph built by
code_graph.py. Manual power iteration (not nx.pagerank's black-box solver) so
every animation frame is an honest, inspectable step of the algorithm.

Run:
    uv run pagerank_gif.py --repo /path/to/httpie --package httpie --anchor httpie/sessions.py --out pagerank.gif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

from code_graph import build_graph


def ego_subgraph(g: nx.DiGraph, anchor: str, radius: int = 2) -> nx.Graph:
    undirected = g.to_undirected()
    return nx.ego_graph(undirected, anchor, radius=radius)


def power_iteration_frames(g: nx.Graph, anchor: str, alpha: float = 0.85, n_iters: int = 12) -> list[dict[str, float]]:
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    adj = nx.to_numpy_array(g, nodelist=nodes)
    degrees = adj.sum(axis=1)
    degrees[degrees == 0] = 1.0
    transition = (adj / degrees[:, None]).T  # column j -> distribution over neighbors

    personalization = np.zeros(n)
    personalization[idx[anchor]] = 1.0

    scores = personalization.copy()
    frames = [dict(zip(nodes, scores))]
    for _ in range(n_iters):
        scores = alpha * (transition @ scores) + (1 - alpha) * personalization
        scores = scores / scores.sum()
        frames.append(dict(zip(nodes, scores)))
    return frames


def render_gif(g: nx.Graph, anchor: str, frames: list[dict[str, float]], out_path: Path, pos: dict):
    max_score = max(max(f.values()) for f in frames)

    tmp_frames = []
    for i, scores in enumerate(frames):
        fig, ax = plt.subplots(figsize=(7, 6))
        sizes = [200 + 4000 * (scores[n] / max_score) for n in g.nodes]
        colors = ["#d62728" if n == anchor else "#1f77b4" for n in g.nodes]
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, width=0.8)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.85)
        labels = {n: Path(n.split("::")[0]).name + ("::" + n.split("::")[1] if "::" in n else "") for n in g.nodes}
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=6)
        ax.set_title(f"Personalized PageRank from anchor — iteration {i}/{len(frames) - 1}")
        ax.axis("off")
        frame_path = out_path.parent / f"_frame_{i:02d}.png"
        fig.savefig(frame_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        tmp_frames.append(frame_path)

    images = [Image.open(p) for p in tmp_frames]
    images[0].save(
        out_path, save_all=True, append_images=images[1:] + [images[-1]] * 4,
        duration=550, loop=0,
    )
    for p in tmp_frames:
        p.unlink()
    print(f"Wrote {out_path} ({len(frames)} iterations, {len(g.nodes)} nodes in ego subgraph)")


def export_interactive_json(g: nx.Graph, anchor: str, frames: list[dict[str, float]], pos: dict, out_path: Path):
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)

    def norm(v, lo, hi):
        return 0.5 if hi == lo else (v - lo) / (hi - lo)

    nodes = [
        {
            "id": n, "kind": d["kind"], "label": d["label"],
            "x": norm(pos[n][0], x_lo, x_hi), "y": norm(pos[n][1], y_lo, y_hi),
        }
        for n, d in g.nodes(data=True)
    ]
    edges = [{"source": u, "target": v} for u, v in g.edges()]
    data = {
        "anchor": anchor,
        "nodes": nodes,
        "edges": edges,
        "frames": [ {n: round(v, 6) for n, v in f.items()} for f in frames ],
    }
    out_path.write_text(json.dumps(data, indent=None))
    print(f"Wrote {out_path} ({len(nodes)} nodes, {len(frames)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("pagerank.gif"))
    args = ap.parse_args()

    full_graph = build_graph(args.repo, args.package)
    sub = ego_subgraph(full_graph, args.anchor, radius=args.radius)
    frames = power_iteration_frames(sub, args.anchor)
    pos = nx.spring_layout(sub, seed=7, k=0.9)
    render_gif(sub, args.anchor, frames, args.out, pos)
    export_interactive_json(sub, args.anchor, frames, pos, args.out.with_suffix(".json"))
