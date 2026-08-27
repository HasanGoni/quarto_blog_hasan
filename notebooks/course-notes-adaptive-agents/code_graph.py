"""Code Knowledge Graph demo — open-source reproduction of Lessons 3-4.

Builds a property graph (files + functions, import/call/co-edit edges) over a
real Python codebase, anchors a natural-language query with TF-IDF cosine
similarity (no paid embeddings API needed), and ranks the graph with
personalized PageRank from that anchor.

Run:
    uv run code_graph.py --repo /path/to/httpie --package httpie --query "where do we build the request headers?"
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def iter_py_files(package_root: Path) -> list[Path]:
    return sorted(p for p in package_root.rglob("*.py") if "__pycache__" not in p.parts)


def module_name(package_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(package_root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_import_edges(files: list[Path], package_root: Path) -> list[tuple[str, str]]:
    mod_to_file = {module_name(package_root, f): f for f in files}
    edges = []
    for f in files:
        src_mod = module_name(package_root, f)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                prefix = "." * (node.level or 0)
                targets = [f"{prefix}{node.module}"] if node.level else [node.module]
                if node.level:
                    # relative import: resolve against the importing package
                    base_parts = src_mod.split(".")[: -node.level]
                    targets = [".".join(base_parts + [node.module])] if node.module else [".".join(base_parts)]
            for t in targets:
                # match by longest known-module prefix (handles `import httpie.cli` etc.)
                candidates = [m for m in mod_to_file if t == m or t.startswith(m + ".") or m.startswith(t + ".")]
                for c in candidates:
                    if c != src_mod:
                        edges.append((str(f.relative_to(package_root.parent)),
                                       str(mod_to_file[c].relative_to(package_root.parent))))
    return sorted(set(edges))


def collect_functions(files: list[Path], package_root: Path) -> dict[str, list[str]]:
    """file(relpath) -> [qualified function/method names defined in it]."""
    out: dict[str, list[str]] = {}
    for f in files:
        rel = str(f.relative_to(package_root.parent))
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
        out[rel] = names
    return out


def build_call_edges(files: list[Path], package_root: Path, func_defs: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """(caller_symbol, callee_symbol, callee_file) using best-effort global name matching."""
    name_to_owners: dict[str, list[str]] = defaultdict(list)
    for rel, names in func_defs.items():
        for n in names:
            name_to_owners[n].append(rel)

    edges = []
    for f in files:
        rel = str(f.relative_to(package_root.parent))
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue

        current_func = [f"{rel}::<module>"]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller = f"{rel}::{node.name}"
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        callee_name = None
                        if isinstance(child.func, ast.Name):
                            callee_name = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            callee_name = child.func.attr
                        if callee_name and callee_name in name_to_owners:
                            owners = name_to_owners[callee_name]
                            # ambiguous global matches (>2 owners) are too noisy to trust — skip
                            if len(owners) <= 2:
                                for owner_file in owners:
                                    callee = f"{owner_file}::{callee_name}"
                                    if callee != caller:
                                        edges.append((caller, callee, owner_file))
    return sorted(set(edges))


def build_coedit_edges(repo_root: Path, package_subdir: str, max_commits: int = 200) -> Counter:
    out = subprocess.run(
        ["git", "log", f"-{max_commits}", "--name-only", "--pretty=format:__COMMIT__", "--", package_subdir],
        cwd=repo_root, capture_output=True, text=True, check=True,
    ).stdout
    commits: list[list[str]] = []
    current: list[str] = []
    for line in out.splitlines():
        if line == "__COMMIT__":
            if current:
                commits.append(current)
            current = []
        elif line.strip().endswith(".py"):
            current.append(f"{package_subdir}/{line.strip()[len(package_subdir) + 1:]}" if line.strip().startswith(package_subdir) else line.strip())
    if current:
        commits.append(current)

    pair_counts: Counter = Counter()
    for files_in_commit in commits:
        uniq = sorted(set(files_in_commit))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair_counts[(uniq[i], uniq[j])] += 1
    return pair_counts


def build_graph(repo_root: Path, package: str, min_coedit: int = 2) -> nx.DiGraph:
    package_root = repo_root / package
    files = iter_py_files(package_root)

    g = nx.DiGraph()
    for f in files:
        rel = str(f.relative_to(repo_root))
        g.add_node(rel, kind="file", label=rel)

    for src, dst in build_import_edges(files, package_root):
        if g.has_node(src) and g.has_node(dst):
            g.add_edge(src, dst, kind="import")

    func_defs = collect_functions(files, package_root)
    for rel, names in func_defs.items():
        for n in names:
            sym = f"{rel}::{n}"
            g.add_node(sym, kind="symbol", label=n)
            g.add_edge(rel, sym, kind="contains")

    for caller, callee, callee_file in build_call_edges(files, package_root, func_defs):
        if g.has_node(caller) and g.has_node(callee):
            g.add_edge(caller, callee, kind="call")

    coedits = build_coedit_edges(repo_root, package)
    for (a, b), count in coedits.items():
        if count >= min_coedit and g.has_node(a) and g.has_node(b):
            g.add_edge(a, b, kind="co_edit", weight=count)
            g.add_edge(b, a, kind="co_edit", weight=count)

    return g


def anchor_from_query(g: nx.DiGraph, query: str) -> tuple[str, dict[str, float]]:
    file_nodes = [n for n, d in g.nodes(data=True) if d["kind"] == "file"]
    repo_root = Path(g.graph["repo_root"])
    texts = [(repo_root / n).read_text(encoding="utf-8", errors="ignore") for n in file_nodes]

    vec = TfidfVectorizer(stop_words="english", max_features=4000)
    doc_matrix = vec.fit_transform(texts)
    query_vec = vec.transform([query])
    sims = cosine_similarity(query_vec, doc_matrix)[0]
    scored = dict(zip(file_nodes, sims))
    anchor = max(scored, key=scored.get)
    return anchor, scored


def run(repo: Path, package: str, query: str, out_json: Path, anchor_override: str | None = None):
    g = build_graph(repo, package)
    g.graph["repo_root"] = str(repo)
    anchor, sims = anchor_from_query(g, query)
    if anchor_override:
        anchor = anchor_override
        sims = {anchor: float("nan")}

    personalization = {n: 0.0 for n in g.nodes}
    personalization[anchor] = 1.0
    undirected = g.to_undirected()
    scores = nx.pagerank(undirected, alpha=0.85, personalization=personalization)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    print(f"Query: {query!r}")
    print(f"Anchor (TF-IDF similarity {sims[anchor]:.3f}): {anchor}")
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("\nTop 15 by personalized PageRank from anchor:")
    for node, score in ranked[:15]:
        kind = g.nodes[node]["kind"]
        print(f"  {score:.5f}  [{kind:6s}]  {node}")

    export = {
        "anchor": anchor,
        "query": query,
        "nodes": [
            {"id": n, "kind": d["kind"], "label": d["label"], "score": scores.get(n, 0.0)}
            for n, d in g.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "kind": d["kind"]}
            for u, v, d in g.edges(data=True)
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(export, indent=None))
    print(f"\nExported graph data -> {out_json}")
    return g, anchor, scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", type=Path, default=Path("graph_data.json"))
    ap.add_argument("--anchor-override", default=None, help="Skip TF-IDF anchoring, force this file node as the anchor")
    args = ap.parse_args()
    run(args.repo, args.package, args.query, args.out, args.anchor_override)
