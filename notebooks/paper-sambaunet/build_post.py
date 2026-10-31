"""Builds posts/series/papers/2026-10-31-sambaunet.qmd from post_template.qmd: code snippets are pulled from sambaunet.py by AST (with real line numbers),
the smoke-test output is the real captured file. Rebuild:  python3 build_post.py"""
import ast, pathlib
H = pathlib.Path(__file__).parent
SRC = (H / "sambaunet.py").read_text(); LINES = SRC.splitlines(); TREE = ast.parse(SRC)
def snippet(spec):
    name, _, meth = spec.partition(".")
    for n in ast.walk(TREE):
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name:
            b = next(m for m in n.body if isinstance(m, ast.FunctionDef) and m.name == meth) if meth else n
            return f"```python\n# sambaunet.py:{b.lineno}-{b.end_lineno}  ({spec})\n" + "\n".join(LINES[b.lineno - 1:b.end_lineno]) + "\n```"
    raise KeyError(spec)
t = (H / "post_template.qmd").read_text()
import re
t = re.sub(r"\{\{SNIP:([\w.]+)\}\}", lambda m: snippet(m.group(1)), t)
t = t.replace("{{OUT}}", (H / "out/smoke_test.txt").read_text().rstrip())
out = H.parents[1] / "posts/series/papers/2026-10-31-sambaunet.qmd"
out.write_text(t); print(out, len(t) // 1024, "KB")
