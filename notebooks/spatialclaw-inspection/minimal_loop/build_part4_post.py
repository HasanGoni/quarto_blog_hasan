"""Assemble the Part 4 post: real code files and real captured outputs pasted in verbatim, prose from the two .md files."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT.parents[1] / "posts/series/spatialclaw-inspection/04-eyes-and-hands.qmd"
tpl = (ROOT / "minimal_loop/part4_template.qmd").read_text()
tpl = tpl.replace("{{RESULT_PROSE}}", (ROOT / "minimal_loop/part4_result_prose.md").read_text().strip())
tpl = tpl.replace("{{TRACE_PROSE}}", (ROOT / "minimal_loop/part4_trace_prose.md").read_text().strip())


def code(m):
    f = ROOT / m.group(1)
    return f"````{{.python}}\n# {m.group(1)}\n{f.read_text().rstrip()}\n````"


def out(m):
    f = ROOT / m.group(1)
    body = re.sub(r"\x1b\[[0-9;]*m", "", f.read_text()).rstrip()
    return ('```{=html}\n<details class="out">\n<summary><strong>Show output</strong></summary>\n```\n\n'
            f"```{{.text}}\n{body}\n```\n\n```{{=html}}\n</details>\n```")


text = re.sub(r"\{\{FILE:(.+?)\}\}", code, tpl)
text = re.sub(r"\{\{OUT:(.+?)\}\}", out, text)
assert "{{" not in text.replace("{{<", ""), [l for l in text.splitlines() if "{{" in l and "{{<" not in l]
POST.write_text(text)
print("wrote", POST, len(text), "bytes")
