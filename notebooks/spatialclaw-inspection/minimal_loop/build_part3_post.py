"""Assemble the Part 3 post from the template: real code files and real captured outputs are pasted in verbatim."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT.parents[1] / "posts/series/spatialclaw-inspection/03-minimal-kernel-loop.qmd"
tpl = (ROOT / "minimal_loop/part3_template.qmd").read_text()


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
POST.write_text(text)
print("wrote", POST, len(text), "bytes")
