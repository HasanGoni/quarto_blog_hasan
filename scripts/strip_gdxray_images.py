#!/usr/bin/env python3
"""Remove GDXray-derived images from posts, keeping text, numbers and plots.

Usage: strip_gdxray_images.py [--apply] POST.qmd ...

An image is removable unless its file name matches KEEP (hand-drawn sketches,
loss curves, a line drawing of a mask). Markdown images, raw <img> tags and
whole ```{=html} blocks (the before/after sliders) that contain a removable
image are replaced by a one-line notice, a notice is added under the front
matter, and the files are moved to private_assets/gdxray_removed/ (gitignored)
so nothing is lost. Dry run by default.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP = re.compile(r"eli5-sketch|loss-curve|shape-signal")
NOTICE = ("*[Figure removed: it showed GDXray pixels, or a model output made from them, and the GDXray "
          "licence does not allow redistribution.]*")
TOP = ("::: {.callout-note}\n"
       "**Images removed.** This post's figures showed frames from [GDXray](https://domingomery.ing.uc.cl/material/gdxray) "
       "or outputs of models trained on them. GDXray is free for research and education only, and redistribution is "
       "not allowed, so the figures are gone. The text, code, numbers and plots are unchanged. The new "
       "[synthetic void dataset](../agentic-factory-segmentation/01-synthetic-void-dataset.qmd) "
       "is what this blog shows from now on.\n"
       ":::\n")
MD_IMG = re.compile(r"!\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^)\s]+)[^)]*\)(\{[^}]*\})?")
HTML_FENCE = re.compile(r"```\{=html\}\n.*?\n```", re.S)
IMG_SRC = re.compile(r"<img[^>]*?src=\"([^\"]+)\"")


def removable(path):
    return not KEEP.search(Path(path).name)


def process(text, post):
    gone = []

    def fence(m):
        srcs = IMG_SRC.findall(m.group(0))
        bad = [s for s in srcs if removable(s)]
        if not bad:
            return m.group(0)
        gone.extend((post.parent / s).resolve() for s in srcs if removable(s))
        return NOTICE

    text = HTML_FENCE.sub(fence, text)

    def md(m):
        if not removable(m.group(1)):
            return m.group(0)
        gone.append((post.parent / m.group(1)).resolve())
        return NOTICE

    text = MD_IMG.sub(md, text)

    def tag(m):
        if not removable(m.group(1)):
            return m.group(0)
        gone.append((post.parent / m.group(1)).resolve())
        return NOTICE

    text = re.sub(r"<img[^>]*?src=\"([^\"]+)\"[^>]*>", tag, text)

    def vid(m):
        gone.append((post.parent / m.group(1)).resolve())
        return NOTICE

    text = re.sub(r"\{\{< video (\S+) >\}\}", vid, text)
    # front-matter thumbnail
    head, sep, body = text.partition("\n---\n")
    def fm(m):
        if removable(m.group(1)):
            gone.append((post.parent / m.group(1)).resolve())
            return ""
        return m.group(0)
    head = re.sub(r"^image:\s*(\S+)\s*\n?", lambda m: fm(m), head + "\n", flags=re.M).rstrip("\n")
    text = head + sep + body
    # collapse runs of notices separated only by blank lines
    text = re.sub(re.escape(NOTICE) + r"(\s*\n)+(?=" + re.escape(NOTICE) + ")", "", text)
    if gone and "**Images removed.**" not in text:
        head, sep, body = text.partition("\n---\n")
        text = head + sep + "\n" + TOP + body
    return text, gone


def main():
    apply = "--apply" in sys.argv
    posts = [Path(a).resolve() for a in sys.argv[1:] if not a.startswith("--")]
    allgone = {}
    for post in posts:
        text = post.read_text()
        new, gone = process(text, post)
        allgone[post] = sorted(set(gone))
        print(f"{post.relative_to(ROOT)}: {len(allgone[post])} files, {text.count(chr(10))} -> {new.count(chr(10))} lines")
        if apply and new != text:
            post.write_text(new)
    if apply:
        for post, files in allgone.items():
            series = post.parent.name
            dest = ROOT / "private_assets" / "gdxray_removed" / series
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                if f.exists():
                    shutil.move(str(f), str(dest / f.name))
                else:
                    print("  missing (moved already?):", f)


if __name__ == "__main__":
    main()
