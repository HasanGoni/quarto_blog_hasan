# CLAUDE.md

Context for any Claude Code (or other agent) session working in this repo, so you don't have
to re-derive it from scratch each time.

## What this is

Hasan Goni's technical blog. Quarto static site, published to quarto.pub, source-controlled here.
Content is computer vision, applied ML, agentic AI, and the tools used day to day — written as
multi-part **series** rather than one-off posts (see `README.md` for the series list).

- Live site: https://hasangoni.quarto.pub/hasan-blog-post
- Publish target/id: `_publish.yml`
- Author identity for frontmatter: `Hasan Goni` / `hasanme1412@gmail.com`

## Adding a new series or post — the pattern

Every series (e.g. `posts/series/papers/`, `posts/series/agentic-development/`,
`posts/series/course-notes/`) follows the same shape:

```
posts/series/<series-name>/
├── index.qmd            # series overview: motivation + "Posts in this Series" list
├── <post-1>.qmd
├── <post-2>.qmd
└── ...
```

Each post's frontmatter includes a `series:` block:

```yaml
series:
  name: "Series Display Name"
  number: 1
```

...and opens with a callout linking back to the index:

```markdown
:::{.callout-tip}
This post is part of the [Series Name](index.qmd) series.
:::
```

**Wiring a new/updated series into the site** requires editing `_quarto.yml` in two places:
1. `website.navbar.left` → the "Series" dropdown menu (one entry per series, links to its `index.qmd`)
2. `website.sidebar.contents` → a `section:` block per series listing every post in reading order

Both are manually maintained lists — nothing here is auto-generated from the directory structure.
`categories.qmd` and `tags.qmd` mostly auto-list from post frontmatter (`categories`/`tags` fields);
`categories.qmd` has a few hardcoded top-level groupings that don't need updating per-series.

**Tone** (from `.cursor/rules/quarto-blog-structure.mdc`): write like an expert data scientist
storytelling to a peer — easy to follow, humor is fine, but any genuinely complex/mathematical
point should get a concrete, easy-to-follow example rather than being hand-waved.

## Rendering and publishing

```bash
quarto render                      # full site, or pass a specific .qmd to render just that file
quarto publish quarto-pub --no-prompt   # push the rendered site to the live quarto.pub URL
```

Always `quarto render` the touched files (and ideally the whole site) before publishing, and check
the tail of the output for `ERROR`/`WARN` lines. **Publishing is a real deploy to a live, public
site — always confirm with the user before running `quarto publish`.**

## Python environments — one `uv`-managed venv per purpose

This repo does **not** use one shared Python environment. Each subproject that needs Python gets
its own purpose-named venv at the location that subproject's tooling expects, kept out of git:

| Venv | Used by | Package manager |
|---|---|---|
| `.venv-quarto` | Rendering the CV-foundations notebooks / general Quarto Python execution | pip + `requirements.txt` |
| `.venv-boogu` | The boogu semiconductor-defect image-editing notebook (`notebooks/_drafts/`) | pip |
| `notebooks/course-notes-adaptive-agents/.venv` | Hands-on demo scripts for the "Online Course Notes" series (code knowledge graph, skill induction, LoRA fine-tuning) | **`uv`**, deps declared in that folder's `pyproject.toml` (`[tool.uv] package = false`) |
| `notebooks/paper-hpma-sam3/.venv` | Real reimplementation of a "Paper of the Week" post (HPMA adapters wired into the real `facebook/sam3` model, trained on real EndoVis2017 data) | **`uv`**, same `pyproject.toml` pattern |

When adding a new Python-dependent subproject, follow the newest pattern: a scoped `pyproject.toml`
next to the code, `uv sync` to create its venv, and add the venv path to `.gitignore`. Don't add
new dependencies to the root `requirements.txt` unless they're genuinely needed by the CV-foundations
render path.

This machine has an NVIDIA GB10 GPU (Grace Blackwell, ~124GB unified memory) — `torch.cuda.is_available()`
is `True` here once a CUDA-enabled `torch` wheel is installed via `uv`/`pip`. There is no passwordless
`sudo`, so installing anything as a system service (e.g. Ollama) isn't viable non-interactively; prefer
Python-native alternatives (e.g. `transformers` running a small local model) over a system daemon.

## Series-specific notes

- **Online Course Notes** (`posts/series/course-notes/`) — consolidated notes from online courses,
  one post per course. When a course post gets a hands-on/reproduction companion post, keep the
  original notes post as the reference summary and add the hands-on version as a numbered follow-up
  in the same series, cross-linked both ways. Demo code + generated assets (plots, GIFs, exported
  graph JSON for interactive figures) live under `notebooks/course-notes-adaptive-agents/`, not
  inside the `posts/` tree.
- **Paper of the Week** (`posts/series/papers/`) — weekly digest, one dated post per week
  (`YYYY-MM-DD.qmd`), consolidating multiple papers into a single post rather than one post per paper.
  A single-paper deep dive with a full, real reimplementation (not pseudocode) is also a valid post
  in this series when a paper warrants it — demo code goes in its own `notebooks/paper-<slug>/`
  `uv` project (same pattern as `course-notes-adaptive-agents`), generated images/GIFs referenced
  from `posts/series/papers/images/`. The sidebar for this series only links to `index.qmd`, not
  every individual post (unlike other series) — that's the existing convention, not an oversight.
