# nbdev tutorial reels: "nbdev, the right way" (design)

Status: revised 2026-10-16 from one long video to a **series of short vertical reels**. Draft for the author's review. Nothing is built yet beyond two spikes (Motion Canvas render, nbdev setup).

## Intent
- **Audience:** people who already use nbdev but use it wrongly (A) and newcomers who want a correct first project (B). First viewers are the author's colleagues at Infineon (the department is standardising on uv and introducing nbdev to the team); later LinkedIn, YouTube Shorts and Instagram Reels.
- **Style:** wrong-vs-right pairs (C). Each reel is one mistake: the hook shows the wrong way, then the fix, then one memorable sentence.
- **Outcome:** after the series a viewer can create a project the right way, work notebook-first, ship a CLI, add libraries with uv, and keep the repo clean with tests and hooks.
- **Voice and delivery:** the author's cloned voice, calm and warm tone per reel (chosen per reel at the end), word-by-word subtitles, per-reel review in the browser. Same approval flow as the management video, with one reel as the unit of review.

## Format
- **Vertical 9:16, 1080x1920, 30 fps, h264 + aac**, 30-60 s per reel (about 8 reels). One reel = one idea = one narration of 5-9 short sentences.
- **Hook in the first 2 seconds:** the wrong way, on screen and in the first sentence ("You edited the generated file? Watch this."). No name chapter: a small name badge ("Hasan Goni") sits in the corner of every reel, and the series title with a number ("nbdev, the right way 3/8") in the first second.
- **Each reel stands alone** (except the setup reel, which later reels assume): a viewer arriving at reel 5 must understand it. Reels name the project state they start from.
- **Safe area:** keep text and subtitles inside the central area, away from the platform UI (roughly the bottom 20% and the right edge on LinkedIn, Shorts and Reels; verify against each platform before the public posting). Subtitles larger than in the landscape videos, placed above the bottom UI band.
- **Terminal text in portrait:** about 40 characters per line, big monospace font. Long commands are wrapped over two lines at fixed break points and typed in step with the words; flags are highlighted when spoken, never shortened without saying so.
- **Internal first, desktop viewers:** colleagues will often watch on a laptop, where a vertical video is small. Design inside the 9:16 frame so a centre crop to 4:5 loses nothing, and check one reel on a desktop before the series is finalised.
- **Release order:** (1) internal at Infineon for feedback, (2) LinkedIn, then YouTube Shorts and Instagram. Public versions contain no internal data or images (the demo repo is generic and public). Whether and when the author posts publicly under their employer's rules is the author's call; nothing is published or sent without being asked.

## Series (one reel each, order as the author chose)
| # | Reel | Hook (wrong) | Fix (right) | Target length |
|---|---|---|---|---|
| 1 | Setup | `uv init`, hand-made venv, activating by hand | `uvx --from nbdev nbdev-new --min-python 3.12`, `uv add --dev nbdev`, `uv run ...`; what each line creates (files only / `.venv` + lock); the folder structure nbdev enforces | 60 s |
| 2 | Explore first, then function | Write the function first, hope it works | Try in cells, see output, wrap in a function, `#\| export` | 45 s |
| 3 | Never edit the generated `.py` | Fix the exported library file | Edit the notebook; where modules land (`#\| default_exp`, `lib_path`/`nbs_path`); what `nbdev-update` can sync back (verify) | 45 s |
| 4 | CLI from a notebook function | Hand-written argparse | A console command from a notebook function (mechanism verified before writing) | 45 s |
| 5 | Libraries with uv | `pip install` plus hand-edited config | `uv add` (own or third-party library); what changes in `pyproject.toml` and `uv.lock` | 45 s |
| 6 | Tests and hooks | No tests, messy notebook diffs, broken merges | Tests in notebooks, `nbdev-test`, `nbdev-install-hooks`, readable merge conflicts | 60 s |
| 7 | Before every commit | Running export, test and clean by hand, or not at all | `nbdev-prepare` | 30 s |
| 8 | The rest of the toolbox | - | Rapid list: `nbdev-preview`, `nbdev-release-gh`, `nbdev-pypi`, `nbdev-changelog`; the docs line: the notebook is the doc | 45 s |
Rule from the management video still applies: every drawn item is also spoken, nothing is said twice, visuals appear exactly when their words are spoken. A short "series 1/8" card on the first reel names the whole list once.

## Truth rule
Every command is run first in the demo repo (`HasanGoni/nbdev-tutorial`, created empty on 2026-10-16) and the on-screen output is the captured real output. Versions pinned in the video: nbdev 3.3.25, uv 0.11.26 (re-check on build day). Anything not verifiable is cut or labelled.

### Already verified (2026-10-16)
- `uvx --from nbdev nbdev-new --min-python 3.12` writes files only (no venv): `nbs/` (`00_core.ipynb`, `index.ipynb`, `_quarto.yml`), library folder, `pyproject.toml` (`[tool.nbdev]` empty = defaults), `.github/workflows/{test,deploy}.yaml`, `README.md`, `LICENSE`, `.gitignore`; `requires-python = ">=3.12"`; repo name/user inferred from git.
- `uv add --dev nbdev` creates `.venv` (Python 3.12), `uv.lock`, `[dependency-groups] dev` and an editable install of the package.
- `uv run nbdev-export` / `uv run nbdev-test` work without activation (`which nbdev-export` is empty outside `uv run`).
- Generated `core.py` carries "AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb"; first notebook cell is `#| default_exp core`.
- Config defaults: `lib_path` = `./<package>`, `nbs_path` = `./nbs`, `doc_path` = `./_docs`.

### To verify before each chapter is scripted
- M2: what happens to a hand-edit of `core.py` on the next `nbdev-export`; behaviour of `nbdev-update`; overriding `lib_path`/`nbs_path` in `[tool.nbdev]`; dotted `default_exp sub.module` for subfolders.
- M3: the exact CLI mechanism in nbdev 3 (expected `fastcore.script.call_parse` plus a `[project.scripts]` entry) and the command that registers it; run the resulting command through `uv run`.
- M4: `uv add` of a PyPI library, of a local/own library (path or git), and what nbdev's `pyproject.toml` dependency handling expects (`dependencies` vs the dev group).
- M5: notebook tests (`nbdev-test`, directives such as `#| eval: false`), `nbdev-install-hooks`, `nbdev-merge`/`nbdev-fix` on a real conflict, `nbdev-prepare` output.
- Close: `nbdev-preview` (needs Quarto: check install), release commands only described, not executed against PyPI.

## Visual design and tool choice
No default tool. Per reel we decide together (skills `narrated-manim-video`, `narrated-motion-canvas-video`). Starting proposal, confirmed reel by reel: **Motion Canvas for all reels**, because the frame is new (portrait), the layout engine handles stacked wrong/right cards and wrapped terminal text, and iteration is quick. Manim only where a reel needs it (for example a zoom tour of the folder tree in reel 1, if it reads better); portrait Manim means redefining the frame, so it must earn its place.
Shared rules: **yellow only for the subtitle highlighter**; other text white or accent cyan `#6EC6FF`; wrong = red/orange card, right = green card; terminal panels replay captured text exactly as run; content stays in the safe area.
Pilot: reels 1 (setup) and 3 (never edit the generated `.py`) first, one command-heavy and one before/after, to settle portrait layout, subtitle size and pacing before the rest.

## Pipeline and files
- Voice, timestamps, per-reel build and approval: reuse the `videos/make_chapter.sh` stages and `approve_chapter.sh` (new slugs, story files such as `videos/nbdev/story_order.txt`). There is no joined cut: the deliverables are the individual reels (calm or warm each, as the author chooses), plus compressed copies for platforms with size limits. Motion Canvas chapters follow the `narrated-motion-canvas-video` recipe (audio + `timeline.json` in, frames out; set the project resolution to 1080x1920 and render with ffmpeg to 30 fps h264/aac; the template is landscape, so the portrait settings are a first task).
- Narration lives in each chapter file's `LINES` (we-voice, short, 1-3 marked words).
- Pronunciation, to be decided by listening tests with the author: "nbdev", "uv", "uvx", "nbdev-new", "pyproject". Set once in `video_speech.spoken()` and recorded in the video-review memory.
- Demo repo `HasanGoni/nbdev-tutorial`: each chapter's state is reproduced in a local clone first. Pushing to GitHub happens only when the author says so.
- A new skill `nbdev-tutorial-video` is NOT planned; learnings go into the two existing video skills.

## Review flow
1. Pilot reels 1 and 3 (calm and warm), reviewed in `chapters/index.html` (portrait players); fix format issues once.
2. Reels 2, 4-8 one by one: verify commands, show the narration script for approval, build both tones, author approves each.
3. The author picks calm or warm per reel. Internal release first; LinkedIn, YouTube and Instagram versions only when asked.

## Success criteria
- Each reel is 30-60 s, understandable alone, with the hook in the first two seconds.
- Every command and output on screen was run by us and matches the installed versions.
- Each chapter: spoken words, subtitles and visuals in sync (frame checks at word boundaries + whisper transcript), no overlap or clipping, no yellow outside subtitles.
- A newcomer following only the series can create the project, export a notebook, run a CLI, add a library and run tests.

## Out of scope (for now)
A joined long video, slides, a blog post version, publishing the demo repo, releasing to PyPI/conda, nbdev docs customisation beyond the closing line.

## Open questions
1. Desktop viewing at Infineon: is a 9:16 master acceptable, or should reels also ship as 4:5 or 1:1? (Design allows a centre crop; decide after the pilot.)
2. Name badge only, or also a short title card on reel 1? (Spec assumes badge only.)
3. Is the 8-reel list right, or should reels 6 and 7 merge?
