# Synthetic Data series — status

Living status doc — update this whenever the series' implementation state changes materially
(a part ships, a draft gets committed, the plan changes). Unlike `specs/` and `plans/` (point-in-
time snapshots of a decision), this file always reflects **right now**, so a new session or agent
picking up this series can get oriented without re-deriving state from git log + working-tree
diffing.

**Read these two first, in order:**
1. `docs/superpowers/specs/2026-08-28-synthetic-data-series-design.md` — the design: what each of
   the 9 parts is and why, in what order, with every honest finding/risk noted inline as the
   series has actually progressed. This is the source of truth for scope and sequencing.
2. This file — what's actually been built/shipped vs. drafted-but-uncommitted vs. not started.

## Series shape (as of 2026-08-29)

**Nine parts**, most recently expanded to add Part 5 (joint image+mask generation, inspired by
NVIDIA's MAISI/NV-Generate-CTMR work) — see the spec's intro list for the full numbered rundown
and each part's own `## Part N` section for implementation detail.

## Per-part status

| Part | Title | Status |
|---|---|---|
| 1 | Physics-based X-ray simulation | **Shipped & published.** Live at `posts/series/synthetic-data/01-physics-based-xray-simulation.qmd`. Went through a full 10-task implementation plan + final whole-branch review + fix wave (see `docs/superpowers/plans/2026-08-28-synthetic-data-part1-physics-xray.md`), fully committed and merged to `main`. |
| 2 | Mask-conditioned diffusion fine-tuning | **Drafted, uncommitted.** Post (`posts/series/synthetic-data/02-mask-conditioned-diffusion-finetuning.qmd`), training code (`train_diffusion_lora.py`, `src/synth_xray/diffusion_data.py`), comparison/diagram scripts, and generated images/GIF all exist in the working tree as untracked/modified files. No placeholder markers left in the post — reads as a complete draft. **Not committed, not rendered/verified, not published.** |
| 3 | ControlNet-conditioned diffusion | **Drafted, uncommitted.** Post (`03-controlnet-conditioned-diffusion.qmd`), `train_controlnet_lora.py`, comparison/diagram scripts, generated images exist. Same caveat as Part 2 — looks complete, not committed/rendered/published. **Not yet wired into `_quarto.yml`'s sidebar** (Part 2 is; Part 3 isn't, per the current uncommitted `_quarto.yml` diff). |
| 4 | Flux-based mask-conditioned generation | Not started. License/gating check is a hard gate before committing to this part — see spec's Open Risks. |
| 5 | Joint image+mask diffusion generation | Not started. Added to the spec 2026-08-29; architecture (channel-expansion fine-tune vs. from-scratch small model vs. adapting a published recipe) deliberately left open, decide at implementation time. |
| 6 | Combining what works | Not started. Depends on Parts 3-5's empirical results. |
| 7 | GAN-based synthesis | Not started. |
| 8 | Cut-paste + Poisson blending baseline | Not started. |
| 9 | Post-training a real detector (the payoff) | Not started. |

## ⚠️ Known inconsistency to resolve before committing Parts 2-3

The uncommitted `posts/series/synthetic-data/index.qmd` and `_quarto.yml` sidebar changes were
written **before** Part 5 (joint image+mask generation) was inserted into the spec. They reflect
the *old* 8-part numbering (Part 3 = ControlNet, Part 4 = Flux, Part 5 = Combining, Part 6 = GAN,
Part 7 = Cut-paste, Part 8 = Payoff) rather than the current 9-part numbering in the spec (Part 5
= joint generation, Combining pushed to Part 6, GAN to 7, cut-paste to 8, payoff to 9).

**Before committing/publishing Parts 2-3**, reconcile `index.qmd`'s "Posts in this series" list
and any in-post cross-references against the spec's current numbering — don't just commit the
existing draft text as-is, it'll publish stale part numbers.

## Uncommitted files in the working tree (as of 2026-08-29, this session)

Modified (tracked): `.gitignore`, `CLAUDE.md`, `_quarto.yml`, `notebooks/synthetic-data-xray/pyproject.toml`,
`notebooks/synthetic-data-xray/uv.lock`, `posts/series/synthetic-data/index.qmd`

Untracked (new): `notebooks/synthetic-data-xray/{generate_controlnet_comparison,generate_diagram_part2,generate_diagram_part3,generate_diffusion_comparison,train_controlnet_lora,train_diffusion_lora}.py`,
`notebooks/synthetic-data-xray/src/synth_xray/diffusion_data.py`,
`posts/series/synthetic-data/{02-mask-conditioned-diffusion-finetuning,03-controlnet-conditioned-diffusion}.qmd`,
`posts/series/synthetic-data/images/{synth-xray-controlnet-eli5-sketch,synth-xray-defect-diffusion-zoom,synth-xray-defect-diffusion,synth-xray-diffusion-eli5-sketch,synth-xray-diffusion-loss-curve}.png`,
`posts/series/synthetic-data/images/synth-xray-diffusion-training-progress.gif`

This is real work from another session, not scratch/garbage — don't discard it. Run `git status`/
`git diff` to see the live picture; this list is a snapshot and will drift as work continues.

## Conventions this series follows (see `CLAUDE.md` for the canonical version)

- **No Mermaid diagrams** in the "Architecture" section (explicit user preference, corrected
  2026-08-29) — use prose/bulleted walkthroughs of the real source code instead. Don't retrofit
  Part 1's existing Mermaid diagram without being asked.
- Whiteboard/xkcd-style diagram for the ELI5 section: matplotlib `plt.xkcd()` +
  `FancyBboxPatch`/`FancyArrowPatch`, per `notebooks/synthetic-data-xray/generate_diagram.py`
  (Part 1's version) — reuse this method, don't introduce a new diagramming approach.
- Full deep-dive checklist (two reading levels, jargon table, real code, real captured output,
  size-optimized GIF, whiteboard diagram) applies to every part by default — see CLAUDE.md's
  "Paper of the Week" series notes, apply it without being asked per post.
- `quarto publish` is a live deploy — always confirm with the user before running it.

## Infra note (unrelated to the series, but relevant if git/network operations mysteriously fail)

GitHub connectivity on this machine was being blocked at the network level (GitHub's IP ranges
specifically — confirmed via `raw.githubusercontent.com` working while `github.com`/`api.github.com`
timed out). Fixed 2026-08-29 by installing and connecting Cloudflare WARP (`warp-cli`). If `git
push`/`fetch` or any GitHub-hosted download starts timing out again, check `warp-cli status` first
before assuming a code/config problem.
