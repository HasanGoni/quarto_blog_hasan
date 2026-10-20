---
name: inspiration
description: Use when the user shares a paper, repo or tool and says "we can use this too", "can we adapt this", "take inspiration from", or asks how an outside method could fit their own use case (image inspection, local/open models, non-commercial vs commercial licence). Reads the real process first, then adapts it. Not for plain paper summaries (use paper-explainer-pages) or for installing a repo as-is.
---

# Inspiration: understand the real process, then adapt it

The goal is never "install their repo". It is: **find the idea that earns its place, prove how much of
the headline it explains, and rebuild only that, under licences and constraints that fit the user's work.**
Source of this method: the SpatialClaw series (`posts/series/spatialclaw-inspection/`).

## Constraints to carry in (ask only what the user has not said)

- Data: image-only inspection, one image or a few crops, known scale, a verdict someone is accountable for.
- Models: local/open weights preferred; no data leaves the machine.
- Licence: blog experiments may study non-commercial work; **work code must not copy it.**
- Publishing: never `quarto publish` without explicit confirmation. No tool names in published text.

## Process (in order, do not skip to the plan)

1. **Read primary sources, not the README.** `gh repo view`, shallow clone into a gitignored
   `notebooks/<topic>/upstream/` (record the commit hash), `curl` the PDF + `pdftotext` the *full* paper.
2. **Trace the real process.** Entry point -> loop -> state -> exit conditions -> fallback behaviour.
   Cite `file:line`. Compare defaults in config with what the paper says: flags that are off by default,
   caps, timeouts. Differences between paper and code are findings.
3. **Run what can run.** Smoke-test the pieces that need no GPU or weights (parsers, safety filters,
   state machines, kernels). Load upstream files verbatim by path, never edit them. Capture real output
   to `outputs/`. Label every stand-in. State what was not run. Parse-only evidence stays parse-only:
   do not claim an exploit works because a checker passed it.
4. **Decompose the headline number** using the paper's own tables: no-tool vs tool vs the new design with
   the same backbone. Say what part is the idea and what part is setup. List the rows where it does **not**
   win. Mark your arithmetic as yours.
5. **Audit licences as a table**: their code, each model/weights checkpoint, each third-party dependency.
   Rule: ideas are free, code is not. If you read their code you cannot call your rebuild clean-room;
   say that. Add "not legal advice, get it reviewed for work use".
6. **Adaptation matrix: keep / swap / drop** against the constraints above. For each swap, say why.
   Ask what a wrong answer costs in production (e.g. a forced guess that says PASS is a shipped
   defect) and change the failure path first.
7. **Pre-register the experiment** against the user's existing baseline, on data they may publish.
   Write the expectation, including where the new idea should tie or lose, before any code.
8. **Plan the series**, one concept per post, each with: diagram (xkcd whiteboard style, no Mermaid),
   one interactive element, daily-life analogy for every jargon term, collapsed real outputs,
   and a narrated warm-tone video (see `narrated-manim-video`, `manim-zoom-scenes`).

## Deliverables

- New standalone series under `posts/series/<name>/` (index + deep dive + adaptation plan). Do not
  fold it into an existing series unless asked. Wire `_quarto.yml` navbar and sidebar.
- Code and captures under `notebooks/<name>/` with its own `uv` project and a README that explains how
  to re-clone the upstream at the pinned commit.
- Render the touched pages and read the tail for ERROR/WARN. Leave publishing for the user.

## Red flags

| Thought | Reality |
|---|---|
| "The README says it works" | Defaults in config often differ from the paper. Read both. |
| "+11 points, so the idea is great" | Decompose first. Often most of it is toolset and prompt. |
| "I'll copy their loop, it's quicker" | Licence. Rebuild from the design. |
| "Static check = sandbox" | Parse-only. Test it, and say what you did not execute. |
| "Out of budget, return the best guess" | In inspection, abstain/REVIEW, never PASS. |
