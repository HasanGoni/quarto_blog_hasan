# nbdev pilot reels (reel 1 Setup, reel 3 Never edit the generated .py) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the portrait (9:16) pipeline and the two pilot reels, each in calm and warm tone, ready for the author's browser review.

**Architecture:** Real commands are run first and captured as JSON facts; the narration (per-reel JSON of sentences) goes through the existing voice stages (Kokoro or Qwen, Chatterbox clone, whisper word times) via a new `make_reel_voice.sh`; a Motion Canvas project (1080x1920) turns audio + `timeline.json` + facts into frames; `make_reel.sh` renders headless and muxes with ffmpeg.

**Tech Stack:** Motion Canvas 3.17 (TypeScript, vite 4, playwright-core headless Chromium), ffmpeg, existing `~/.hermes/scripts` voice code, nbdev 3.3.x, uv 0.11.x.

**Spec:** `docs/superpowers/specs/2026-10-16-nbdev-tutorial-video-design.md`

## Global Constraints
- Reel: 1080x1920, 30 fps, h264 + aac, 30-60 s, hook (wrong way) in the first 2 s, stands alone.
- Yellow `#F6D743` only for the subtitle highlighter; other text white or accent cyan `#6EC6FF`; wrong = red/orange card, right = green card.
- Every command and output on screen is captured from a real run (nbdev 3.3.25, uv 0.11.26 at build time); nothing invented.
- Every drawn item is also spoken; visuals appear on their spoken word; no minimap; text inside the safe area (clear of bottom 20% and right edge).
- Pronunciations set by listening test and kept in `video_speech.spoken()`; name badge "Hasan Goni" in a corner of every reel.
- Disk is tight (about 3 GB free): no new large downloads, remove frames after muxing. Nothing is pushed to GitHub or sent anywhere.
- Existing Manim chapters and `make_chapter.sh` stay untouched.

## Review Focus
- A narration word that has no timestamp (name, acronym, `uvx`) must not stall the word clock: the timeline must have one time per displayed word.
- A terminal line longer than the panel must wrap at its planned break, never clip.
- Calm and warm builds of one reel must not overwrite each other's audio or timeline.
- A render whose audio is shorter than the visuals must keep the full video length.
- A command in the facts that fails or prints something different from the script must stop the build.

## File structure (all under `notebooks/paper-explainers-2026-10-09/videos/nbdev/`)
- `facts/capture_reel1.sh`, `facts/capture_reel3.sh` -> `facts/reel1.json`, `facts/reel3.json` (real commands, exit codes, outputs, file trees).
- `reels/<slug>.json`: `{"slug","title","lines":{"R1":"sentence with **marked** words"}}`.
- `reel_voice.py`: `lines_to_missing(reel_json, nd) -> None`, `build_audio_timeline(reel_json, nd, out_dir, suffix) -> end_seconds`.
- `make_reel_voice.sh <slug> <plain|warm>`: voice stages + audio + `timeline.json`.
- `mc/`: Motion Canvas project (`src/lib/{clock,subtitles,panels}.tsx`, `src/scenes/reel1.tsx`, `reel3.tsx`, `src/project.ts`), `render.mjs`.
- `make_reel.sh <slug> <plain|warm>`: voice + render + mux -> `out/<slug>.mp4` / `.warm.mp4`; `reels_index.py` -> `out/index.html`.

## Tasks

### Task 1: Capture the real facts for both reels
**Files:** Create `facts/capture_reel1.sh`, `facts/capture_reel3.sh`, outputs `facts/*.json`.
**Interfaces:** Produces `reel1.json` = `{"steps":[{"cmd","exit","out","tree"}]}`; `reel3.json` adds `edit`, `before`, `after_export`, `after_update`.
- [ ] Reel 1: in a fresh clone of the empty demo repo run `uvx --from nbdev nbdev-new --min-python 3.12 ...`, record the tree; run `uv add --dev nbdev`, record tree, `pyproject.toml` dev group, `.python-version`/`.venv`; run `uv run nbdev-export`, record exit and `which nbdev-export` (empty).
- [ ] Reel 3: run `uv run nbdev-export`, hand-edit `core.py` (add a line), run `uv run nbdev-export` again, record whether the edit survives; run `uv run nbdev-update` in the edited state and record what happens to the notebook; record `lib_path`/`nbs_path` via `get_config()`.
- [ ] Script exits non-zero if any expected string is absent. Run both; commit scripts and JSON.

### Task 2: Reel voice stages
**Files:** Create `reel_voice.py`, `make_reel_voice.sh`, `reels/setup.json`, `reels/never_edit_py.json`.
**Interfaces:** Consumes `video_speech.{spoken,tts_key,word_times,strip_markup}` and manifests under `assets/um/narration_clone[_warm]`. Produces `out/<slug>/audio<sfx>.wav` (sentences concatenated with 0.35 s gaps) and `out/<slug>/timeline<sfx>.json` in the same shape the Manim chapters use (`beats[{key,text,t0,dur,times}]`, `end`).
- [ ] Write a test `test_reel_voice.py`: fake manifest with 2 sentences, assert `times` length equals the number of display words and `t0` of beat 2 = dur1 + 0.35.
- [ ] Implement `reel_voice.py`, run the test, then `make_reel_voice.sh` (steps copied from `make_chapter.sh` stage 2, `REEL_LINES` instead of the Manim collect pass). Exit non-zero on any missing audio.
- [ ] Pronunciation test of "nbdev", "uv", "uvx", "pyproject" with Kokoro and Qwen plus whisper; fix in `spoken()` only after listening results are checked (record in the video-review memory).
- [ ] Commit.

### Task 3: Portrait Motion Canvas project
**Files:** Create `mc/*` from the skill template, `src/lib/clock.tsx` (`wordClock(timeline)` returning `at(t)`), `src/lib/subtitles.tsx` (`addSubtitles(view, beats, bottomY)` word-by-word, marked words yellow background + `#111` text, punctuation attached), `src/lib/panels.tsx` (`terminal(lines, width)`, `card(kind: 'wrong'|'right', title)`, `badge()`), `render.mjs` taking `PORT`.
**Interfaces:** `makeReel(timeline, build)` runs `build(view, at, beats)`; every lib function takes the word clock so visuals fire with `yield`.
- [ ] `setup.sh` the template into `mc/`, set resolution 1080x1920 and exporter to PNG sequence in `src/project.meta`, move Chromium to `~/.cache/pw-browsers`.
- [ ] Render the template with a test timeline; check frame size 1080x1920 and subtitle placement above the bottom 20%.
- [ ] Commit (node_modules and output ignored).

### Task 4: `make_reel.sh` and the review page
**Files:** Create `make_reel.sh`, `reels_index.py`; extend `.gitignore`.
- [ ] `make_reel.sh <slug> <plain|warm>`: call Task 2, copy audio and timeline into `mc/src`, start vite on a free strict port, run warm-up then `render.mjs`, kill the port, ffmpeg frames + audio with `apad`, `-t end`, 1080x1920 30 fps, delete frames; print `DONE: <path>`. Fails loudly on each step.
- [ ] `reels_index.py`: portrait players, calm/warm side by side, sentence list. Commit.

### Task 5: Reel 1 (Setup)
**Files:** `reels/setup.json`, `mc/src/scenes/reel1.tsx`.
- [ ] Narration (about 60 s): hook with the wrong way (`uv init`, hand-made venv), the two lines, what each creates (files only / `.venv` + lock + dev group), `uv run` needs no activation, the folder structure (`nbs/`, library folder, `pyproject.toml`, workflows). Every item drawn is spoken.
- [ ] Scene: wrong card first, terminal panel with the real commands wrapped at planned breaks (typed in step with the words), file tree that fills in per command from `facts/reel1.json`.
- [ ] `./make_reel.sh setup plain` and `warm`; frame checks at word boundaries (ffmpeg `-ss`), whisper transcript of both, duration 30-60 s, no yellow outside subtitles. Commit.

### Task 6: Reel 3 (Never edit the generated .py)
**Files:** `reels/never_edit_py.json`, `mc/src/scenes/reel3.tsx`.
- [ ] Narration (about 45 s) written only after Task 1 shows what really happens on re-export and `nbdev-update`; shows header "AUTOGENERATED! DO NOT EDIT!", the edit, the result, and the fix (edit the notebook); mentions where modules land (`#| default_exp`, `lib_path`) only if verified.
- [ ] Scene: wrong card (edit `core.py`) then right card (edit notebook cell), real before/after text from `facts/reel3.json`.
- [ ] Build calm and warm, same checks as Task 5. Commit.

### Task 7: Review handoff and skills
- [ ] `reels_index.py`; open `out/index.html` in Firefox and give the author the sentence keys.
- [ ] Add the portrait findings (resolution settings, safe area, wrapped terminal, new scripts) to `narrated-motion-canvas-video` and a memory note. Commit.

## Self-review
Spec coverage: format, hook, badge, safe area, terminal wrapping, truth rule, per-reel calm/warm, pilot reels 1 and 3, colour rule, no joined cut, no publishing: each is in a task or a global constraint. Open spec questions (4:5 crop, badge only, 6/7 merge) are decided after the pilot. No placeholders: narration text is written inside Tasks 5 and 6 once facts exist, by design (the narration depends on captured output).
