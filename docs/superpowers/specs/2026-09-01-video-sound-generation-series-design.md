# Video & Sound Generation series — design

Date: 2026-09-01
Status: Part 1 shipped (2026-09-01)

## Motivation

Hasan asked about applying joint video+audio generation to his X-ray inspection work, prompted
by a LinkedIn post about DreamX-Creator (a 7B joint video+audio diffusion model from Zhejiang
University / GD-ML). Checked before committing to anything: DreamX-Creator's weights are not
released yet ("planning to release," per the team's own post) — real, current fact, not assumed.
Two real, released, non-gated alternatives exist as of this writing: LTX-2 (Lightricks, 19B,
released 2026-01-05) and MOVA (OpenMOSS, 32B, released 2026-01-29). MOVA's 32B doesn't fit this
machine's real memory budget alongside the persistent vLLM process (~64GB needed at bf16 vs.
~50GB actually free); LTX-2 does, especially using its provided fp8/fp4 quantized checkpoints
(non-bf16 variants shipped directly by Lightricks, not something this repo has to produce itself).
Decision: start with LTX-2.

This is a new, standalone series — not a Synthetic Data series extension. Video/audio generation
is a different problem class (temporal + acoustic, not static 2D images), and neither GDXray nor
any dataset used elsewhere in this repo has a video or audio modality to build on.

## Non-goals

- Not claiming DreamX-Creator was used — it isn't released. Revisit if/when it ships; note the
  substitution plainly in Part 1 rather than pretending LTX-2 was the original plan.
- Not attempting MOVA on this hardware without first stopping the persistent vLLM process — that
  decision belongs to the user (vLLM is his active coding-assistant backend), not something to do
  unilaterally. If MOVA becomes worth revisiting, ask first.
- Not claiming any real acoustic signal exists in X-ray imaging itself. The "sound" angle for an
  X-ray-domain post has to be a real, defensible one (see Part 1 framing below) — sonifying pixel
  data or inventing synthetic audio with no real-world referent would be dishonest dressing-up,
  not a real use case.

## Part 1 — X-ray inspection video+audio generation (real, non-gated LTX-2)

**The real, defensible framing, not a strained one**: real non-destructive-testing (NDT)
inspection lines commonly use continuous/video-style X-ray scanning (radioscopy, conveyor-based
industrial CT/X-ray scanners) rather than single static shots — and those real scanning rigs
produce real ambient mechanical/electronic sound (conveyor motors, detector cooling, alarms).
"Generate a synthetic inspection-scan video with plausible ambient scanning sound from a real
static radiograph" is a real, sensible extension of this blog's existing synthetic-data-for-
industrial-inspection theme (see the Synthetic Data series), not an arbitrary application of a
video model to an unrelated domain.

- **Seed image**: a real GDXray Welds radiograph (same source, same real defect crops used
  throughout the Synthetic Data series — direct cross-link both ways).
- **Model**: `Lightricks/LTX-2` via `diffusers`' real `LTX2Pipeline` (`image-text-to-audio-video`
  task, confirmed supported directly on the model card; `diffusers>=0.40.0` on PyPI already
  contains `LTX2Pipeline` — checked against the `v0.40.0` git tag directly, no need for a
  git-main install).

  **Real memory picture, checked component-by-component via the HF tree API (2026-09-01) —
  materially worse than the first-pass ~50GB estimate**: current actual free memory is
  25GB free / 37GB available (`free -h`), tighter than assumed, since vLLM's resident set is
  72.6GB (`nvidia-smi`) on top of desktop/browser overhead. The video/audio transformer itself
  is available pre-quantized (fp4 ≈ 20GB, fp8 ≈ 27GB, bf16 ≈ 43GB) — but the text encoder is
  **Gemma3ForConditionalGeneration in bf16, ≈ 54GB on disk**, bigger than the transformer, and
  the repo ships no quantized variant of it. Naive full-pipeline load (fp4 transformer + bf16
  text encoder + small VAE/audio-VAE/vocoder/connector components) is ~80GB — doesn't fit
  25-37GB free even with `enable_sequential_cpu_offload`, and on this GB10's true unified-memory
  architecture (CPU and GPU share one physical DRAM pool over NVLink-C2C), "CPU offload" doesn't
  get the usual discrete-GPU win of a separate memory pool to spill into — it only helps by not
  instantiating every component at once, not by escaping the shared RAM ceiling.

  **Decision (confirmed with the user via a real tradeoff, not assumed)**: quantize the Gemma3
  text encoder ourselves at load time via `transformers.BitsAndBytesConfig` (int4, with an int8
  fallback), keep the fp4 transformer, and empirically verify real resident memory with a
  standalone probe script (`probe_text_encoder_memory.py`, loads *only* the text encoder +
  tokenizer and reports `torch.cuda.max_memory_allocated` after a real forward pass) before
  committing to the full ~80GB pipeline download. Ruled out for now: asking the user to stop
  vLLM temporarily (their active coding-assistant backend — not this project's call to make
  unilaterally, consistent with the MOVA non-goal above) and re-scoping to a smaller model
  (deferred unless the quantization approach demonstrably doesn't fit).

  **Probe result (2026-09-01, real, measured)**: int4-quantized Gemma3 text encoder loaded on
  GPU at only **8.34GB resident** (`torch.cuda.max_memory_allocated`) — download was the full
  54GB bf16 (int4 quantization happens at load time, not download time), but the compressed
  in-memory footprint is far under budget, with a verified-correct forward pass
  (`hidden_states[-1].shape == [1, 18, 3840]`). This validates the quantize-ourselves approach;
  proceeding to int4-quantize the transformer the same way (`generate_video_sound.py`) and build
  the full pipeline (small components — vae/audio_vae/connectors/vocoder — kept bf16, all tiny).

  **First real end-to-end smoke test (2026-09-01, `generate_video_sound.py`, real GDXray Welds
  seed image)**: `enable_sequential_cpu_offload()` turned out incompatible with bitsandbytes
  quantization (`NotImplementedError: Cannot copy out of meta tensor`) — a known limitation, not
  a bug to route around; removed it and loaded both quantized components directly onto
  `cuda:0` via `device_map={"": device}` instead, since the real numbers already showed no
  offloading was needed. Also needed `av` (PyAV) added as a dependency for the pipeline's
  internal image-conditioning re-compression step. Real result at smoke settings
  (384×256, 25 frames, 8 steps): pipeline build **24.52GB** resident, generation peak
  **29.95GB** — comfortably inside the 25-37GB budget. Output verified as a genuine H.264
  video stream (384×256, ~1.04s) + AAC audio stream (24kHz, ~1.01s) via `ffprobe`, and a
  decoded mid-clip frame visually resembles real weld-radiograph texture rather than noise
  (expected to be rough at only 8 steps — this was a mechanics check, not a quality bar).
  Next: scale toward real deliverable settings (higher resolution/frame count/step count)
  while watching peak memory, since 29.95GB already used most of the available margin under
  vLLM's real, current resident footprint.

  **Real finding (2026-09-01, three incremental scaling tests + one production run)**: peak
  memory held flat at exactly **29.95GB** across 384x256/25f/8steps, 512x320/25f/8steps,
  512x320/49f/8steps, and the real production run 768x512/121f/30steps -- strong evidence
  `vae.enable_tiling()` (plus the transformer's own internal chunking) genuinely bounds peak
  memory to something close to a fixed ceiling rather than something that scales with output
  size. This is a useful, transferable finding independent of the content result. During this
  testing, real `NVRM: Out of memory` kernel-log entries appeared at the memory-pressure peaks
  (transient CUDA allocation-failure-and-retry, not an actual process kill -- confirmed via
  `journalctl -k` showing no OOM-killer entries and every expected GPU process, including vLLM,
  still alive throughout) -- flagged to the user directly rather than assumed benign, since a
  process did visibly get killed on their end that this session's logs couldn't identify; user
  confirmed to proceed.

  **Part 1 real production result (768x512, 121 frames/5.04s, 30 steps, real GDXray Welds seed
  frame)**: pipeline build 24.52GB resident, generation peak 29.95GB, real H.264 video + AAC
  audio output (`ffprobe`-verified), 360KB file (well under quarto.pub's ~1MB limit), real
  non-silent audio (`ffmpeg volumedetect`: mean -44.5dB, max -31.4dB, no clipping). **Honest
  finding, confirmed by the user's explicit choice to report it as-is rather than attempt a
  mitigation**: frame 0 (the real image-conditioning anchor) faithfully preserves the X-ray
  radiograph texture; by mid-clip the video has drifted into ordinary color industrial
  photography with no visible trace of the X-ray domain by the end. A single real conditioning
  frame is not enough to hold a 19B general-purpose video model inside a narrow visual domain for
  five seconds of generated motion -- the dominant natural-video training prior reasserts itself
  almost immediately past the anchored frame. Shipped as Part 1 of the new series
  (`posts/series/video-sound-generation/01-xray-inspection-video-sound.qmd`), wired into
  `_quarto.yml` (navbar + sidebar), `README.md`, and cross-linked both ways with the Synthetic
  Data series index.
- **Prompt**: describes a real, plausible inspection scenario (a scan head or conveyor moving
  across the weld, revealing the defect) plus real, plausible ambient industrial sound (motor
  hum, mechanical clicks) — not narration, not music, not anything with no real-world referent.
- **Deliverable bar**: same as every other deep-dive post in this repo (real code, no pseudocode;
  real captured output; an actual generated video+audio clip, not a mockup; jargon-buster table;
  two reading levels; a whiteboard-style ELI5 diagram; honest limitations section). A generated
  video+audio clip is the headline deliverable here, analogous to the GIF in prior posts — check
  its file size against quarto.pub's upload limit the same way, and check whether quarto.pub /
  the Quarto HTML output can even embed video+audio inline before assuming it can.
- **Honest evaluation, not hype**: does the generated video actually look like a plausible
  inspection scan (temporal coherence, defect staying visually consistent across frames, no
  obvious artifacts), and does the generated audio actually sound like plausible industrial
  scanning equipment rather than generic/wrong audio? Report both dimensions honestly, the same
  way every prior post in this repo reports what didn't work alongside what did.

## Tooling

New `uv`-managed subproject: `notebooks/video-sound-xray/` (or similar — finalize the exact name
at implementation time), following the same scoped-`pyproject.toml`/gitignored-`.venv` pattern as
every other notebooks/ subproject in this repo. Reuses `synth_xray`'s real GDXray data utilities
(`data.py`, `groundtruth.py`) for the seed image rather than reimplementing GDXray access.

## Series placement

New entry in `_quarto.yml`'s navbar "Series" dropdown and sidebar, and a new `README.md` series
list entry, following the exact same wiring pattern as every other series in this repo (see
`CLAUDE.md`'s "Adding a new series or post" section). Cross-link from the Synthetic Data series'
`index.qmd` given the thematic overlap (industrial inspection, GDXray), without merging the two
series.

## Part 2 — video conditioning vs. Part 1's single-image conditioning (shipped 2026-09-02)

User asked about a technique they'd heard of ("insert this video and next time it is better") --
real web research confirmed this matches LTX-2's own documented "video conditioning" / "Extend
Video" feature (a "Video Strength" + "Context Duration" mechanism, real sources: MindStudio's
LTX-2.3 video-to-video writeup, Scenario's LTX-2 Extend Video docs, LTX's own character-
consistency blog post), exposed in `diffusers` as `LTX2ConditionPipeline` +
`LTX2VideoCondition(frames=<PIL image(s) or video>, index=int, strength=float)`. Real, testable
hypothesis: does conditioning on *many* real frames (not just one) hold the X-ray domain longer
than Part 1's single-frame conditioning did?

**Method** (`generate_video_sound_conditioned.py`): same model, same int4 quantization strategy,
same prompt/negative prompt, same output settings (768x512, 121 frames, 30 steps) as Part 1 --
only variable changed is the conditioning signal: the real seed image repeated 41 times (~1.7s)
as a `LTX2VideoCondition(frames=[seed]*41, index=0, strength=1.0)`, vs. Part 1's single frame.
`LTX2ConditionPipeline` shares the exact same component set as `LTX2ImageToVideoPipeline`, so
`build_pipeline` needed only a pipeline-class swap, not new loading logic.

**Real result, quantified, not just eyeballed**: mean color saturation (RGB max-min/max) at
matched frame indices (0/60/119) -- Part 1 late frames: 0.097-0.101; Part 2 late frames:
0.0033-0.0086, roughly 12-30x lower. Video conditioning **dramatically reduced color drift** --
the drifted-to content stayed monochrome/grayscale throughout, unlike Part 1's clear shift to
full color. But it did **not** fix content/semantic drift: by mid-clip the visible content still
shifts from X-ray radiograph texture to what reads as black-and-white industrial-machinery
footage -- a real, different failure mode (grayscale drift, not color drift), not a full fix.
**Real tradeoff, also measured, not assumed**: audio got dramatically quieter under the longer
video condition (`ffmpeg volumedetect`: mean -71.3dB / max -66.8dB, vs. Part 1's mean -44.5dB /
max -31.4dB) -- holding the visual domain harder seems to have suppressed the audio branch's
generative freedom. Memory and generation time were unchanged (24.52GB build / 29.95GB peak /
~1154s generation, matching Part 1 almost exactly) -- video conditioning cost nothing extra on
either axis here, thanks to the same tiling-bounded memory ceiling found in Part 1.

## Open risks / follow-ups (not blockers)

- Whether `diffusers`' `LTX2Pipeline` support is mature/stable given how recently LTX-2 shipped
  (2026-01-05) — verify with a real smoke test before committing to a full pipeline, same
  diligence as every other new-model integration in this repo.
- **Resolved (2026-09-01)**: Quarto's `{{< video file.mp4 >}}` shortcode does embed real
  video+audio inline — verified with an actual `quarto render` of a synthetic ffmpeg-generated
  mp4 (H.264 + AAC), producing a working `video.js`-backed `<video><source></video>` element.
  No fallback needed; a real generated `.mp4` from the pipeline is directly embeddable.
- DreamX-Creator's actual release — revisit as a lighter-weight (7B) alternative once real weights
  ship, rather than assuming LTX-2 is the permanent choice for this series.
