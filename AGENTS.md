## Learned User Preferences

- For multi-panel figures in paper posts, use `layout-ncol` with separate full-resolution panels instead of wide strip images inside `.column-page` (strips squash panels and look blurry).
- In `.qmd` posts, collapse long real-code blocks by default; prefer HTML `<details>` over markdown `code-fold` or collapsed callouts for fenced code (those often render expanded).
- Paper deep-dive posts should include lightweight GIFs for modality or before/after demos when stills are hard to scan; keep GIFs well under quarto.pub's ~1 MB upload limit.
- When a paper post runs an off-domain zero-shot experiment, frame it honestly as an OOD stress test — not a deploy or fine-tune recommendation for that dataset.
- In jargon/glossary sections, don't use a separate "Daily-life version:" label — weave analogies into each definition with openers like "Think of...", "For example...", or "Suppose...".
- Never write company or client data outputs into the repo; save inference artifacts to paths outside the git tree.
- Agentic Zero to Advanced LLM demos should show the usual `.env` / `OPENAI_API_KEY` cloud path and, when using a local OpenAI-compatible server (e.g. vLLM), document `OPENAI_BASE_URL` + model id with real captured local output.

## Learned Workspace Facts

- Marigold V2 paper post (`2026-09-11-marigold-v2-depth-on-xrays.qmd`): Case 2 shows depth, normals, and albedo from the main Marigold V2 LoRA checkpoint; Case 3 compares base vs LoRA for depth and normals only (no albedo base-vs-LoRA row).
- GDXray Castings defect bounding boxes are not valid supervision for Marigold-style dense geometry LoRA (depth/normals/albedo maps); the post uses released checkpoints zero-shot on X-ray crops.
- Marigold V2 LoRA training expects paired RGB natural images plus dense per-pixel depth (Hypersim + Virtual KITTI 2), normals (Hypersim), or albedo (Hypersim) — not grayscale X-ray transmission images or bbox labels.
- `notebooks/paper-marigold-v2/` is the uv project for the Marigold V2 paper reimplementation; GIFs for the post are built with `make_gifs.py` into `posts/series/papers/images/`.
- Agentic Zero to Advanced Part 2 (`02-dataset-baseline.qmd`) uses GDXray Castings series C0001 (72 rotated views of one aluminum wheel) with a classical top-hat/morphology defect-detection baseline.
- GDXray Castings `ground_truth.txt` is single-class detection — each bbox marks defect location only (no porosity/inclusion/crack type labels); defects are tiny (often 10–31 px) and low-contrast.
- `notebooks/agentic-zero-to-advanced/` is the uv project for the Agentic Zero to Advanced series; `baseline_detector.py` regenerates Part 2 figures into `posts/series/agentic-zero-to-advanced/images/`; Part 3 uses `detector_tool.py` + `first_tool_call.py` (aisuite) with `.env.example` for cloud vs local vLLM.
- Agentic Part 3 (`03-first-tool-call.qmd`) wraps the Part 2 detector as `inspect_casting_xray()`, shows honest tool metrics (e.g. 0 TP / 3 FP / 3 FN on `C0001_0001.png`), and stops at interpretation rather than improving the baseline.
