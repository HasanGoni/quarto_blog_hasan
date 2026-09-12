## Learned User Preferences

- For multi-panel figures in paper posts, use `layout-ncol` with separate full-resolution panels instead of wide strip images inside `.column-page` (strips squash panels and look blurry).
- In paper deep-dive `.qmd` posts, collapse long real-code blocks by default; prefer HTML `<details>` over markdown `code-fold` or collapsed callouts for fenced code (those often render expanded).
- Paper deep-dive posts should include lightweight GIFs for modality or before/after demos when stills are hard to scan; keep GIFs well under quarto.pub's ~1 MB upload limit.
- When a paper post runs an off-domain zero-shot experiment, frame it honestly as an OOD stress test — not a deploy or fine-tune recommendation for that dataset.

## Learned Workspace Facts

- Marigold V2 paper post (`2026-09-11-marigold-v2-depth-on-xrays.qmd`): Case 2 shows depth, normals, and albedo from the main Marigold V2 LoRA checkpoint; Case 3 compares base vs LoRA for depth and normals only (no albedo base-vs-LoRA row).
- GDXray Castings defect bounding boxes are not valid supervision for Marigold-style dense geometry LoRA (depth/normals/albedo maps); the post uses released checkpoints zero-shot on X-ray crops.
- Marigold V2 LoRA training expects paired RGB natural images plus dense per-pixel depth (Hypersim + Virtual KITTI 2), normals (Hypersim), or albedo (Hypersim) — not grayscale X-ray transmission images or bbox labels.
- `notebooks/paper-marigold-v2/` is the uv project for the Marigold V2 paper reimplementation; GIFs for the post are built with `make_gifs.py` into `posts/series/papers/images/`.
