# spatialclaw-inspection

Smoke tests and diagrams for the "SpatialClaw for Inspection" series.

The upstream code is NOT vendored (non-commercial NVIDIA licence). To reproduce:

    git clone https://github.com/NVlabs/SpatialClaw.git upstream/repo
    git -C upstream/repo checkout f37b163
    uv sync
    .venv/bin/python -I smoke_safety.py
    .venv/bin/python -I smoke_kernel.py

Captured outputs: `outputs/`. Diagrams: `generate_diagrams.py`.

## Part 1 video (warm voice, approved 2026-10-20)

Scenes: `video/spatialclaw_video.py` (classes `Sc1`..`Sc4`), script: `video/script.md`, stand-in pad: `video/make_scene_assets.py`.
Approved clips: `video/chapters/sc{1..4}.warm.mp4` (about 22-30 s each). Rebuild from `notebooks/paper-explainers-2026-10-09/videos/`
after symlinking the scene module there (absolute target; that directory lives on the Elements drive):

    ln -sf <repo>/notebooks/spatialclaw-inspection/video/spatialclaw_video.py spatialclaw_video.py
    VC=<repo>/notebooks/voice-clone ./make_chapter.sh spatialclaw_video.py Sc1 sc1 clone warm

## Part 3: the minimal kernel loop

Own code (no upstream code): `minimal_loop/kernel_session.py`, `minimal_loop/loop.py`.

    .venv/bin/python -I minimal_loop/test_paths.py        # 9 control-flow paths, scripted stand-in model, real kernel
    .venv/bin/python -I minimal_loop/run_dev.py 12        # real run: needs the local vLLM server on :8000 (qwen3.8-27b)
    .venv/bin/python -I minimal_loop/trivial_baselines.py # context numbers for the same 12 images
    .venv/bin/python minimal_loop/build_trace_widget.py && .venv/bin/python minimal_loop/build_part3_post.py   # rebuild the post

Data: `../agentic-factory-segmentation/data/void_dataset_v2/val.npz` (DEV split only; the test split is reserved for Part 6).
Results: `out/run_dev.json` (every cell of every episode), `outputs/*.txt` (captured stdout).
The model-written cells run UNSANDBOXED (Part 5 adds a container); see the post for the audit of all 93 executed cells.
Video: `video/part3_video.py` (classes `P3a`..`P3c`), clips in `video/chapters/p3{a,b,c}.warm.mp4`.

## Part 4 (eyes and hands)
`minimal_loop/`: `kernel_tools.py` (show + 4 tools, run inside the kernel), `session4.py`, `loop4.py`, `test_paths4.py`, `part_mask_check.py`,
`run_dev4.py A|B|C [N] [r2]`, `summarize_part4.py`, `build_trace_widget4.py`, `build_part4_post.py`. Runs: `out/run_dev4_{A,B,C}[r2].json`
(+ `out/part4_*_images/`). Video: `video/part4_video.py` (P4a-c), build with `make_chapter.sh part4_video.py P4a p4a clone warm`
(symlink the module into `paper-explainers-2026-10-09/videos/` with an absolute target, set `VC`).
