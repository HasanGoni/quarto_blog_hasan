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
