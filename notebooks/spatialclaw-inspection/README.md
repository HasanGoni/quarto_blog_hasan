# spatialclaw-inspection

Smoke tests and diagrams for the "SpatialClaw for Inspection" series.

The upstream code is NOT vendored (non-commercial NVIDIA licence). To reproduce:

    git clone https://github.com/NVlabs/SpatialClaw.git upstream/repo
    git -C upstream/repo checkout f37b163
    uv sync
    .venv/bin/python -I smoke_safety.py
    .venv/bin/python -I smoke_kernel.py

Captured outputs: `outputs/`. Diagrams: `generate_diagrams.py`.
