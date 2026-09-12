#!/usr/bin/env bash
# Real Marigold V2 inference on the real GDXray+ Castings X-ray crops in xray_inputs/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

uv run python repo/scripts/infer.py --modality depth   --image_dir xray_inputs --output_dir output/depth
uv run python repo/scripts/infer.py --modality normals --image_dir xray_inputs --output_dir output/normals
uv run python repo/scripts/infer.py --modality albedo  --image_dir xray_inputs --output_dir output/albedo
