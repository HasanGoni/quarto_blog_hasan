#!/usr/bin/env bash
# Build ONE Motion Canvas chapter (voice from the Manim voice stub, headless render, mux to 1280x720 / 30 fps so it joins Manim chapters).
# Usage: ./make_mc_chapter.sh <slug> <VoiceClass>     e.g.  ./make_mc_chapter.sh p8_tools ToolsVoice     -> chapters/<slug>.warm.mp4
# Needs: video/mc (setup.sh), video/p8_voice.py, mc/src/scenes/<slug>.tsx and mc/src/<slug>.ts. Every step fails loudly; success ends with DONE.
set -euo pipefail
cd "$(dirname "$0")"; HERE=$PWD; SLUG=$1; CLS=$2
[ -n "${SKIP_VOICE:-}" ] || ./make_chapter.sh p8_voice "$CLS" "$SLUG" clone warm | tail -n 2
mkdir -p "mc/src/$SLUG"
ffmpeg -y -loglevel error -i "chapters/$SLUG.warm.mp4" -vn -ac 1 -ar 24000 "mc/src/$SLUG/audio.wav"
cp "assets/um/timeline_$SLUG.warm.json" "mc/src/$SLUG/timeline.json"
cd mc
export PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright-mc}
PORT=$((9100 + RANDOM % 500)); LOG=$(mktemp); trap 'kill $(lsof -ti:$PORT) 2>/dev/null || true; rm -f "$LOG"' EXIT
rm -rf output; (npx vite --port $PORT --strictPort --host 127.0.0.1 > "$LOG" 2>&1 &)
for i in $(seq 1 40); do grep -q "ready in" "$LOG" && break; sleep 0.5; done
grep -q "ready in" "$LOG" || { echo "ERROR: vite did not start" >&2; tail -n 15 "$LOG" >&2; exit 1; }
PORT=$PORT PROJECT=$SLUG node shot.mjs > /dev/null 2>&1 || true      # warm-up: the first load after a start re-optimises dependencies
PORT=$PORT PROJECT=$SLUG node render.mjs 2>&1 | tail -n 3
END=$(python3 -c "import json;print(json.load(open('src/$SLUG/timeline.json'))['end'])")
N=$(ls output/*/*.png 2>/dev/null | wc -l); WANT=$(python3 -c "print(round($END*30))")
[ "$N" -ge $((WANT - 2)) ] || { echo "ERROR: rendered $N frames, expected about $WANT" >&2; grep -i -E "error|undefined" "$LOG" | tail -n 8 >&2; exit 1; }
FR=$(ls -d output/*/ | head -n 1)
ffmpeg -y -loglevel error -framerate 30 -i "${FR}%06d.png" -i "src/$SLUG/audio.wav" -af apad -vf "scale=1280:720" -c:v libx264 -crf 22 -pix_fmt yuv420p -c:a aac -ac 1 -b:a 64k -t "$END" -movflags +faststart "../chapters/$SLUG.mc.warm.mp4"
rm -rf output
echo "DONE: $(ls -la ../chapters/$SLUG.mc.warm.mp4 | awk '{print $5" bytes "$9}')  ${END}s"
