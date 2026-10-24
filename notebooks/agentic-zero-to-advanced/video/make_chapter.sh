#!/usr/bin/env bash
# Build ONE narrated chapter as its own small video (visuals + audio + word-by-word subtitles with yellow highlighter).
# Usage:  ./make_chapter.sh <module>[.py] <SceneClass> <slug> [kokoro|clone] [plain|warm]      (about 2-3 min; run it in the FOREGROUND and read the last lines)
#   clone (default) = the author's own cloned voice.   plain (default) = calm: Kokoro speech converted to the author's timbre  ->  chapters/<slug>.mp4
#                                                      warm = Qwen3 emotional speech converted to the author's timbre         ->  chapters/<slug>.warm.mp4
#   kokoro          = the preset/blended Kokoro voice from video_voice.json
# Every step fails loudly (non-zero exit + the error text); success ends with DONE: <path>.
# Review the chapter, give feedback, rebuild only that chapter; approve with ./approve_chapter.sh <slug>; join with ./join_approved.sh
set -euo pipefail
cd "$(dirname "$0")"; HERE=$PWD
MOD=${1%.py}; CLS=$2; SLUG=$3; ENGINE=${4:-clone}; STYLE=${5:-plain}
[ -f "$MOD.py" ] || { echo "ERROR: $MOD.py not found in $HERE" >&2; exit 2; }
SFX=""; [ "$STYLE" = plain ] || SFX=".$STYLE"
TAG="$SLUG$SFX"
export VIDEO_VOICE_ENGINE=$ENGINE VIDEO_VOICE_STYLE=$STYLE HF_HUB_DISABLE_XET=1 CHAPTER_SUFFIX=$SFX
S=~/.hermes/scripts; MANIM=~/.hermes/video/.venv/bin/manim; VC=$HERE/../../voice-clone
RES=${RES:-1280,720}; FPS=${FPS:-30}
if [ "$ENGINE" = clone ]; then
  if [ "$STYLE" = plain ]; then ND=assets/um/narration_clone; else ND=assets/um/narration_clone_$STYLE; fi
else ND=assets/um/narration; fi
mkdir -p assets/um/narration assets/um/narration_clone chapters "$ND"
rm -f $ND/missing.json
LOGD=$(mktemp -d); trap 'rm -rf "$LOGD"' EXIT
fail() { echo "ERROR in step: $1" >&2; tail -n 15 "$LOGD/$2" >&2; exit 1; }
echo "preflight: $(free -g | awk '/Mem:/{print $7" GB RAM available"}'), $(df -h . | awk 'NR==2{print $4" disk free"}'), voice=$ENGINE/$STYLE"
# 1) quick collect pass: which sentences still need a voice (also catches scene-code errors in seconds)
(cd ~/.hermes/video && $MANIM render --disable_caching --media_dir "$HERE/_media_c1_$TAG" -ql --progress_bar none "$HERE/$MOD.py" "$CLS") > "$LOGD/c1.log" 2>&1 || fail "scene code (collect pass)" c1.log
# 2) voice
if [ -f $ND/missing.json ]; then
  if [ "$ENGINE" = clone ] && [ "$STYLE" != plain ]; then
    # source stage: Qwen3 emotional speech (no word times; they are recovered with whisper after the conversion)
    $VC/qwen/.venv/bin/python $S/tts_batch_qwen.py $ND/missing.json assets/um/src_qwen_$STYLE > "$LOGD/q.log" 2>&1 || fail "Qwen3 speech (warm source)" q.log
    tail -n 1 "$LOGD/q.log"; SRC=assets/um/src_qwen_$STYLE
  else
    # Kokoro makes the words + word timings (on CPU: small model, never competes with vLLM for GPU memory)
    CUDA_VISIBLE_DEVICES="" VIDEO_VOICE_ENGINE=kokoro ~/.hermes/tts/.venv/bin/python $S/tts_batch.py $ND/missing.json assets/um/narration > "$LOGD/k.log" 2>&1 || fail "Kokoro speech" k.log
    tail -n 1 "$LOGD/k.log"; SRC=assets/um/narration
  fi
  if [ "$ENGINE" = clone ]; then
    $VC/.venv/bin/python $S/tts_batch_clone.py $ND/missing.json $ND $SRC > "$LOGD/v.log" 2>&1 || fail "voice conversion (clone)" v.log
    grep -E "converted|no Kokoro" "$LOGD/v.log" || true
    grep -q "no Kokoro" "$LOGD/v.log" && fail "voice conversion: a sentence had no source audio" v.log
  fi
  if [ "$STYLE" != plain ]; then
    ~/.hermes/tts/.venv/bin/python fill_word_ts.py $ND > "$LOGD/w.log" 2>&1 || fail "word timestamps (whisper)" w.log
    tail -n 1 "$LOGD/w.log"
  fi
  # every sentence must now have audio, otherwise the video would silently get placeholder timing
  VIDEO_VOICE_ENGINE=$ENGINE VIDEO_VOICE_STYLE=$STYLE python3 - "$ND" <<'PY' || { echo "ERROR: narration audio missing for the sentences above" >&2; exit 1; }
import json, sys
sys.path.insert(0, str(__import__("pathlib").Path.home() / ".hermes" / "scripts"))
from video_speech import spoken, tts_key
nd = sys.argv[1]
man = json.load(open(f"{nd}/manifest.json"))
miss = [t for t in json.load(open(f"{nd}/missing.json")) if tts_key(spoken(t)) not in man and tts_key(t) not in man]
for t in miss:
    print("  no audio for:", t[:90])
sys.exit(1 if miss else 0)
PY
fi
# 3) base video (visuals + audio), 4) transparent subtitle overlay, 5) composite
(cd ~/.hermes/video && $MANIM render --disable_caching --media_dir "$HERE/_media_c2_$TAG" -r $RES --fps $FPS --progress_bar none "$HERE/$MOD.py" "$CLS") > "$LOGD/b.log" 2>&1 || fail "base video" b.log
(cd ~/.hermes/video && SUBTITLE_TIMELINE="$HERE/assets/um/timeline_$TAG.json" $MANIM render --disable_caching -t --format=webm --media_dir "$HERE/_media_c3_$TAG" -r $RES --fps $FPS --progress_bar none "$HERE/subtitle_overlay.py" SubtitleOverlay) > "$LOGD/s.log" 2>&1 || fail "subtitle overlay" s.log
BASE=$(ls "$HERE"/_media_c2_$TAG/videos/$MOD/*/$CLS.mp4); SUBS=$(ls "$HERE"/_media_c3_$TAG/videos/subtitle_overlay/*/SubtitleOverlay.webm)
ffmpeg -y -loglevel error -i "$BASE" -c:v libvpx-vp9 -i "$SUBS" -filter_complex "[0:v][1:v]overlay=format=auto[v]" -map "[v]" -map 0:a -c:v libx264 -crf 22 -preset medium -pix_fmt yuv420p -c:a copy -movflags +faststart chapters/$TAG.mp4
rm -rf _media_c1_$TAG _media_c2_$TAG _media_c3_$TAG
echo "DONE: $(ls -la chapters/$TAG.mp4 | awk '{print $5" bytes "$9}')"
