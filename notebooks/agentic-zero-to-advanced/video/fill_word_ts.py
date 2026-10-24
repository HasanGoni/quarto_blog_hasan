#!/usr/bin/env python3
"""Fill the word timestamps of a narration folder whose manifest entries have none (the "warm" voice style).

The warm source (Qwen3 speech) has no word times, and the voice conversion keeps none, so every manifest entry
has ts == []. Without them the subtitles fall back to an even spread. This transcribes each clip with faster-whisper
(CPU, so it never competes with vLLM for GPU memory), aligns whisper's words to the SPOKEN text, and writes
[[word, start, end], ...] back into manifest.json, the same shape Kokoro's timestamps have.

Usage (tts venv):  python fill_word_ts.py <narration_dir>
"""
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
from faster_whisper import WhisperModel  # noqa: E402
from video_speech import spoken  # noqa: E402

nd = Path(sys.argv[1])
mpath = nd / "manifest.json"
man = json.load(open(mpath))
norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
todo = [k for k, e in man.items() if not e.get("ts") and (nd / e["file"]).exists()]
if not todo:
    print("0 word timestamps to fill")
    sys.exit(0)
model = WhisperModel("small", device="cpu", compute_type="int8")
filled = covered = total = 0
for k in todo:
    e = man[k]
    segs, _ = model.transcribe(str(nd / e["file"]), language="en", word_timestamps=True)
    heard = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words if norm(w.word)]
    said = re.findall(r"\S+", spoken(e["text"]))
    a, b = [norm(w) for w in said], [norm(h[0]) for h in heard]
    ts = []
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for i in range(blk.size):
            ts.append([said[blk.a + i], round(heard[blk.b + i][1], 2), round(heard[blk.b + i][2], 2)])
    e["ts"] = ts
    filled += 1; covered += len(ts); total += len(said)
json.dump(man, open(mpath, "w"), indent=1)
print(f"{filled} clips filled with whisper word times ({covered}/{total} spoken words aligned)")
