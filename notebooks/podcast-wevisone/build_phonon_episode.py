#!/usr/bin/env python3
"""WeVisOne-VL podcast with Phonon preset voices. Usage: build_phonon_episode.py HOST_A_VOICE HOST_B_VOICE [out_name]"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, subprocess
os.environ["VIDEO_VOICE_STYLE"] = "warm"          # no Kokoro phoneme markup in spoken()
sys.path.insert(0, str(Path.home() / ".hermes/scripts"))
from video_speech import spoken
import phonon_tts as P
a_voice, b_voice = sys.argv[1], sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else f"wevisone_phonon_{a_voice}_{b_voice}"
here = Path(__file__).parent; SR = 24000; GAP = 0.35
script = json.load(open(here / "script.json"))
parts, t0 = [], time.time()
for i, (who, text) in enumerate(script):
    v = a_voice if who == "A" else b_voice
    cf = here / "assets" / "phonon_cache" / (hashlib.sha1(f"{v}|{spoken(text)}".encode()).hexdigest()[:16] + ".npy")
    if cf.exists():                                   # edited scripts only re-synthesise changed lines
        a = np.load(cf)
    else:
        a = P.synth(spoken(text), v)
        a = np.asarray(a[0] if isinstance(a, list) else a, dtype=np.float32).squeeze()
        a = a * (10 ** (-22 / 20) / (np.sqrt(np.mean(a ** 2)) + 1e-9))
        cf.parent.mkdir(parents=True, exist_ok=True); np.save(cf, a)
    parts += [np.clip(a, -0.95, 0.95), np.zeros(int(SR * GAP), np.float32)]
    print(i, who, f"{len(a)/SR:.1f}s", flush=True)
y = np.concatenate(parts); out = here / "out"; out.mkdir(exist_ok=True)
sf.write(out / f"{name}.wav", y, SR)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / f"{name}.wav"), "-codec:a", "libmp3lame", "-q:a", "3", str(out / f"{name}.mp3")], check=True)
print(f"DONE {name}: {len(y)/SR/60:.1f} min audio in {time.time()-t0:.0f}s")
