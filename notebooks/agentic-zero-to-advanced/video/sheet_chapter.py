"""Contact sheet of a finished chapter: one frame late in every narrated sentence. Usage: python3 sheet_chapter.py <chapter.mp4> <timeline.json> <out.png> [cols]"""
import json, subprocess, sys
mp4, tl, out = sys.argv[1:4]; cols = int(sys.argv[4]) if len(sys.argv) > 4 else 3
beats = json.load(open(tl))["beats"]
ts = [b["t0"] + 0.88 * b["dur"] for b in beats]
sel = "+".join(f"eq(n\\,{int(t * 30)})" for t in ts)
rows = (len(ts) + cols - 1) // cols
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", f"select='{sel}',scale=640:-1,tile={cols}x{rows}", "-frames:v", "1", "-vsync", "0", out], check=True)
print(len(ts), "frames:", [round(t, 1) for t in ts])
