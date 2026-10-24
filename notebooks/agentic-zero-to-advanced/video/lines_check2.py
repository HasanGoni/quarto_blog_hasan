"""Per-clip check of narration against its script: transcribe each warm clip with faster-whisper and diff it with what should be said.
Usage (tts venv): python lines_check2.py <module.py> <DICTNAME> [<DICTNAME> ...]     Prints the ratio for every line and what was heard when it is not perfect."""
import ast, difflib, json, re, sys
sys.path.insert(0, str(__import__("pathlib").Path.home() / ".hermes" / "scripts"))
from faster_whisper import WhisperModel
from video_speech import spoken, strip_markup
tree = ast.parse(open(sys.argv[1]).read())
lines = {}
for n in tree.body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in sys.argv[2:]:
        lines.update(ast.literal_eval(n.value))
ND = "assets/um/narration_clone_warm"
man = json.load(open(f"{ND}/manifest.json"))
by = {strip_markup(v["text"]): v for v in man.values()}
m = WhisperModel("small", device="cpu", compute_type="int8")
norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
for k, t in lines.items():
    e = by.get(strip_markup(t)) or by.get(strip_markup(spoken(t)))
    if not e:
        print(k, "NO AUDIO"); continue
    segs, _ = m.transcribe(f"{ND}/{e['file']}", language="en")
    heard = " ".join(s.text.strip() for s in segs)
    a, b = [norm(w) for w in spoken(t).split()], [norm(w) for w in heard.split()]
    r = difflib.SequenceMatcher(None, a, b).ratio()
    print(f"{k} {r:.2f}" + (f" | {heard}" if r < 0.999 else ""))
