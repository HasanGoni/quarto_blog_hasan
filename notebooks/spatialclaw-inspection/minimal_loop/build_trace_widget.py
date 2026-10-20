"""Build the interactive trace viewer for Part 3 from the REAL run (out/run_dev.json). Output: an include file for the post.

The viewer lets you pick an image, then step through the agent's cells: purpose, code, what the kernel printed.
Ground truth is shown next to the agent's number. Nothing here is invented: every string comes from run_dev.json.
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.load(open(ROOT / "out/run_dev.json"))
rows = {r["image"]: r for r in data["rows"]}
eps = []
for t in data["traces"]:
    r = rows[t["image"]]
    eps.append(dict(image=t["image"], gt=round(r["gt_pct"], 2), agent=None if r["agent_pct"] is None else round(r["agent_pct"], 2),
                    status=r["agent_status"], gt_status=r["gt_status"], reason=r["reason"], cracks=r["has_cracks"],
                    cells=[dict(step=c["step"], purpose=c["purpose"], code=c["code"], out=(c["stdout"] or "")[:900], err=c["error"]) for c in t["cells"]]))
payload = json.dumps(eps).replace("</", "<\\/")
page = '''```{=html}
<div id="tv" style="border:1px solid #ccc;border-radius:6px;padding:12px;margin:1em 0;">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <label for="tv-img"><b>Image</b></label><select id="tv-img"></select>
    <button id="tv-prev" type="button">&larr; Previous cell</button>
    <button id="tv-next" type="button">Next cell &rarr;</button>
    <span id="tv-pos" style="color:#666;"></span>
  </div>
  <div id="tv-head" style="margin:8px 0;font-weight:bold;"></div>
  <div id="tv-purpose" style="color:#555;margin-bottom:4px;"></div>
  <pre id="tv-code" style="white-space:pre-wrap;margin:0 0 6px;"></pre>
  <pre id="tv-out" style="white-space:pre-wrap;margin:0 0 6px;color:#1b5e20;"></pre>
  <pre id="tv-err" style="white-space:pre-wrap;margin:0 0 6px;color:#b71c1c;"></pre>
  <div id="tv-end" style="margin-top:8px;"></div>
</div>
<script>
(function(){
  var E = __PAYLOAD__, sel = document.getElementById("tv-img"), i = 0, k = 0;
  E.forEach(function(e, j){ var o = document.createElement("option"); o.value = j; o.textContent = e.image + " (ground truth " + e.gt + " %)"; sel.appendChild(o); });
  function show(){
    var e = E[i], c = e.cells[k];
    document.getElementById("tv-pos").textContent = "cell " + (k+1) + " of " + e.cells.length;
    document.getElementById("tv-head").textContent = e.image + (e.cracks ? "  (contains cracks)" : "");
    document.getElementById("tv-purpose").textContent = c.purpose;
    document.getElementById("tv-code").textContent = c.code || "(no code block in the reply)";
    document.getElementById("tv-out").textContent = c.out;
    document.getElementById("tv-err").textContent = c.err;
    var last = k === e.cells.length - 1;
    document.getElementById("tv-end").textContent = last ? ("End of episode. Agent number: " + (e.agent === null ? "none" : e.agent + " %") + "  |  ground truth: " + e.gt + " %  |  verdict: " + e.status + " (" + e.reason + "), truth would be " + e.gt_status) : "";
  }
  sel.onchange = function(){ i = +sel.value; k = 0; show(); };
  document.getElementById("tv-prev").onclick = function(){ k = Math.max(0, k-1); show(); };
  document.getElementById("tv-next").onclick = function(){ k = Math.min(E[i].cells.length-1, k+1); show(); };
  show();
})();
</script>
```
'''.replace("__PAYLOAD__", payload)
out = ROOT.parents[1] / "posts/series/spatialclaw-inspection/_trace_widget.md"
out.write_text(page)
print("wrote", out, len(page), "bytes,", len(eps), "episodes")
