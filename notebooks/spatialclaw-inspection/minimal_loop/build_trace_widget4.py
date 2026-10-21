"""Interactive trace viewer for Part 4, built from the REAL runs (out/run_dev4_{A,B,C}.json).

Pick an image and a condition, then step through the agent's cells: purpose, code, what the kernel printed, and the
pictures the model was shown. Every string and picture comes from the saved runs. Pictures are re-saved as small JPEGs
under posts/series/spatialclaw-inspection/images/part4/ (quarto.pub rejects large files).
"""
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT.parents[1] / "posts/series/spatialclaw-inspection"
PIC = SITE / "images/part4"
PIC.mkdir(parents=True, exist_ok=True)
runs = {c: json.load(open(ROOT / f"out/run_dev4_{c}.json")) for c in "ABC"}
eps = {}
n_pics = 0
for c, data in runs.items():
    rows = {r["image"]: r for r in data["rows"]}
    for t in data["traces"]:
        r = rows[t["image"]]
        cells = []
        for cell in t["cells"]:
            pics = []
            for im in cell.get("images", []):
                src = ROOT / f"out/part4_{c}_images" / im["file"]
                dst = PIC / f"{c}_{im['file'].replace('.png', '.jpg')}"
                Image.open(src).convert("RGB").save(dst, quality=72)
                pics.append(dict(f=f"images/part4/{dst.name}", t=im["title"]))
                n_pics += 1
            cells.append(dict(step=cell["step"], purpose=cell["purpose"], code=cell["code"], out=(cell["stdout"] or "")[:700], err=cell["error"], pics=pics))
        eps.setdefault(t["image"], dict(image=t["image"], gt=round(r["gt_pct"], 2), gt_status=r["gt_status"], runs={}))["runs"][c] = dict(
            agent=None if r["agent_pct"] is None else round(r["agent_pct"], 2), status=r["agent_status"], reason=r["reason"], cells=cells)
payload = json.dumps(list(eps.values())).replace("</", "<\\/")
page = '''```{=html}
<div id="tv" style="border:1px solid #ccc;border-radius:6px;padding:12px;margin:1em 0;">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <label for="tv-img"><b>Image</b></label><select id="tv-img"></select>
    <label for="tv-cond"><b>Condition</b></label>
    <select id="tv-cond"><option value="A">A: text only, 14 turns</option><option value="B">B: + show()</option><option value="C" selected>C: + show() + tools</option></select>
    <button id="tv-prev" type="button">&larr; Previous cell</button>
    <button id="tv-next" type="button">Next cell &rarr;</button>
    <span id="tv-pos" style="color:#666;"></span>
  </div>
  <div id="tv-head" style="margin:8px 0;font-weight:bold;"></div>
  <div id="tv-purpose" style="color:#555;margin-bottom:4px;"></div>
  <pre id="tv-code" style="white-space:pre-wrap;margin:0 0 6px;"></pre>
  <pre id="tv-out" style="white-space:pre-wrap;margin:0 0 6px;color:#1b5e20;"></pre>
  <pre id="tv-err" style="white-space:pre-wrap;margin:0 0 6px;color:#b71c1c;"></pre>
  <div id="tv-pics" style="display:flex;gap:8px;flex-wrap:wrap;"></div>
  <div id="tv-end" style="margin-top:8px;"></div>
</div>
<script>
(function(){
  var E = __PAYLOAD__, sel = document.getElementById("tv-img"), cs = document.getElementById("tv-cond"), i = 0, k = 0;
  E.forEach(function(e, j){ var o = document.createElement("option"); o.value = j; o.textContent = e.image + " (ground truth " + e.gt + " %)"; sel.appendChild(o); });
  function show(){
    var e = E[i], r = e.runs[cs.value], c = r.cells[k];
    document.getElementById("tv-pos").textContent = "cell " + (k+1) + " of " + r.cells.length;
    document.getElementById("tv-head").textContent = e.image + "  |  condition " + cs.value;
    document.getElementById("tv-purpose").textContent = c.purpose;
    document.getElementById("tv-code").textContent = c.code || "(no code block in the reply)";
    document.getElementById("tv-out").textContent = c.out;
    document.getElementById("tv-err").textContent = c.err;
    var box = document.getElementById("tv-pics"); box.innerHTML = "";
    c.pics.forEach(function(p){ var d = document.createElement("figure"); d.style.margin = "0"; var im = document.createElement("img"); im.src = p.f; im.alt = p.t || "picture shown to the model"; im.style.width = "224px"; im.style.border = "1px solid #bbb"; var cap = document.createElement("figcaption"); cap.style.fontSize = "0.8em"; cap.style.color = "#555"; cap.textContent = "model saw: " + (p.t || "(untitled)"); d.appendChild(im); d.appendChild(cap); box.appendChild(d); });
    var last = k === r.cells.length - 1;
    document.getElementById("tv-end").textContent = last ? ("End of episode. Agent number: " + (r.agent === null ? "none" : r.agent + " %") + "  |  ground truth: " + e.gt + " %  |  verdict: " + r.status + " (" + r.reason + "), truth would be " + e.gt_status) : "";
  }
  sel.onchange = function(){ i = +sel.value; k = 0; show(); };
  cs.onchange = function(){ k = 0; show(); };
  document.getElementById("tv-prev").onclick = function(){ k = Math.max(0, k-1); show(); };
  document.getElementById("tv-next").onclick = function(){ k = Math.min(E[i].runs[cs.value].cells.length-1, k+1); show(); };
  show();
})();
</script>
```
'''.replace("__PAYLOAD__", payload)
out = SITE / "_trace_widget4.md"
out.write_text(page)
print("wrote", out, len(page), "bytes,", len(eps), "images,", n_pics, "pictures")
