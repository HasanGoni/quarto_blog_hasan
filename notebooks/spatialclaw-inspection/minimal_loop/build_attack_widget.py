"""Interactive attack explorer for Part 5, built from the REAL attack run (out/part5_attacks.json)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.load(open(ROOT / "out/part5_attacks.json"))
payload = json.dumps(data).replace("</", "<\\/")
page = '''```{=html}
<div id="av" style="border:1px solid #ccc;border-radius:6px;padding:12px;margin:1em 0;">
  <label for="av-sel"><b>Attack</b></label> <select id="av-sel" style="max-width:100%;"></select>
  <pre id="av-code" style="white-space:pre-wrap;margin:8px 0;"></pre>
  <div style="display:flex;gap:12px;flex-wrap:wrap;">
    <div style="flex:1 1 280px;border:1px solid #e03131;border-radius:4px;padding:8px;"><b style="color:#e03131;">Unsandboxed kernel (Parts 3 and 4)</b><div id="av-un" style="margin-top:4px;"></div></div>
    <div style="flex:1 1 280px;border:1px solid #2f9e44;border-radius:4px;padding:8px;"><b style="color:#2f9e44;">Container (Part 5)</b><div id="av-co" style="margin-top:4px;"></div></div>
  </div>
</div>
<script>
(function(){
  var D = __PAYLOAD__, sel = document.getElementById("av-sel");
  D.forEach(function(d, i){ var o = document.createElement("option"); o.value = i; o.textContent = d.name; sel.appendChild(o); });
  function show(){
    var d = D[+sel.value];
    document.getElementById("av-code").textContent = d.code;
    document.getElementById("av-un").textContent = d.unsandboxed + (d.unsandboxed_next ? "  [" + d.unsandboxed_next + "]" : "");
    document.getElementById("av-co").textContent = d.container + (d.container_next ? "  [" + d.container_next + "]" : "");
  }
  sel.onchange = show; show();
})();
</script>
```
'''.replace("__PAYLOAD__", payload)
out = ROOT.parents[1] / "posts/series/spatialclaw-inspection/_attack_widget.md"
out.write_text(page)
print("wrote", out, len(page), "bytes")
