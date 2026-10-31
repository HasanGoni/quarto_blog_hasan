"""Builds explainer.html (single self-contained interactive page) from the REAL files: sambaunet.py code (by AST, with line numbers),
out/smoke_test.txt (real captured output), the paper's own figures. Rebuild:  python3 build_page.py"""
import ast, base64, html, json, pathlib
H = pathlib.Path(__file__).parent
SRC = (H / "sambaunet.py").read_text(); LINES = SRC.splitlines(); TREE = ast.parse(SRC)
OUT = (H / "out/smoke_test.txt").read_text()

def b64(p): return "data:image/jpeg;base64," + base64.b64encode((H / p).read_bytes()).decode()

def snippet(name, cls=None):
    for n in ast.walk(TREE):
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name:
            body = n
            if cls:
                body = next(m for m in n.body if isinstance(m, ast.FunctionDef) and m.name == cls)
            a, b = body.lineno, body.end_lineno
            return {"title": f"sambaunet.py:{a}-{b}  {name}" + (f".{cls}" if cls else ""), "code": "\n".join(LINES[a - 1:b])}
    raise KeyError(name)

def out_lines(*keys):
    return "\n".join(l for l in OUT.splitlines() if any(k in l for k in keys))

STAGES = {
 "in": dict(t="1 · Input", shape="(B,3,256,256)", analogy="A photo of a mole under a dermatoscope, shrunk to a thumbnail the size both brains can read.",
   text="Every image is resized to 256×256 (paper §4.2, chosen to fit the VMamba-S backbone). Both encoders see the SAME picture at the same time — they are two parallel experts, not a pipeline.",
   code=[], out=out_lines("input")),
 "sam": dict(t="2 · SAM encoder (frozen)", shape="3 taps: S6, S9, S12 → (B,768,16,16)", analogy="A very well-travelled tourist who has seen a billion photos: knows what an 'object with a boundary' looks like, but has never read a dermatology book.",
   text="A ViT-B with 12 transformer blocks. Blocks 3, 6, 9, 12 use global attention (the others look inside 14×14 windows) — we checked this in the HF config, and it means the paper's taps (6, 9, 12) sit exactly on global-attention blocks. Its weights stay FROZEN (the snowflake in Fig. 1a). We feed it 256×256, so it has 16×16 tokens. The paper does not say how it handles SAM's native 1024 input — our choice, and the trapezoid 'Upsample' in Fig. 1b hints at it.",
   code=[snippet("CoSESamEncoder", "forward")], out=out_lines("SAM tap", "S6:", "S9:", "S12:", "frozen SAM")),
 "cose": dict(t="3 · CoSE: LoRA + Adapters", shape="+2.53 M trainable parameters", analogy="Instead of re-educating the tourist, you hand them a small notebook (LoRA: tweaks how they look) and a pair of glasses (Adapter: tweaks what they conclude). The brain stays untouched.",
   text="Eq. 1: LoRA adds W_up(W_down(x)) to the Q and V projections. Eq. 2: two Adapters per block — one after attention, one in the MLP path. Both start as exact identities (up-projection = 0), so on day one the model equals plain SAM. We verified: max |CoSE − SAM| = 0.00e+00. A side effect worth knowing: at step 0 the LoRA 'down' matrices get ZERO gradient (the 'up' is zero) — normal for LoRA, but it fooled our first gradient check.",
   code=[snippet("LoRAqkv"), snippet("Adapter"), snippet("AttnWithAdapter"), snippet("MLPWithAdapter")], out=out_lines("max|CoSE", "LoRA + Adapters", "total params", "frozen SAM", "trainable total", "grad non-zero")),
 "mamba": dict(t="4 · Mamba (VSS) encoder", shape="M1 (96,64,64) · M2 (192,32,32) · M3 (384,16,16)", analogy="A careful inspector who walks through the image line by line, carrying a small notebook of what they've seen so far. Four inspectors walk four different routes so no pixel is only seen from one side.",
   text="Patch-embed (4×4) → [VSS block ×2 → Patch Merging] ×3 → VSS block ×2. Each VSS block (Fig. 1c) = LayerNorm → Linear → depth-wise conv → SS2D → LayerNorm, gated by a second Linear branch. SS2D scans the feature map in 4 directions with a selective state-space (S6) recurrence: linear cost in the number of pixels, where attention is quadratic. The skip features M1, M2, M3 are taken before each merge. Paper starts from VMamba-S weights; ours are random (stand-in) and the scan is plain PyTorch, not the CUDA kernel.",
   code=[snippet("selective_scan_ref"), snippet("SS2D", "forward"), snippet("VSSBlock")], out=out_lines("chunked vs sequential", "Mamba skip", "Mamba UNet")),
 "hcem": dict(t="5 · HCEM (the paper's new module; Fig. 1b calls it 'CPEM')", shape="(S_i, M_i) → guided skip (C,H,W)", analogy="A translator at a meeting. The tourist (SAM) speaks in 'what is this object' terms, the inspector (Mamba) in 'where exactly is the edge' terms. HCEM puts them in one room and lets a volume knob decide which channels deserve attention.",
   text="Eq. 3: upsample S_i to M_i's size (MS), concatenate along channels, 1×1 conv. Eq. 4: F_att = σ(AvgPool(F) + MaxPool(F)) — a channel-attention gate, CBAM-style. The gated features are added back residually, concatenated once more with M_i and squeezed by a 1×1 conv. Pairs: (S6,M1) (S9,M2) (S12,M3). Try the stepper below.",
   code=[snippet("HCEM")], out=""),
 "dec": dict(t="6 · Mamba decoder + head", shape="(B,1,256,256) mask logits", analogy="The inspector now walks back out, but at every doorway the translator hands over the tourist's notes (guided skip).",
   text="Bottleneck VSS ×2 → [Patch Expand → add HCEM output → VSS ×2] ×3 → ×4 upsample → 1×1 conv. The paper's Fig. 1a labels the decoder blocks 'Patch Merging' and draws a concat symbol (⊗); resolution must go UP, so we used VM-UNet-style Patch Expanding and additive fusion. Both are our reading. The optional SAM mask-decoder output S_out (⊕ in Fig. 1a) is 'selectively incorporated based on performance evaluation' — not implemented here.",
   code=[snippet("SAMBAUNet", "forward")], out=out_lines("decoder out", "logits")),
 "loss": dict(t="7 · Loss and a training step", shape="L = λ·BCE + λ·Dice", analogy="Two judges: one marks every pixel right/wrong (BCE), one scores how much the blob overlaps (Dice). Dice stops the model from winning by calling everything 'skin'.",
   text="Eq. 5/6 with λ 'identical to VM-UNet' — values not printed in the paper; we used 1.0/1.0. Paper recipe: AdamW, lr 1e-3, CosineAnnealingLR (min 1e-4), batch 32, 300 epochs, one RTX A6000. Our smoke test is ONE real image (the lesion printed in Fig. 1) for 30 steps: it proves gradients reach the right parameters and the frozen SAM weight does not move. The Dice 0.993 is memorisation of one picture, NOT a benchmark.",
   code=[snippet("bce_dice_loss")], out=out_lines("loss ", "step ", "after 30", "peak GPU", "unchanged", "grad non-zero")),
}
for s in STAGES.values():
    s["code"] = [{"title": c["title"], "code": c["code"]} for c in s["code"]]

IMG = {k: b64(f"web/{k}.jpg") for k in ["fig1a", "fig1b", "fig1c", "fig2", "table3", "lesion", "S6", "S9", "S12"]}

PAGE = (H / "page_template.html").read_text()
PAGE = PAGE.replace("/*STAGES*/", "const STAGES=" + json.dumps(STAGES) + ";").replace("/*IMG*/", "const IMG=" + json.dumps(IMG) + ";")
PAGE = PAGE.replace("<!--FULLOUT-->", html.escape(OUT))
(H / "explainer.html").write_text(PAGE)
print("explainer.html", len(PAGE) // 1024, "KB")
