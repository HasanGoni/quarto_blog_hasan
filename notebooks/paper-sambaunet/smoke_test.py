"""Smoke test for our SAMBA-UNet re-implementation. One real image (the lesion + mask printed in the paper's Figure 1).
NOT a benchmark: it only proves the wiring runs, gradients reach the right parameters and the frozen SAM stays frozen."""
import time, json, hashlib, torch, numpy as np
from PIL import Image
from transformers import SamModel
import torch.nn.functional as F
from sambaunet import SAMBAUNet, CoSESamEncoder, bce_dice_loss, selective_scan_ref, selective_scan_seq

torch.manual_seed(0)
dev = "cuda"
img = Image.open("assets/emb/p7-009.png").convert("RGB").resize((256, 256), Image.BICUBIC)       # paper Fig.1 lesion image
msk = Image.open("assets/emb/p7-000.png").convert("L").resize((256, 256), Image.BILINEAR)         # paper Fig.1 mask (GT or prediction: unknown)
x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255)[None].to(dev)
y = (torch.from_numpy(np.asarray(msk)).float().div(255)[None, None].to(dev) > 0.5).float()
xb, yb = x.repeat(2, 1, 1, 1), y.repeat(2, 1, 1, 1)                                                   # batch of 2 (same image twice)
print("input", tuple(xb.shape), "mask", tuple(yb.shape), "mask fg fraction %.3f" % y.mean().item())

# --- 0) the chunked selective scan must equal the plain sequential one ---------------------------------------
with torch.no_grad():
    b_, d_, n_, L_ = 2, 8, 16, 256
    u_, dl_ = torch.randn(b_, d_, L_, device=dev), torch.rand(b_, d_, L_, device=dev) * 0.1
    A_ = -torch.rand(d_, n_, device=dev) * 16; B_, C_, D_ = torch.randn(b_, n_, L_, device=dev), torch.randn(b_, n_, L_, device=dev), torch.randn(d_, device=dev)
    print("chunked vs sequential selective scan: max abs diff = %.2e" % (selective_scan_ref(u_, dl_, A_, B_, C_, D_) - selective_scan_seq(u_, dl_, A_, B_, C_, D_)).abs().max().item())

model = SAMBAUNet().to(dev)

# --- 1) CoSE is the identity at init: LoRA/Adapter output heads are zero-initialised ---------------
ref = SamModel.from_pretrained("facebook/sam-vit-base").vision_encoder.to(dev).eval()
with torch.no_grad():
    mean, std = model.sam_mean, model.sam_std
    z = (xb - mean) / std
    h = ref.patch_embed.projection(z).permute(0, 2, 3, 1); h = h + F.interpolate(ref.pos_embed.permute(0, 3, 1, 2), size=h.shape[1:3], mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
    ref_feats = {}
    for i, l in enumerate(ref.layers, 1):
        h = l(h)
        if i in (6, 9, 12): ref_feats[i] = h.permute(0, 3, 1, 2)
    model.eval(); mine = model.sam(z)
    for i in (6, 9, 12):
        print(f"S{i}: shape {tuple(mine[i].shape)}  max|CoSE - original SAM| = {(mine[i]-ref_feats[i]).abs().max().item():.2e}")
del ref

# --- 2) parameter budget ----------------------------------------------------------------------------
def n(ps): return sum(p.numel() for p in ps)
tr = [p for p in model.parameters() if p.requires_grad]
sam_frozen = [p for p in model.sam.parameters() if not p.requires_grad]
lora_ad = [p for nme, p in model.sam.named_parameters() if p.requires_grad]
print(f"total params           {n(model.parameters())/1e6:8.2f} M")
print(f"frozen SAM encoder     {n(sam_frozen)/1e6:8.2f} M")
print(f"LoRA + Adapters (CoSE) {n(lora_ad)/1e6:8.2f} M  trainable")
print(f"Mamba UNet + HCEM      {(n(tr)-n(lora_ad))/1e6:8.2f} M  trainable (random init here; the paper starts from VMamba-S)")
print(f"trainable total        {n(tr)/1e6:8.2f} M  ({100*n(tr)/n(model.parameters()):.1f}% of all)")

# --- 3) forward + shapes -------------------------------------------------------------------------------
model.train(); t = time.time()
logits, dbg = model(xb, return_debug=True); torch.cuda.synchronize()
print(f"forward {time.time()-t:.2f}s  logits {tuple(logits.shape)}")
for k, v in dbg["S"].items(): print(f"  SAM tap S{k:<2}: {tuple(v.shape)}")
for i, m in enumerate(dbg["M"], 1): print(f"  Mamba skip M{i}: {tuple(m.shape)}")
for k in sorted(k for k in dbg if k.startswith("dec")): print(f"  decoder out {k}: {tuple(dbg[k].shape)}")

# --- 4) one optimisation step: do the gradients reach the right parameters? -------------------------------
sam_w0 = hashlib.md5(model.sam.enc.layers[0].attn.attn.qkv.qkv.weight.detach().cpu().numpy().tobytes()).hexdigest()
opt = torch.optim.AdamW(tr, lr=1e-3)
loss = bce_dice_loss(logits, yb); t = time.time(); loss.backward(); torch.cuda.synchronize()
print(f"loss {loss.item():.4f}   backward {time.time()-t:.2f}s")
groups = {"LoRA down (q)": [p for nme, p in model.named_parameters() if "down_q" in nme],
          "LoRA up (q)": [p for nme, p in model.named_parameters() if "up_q" in nme],
          "Adapter up": [p for nme, p in model.named_parameters() if "adapter.up" in nme],
          "HCEM": list(model.hcem.parameters()), "VSS encoder": list(model.enc.parameters())}
for nme, ps in groups.items():
    print(f"  grad non-zero: {nme:14s} {sum(p.grad is not None and p.grad.abs().sum().item()>0 for p in ps)}/{len(ps)} tensors")
print("  frozen SAM qkv has grad:", model.sam.enc.layers[0].attn.attn.qkv.qkv.weight.grad is not None)
opt.step(); opt.zero_grad()
sam_w1 = hashlib.md5(model.sam.enc.layers[0].attn.attn.qkv.qkv.weight.detach().cpu().numpy().tobytes()).hexdigest()
print("  frozen SAM weight unchanged after step:", sam_w0 == sam_w1)

# --- 5) sanity: overfit this one image for a few steps (NOT a result; shows the loss can go down) -------------
losses = [loss.item()]
for s in range(1, 31):
    out = model(xb); l = bce_dice_loss(out, yb); l.backward(); opt.step(); opt.zero_grad(); losses.append(l.item())
    if s % 5 == 0: print(f"  step {s:2d}  loss {l.item():.4f}")
p = (torch.sigmoid(model(xb)[:1]) > 0.5).float()
dice = (2 * (p * y).sum() / (p.sum() + y.sum())).item()
print(f"after 30 steps on ONE image: dice vs mask = {dice:.3f}  (memorisation check only, not generalisation)")
json.dump({"losses": losses, "dice_train_image": dice, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}, open("out/overfit.json", "w"))
print("peak GPU memory %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
torch.save({"pred": p.cpu()}, "out/pred.pt")
