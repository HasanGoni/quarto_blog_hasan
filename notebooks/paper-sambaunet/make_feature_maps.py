"""Real, pre-trained SAM ViT-B features of the paper's lesion image at the three layers the paper taps (6, 9, 12).
CoSE at init == original SAM (verified in smoke_test.py), so these are what S6/S9/S12 look like before fine-tuning."""
import json, numpy as np, torch
from PIL import Image
from sambaunet import CoSESamEncoder
dev = "cuda"
cfg = CoSESamEncoder().enc.config
facts = {"layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "window": cfg.window_size,
         "global_attn_indexes_0based": list(cfg.global_attn_indexes), "image_size": cfg.image_size, "patch": cfg.patch_size, "neck_out": cfg.output_channels}
print(facts)
enc = CoSESamEncoder().to(dev).eval()
img = Image.open("assets/emb/p7-009.png").convert("RGB").resize((256, 256), Image.BICUBIC)
x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255)[None].to(dev)
m = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); s = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
with torch.no_grad(): S = enc((x - m) / s)
# one shared PCA over the three layers so colours are comparable; first 3 PCs -> RGB
F = {k: v[0].flatten(1).T.cpu().numpy() for k, v in S.items()}
allf = np.concatenate(list(F.values())); mu = allf.mean(0); U, Sg, Vt = np.linalg.svd(allf - mu, full_matrices=False)
for k, f in F.items():
    p = (f - mu) @ Vt[:3].T; lo, hi = np.percentile(allf @ Vt[:3].T, 1, axis=0), np.percentile(allf @ Vt[:3].T, 99, axis=0)
    rgb = np.clip((p - (lo - mu @ Vt[:3].T)) / (hi - lo), 0, 1).reshape(16, 16, 3) if False else None
projs = {k: (f - mu) @ Vt[:3].T for k, f in F.items()}
cat = np.concatenate(list(projs.values())); lo, hi = np.percentile(cat, 1, 0), np.percentile(cat, 99, 0)
for k, p in projs.items():
    rgb = (np.clip((p - lo) / (hi - lo), 0, 1).reshape(16, 16, 3) * 255).astype(np.uint8)
    Image.fromarray(rgb).resize((256, 256), Image.NEAREST).save(f"out/sam_S{k}_pca.png")
    n = np.linalg.norm(F[k], axis=1).reshape(16, 16); n = (n - n.min()) / (n.max() - n.min())
    print(f"S{k}: explained var of 3 PCs shared = {(Sg[:3]**2).sum()/(Sg**2).sum():.2f}; token-norm range {np.linalg.norm(F[k],axis=1).min():.1f}-{np.linalg.norm(F[k],axis=1).max():.1f}")
img.save("out/lesion_256.png")
json.dump(facts, open("out/sam_facts.json", "w"))
