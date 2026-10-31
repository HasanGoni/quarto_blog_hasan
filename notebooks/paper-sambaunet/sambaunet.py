"""SAMBA-UNet (arXiv 2610.40029) -- a reference re-implementation written by us.

The paper says "code will be released upon acceptance", so there is NO official repo.
Everything below is built from the paper's text, equations and Figure 1. Where the paper is silent
we made a choice and marked it  # ASSUMPTION  so you can see exactly where we guessed.

Real parts : SAM ViT-B encoder weights (facebook/sam-vit-base, via transformers).
Stand-ins  : the Mamba branch is randomly initialised (the paper starts from VMamba-S weights),
             and the selective scan is a slow pure-PyTorch reference (the official VMamba uses a CUDA kernel).
"""
from __future__ import annotations
import math
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import SamModel


# ----------------------------------------------------------------------------------------------
# 1) CoSE: LoRA on Q,V  (Eq. 1)  +  two Adapters per ViT block (Eq. 2)
# ----------------------------------------------------------------------------------------------
class LoRAqkv(nn.Module):
    """Wraps SAM's fused qkv Linear. Adds  W_up(W_down(x))  to the Q and V slices only (Eq. 1)."""
    def __init__(self, qkv: nn.Linear, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        self.qkv, d = qkv, qkv.in_features
        self.down_q, self.up_q = nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)
        self.down_v, self.up_v = nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)
        for up in (self.up_q, self.up_v):
            nn.init.zeros_(up.weight)              # start exactly at the pre-trained SAM (no change at step 0)
        self.scale, self.d = alpha / rank, d

    def forward(self, x):
        out = self.qkv(x)                           # HF layout: [q | k | v], each d wide (checked in modeling_sam.py)
        dq = self.up_q(self.down_q(x)) * self.scale
        dv = self.up_v(self.down_v(x)) * self.scale
        zero = torch.zeros_like(dq)
        return out + torch.cat([dq, zero, dv], dim=-1)


class Adapter(nn.Module):
    """Eq. 2:  X_out = W_up(GELU(W_down(X_in)))  (bottleneck MLP)."""
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.down, self.up, self.act = nn.Linear(d, hidden), nn.Linear(hidden, d), nn.GELU()
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)   # identity at init

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class AttnWithAdapter(nn.Module):
    """Adapter #1: after MHSA, before the residual add.  # ASSUMPTION: added as x + Adapter(x)."""
    def __init__(self, attn, d):
        super().__init__(); self.attn, self.adapter = attn, Adapter(d)

    def forward(self, hidden_states, **kw):
        out, w = self.attn(hidden_states, **kw)
        return out + self.adapter(out), w


class MLPWithAdapter(nn.Module):
    """Adapter #2: in the MLP's residual path.  # ASSUMPTION: parallel branch  mlp(x) + Adapter(x)  (Houlsby/MedSAM style)."""
    def __init__(self, mlp, d):
        super().__init__(); self.mlp, self.adapter = mlp, Adapter(d)

    def forward(self, x):
        return self.mlp(x) + self.adapter(x)


class CoSESamEncoder(nn.Module):
    """SAM ViT-B encoder, frozen, with LoRA + Adapters (the only trainable parts). Returns S_6, S_9, S_12."""
    def __init__(self, name="facebook/sam-vit-base", rank=4, taps=(6, 9, 12)):
        super().__init__()
        enc = SamModel.from_pretrained(name).vision_encoder
        for p in enc.parameters():
            p.requires_grad_(False)                       # frozen foundation model
        d = enc.config.hidden_size
        for layer in enc.layers:
            layer.attn.qkv = LoRAqkv(layer.attn.qkv, rank)
            layer.attn = AttnWithAdapter(layer.attn, d)
            layer.mlp = MLPWithAdapter(layer.mlp, d)
        self.enc, self.taps, self.d = enc, taps, d

    def forward(self, x256):
        """x256: (B,3,256,256) already SAM-normalised. # ASSUMPTION: SAM sees 256x256 -> 16x16 tokens
        (the paper's images are 256x256 and Figure 1(b) has an *Upsample* block, which only makes sense for 16x16 tokens)."""
        e = self.enc
        h = e.patch_embed.projection(x256).permute(0, 2, 3, 1)  # HF patch_embed insists on 1024px; the conv itself has no fixed size
        #                                                      # (B,16,16,768)
        pe = F.interpolate(e.pos_embed.permute(0, 3, 1, 2), size=h.shape[1:3], mode="bicubic", align_corners=False)
        h = h + pe.permute(0, 2, 3, 1)                                                  # resize abs. pos-embed 64x64 -> 16x16
        feats = {}
        for i, layer in enumerate(e.layers, start=1):
            h = layer(h)
            if i in self.taps:
                feats[i] = h.permute(0, 3, 1, 2).contiguous()                           # (B,768,16,16)
        return feats


# ----------------------------------------------------------------------------------------------
# 2) Mamba side: VSS block (Fig. 1c) with SS2D. Pure-PyTorch chunked selective scan (no CUDA kernel).
# ----------------------------------------------------------------------------------------------
def selective_scan_seq(u, delta, A, B, C, D):
    """Plain sequential S6 scan (slow, kept as the ground truth to test the chunked version against).
    u,delta:(b,d,L)  A:(d,n)  B,C:(b,n,L)  D:(d).   h_t = exp(dt*A) h_{t-1} + dt*B_t*u_t ;  y_t = C_t h_t + D u_t"""
    b, d, L = u.shape; n = A.shape[1]
    dA = torch.exp(delta.unsqueeze(-1) * A[None, :, None, :])                        # (b,d,L,n)
    dBu = delta.unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1) * u.unsqueeze(-1)    # (b,d,L,n)
    h = u.new_zeros(b, d, n); ys = []
    for t in range(L):
        h = dA[:, :, t] * h + dBu[:, :, t]
        ys.append((h * C[:, :, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, -1) + u * D[None, :, None]


def selective_scan_ref(u, delta, A, B, C, D, T=64):
    """Same maths as selective_scan_seq, but two-level (chunked) so it takes ~2*sqrt(L) sequential steps instead of L:
    1) inside every chunk of T steps, scan from a zero state (all chunks in parallel);
    2) scan the chunk-end states across chunks (carry);  3) add carry * running decay back into every step."""
    b, d, L = u.shape; n = A.shape[1]; nc = L // T
    assert L % T == 0
    dA = torch.exp(delta.unsqueeze(-1) * A[None, :, None, :]).view(b, d, nc, T, n)
    dBu = (delta.unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1) * u.unsqueeze(-1)).view(b, d, nc, T, n)
    h = torch.zeros_like(dBu[:, :, :, 0]); P = torch.ones_like(h); hs, Ps = [], []
    for t in range(T):                                        # 1) local scan, parallel over chunks
        h = dA[:, :, :, t] * h + dBu[:, :, :, t]; P = dA[:, :, :, t] * P
        hs.append(h); Ps.append(P)
    hs, Ps = torch.stack(hs, 3), torch.stack(Ps, 3)           # (b,d,nc,T,n)
    E, Ptot = hs[:, :, :, -1], Ps[:, :, :, -1]
    c = torch.zeros_like(E[:, :, 0]); carries = []
    for k in range(nc):                                       # 2) carry across chunks
        carries.append(c); c = Ptot[:, :, k] * c + E[:, :, k]
    carry = torch.stack(carries, 2).unsqueeze(3)              # (b,d,nc,1,n)
    full = (hs + Ps * carry).reshape(b, d, L, n)              # 3) true state at every step
    y = (full * C.permute(0, 2, 1).unsqueeze(1)).sum(-1)
    return y + u * D[None, :, None]


class SS2D(nn.Module):
    """2D selective scan: unfold the feature map in 4 directions (row, column, and both reversed), run S6 on each, merge."""
    def __init__(self, d_model, d_state=16, expand=2, dt_rank=None):
        super().__init__()
        self.d_inner, self.n = expand * d_model, d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.dw = nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner)
        K = 4
        self.x_proj = nn.Parameter(torch.randn(K, self.dt_rank + 2 * d_state, self.d_inner) * self.d_inner ** -0.5)
        self.dt_w = nn.Parameter(torch.randn(K, self.d_inner, self.dt_rank) * self.dt_rank ** -0.5)
        self.dt_b = nn.Parameter(torch.full((K, self.d_inner), -4.0))             # softplus(-4) ~ 0.018 initial step size
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(K * self.d_inner, 1))
        self.Ds = nn.Parameter(torch.ones(K * self.d_inner))
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):                                   # x: (B,H,W,C)
        B, H, W, _ = x.shape; L = H * W; K = 4
        xz = self.in_proj(x); x, z = xz.chunk(2, dim=-1)
        x = F.silu(self.dw(x.permute(0, 3, 1, 2)))          # (B,Dm,H,W)
        xs = torch.stack([x.flatten(2), x.transpose(2, 3).flatten(2)], 1)         # row-major and column-major order
        xs = torch.cat([xs, xs.flip(-1)], 1)                                      # + both reversed -> (B,4,Dm,L)
        dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj)
        dts, Bs, Cs = dbl.split([self.dt_rank, self.n, self.n], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_w)
        dts = F.softplus(dts + self.dt_b[None, :, :, None])
        A = -torch.exp(self.A_log.float())
        y = self._scan4(xs, dts, A, Bs, Cs)                 # (B,K,Dm,L)
        y = y[:, 0] + y[:, 2].flip(-1) + (y[:, 1] + y[:, 3].flip(-1)).view(B, self.d_inner, W, H).transpose(2, 3).flatten(2)
        y = self.norm(y.transpose(1, 2).reshape(B, H, W, -1))
        return self.out_proj(y * F.silu(z))

    def _scan4(self, xs, dts, A, Bs, Cs):
        # every direction has its own B, C, A, D (as in VMamba): one scan per direction
        B_, K, Dm, L = xs.shape
        A4 = A.view(K, Dm, self.n); D4 = self.Ds.view(K, Dm)
        outs = [selective_scan_ref(xs[:, k], dts[:, k], A4[k], Bs[:, k], Cs[:, k], D4[k]) for k in range(K)]
        return torch.stack(outs, 1)                          # (B,K,Dm,L)


class VSSBlock(nn.Module):
    """Fig. 1(c): LN -> [Linear -> DW-Conv -> act -> SS2D -> LN] gated by [Linear -> act] -> Linear, + residual.
    (SS2D above already contains the in-proj/DW-conv/LN/gate/out-proj, exactly the structure drawn in the figure.)"""
    def __init__(self, d):
        super().__init__(); self.norm, self.ss2d = nn.LayerNorm(d), SS2D(d)

    def forward(self, x):
        return x + self.ss2d(self.norm(x))


class Stage(nn.Module):
    def __init__(self, d, n=2):
        super().__init__(); self.blocks = nn.Sequential(*[VSSBlock(d) for _ in range(n)])
    def forward(self, x): return self.blocks(x)


class PatchMerge(nn.Module):
    def __init__(self, d):
        super().__init__(); self.norm, self.red = nn.LayerNorm(4 * d), nn.Linear(4 * d, 2 * d, bias=False)
    def forward(self, x):                                   # (B,H,W,C) -> (B,H/2,W/2,2C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.red(self.norm(x))


class PatchExpand(nn.Module):
    """# ASSUMPTION: the decoder column in Fig. 1 is labelled 'Patch Merging' but must go up in resolution -> we use VM-UNet's Patch Expanding."""
    def __init__(self, d):
        super().__init__(); self.expand, self.norm = nn.Linear(d, 2 * d, bias=False), nn.LayerNorm(d // 2)
    def forward(self, x):                                   # (B,H,W,C) -> (B,2H,2W,C/2)
        B, H, W, C = x.shape; x = self.expand(x).view(B, H, W, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5).reshape(B, 2 * H, 2 * W, C // 2)
        return self.norm(x)


# ----------------------------------------------------------------------------------------------
# 3) HCEM (paper text Eq. 3-4 + Fig. 1b, labelled "CPEM" in the figure)
# ----------------------------------------------------------------------------------------------
class HCEM(nn.Module):
    def __init__(self, c_sam, c_mamba):
        super().__init__()
        self.fuse = nn.Conv2d(c_sam + c_mamba, c_mamba, 1)        # 1x1 conv after the concat (Eq. 3 + text)
        self.out = nn.Conv2d(2 * c_mamba, c_mamba, 1)             # concat with M_i again, 1x1 conv to unify channels

    def forward(self, s, m):                                      # s: SAM feature (B,768,16,16), m: Mamba skip (B,C,H,W)
        s = F.interpolate(s, size=m.shape[-2:], mode="bilinear", align_corners=False)   # MS(.) = Upsample (Fig. 1b)
        f = self.fuse(torch.cat([s, m], 1))                                              # F_cat -> 1x1 conv
        att = torch.sigmoid(F.adaptive_avg_pool2d(f, 1) + F.adaptive_max_pool2d(f, 1))   # Eq. 4: sigma(AvgPool + MaxPool)
        f = f + f * att                                           # "applied to the fused features via residual connections"
        return self.out(torch.cat([f, m], 1))


# ----------------------------------------------------------------------------------------------
# 4) The whole network (Fig. 1a)
# ----------------------------------------------------------------------------------------------
class SAMBAUNet(nn.Module):
    def __init__(self, n_classes=1, dims=(96, 192, 384, 768), rank=4):
        super().__init__()
        self.sam = CoSESamEncoder(rank=rank)
        self.patch_embed = nn.Sequential(nn.Conv2d(3, dims[0], 4, 4), nn.GroupNorm(1, dims[0]))
        self.enc = nn.ModuleList([Stage(d) for d in dims])
        self.merge = nn.ModuleList([PatchMerge(d) for d in dims[:3]])
        self.hcem = nn.ModuleList([HCEM(self.sam.d, d) for d in dims[:3]])        # (S6,M1) (S9,M2) (S12,M3)
        self.expand = nn.ModuleList([PatchExpand(d) for d in (dims[3], dims[2], dims[1])])
        self.dec_fuse = nn.ModuleList([nn.Linear(d, d) for d in (dims[2], dims[1], dims[0])])
        self.dec = nn.ModuleList([Stage(d) for d in (dims[3], dims[2], dims[1], dims[0])])
        self.final = nn.Sequential(nn.Upsample(scale_factor=4, mode="bilinear"), nn.Conv2d(dims[0], n_classes, 1))
        self.register_buffer("sam_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sam_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img, return_debug=False):               # img (B,3,256,256) in [0,1]
        dbg = {}
        S = self.sam((img - self.sam_mean) / self.sam_std)    # {6,9,12}: (B,768,16,16)
        x = self.patch_embed(img).permute(0, 2, 3, 1)         # (B,64,64,96)
        skips = []
        for i, st in enumerate(self.enc):
            x = st(x)
            if i < 3:
                skips.append(x.permute(0, 3, 1, 2)); x = self.merge[i](x)
        x = self.dec[0](x)                                    # bottleneck (B,8,8,768)
        for j, (sk, si) in enumerate(zip(reversed(skips), (12, 9, 6))):
            m_idx = 2 - j
            g = self.hcem[m_idx](S[si], sk)                   # HCEM(Sa_i, M_i) -> (B,C,H,W)
            x = self.expand[j](x)
            x = self.dec_fuse[j](x + g.permute(0, 2, 3, 1))   # # ASSUMPTION: guided skip is ADDED to the up-sampled decoder feature (VM-UNet style), figure shows a concat symbol
            x = self.dec[j + 1](x)
            dbg[f"dec{j}"] = x.detach()
        logits = self.final(x.permute(0, 3, 1, 2))
        return (logits, {**dbg, "S": S, "M": skips}) if return_debug else logits


def bce_dice_loss(logits, target, l1=1.0, l2=1.0, eps=1.0):
    """Eq. 5. The paper sets lambda 'identical to VM-UNet' without printing them -> 1.0 / 1.0 here.  # ASSUMPTION"""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)
    return l1 * bce + l2 * dice
