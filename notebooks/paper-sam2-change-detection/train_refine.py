"""Stage 1 (continued): a lightweight CNN trained with an uncertainty-aware loss to refine the
real SAM2-derived pseudo-labels into full, dense change masks. Stage 2: a teacher-student
self-training loop that periodically refreshes the pseudo-labels using the (improving) model
itself, closing the loop between "labels train the model" and "the model improves the labels."

Real ground truth is used ONLY to measure IoU for reporting -- never as a training signal, to
keep this an honest point-supervised (not fully-supervised) reimplementation.
"""
import glob
import json
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

IMG_SIZE = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
IMG_OUT = "../../posts/series/papers/images"
OUT = "out"
os.makedirs(OUT, exist_ok=True)


def load_split(split):
    files = sorted(glob.glob(f"data/pseudolabels/{split}_*.npz"))
    data = []
    for f in files:
        d = np.load(f)
        data.append({
            "img_a": d["img_a"], "img_b": d["img_b"],
            "pseudo": d["pseudo"], "conf": d["conf"], "gt": d["gt"],
        })
    return data


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())
        self.e1 = block(6, 32); self.e2 = block(32, 64); self.e3 = block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.b = block(128, 128)
        self.d3 = block(128 + 128, 64); self.d2 = block(64 + 64, 32); self.d1 = block(32 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.out(d1)


def to_input(img_a, img_b):
    a = torch.from_numpy(img_a).permute(2, 0, 1).float() / 255.0
    b = torch.from_numpy(img_b).permute(2, 0, 1).float() / 255.0
    return torch.cat([a, b], 0).unsqueeze(0).to(device)


def weighted_bce_dice(pred_logits, target, weight):
    """Plain per-pixel BCE (even weighted by confidence) is dominated by the sheer pixel count
    of the "no change" background -- change regions are a tiny fraction of any 256x256 image, so
    a model minimizing per-pixel loss can (and, empirically here, does) collapse to predicting
    "nothing changed" everywhere. Two standard fixes, both applied: (1) balance the BCE weight by
    each image's actual positive/negative pixel ratio, not just pseudo-label confidence; (2) add
    a Dice loss term, which scores shape overlap rather than raw pixel counts and so isn't fooled
    by class imbalance the way per-pixel BCE is."""
    n_pos = target.sum().clamp(min=1.0)
    n_neg = (1 - target).sum().clamp(min=1.0)
    pos_weight = (n_neg / n_pos).clamp(max=50.0)
    balance = target * pos_weight + (1 - target)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    bce = (bce * weight * balance).mean()

    probs = torch.sigmoid(pred_logits)
    inter = (probs * target).sum()
    dice = 1 - (2 * inter + 1) / (probs.sum() + target.sum() + 1)
    return bce + dice


def iou_np(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(inter / union) if union > 0 else 1.0


@torch.no_grad()
def eval_iou(model, data, thresh=0.5):
    """Reports both the overall mean IoU AND the mean IoU restricted to examples that actually
    have real change -- roughly half of LEVIR-CD crops have none, and a model that collapses to
    predicting "no change" everywhere scores a deceptively high *overall* IoU (1.0 on every
    empty-ground-truth example) while catching zero real changes. The restricted number is the
    one that actually reflects whether the model learned anything."""
    model.eval()
    ious, ious_nonempty = [], []
    for d in data:
        x = to_input(d["img_a"], d["img_b"])
        pred = torch.sigmoid(model(x))[0, 0].cpu().numpy() > thresh
        v = iou_np(pred, d["gt"])
        ious.append(v)
        if d["gt"].sum() > 0:
            ious_nonempty.append(v)
    model.train()
    return float(np.mean(ious)), float(np.mean(ious_nonempty)) if ious_nonempty else None


@torch.no_grad()
def ema_update(teacher, student, decay=0.99):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(decay).add_(sp, alpha=1 - decay)


if __name__ == "__main__":
    train_data = load_split("train")
    val_data = load_split("val")
    print(f"train pairs: {len(train_data)}  val pairs: {len(val_data)}")

    # both the overall and genuine-change-only IoU -- the overall number is inflated by examples
    # with no real change at all (pseudo-label and GT both trivially empty, IoU=1.0 by convention)
    raw_ious_all = [iou_np(d["pseudo"], d["gt"]) for d in train_data]
    raw_ious_nonempty = [v for v, d in zip(raw_ious_all, train_data) if d["gt"].sum() > 0]
    raw_pseudo_iou = float(np.mean(raw_ious_all))
    raw_pseudo_iou_nonempty = float(np.mean(raw_ious_nonempty)) if raw_ious_nonempty else None
    print(f"Raw pseudo-label IoU vs real GT (train, no training at all): {raw_pseudo_iou:.3f}  "
          f"(genuine-change only: {raw_pseudo_iou_nonempty:.3f}, n={len(raw_ious_nonempty)})")

    student = TinyUNet().to(device)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)

    STAGE1_STEPS = int(os.environ.get("STAGE1_STEPS", 1500))
    losses = []
    val_ious_stage1 = []
    rng = np.random.default_rng(0)

    print("--- Stage 1: training on real-SAM2-derived pseudo-labels ---")
    order = rng.permutation(len(train_data))
    step = 0
    while step < STAGE1_STEPS:
        for k in order:
            if step >= STAGE1_STEPS:
                break
            d = train_data[int(k)]
            x = to_input(d["img_a"], d["img_b"])
            target = torch.from_numpy(d["pseudo"]).float().unsqueeze(0).unsqueeze(0).to(device)
            conf = torch.from_numpy(d["conf"]).float().unsqueeze(0).unsqueeze(0).to(device)
            weight = 0.4 + 0.6 * conf  # uncertainty-aware: low-confidence pixels count less
            pred = student(x)
            loss = weighted_bce_dice(pred, target, weight)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if step % 200 == 0:
                val_iou, val_iou_nonempty = eval_iou(student, val_data)
                val_ious_stage1.append((step, val_iou, val_iou_nonempty))
                print(f"step {step:4d}  loss {loss.item():.4f}  val IoU {val_iou:.3f}  "
                      f"(genuine-change only: {val_iou_nonempty})")
            step += 1
        order = rng.permutation(len(train_data))

    stage1_final_iou, stage1_final_iou_nonempty = eval_iou(student, val_data)
    print(f"Stage 1 final val IoU: {stage1_final_iou:.3f}  (genuine-change only: {stage1_final_iou_nonempty:.3f})")
    torch.save(student.state_dict(), f"{OUT}/student_stage1.pt")

    print("--- Stage 2: teacher-student self-training ---")
    teacher = copy.deepcopy(student)
    ROUNDS = int(os.environ.get("ROUNDS", 4))
    STEPS_PER_ROUND = int(os.environ.get("STEPS_PER_ROUND", 400))
    stage2_ious = [(0, stage1_final_iou, stage1_final_iou_nonempty)]

    for r in range(ROUNDS):
        # teacher relabels the training set -- pseudo-labels refreshed from the model, not SAM2
        teacher.eval()
        refreshed = []
        with torch.no_grad():
            for d in train_data:
                x = to_input(d["img_a"], d["img_b"])
                prob = torch.sigmoid(teacher(x))[0, 0].cpu().numpy()
                refreshed.append({**d, "pseudo": prob > 0.5, "conf": np.clip(np.abs(prob - 0.5) * 2, 0.3, 1.0)})

        order = rng.permutation(len(refreshed))
        for step_in_round, k in enumerate(order[:STEPS_PER_ROUND]):
            d = refreshed[int(k)]
            x = to_input(d["img_a"], d["img_b"])
            target = torch.from_numpy(d["pseudo"]).float().unsqueeze(0).unsqueeze(0).to(device)
            conf = torch.from_numpy(d["conf"]).float().unsqueeze(0).unsqueeze(0).to(device)
            weight = 0.4 + 0.6 * conf
            pred = student(x)
            loss = weighted_bce_dice(pred, target, weight)
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(teacher, student, decay=0.98)

        round_iou, round_iou_nonempty = eval_iou(student, val_data)
        stage2_ious.append((r + 1, round_iou, round_iou_nonempty))
        print(f"round {r+1}/{ROUNDS}  val IoU {round_iou:.3f}  (genuine-change only: {round_iou_nonempty:.3f})")

    torch.save(student.state_dict(), f"{OUT}/student_final.pt")
    print("Saved student_final.pt")

    with open(f"{OUT}/refine_results.json", "w") as f:
        json.dump({
            "raw_pseudo_iou": raw_pseudo_iou,
            "raw_pseudo_iou_nonempty": raw_pseudo_iou_nonempty,
            "stage1_final_val_iou": stage1_final_iou,
            "stage1_final_val_iou_nonempty": stage1_final_iou_nonempty,
            "stage1_curve": val_ious_stage1,
            "stage2_curve": stage2_ious,
        }, f, indent=2)

    plt.figure(figsize=(5.5, 3.8))
    plt.plot(losses)
    plt.xlabel("step"); plt.ylabel("weighted BCE + Dice loss"); plt.title("Stage 1: refinement CNN training")
    plt.tight_layout()
    plt.savefig(f"{IMG_OUT}/sam2cd-stage1-loss-curve.png", dpi=130)

    plt.figure(figsize=(6.5, 4.2))
    xs1 = [s for s, _, _ in val_ious_stage1]
    ys1_all = [v for _, v, _ in val_ious_stage1]
    ys1_ne = [v for _, _, v in val_ious_stage1]
    plt.plot(xs1, ys1_all, marker="o", label="Stage 1 (all examples)")
    plt.plot(xs1, ys1_ne, marker="s", label="Stage 1 (genuine-change examples only)")
    plt.axhline(raw_pseudo_iou, linestyle="--", color="gray", label="Raw SAM2 pseudo-label (all)")
    plt.axhline(raw_pseudo_iou_nonempty, linestyle=":", color="gray", label="Raw SAM2 pseudo-label (genuine-change only)")
    plt.xlabel("training step"); plt.ylabel("IoU vs real ground truth")
    plt.legend(fontsize=8); plt.title("Change-detection quality vs real GT")
    plt.tight_layout()
    plt.savefig(f"{IMG_OUT}/sam2cd-stage1-iou-curve.png", dpi=130)

    plt.figure(figsize=(6.5, 4.2))
    xs2 = [r for r, _, _ in stage2_ious]
    ys2_all = [v for _, v, _ in stage2_ious]
    ys2_ne = [v for _, _, v in stage2_ious]
    plt.plot(xs2, ys2_all, marker="o", color="darkorange", label="all examples")
    plt.plot(xs2, ys2_ne, marker="s", color="firebrick", label="genuine-change examples only")
    plt.xlabel("self-training round"); plt.ylabel("IoU vs real ground truth")
    plt.legend(); plt.title("Stage 2: teacher-student self-training")
    plt.tight_layout()
    plt.savefig(f"{IMG_OUT}/sam2cd-stage2-curve.png", dpi=130)

    print("Done.")
