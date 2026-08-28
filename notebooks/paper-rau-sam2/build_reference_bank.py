"""Builds the real reference bank from real CAMUS echocardiography data:
DINOv2 embeddings for retrieval (Fig. 2a) and SAM2-encoder masked-pooled
per-label memory vectors (the reference-mask memory that Eq. 7 attends over).

CAMUS-Lite (YongchengYAO/CAMUS-Lite on HF) is a public subset of the real
CAMUS cardiac ultrasound dataset — this is literally the paper's own OOD
evaluation dataset, not a substitute picked for convenience. Each file is a
NIfTI half-cardiac-cycle sequence (~22 frames); we use frame 0 (roughly
end-diastole) as the reference frame with its ground-truth mask.
Labels: 1 = left ventricular myocardium, 2 = left ventricle, 3 = left atrium.

Run:
    uv run build_reference_bank.py --n-references 30 --out reference_bank.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel, Sam2Model, Sam2Processor

from camus_data import LABELS, download_camus, list_patients, load_frame_pair
from rau_modules import ReferenceBank


@torch.no_grad()
def build(n_references: int, out_path: Path, device: str, cache_dir: Path):
    img_dir, mask_dir = download_camus(cache_dir)
    patients = list_patients(img_dir, mask_dir)
    print(f"Found {len(patients)} CAMUS patient/view volumes")
    ref_patients = patients[:n_references]

    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    sam2 = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device).eval()
    sam2_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

    dino_embeds = []
    memory_vectors = []

    for i, (img_path, mask_path, pid) in enumerate(ref_patients):
        ref_img, ref_mask, _, _ = load_frame_pair(img_path, mask_path, ref_frame=0, target_frame=0)

        dino_inputs = dino_proc(images=ref_img, return_tensors="pt").to(device)
        dino_out = dino.forward(**dino_inputs)

        sam_inputs = sam2_proc(images=ref_img, return_tensors="pt").to(device)
        vout = sam2.get_image_features(sam_inputs.pixel_values, return_dict=True)
        # fpn_hidden_states[-1]: (seq_len, batch, channels) -> (channels, h, w),
        # the same reshape Sam2Model.forward applies before the mask decoder.
        h, w = sam2.backbone_feature_sizes[-1]
        feat = vout.fpn_hidden_states[-1].permute(1, 2, 0).view(1, -1, h, w)[0]  # (256, h, w)
        mask_resized = torch.nn.functional.interpolate(
            torch.from_numpy(ref_mask).float()[None, None], size=(h, w), mode="nearest"
        )[0, 0].to(device)

        entry = {}
        for lbl in LABELS:
            m = mask_resized == lbl
            if m.sum() > 0:
                entry[lbl] = feat[:, m].mean(dim=1).cpu()

        if entry:
            memory_vectors.append(entry)
            dino_embeds.append(dino_out.pooler_output[0].cpu())

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(ref_patients)}")

    bank = ReferenceBank(dino_embeds=torch.stack(dino_embeds), memory_vectors=memory_vectors)
    bank.save(out_path)
    print(f"Saved reference bank ({len(memory_vectors)} entries) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-references", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("reference_bank.pt"))
    ap.add_argument("--cache-dir", type=Path, default=Path("camus_cache"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    build(args.n_references, args.out, args.device, args.cache_dir)
