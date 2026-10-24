"""Context-engineering post, part 2: visual retrieval with DINOv3 + LanceDB as one more tool in the harness.

    uv run python ctx_visual.py build      # embed the 600 test images, write cases/*.md, fill LanceDB
    uv run python ctx_visual.py quality    # does a DINOv3 neighbour share the part class more than chance?

`similar_cases(image_id, k)` returns the k most similar test images (cosine, DINOv3 CLS embedding).
 - payload version:  the k full case records, as text
 - paths version:    the k file paths under cases/ (the article's "return pointers, not payloads")

The agent is scored on READING those records correctly (exact ground truth from the same table).
Retrieval quality is measured separately by `quality`, because a score on "did it find the neighbours
I defined as right" would be circular.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch

import void_data as vd
from ctx_corpus import ARCHIVE

DB_DIR = ARCHIVE.parent / "ctx_lancedb"
MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"


def case_markdown(row: dict, voids: list[dict]) -> str:
    lines = [f"# Case {row['id']}", "", f"part: {row['part_name']}", f"n_voids: {row['n_voids']}",
             f"void_area_total_px: {row['void_area']}", f"part_area_px: {row['part_area']}",
             f"clustered: {row['cluster']}", "", "## Voids"]
    for v in voids:
        lines.append(f"- area_px {v['area_px']}, eq_diam_px {v['eq_diam_px']:.1f}, contrast {v['contrast']:.3f}, "
                     f"centre ({v['cx']:.0f}, {v['cy']:.0f})")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def embed(images: np.ndarray) -> np.ndarray:
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).eval().cuda()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).cuda()
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).cuda()
    out = []
    for i in range(0, len(images), 50):
        x = torch.from_numpy(images[i:i + 50]).float().div(255).cuda()
        x = x.unsqueeze(1).repeat(1, 3, 1, 1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
        x = ((x - mean) / std)  # 224x224 already, the size the model was trained at
        out.append(torch.nn.functional.normalize(model(x), dim=-1).cpu().numpy())
    return np.concatenate(out)


def build() -> None:
    import lancedb
    rows = {r["id"]: r for r in csv.DictReader(open(ARCHIVE / "exports" / "inspection_log.csv"))}
    ids = vd.split_ids("test")
    (ARCHIVE / "cases").mkdir(exist_ok=True)
    for i in ids:
        (ARCHIVE / "cases" / f"{i}.md").write_text(case_markdown(rows[i], vd._voids_by_id().get(i, [])))
    vecs = embed(vd._arrays("test")[0])
    db = lancedb.connect(str(DB_DIR))
    db.create_table("frames", data=[{"id": i, "part_name": rows[i]["part_name"], "vector": v.tolist()}
                                    for i, v in zip(ids, vecs)], mode="overwrite")
    print(f"embedded {len(ids)} test images with {MODEL_NAME}, dim {vecs.shape[1]}; cases/ written; table at {DB_DIR}")


_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        import lancedb
        _TABLE = lancedb.connect(str(DB_DIR)).open_table("frames")
    return _TABLE


def neighbours(image_id: str, k: int) -> list[str]:
    t = table()
    q = t.search().where(f"id = '{image_id}'").limit(1).select(["vector"]).to_list()[0]["vector"]
    hits = t.search(q).metric("cosine").limit(k + 1).select(["id"]).to_list()
    return [h["id"] for h in hits if h["id"] != image_id][:k]  # the query image is its own nearest neighbour


def similar(image_id: str, k: int, *, paths: bool) -> str:
    ids = neighbours(image_id, k)
    if paths:
        return "\n".join(f"cases/{i}.md" for i in ids)
    return "\n\n=====\n\n".join((ARCHIVE / "cases" / f"{i}.md").read_text() for i in ids)


def quality(k: int = 4) -> None:
    rows = {r["id"]: r for r in csv.DictReader(open(ARCHIVE / "exports" / "inspection_log.csv"))}
    ids = vd.split_ids("test")
    rng = np.random.default_rng(0)
    same = chance = 0
    for i in ids:
        nb = neighbours(i, k)
        same += sum(rows[n]["part_name"] == rows[i]["part_name"] for n in nb)
        rnd = rng.choice([x for x in ids if x != i], k, replace=False)
        chance += sum(rows[n]["part_name"] == rows[i]["part_name"] for n in rnd)
    n = len(ids) * k
    print(f"share of the {k} nearest DINOv3 neighbours with the same part class: {same / n:.3f} "
          f"(random neighbours: {chance / n:.3f}; {len(ids)} queries)")


if __name__ == "__main__":
    {"build": build, "quality": quality}[sys.argv[1]]()
