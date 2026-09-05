"""Shared constants and caption parsing for the DPA reimplementation.

Real captions in Kelvin878/mvtec look like "An image with 1 crack defect on the capsule" --
parsed into (defect_type, category) so the defect type can be used as a product-agnostic prompt
("a crack defect", no category name) while the category is used to build the held-out zero-shot
transfer test.
"""
import re

IMG_SIZE = 256

# (defect_type, category) combinations excluded ENTIRELY from training -- a genuine zero-shot
# test, since the model never sees this exact defect-on-product pairing during fine-tuning, but
# the real ground-truth image still exists in the dataset for comparison at eval time.
HELD_OUT_COMBOS = {("crack", "capsule"), ("color", "pill"), ("hole", "hazelnut")}


def parse_caption(text: str):
    m = re.match(r"An image with \d+ (.+) defect on the (\w+)", text)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def product_agnostic_prompt(defect_type: str) -> str:
    return f"a {defect_type} defect"
