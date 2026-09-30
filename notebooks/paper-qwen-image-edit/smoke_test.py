"""Smoke test: confirm QwenImage21Pipeline loads and can do one small edit."""
import time
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

t0 = time.time()
pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
print(f"Model load time: {time.time()-t0:.1f}s")
print(f"GPU mem allocated after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

img = Image.open(
    "/home/hasan-spark/.cache/kagglehub/datasets/samarthgoel2604/"
    "gdxray-without-synthetic-for-defectd/versions/1/Castings kaggle/Castings (1)/Castings/C0001/C0001_0001.png"
).convert("RGB")
print("source size:", img.size)

t1 = time.time()
out = pipe(
    prompt="Add a small dark circular void defect near the center of the image, matching the grayscale X-ray noise texture of the surrounding metal.",
    image=img,
    num_inference_steps=20,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
print(f"Inference time: {time.time()-t1:.1f}s")
print(f"GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
out.save("/home/hasan-spark/workspace/projects/git_data/quarto_blog_hasan/notebooks/paper-qwen-image-edit/out/smoke_test.png")
print("Saved smoke_test.png, size:", out.size)
