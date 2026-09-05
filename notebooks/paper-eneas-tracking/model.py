"""Real SAM2 point-prompted segmentation, real DINOv2 identity embedding, and the real
semantic-verification fallback (Qwen2-VL asked directly whether two crops show the same object)
-- the three pieces ENEAS's semantic verification layer combines.
"""
import torch
import torch.nn.functional as F
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

_sam2_model = None
_sam2_processor = None
_dino_model = None
_dino_processor = None
_qwen_model = None
_qwen_processor = None


def get_sam2():
    global _sam2_model, _sam2_processor
    if _sam2_model is None:
        from transformers import Sam2Model, Sam2Processor
        _sam2_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
        _sam2_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").to(device).eval()
    return _sam2_model, _sam2_processor


@torch.no_grad()
def sam2_segment(image: Image.Image, point) -> tuple:
    model, processor = get_sam2()
    inputs = processor(images=image, input_points=[[[list(point)]]], input_labels=[[[1]]],
                        return_tensors="pt").to(device)
    out = model(**inputs)
    best = out.iou_scores[0, 0].argmax().item()
    mask = processor.post_process_masks(out.pred_masks, inputs["original_sizes"])[0][0, best].cpu().numpy()
    return mask > 0.0, float(out.iou_scores[0, 0, best].item())


def get_dino():
    global _dino_model, _dino_processor
    if _dino_model is None:
        from transformers import AutoImageProcessor, AutoModel
        _dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        _dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    return _dino_model, _dino_processor


@torch.no_grad()
def dino_embed(crop: Image.Image) -> torch.Tensor:
    model, processor = get_dino()
    inputs = processor(images=crop, return_tensors="pt").to(device)
    cls = model(**inputs).last_hidden_state[0, 0]
    return F.normalize(cls, dim=0)


def get_qwen():
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        _qwen_processor = AutoProcessor.from_pretrained(model_id)
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(device).eval()
    return _qwen_model, _qwen_processor


@torch.no_grad()
def vlm_same_object(reference_crop: Image.Image, candidate_crop: Image.Image) -> bool:
    """The real conditional VLM refinement step: only invoked when embedding similarity is
    ambiguous, exactly as ENEAS describes -- fast embedding matching first, selective VLM
    reasoning only when needed, not on every frame."""
    from qwen_vl_utils import process_vision_info
    model, processor = get_qwen()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "[Reference crop]"}, {"type": "image", "image": reference_crop},
        {"type": "text", "text": "[Candidate crop]"}, {"type": "image", "image": candidate_crop},
        {"type": "text", "text": "Do the reference crop and the candidate crop show the same "
                                  "individual physical object (not just the same category)? "
                                  "Answer with exactly one word: yes or no."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)
    gen = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    answer = processor.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return "yes" in answer.strip().lower()
