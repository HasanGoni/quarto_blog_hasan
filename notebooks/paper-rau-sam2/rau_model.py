"""Wires a real Qwen2.5-VL-3B-Instruct + real SAM2 together per RAU's
Section 3.3 fusion design.

Substitution from the paper, stated up front: the paper uses Qwen2.5-VL-7B;
we use its 3B sibling (same architecture family, same processor/chat-template
API) so the whole pipeline downloads and trains in minutes instead of hours
on this machine, while remaining a real, unmodified Qwen2.5-VL checkpoint.

Mechanism (real, not simplified):
1. A new special token `<SEG>` is added to the tokenizer and the model's
   embedding matrix is resized — the same trick LISA-style segmentation VLMs
   use. Only this new token's embedding row is trainable; every other VLM
   weight stays frozen (matches the paper's Section 3.3 "Training Setting").
2. Given [reference_image, target_image] + a text prompt asking the VLM to
   locate a named region, teacher-forced with a completion ending in
   `<SEG>`, we take the real last-hidden-state at the `<SEG>` position —
   h_seg (Eq. 6's h_i^<Seg>).
3. h_seg is projected (SegQueryProjection) and fused via dot-product
   attention (rau_modules.memory_attention) against the retrieved
   reference's per-label memory vectors (Eq. 7) -> z.
4. z is passed into the real `Sam2Model.forward(..., target_embedding=z)` —
   SAM2's own public, documented semantic-prompting interface (the same one
   PerSAM uses) — no internal SAM2 hacking required this time.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Sam2Model, Sam2Processor

from rau_modules import ReferenceBank, SegQueryProjection, memory_attention

SEG_TOKEN = "<SEG>"


def _patched_two_way_transformer_forward(self, point_embeddings, image_embeddings,
                                          image_positional_embeddings, attention_similarity,
                                          target_embedding=None, **kwargs):
    """Real transformers Sam2TwoWayTransformer.forward with one change:
    `queries += target_embedding` (in-place) -> `queries = queries + target_embedding`
    (out-of-place). The library's own version breaks autograd whenever
    target_embedding requires grad, e.g. when it's produced by a trainable
    upstream module (our fused VLM query) instead of a frozen inference-time
    constant — it was never exercised with a trainable target_embedding
    before, since target_embedding's only prior use (PerSAM) is inference-only."""
    image_embeddings = image_embeddings.flatten(2).transpose(1, 2).unsqueeze(1)
    image_positional_embeddings = image_positional_embeddings.flatten(2).transpose(1, 2).unsqueeze(1)

    queries = point_embeddings
    keys = image_embeddings

    for layer in self.layers:
        if target_embedding is not None:
            queries = queries + target_embedding  # was `queries += target_embedding`

        queries, keys, _ = layer(
            queries=queries, keys=keys,
            query_point_embedding=point_embeddings, key_point_embedding=image_positional_embeddings,
            attention_similarity=attention_similarity, **kwargs,
        )

    query = queries + point_embeddings
    key = keys + image_positional_embeddings
    attn_out, _ = self.final_attn_token_to_image(query=query, key=key, value=keys)
    queries = queries + attn_out
    queries = self.layer_norm_final_attn(queries)
    return queries, keys


class RAU(nn.Module):
    def __init__(self, vlm_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", sam2_name: str = "facebook/sam2.1-hiera-large"):
        super().__init__()
        # No device_map="auto": accelerate's CPU-offload hooks cache the
        # pre-resize embedding shape, and resize_token_embeddings() below
        # then trips a shape-mismatch crash the first time an offloaded
        # embedding row gets swapped back onto the GPU (hit in GRPO's
        # .generate(), which reloads the model fresh and can get offloaded
        # differently than SFT's run did). The 3B model comfortably fits in
        # a single device's memory, so just place it there directly.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(vlm_name, dtype=torch.bfloat16).to(device)
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_name)

        added = self.vlm_processor.tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})
        if added:
            self.vlm.resize_token_embeddings(len(self.vlm_processor.tokenizer))
        self.seg_token_id = self.vlm_processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)

        for p in self.vlm.parameters():
            p.requires_grad_(False)
        # Only the new <SEG> embedding row is trainable (paper: "jointly train
        # the embeddings of the <Seg> tokens ... freezing the other VLM weights").
        embed = self.vlm.get_input_embeddings()
        embed.weight.requires_grad_(False)
        self._seg_embed_row = nn.Parameter(embed.weight.data[self.seg_token_id].clone())
        self._orig_embed_forward = embed.forward
        embed.forward = self._patched_embed_forward(embed)

        # Qwen2.5-VL ties input/output embeddings (config.tie_word_embeddings),
        # so lm_head.weight IS embed.weight (same Parameter, frozen). Without
        # also patching lm_head's forward, gradients from a next-token loss
        # on <SEG> would flow to the frozen shared weight, not to our
        # trainable _seg_embed_row, and the model would never learn to
        # actually choose to emit <SEG>.
        self._orig_lm_head_forward = self.vlm.lm_head.forward
        self.vlm.lm_head.forward = self._patched_lm_head_forward(self.vlm.lm_head)

        self.sam2 = Sam2Model.from_pretrained(sam2_name).to(self.vlm.device)
        self.sam2_processor = Sam2Processor.from_pretrained(sam2_name)
        for p in self.sam2.parameters():
            p.requires_grad_(False)
        for p in self.sam2.mask_decoder.parameters():
            p.requires_grad_(True)  # paper: SAM2 decoder is trained too
        self.sam2.mask_decoder.transformer.forward = types.MethodType(
            _patched_two_way_transformer_forward, self.sam2.mask_decoder.transformer
        )

        vlm_dim = self.vlm.config.text_config.hidden_size
        self.projection = SegQueryProjection(vlm_dim=vlm_dim, sam2_dim=256).to(self.vlm.device, dtype=torch.float32)

    def _patched_embed_forward(self, embed_module):
        orig = self._orig_embed_forward
        def fwd(input_ids):
            out = orig(input_ids)
            mask = (input_ids == self.seg_token_id).unsqueeze(-1)
            seg_row = self._seg_embed_row.to(out.dtype).view(*([1] * (out.dim() - 1)), -1).expand_as(out)
            return torch.where(mask, seg_row, out)
        return fwd

    def _patched_lm_head_forward(self, lm_head_module):
        orig = self._orig_lm_head_forward
        def fwd(hidden_states):
            logits = orig(hidden_states)
            seg_logit = (hidden_states * self._seg_embed_row.to(hidden_states.dtype)).sum(-1, keepdim=True)
            vocab_size = logits.shape[-1]
            col_mask = torch.arange(vocab_size, device=logits.device) == self.seg_token_id
            col_mask = col_mask.view(*([1] * (logits.dim() - 1)), vocab_size)
            return torch.where(col_mask, seg_logit.expand_as(logits), logits)
        return fwd

    def trainable_parameters(self):
        return [self._seg_embed_row] + list(self.projection.parameters()) + list(self.sam2.mask_decoder.parameters())

    def load_sft_checkpoint(self, path):
        ckpt = torch.load(path, weights_only=True, map_location=self.vlm.device)
        with torch.no_grad():
            self._seg_embed_row.copy_(ckpt["seg_embed_row"].to(self._seg_embed_row.device))
        self.projection.load_state_dict(ckpt["projection"])
        self.sam2.mask_decoder.load_state_dict(ckpt["mask_decoder"])

    def get_seg_hidden_state(self, reference_image, target_image, prompt_text: str, seg_completion: str = " <SEG>",
                              return_lm_loss: bool = False):
        """Real forward pass through Qwen2.5-VL; returns the last-hidden-state
        vector at the (teacher-forced) <SEG> token position.

        If return_lm_loss, also returns the real next-token cross-entropy
        loss for predicting <SEG> at that position — without this, nothing
        trains the model to *choose* to emit <SEG> during free generation
        (only the segmentation loss shapes the embedding row's content, not
        the output distribution over it); needed so a later GRPO stage has a
        policy that spontaneously produces <SEG> often enough to sample from."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": reference_image},
                {"type": "image", "image": target_image},
                {"type": "text", "text": prompt_text},
            ],
        }]
        chat_text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = chat_text + seg_completion

        inputs = self.vlm_processor(
            text=[full_text], images=[reference_image, target_image], return_tensors="pt"
        ).to(self.vlm.device)

        outputs = self.vlm(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden)

        seg_positions = (inputs["input_ids"][0] == self.seg_token_id).nonzero(as_tuple=True)[0]
        seg_pos = seg_positions[-1]
        h_seg = last_hidden[seg_pos]
        if not return_lm_loss:
            return h_seg.float()

        lm_logits = outputs.logits[0, seg_pos - 1].float()  # predicts token AT seg_pos
        lm_loss = torch.nn.functional.cross_entropy(lm_logits.unsqueeze(0), seg_pos.new_tensor([self.seg_token_id]))
        return h_seg.float(), lm_loss

    def segment(self, target_image, memory_vectors: dict[int, torch.Tensor], h_seg: torch.Tensor):
        q = self.projection(h_seg.unsqueeze(0)).squeeze(0)
        z = memory_attention(q, memory_vectors)  # (256,)

        sam_inputs = self.sam2_processor(images=target_image, return_tensors="pt").to(self.vlm.device)
        target_embedding = z.to(self.sam2.dtype).view(1, 1, 1, -1)
        outputs = self.sam2(**sam_inputs, target_embedding=target_embedding, multimask_output=False)
        return outputs, sam_inputs
