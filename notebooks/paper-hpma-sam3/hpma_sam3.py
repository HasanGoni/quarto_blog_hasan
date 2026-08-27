"""Wires HPMA's coupling adapters into the REAL facebook/sam3 forward pass.

SAM3's own code isn't forked — we monkey-patch two bound methods on an
already-loaded `Sam3Model` instance:

1. `model.get_text_features` — wrapped so the global adapter's cross-attention
   modifies the real text-token embeddings (`.pooler_output`, shape
   (B, 32, 256)) before they reach the DETR encoder/decoder.
2. `model.detr_decoder.forward` — swapped for a subclass whose forward body
   is an exact copy of the real `Sam3DetrDecoder.forward` (transformers
   5.16.1) with exactly one line changed: the structural adapter's delta is
   added to `query_embeds`.

Both patches preserve autograd — gradients flow from the loss back through
the adapters, through the (frozen) SAM3 weights untouched, exactly as the
paper intends: SAM3 stays frozen (except the top few blocks per the paper;
we keep 100% of SAM3 frozen for this reduced demo — see train_hpma.py), only
the adapters and prototypes-derived deltas are learned.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn
from transformers import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import (
    Sam3DetrDecoder,
    Sam3DETRDecoderOutput,
    create_bidirectional_mask,
)
from transformers.utils.generic import can_return_tuple

from hpma_modules import GlobalCouplingAdapter, StructuralCouplingAdapter, PrototypeBank


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def _hpma_decoder_forward(self, vision_features, text_features, vision_pos_encoding,
                           text_mask=None, spatial_shapes=None, **kwargs):
    """Exact copy of Sam3DetrDecoder.forward (transformers 5.16.1) with the
    structural-adapter delta added to query_embeds. See class docstring."""
    import torch.nn.functional as F

    batch_size = vision_features.shape[0]

    query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
    if getattr(self, "hpma_structural_prototypes", None) is not None:
        query_embeds = self.hpma_structural_adapter(query_embeds, self.hpma_structural_prototypes)

    reference_boxes = self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1)
    reference_boxes = reference_boxes.sigmoid()
    presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)

    hidden_states = torch.cat([presence_token, query_embeds], dim=1)

    text_cross_attn_mask = None
    if text_mask is not None:
        text_cross_attn_mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=hidden_states,
            attention_mask=text_mask, encoder_hidden_states=text_features,
        )

    intermediate_outputs = []
    intermediate_boxes = [reference_boxes]
    intermediate_presence_logits = []

    for layer in self.layers:
        reference_points_input = reference_boxes.unsqueeze(2)
        query_sine_embed = self.position_encoding.encode_boxes(reference_points_input[:, :, 0, :])
        query_pos = self.ref_point_head(query_sine_embed)

        vision_cross_attn_mask = None
        if spatial_shapes is not None and spatial_shapes.shape[0] == 1:
            spatial_shape = (spatial_shapes[0, 0], spatial_shapes[0, 1])
            rpb_matrix = self._get_rpb_matrix(reference_boxes, spatial_shape)
            vision_cross_attn_mask = F.pad(rpb_matrix, (0, 0, 1, 0), mode="constant", value=0)

        hidden_states = layer(
            hidden_states, query_pos=query_pos, text_features=text_features,
            vision_features=vision_features, vision_pos_encoding=vision_pos_encoding,
            text_cross_attn_mask=text_cross_attn_mask, vision_cross_attn_mask=vision_cross_attn_mask,
            **kwargs,
        )

        query_hidden_states = hidden_states[:, 1:]
        reference_boxes_before_sigmoid = _inverse_sigmoid(reference_boxes)
        delta_boxes = self.box_head(self.output_layer_norm(query_hidden_states))
        new_reference_boxes = (delta_boxes + reference_boxes_before_sigmoid).sigmoid()
        reference_boxes = new_reference_boxes.detach()

        intermediate_outputs.append(self.output_layer_norm(query_hidden_states))
        intermediate_boxes.append(new_reference_boxes)

        presence_hidden = hidden_states[:, :1]
        presence_logits = self.presence_head(self.presence_layer_norm(presence_hidden)).squeeze(-1)
        presence_logits = presence_logits.clamp(
            min=-self.clamp_presence_logit_max_val, max=self.clamp_presence_logit_max_val
        )
        intermediate_presence_logits.append(presence_logits)

    intermediate_outputs = torch.stack(intermediate_outputs)
    intermediate_boxes = torch.stack(intermediate_boxes[:-1])
    intermediate_presence_logits = torch.stack(intermediate_presence_logits)

    return Sam3DETRDecoderOutput(
        intermediate_hidden_states=intermediate_outputs,
        reference_boxes=intermediate_boxes,
        presence_logits=intermediate_presence_logits,
    )


class HPMASam3(nn.Module):
    def __init__(self, model_name: str = "facebook/sam3", dim: int = 256):
        super().__init__()
        self.sam3 = Sam3Model.from_pretrained(model_name)
        self.processor = Sam3Processor.from_pretrained(model_name)
        for p in self.sam3.parameters():
            p.requires_grad_(False)
        self.sam3.eval()

        self.global_adapter = GlobalCouplingAdapter(dim=dim)
        self.structural_adapter = StructuralCouplingAdapter(dim=dim)

        self._active_global_prototypes = None
        self._orig_get_text_features = self.sam3.get_text_features
        self.sam3.get_text_features = self._patched_get_text_features

        self.sam3.detr_decoder.forward = types.MethodType(_hpma_decoder_forward, self.sam3.detr_decoder)
        self.sam3.detr_decoder.hpma_structural_adapter = self.structural_adapter
        self.sam3.detr_decoder.hpma_structural_prototypes = None

    def _patched_get_text_features(self, input_ids, attention_mask=None, **kwargs):
        text_outputs = self._orig_get_text_features(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if self._active_global_prototypes is not None:
            text_outputs.pooler_output = self.global_adapter(text_outputs.pooler_output, self._active_global_prototypes)
        return text_outputs

    def set_active_category(self, prototypes: PrototypeBank, category: int):
        """Point both trainable adapters at this category's frozen prototypes
        for the next forward pass."""
        self._active_global_prototypes = prototypes.vectors["global"][category]
        self.sam3.detr_decoder.hpma_structural_prototypes = prototypes.vectors["structural"][category]

    def forward(self, pixel_values, input_ids, attention_mask=None, vision_embeds=None):
        return self.sam3(
            pixel_values=pixel_values if vision_embeds is None else None,
            vision_embeds=vision_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def adapter_parameters(self):
        return list(self.global_adapter.parameters()) + list(self.structural_adapter.parameters())
