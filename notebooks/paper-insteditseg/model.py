"""Model components: SD1.5 U-Net adapted for instruction-driven image editing, plus the
DINO Feature Guidance Block from InstEditSeg (arXiv:2609.02004).

Two deliberate, honestly-noted adaptations from the paper:
1. DINOv3 is used in the paper; it's a gated checkpoint under Meta's custom research license.
   DINOv2-small (Apache-2.0, ungated) is substituted here -- same self-supervised-features idea,
   actually reproducible by anyone reading this post.
2. Image conditioning uses the InstructPix2Pix technique (expand the U-Net's first conv to also
   take the original image's latent, zero-init the new weight half) rather than the paper's exact
   internal conditioning path, which isn't public -- this is a standard, well-documented way to
   condition a pretrained SD U-Net on a full reference image for instruction-driven editing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel

SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DINO_ID = "facebook/dinov2-small"
DINO_DIM = 384
UNET_CHANNELS = [320, 640, 1280, 1280]  # SD1.5 down_block output channels, 4 resolution levels


def expand_conv_in_for_image_conditioning(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    """InstructPix2Pix-style: double conv_in's input channels (4 -> 8) so the U-Net can also
    see the original image's clean latent concatenated with the noisy target latent. The new
    half of the weight is zero-initialized, so at step 0 the model behaves exactly like the
    pretrained checkpoint (image-conditioning contributes nothing until LoRA fine-tuning teaches
    it to)."""
    old_conv = unet.conv_in
    new_conv = nn.Conv2d(old_conv.in_channels * 2, old_conv.out_channels,
                          kernel_size=old_conv.kernel_size, padding=old_conv.padding)
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :old_conv.in_channels] = old_conv.weight
        new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv
    unet.config.in_channels = old_conv.in_channels * 2
    return unet


class DinoFeatureGuidance(nn.Module):
    """The paper's "DINO Feature Guidance Block": a multi-scale feature pyramid built from a
    frozen DINO backbone, injected into the diffusion U-Net via zero-initialized convolutions
    so pretrained weights are never disrupted at initialization -- injected through diffusers'
    native `down_intrablock_additional_residuals` (the same documented mechanism T2I-Adapter
    uses), verified against the installed diffusers UNet2DConditionModel.forward signature."""

    def __init__(self, dino_dim: int = DINO_DIM, unet_channels=UNET_CHANNELS):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(dino_dim, ch, kernel_size=1) for ch in unet_channels
        ])
        for proj in self.projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, dino_grid: torch.Tensor, latent_size: int):
        """dino_grid: (B, dino_dim, H, W) spatial DINO features.
        Returns a list of 4 residual tensors at latent_size, latent_size/2, /4, /8 -- matching
        SD1.5's 4 down-block resolutions."""
        residuals = []
        size = latent_size
        for proj in self.projections:
            feat = F.interpolate(dino_grid, size=(size, size), mode="bilinear", align_corners=False)
            residuals.append(proj(feat))
            size = size // 2
        return residuals


def dino_spatial_features(dino_model, images_0_1: torch.Tensor) -> torch.Tensor:
    """images_0_1: (B, 3, H, W) in [0, 1]. Returns (B, dino_dim, h_patch, w_patch)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=images_0_1.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images_0_1.device).view(1, 3, 1, 1)
    x = (images_0_1 - mean) / std
    with torch.no_grad():
        out = dino_model(pixel_values=x).last_hidden_state  # (B, 1+N, D)
    tokens = out[:, 1:, :]  # drop CLS
    n = tokens.shape[1]
    h = w = int(n ** 0.5)
    grid = tokens[:, : h * w, :].transpose(1, 2).reshape(tokens.shape[0], -1, h, w)
    return grid.float()


def load_models(device: str):
    vae = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet")
    unet = expand_conv_in_for_image_conditioning(unet).to(device)
    text_encoder = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
    dino_model = AutoModel.from_pretrained(DINO_ID).to(device).eval()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    dino_model.requires_grad_(False)

    return vae, unet, text_encoder, tokenizer, noise_scheduler, dino_model
