# Semantic Foveated Diffusion

A two-stage, text-guided mixed-resolution **image generation** pipeline built on [Foveated Diffusion](https://arxiv.org/abs/2603.23491) ([`bchao1/foveated_diffusion`](https://github.com/bchao1/foveated_diffusion)).

Whereas standard Foveated Diffusion requires a **pre-specified** spatial resolution mask (such as a fixed circle or eye-tracker gaze point), **Semantic Foveated Diffusion** automatically infers *where* to allocate high-resolution tokens based on a user-specified text description of the important image regions.

---

## Method & Pipeline Overview

### Inputs
1. **Main Image Prompt:** Describes the full scene to generate (e.g., *"A medieval knight standing in a mossy forest holding a glowing runesword"*).
2. **Focus Prompt:** Specifies which semantic part(s) of the scene should receive high-resolution compute (e.g., *"the glowing runesword"*).

### Stage 1: Coarse Low-Resolution Generation & Semantic Mask Inference
1. **Low-Resolution Draft Generation:**
   Generate a coarse low-resolution image (at half spatial resolution, which uses 4x fewer tokens than full high-resolution generation) conditioned on the main prompt.
2. **Vision-Language Feature Matching:**
   Pass the coarse image and the focus prompt through a lightweight pretrained vision-language localizer (such as CLIPSeg or dense CLIP/SigLIP patch similarity) to produce a 2D spatial relevance heatmap over the image.
3. **Block-Aligned Foveation Mask:**
   Convert the heatmap into a binary foveation mask on the token-block grid (e.g., by selecting the top-scoring blocks or applying a similarity threshold, with optional padding so object edges are not clipped). This mask marks which regions should use high-resolution (HR) tokens and which peripheral regions can stay as low-resolution (LR) tokens.

### Stage 2: Selective High-Resolution Re-Encoding & Mixed-Resolution Denoising
1. **Selective Re-Encoding into Mixed-Resolution Tokens:**
   - Keep the peripheral regions outside the mask as compact low-resolution latent tokens (1 token per 2x2 block).
   - Upsample the coarse image to full resolution, encode it with the VAE encoder, and retain all 4 high-resolution tokens per 2x2 block inside the inferred mask.
   - Merge the high-resolution focus tokens and low-resolution peripheral tokens into a single compact, variable-length mixed-resolution token sequence.
2. **Partial Re-Noising & Mixed-Resolution Denoising:**
   Add an intermediate amount of noise to the merged token sequence (preserving the coarse global layout from Stage 1 while allowing sharp details to be generated inside the focus region), and denoise back to a clean latent sequence using Foveated Diffusion's mixed-resolution DiT and pretrained Foveated LoRA weights.
3. **Decode & Blend:**
   Split the denoised mixed-resolution tokens back into high-resolution and low-resolution grids, decode both with the VAE, and smoothly blend them using the inferred mask to produce the final full-resolution image.
