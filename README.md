# Adaptive Foveated Diffusion (`adaptive_foveated_diffusion_vlr`)

A two-stage, text-guided mixed-resolution **image generation** pipeline built on [Foveated Diffusion](https://arxiv.org/abs/2603.23491) ([`bchao1/foveated_diffusion`](https://github.com/bchao1/foveated_diffusion)).

Whereas standard Foveated Diffusion requires a **pre-specified** spatial resolution mask (such as a fixed circle or eye-tracker gaze point), **Adaptive Foveated Diffusion** automatically infers *where* to allocate high-resolution tokens based on a user-specified text description of the important image regions.

---

## Concise Method & Pipeline Overview

### Inputs
1. **Main Image Prompt ($P$):** Describes the full scene (e.g., *"A medieval knight standing in a mossy forest holding a glowing runesword"*).
2. **Focus Prompt ($P_{\text{focus}}$):** Specifies which part(s) of the scene should receive high-resolution compute (e.g., *"the glowing runesword"*).

### Stage 1: Coarse Low-Resolution Generation & Mask Inference
1. **Low-Resolution Draft Generation:**
   Generate a coarse low-resolution image $\hat{x}^{\text{low}}$ (e.g., at $1/2\times$ spatial resolution, which uses $4\times$ fewer tokens than full high-resolution generation) conditioned on the main prompt $P$.
2. **Vision-Language Feature Matching:**
   Pass the coarse image $\hat{x}^{\text{low}}$ and the focus prompt $P_{\text{focus}}$ through a lightweight pretrained vision-language localizer (e.g., CLIPSeg or dense CLIP/SigLIP patch similarity) to produce a 2D spatial relevance heatmap over the image.
3. **Block-Aligned Foveation Mask ($M$):**
   Binarize the heatmap (starting with a simple top-$k\%$ token budget or similarity threshold, plus optional dilation so object edges aren't clipped) on the token-block grid to produce a binary foveation mask $M \in \{0, 1\}^{h \times w}$, where $M=1$ marks high-resolution (HR) blocks and $M=0$ marks peripheral low-resolution (LR) blocks.

### Stage 2: Selective High-Resolution Re-Encoding & Mixed-Resolution Denoising
1. **Selective Re-Encoding into Mixed-Resolution Tokens:**
   - Keep the peripheral regions ($M=0$) as compact low-resolution latent tokens $\hat{z}_0^{\text{low}} = E(\hat{x}^{\text{low}})$ (1 token per $2\times 2$ block).
   - Upsample $\hat{x}^{\text{low}}$ to full resolution $(H, W)$, encode it with the VAE encoder $\hat{z}_0^{\text{high}} = E(\text{Up}(\hat{x}^{\text{low}}))$, and retain all 4 HR tokens per $2\times 2$ block inside the foveal mask ($M=1$).
   - Merge these into a single compact, variable-length mixed-resolution token sequence $\hat{z}_0^{\text{fov}} = \text{Merge}(\hat{z}_0^{\text{high}}, \hat{z}_0^{\text{low}}, M)$ of length $L = m + (h\cdot w - m)/4 \ll h\cdot w$.
2. **Partial Re-Noising & Mixed-Resolution Denoising:**
   Add intermediate noise to $\hat{z}_0^{\text{fov}}$ up to a refinement noise level $t_{\text{switch}} \in (0, 1)$ (preserving the coarse global layout from Stage 1 while allowing fine high-frequency detail to be synthesized inside $M$), and denoise back to $t=0$ using Foveated Diffusion's mixed-resolution DiT (with Cross-Resolution Phase-Aligned RoPE attention and pretrained Foveated LoRA weights—requiring **zero new DiT training**).
3. **Decode & Blend:**
   Split the clean mixed-resolution tokens back into HR and LR grids, decode with the VAE, and blend using the smoothed mask $M$ to produce the final $H \times W$ image.
