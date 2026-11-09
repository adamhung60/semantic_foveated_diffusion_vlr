"""Open-vocabulary semantic mask localizer for Stage 1 of Semantic Foveated Diffusion.

Implements Stage 1 Steps 2-3 of method.tex:
  1. Vision-Language Feature Matching via CLIPSeg (CIDAS/clipseg-rd64-refined)
     to produce a 2D spatial relevance heatmap S in [0, 1]^(h/2 x w/2).
  2. Block-Aligned Foveation Mask Construction on the (h/2 x w/2) low-resolution
     token grid (top-k% token budget or adaptive thresholding + morphological dilation),
     expanded 2x2 to the (h x w) high-resolution token grid M in {0, 1}^(h x w).
"""

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


@dataclass
class SemanticMaskOutput:
    """Container for inferred semantic foveation masks and diagnostics."""
    heatmap_low: torch.Tensor          # [h_low, w_low] continuous relevance in [0, 1]
    heatmap_full: torch.Tensor         # [H, W] continuous relevance in [0, 1]
    mask_low: torch.Tensor             # [h_low, w_low] binary block mask in {0, 1}
    mask_token_grid: torch.Tensor      # [h_high, w_high] block-aligned binary HR token mask in {0, 1}
    mask_full_res: torch.Tensor        # [H, W] binary pixel-space mask in {0, 1}
    hr_block_ratio: float              # Fraction of (h_low x w_low) blocks selected as HR
    effective_token_ratio: float       # L / (h_high * w_high) = hr_ratio + (1 - hr_ratio) / lr_factor^2
    num_tokens: int                    # Total mixed-res token sequence length L
    inference_time_s: float            # Wall-clock time for VLM mask inference


class CLIPSegLocalizer:
    """Lightweight open-vocabulary vision-language localizer using CLIPSeg."""

    def __init__(
        self,
        model_id: str = "CIDAS/clipseg-rd64-refined",
        device: Union[str, torch.device] = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.processor = CLIPSegProcessor.from_pretrained(model_id, use_fast=False)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def compute_heatmap(
        self,
        draft_image: Image.Image,
        focus_prompt: Union[str, List[str]],
    ) -> torch.Tensor:
        """Compute a continuous 2D relevance heatmap in [0, 1] from the draft image and focus prompt(s)."""
        prompts = [focus_prompt] if isinstance(focus_prompt, str) else list(focus_prompt)
        images = [draft_image.convert("RGB")] * len(prompts)

        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # [N, H_clipseg, W_clipseg] or [H_clipseg, W_clipseg]
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)

        probs = torch.sigmoid(logits.float())
        # Aggregate across multiple focus phrases via elementwise max
        heatmap = probs.max(dim=0).values  # [H_clipseg, W_clipseg]

        h_min = heatmap.min()
        h_max = heatmap.max()
        if (h_max - h_min) > 1e-6:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = torch.zeros_like(heatmap)

        return heatmap

    @torch.no_grad()
    def infer_mask(
        self,
        draft_image: Image.Image,
        focus_prompt: Union[str, List[str]],
        target_height: int = 1024,
        target_width: int = 1024,
        lr_factor: int = 2,
        budget_ratio: Optional[float] = 0.25,
        threshold: float = 0.50,
        selection_mode: str = "top_k",
        dilation_blocks: int = 1,
        smooth_sigma_blocks: float = 0.85,
    ) -> SemanticMaskOutput:
        """Infer a block-aligned binary foveation mask M in {0, 1}^(h x w) from the Stage-1 draft.

        Args:
            draft_image: Coarse low-resolution RGB image from Stage 1.
            focus_prompt: Text specification of the semantic region(s) of interest.
            target_height: Full target image height H (default 1024 -> h = 64).
            target_width: Full target image width W (default 1024 -> w = 64).
            lr_factor: Spatial downsampling factor d in periphery (default 2 -> 32x32 blocks).
            budget_ratio: Target fraction of high-resolution blocks in (0, 1] when selection_mode="top_k".
            threshold: Relevance threshold in (0, 1) when selection_mode="threshold".
            selection_mode: "top_k" (exact token budget) or "threshold" (adaptive object size).
            dilation_blocks: Morphological dilation radius on the low-res block grid (for "threshold" mode
                or post-threshold expansion).
            smooth_sigma_blocks: Gaussian smoothing sigma on the (h/2 x w/2) block grid so top-k selection
                forms a smooth, dilated envelope around object boundaries without clipping edges.
        """
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        h_high, w_high = target_height // 16, target_width // 16
        h_low, w_low = h_high // lr_factor, w_high // lr_factor
        num_blocks = h_low * w_low

        raw_heatmap = self.compute_heatmap(draft_image, focus_prompt)  # [352, 352]
        heatmap_4d = raw_heatmap.unsqueeze(0).unsqueeze(0)

        # Pool onto the (h_low, w_low) low-resolution token grid
        heatmap_low = F.interpolate(
            heatmap_4d, size=(h_low, w_low), mode="bilinear", align_corners=False
        )

        # Slight spatial smoothing on the block grid so selected blocks envelop object boundaries cleanly
        if smooth_sigma_blocks > 0:
            k_size = max(3, int(round(smooth_sigma_blocks * 4)) | 1)
            coords = torch.arange(k_size, device=self.device, dtype=torch.float32) - (k_size - 1) / 2.0
            g1d = torch.exp(-(coords ** 2) / (2.0 * smooth_sigma_blocks ** 2))
            g1d = g1d / g1d.sum()
            pad = k_size // 2
            heatmap_low = F.conv2d(heatmap_low, g1d.view(1, 1, k_size, 1), padding=(pad, 0))
            heatmap_low = F.conv2d(heatmap_low, g1d.view(1, 1, 1, k_size), padding=(0, pad))

        heatmap_low = heatmap_low.squeeze(0).squeeze(0)
        h_min, h_max = heatmap_low.min(), heatmap_low.max()
        if (h_max - h_min) > 1e-6:
            heatmap_low = (heatmap_low - h_min) / (h_max - h_min)

        if selection_mode == "top_k":
            assert budget_ratio is not None and 0.0 < budget_ratio <= 1.0
            # Optional soft morphological dilation before top-k ranking so edges are prioritized
            if dilation_blocks > 0:
                k_dil = 2 * dilation_blocks + 1
                dilated = F.max_pool2d(
                    heatmap_low.unsqueeze(0).unsqueeze(0),
                    kernel_size=k_dil,
                    stride=1,
                    padding=dilation_blocks,
                ).squeeze(0).squeeze(0)
                score_grid = 0.7 * heatmap_low + 0.3 * dilated
            else:
                score_grid = heatmap_low

            k_blocks = max(1, min(num_blocks, int(round(budget_ratio * num_blocks))))
            flat_scores = score_grid.view(-1)
            topk_indices = torch.topk(flat_scores, k=k_blocks, largest=True).indices
            mask_low = torch.zeros_like(flat_scores)
            mask_low[topk_indices] = 1.0
            mask_low = mask_low.view(h_low, w_low)
        elif selection_mode == "threshold":
            mask_low = (heatmap_low >= threshold).float()
            if mask_low.sum() == 0:
                # Fallback to top 10% blocks if nothing exceeds threshold
                k_blocks = max(1, int(round(0.10 * num_blocks)))
                flat_scores = heatmap_low.view(-1)
                topk_indices = torch.topk(flat_scores, k=k_blocks, largest=True).indices
                mask_low = torch.zeros_like(flat_scores)
                mask_low[topk_indices] = 1.0
                mask_low = mask_low.view(h_low, w_low)
            if dilation_blocks > 0:
                k_dil = 2 * dilation_blocks + 1
                mask_low = F.max_pool2d(
                    mask_low.unsqueeze(0).unsqueeze(0),
                    kernel_size=k_dil,
                    stride=1,
                    padding=dilation_blocks,
                ).squeeze(0).squeeze(0)
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        # Expand each selected (h_low, w_low) block to an (lr_factor x lr_factor) block on (h_high, w_high)
        mask_token_grid = F.interpolate(
            mask_low.unsqueeze(0).unsqueeze(0),
            size=(h_high, w_high),
            mode="nearest",
        ).squeeze(0).squeeze(0)

        mask_full_res = F.interpolate(
            mask_token_grid.unsqueeze(0).unsqueeze(0),
            size=(target_height, target_width),
            mode="nearest",
        ).squeeze(0).squeeze(0)

        heatmap_full = F.interpolate(
            heatmap_4d,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_time_s = time.perf_counter() - t0

        num_hr_tokens = int(mask_token_grid.sum().item())
        total_hr_tokens = h_high * w_high
        num_lr_tokens = (total_hr_tokens - num_hr_tokens) // (lr_factor ** 2)
        num_tokens = num_hr_tokens + num_lr_tokens
        hr_block_ratio = float(mask_low.mean().item())
        effective_token_ratio = float(num_tokens / total_hr_tokens)

        return SemanticMaskOutput(
            heatmap_low=heatmap_low,
            heatmap_full=heatmap_full,
            mask_low=mask_low,
            mask_token_grid=mask_token_grid,
            mask_full_res=mask_full_res,
            hr_block_ratio=hr_block_ratio,
            effective_token_ratio=effective_token_ratio,
            num_tokens=num_tokens,
            inference_time_s=inference_time_s,
        )
