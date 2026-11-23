"""Foveation masks that ignore the focus prompt, matched to the semantic mask's token budget.

Every mask here selects exactly `k` blocks on the (h/2 x w/2) block grid, the same number as the CLIPSeg
mask it is compared against. Token count and latency are therefore equal, and only the placement differs.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

GENERIC_PROMPT = "the main subject of the photo"


@dataclass
class BlockMask:
    mask_low: torch.Tensor         # [h_low, w_low] binary block mask
    mask_token_grid: torch.Tensor  # [h_high, w_high] binary HR-token mask
    mask_full_res: torch.Tensor    # [H, W] binary pixel mask
    num_tokens: int                # mixed-resolution sequence length


def expand_block_mask(mask_low: torch.Tensor, target_height: int = 1024, target_width: int = 1024, lr_factor: int = 2) -> BlockMask:
    mask_low = mask_low.float()
    h_high, w_high = target_height // 16, target_width // 16
    token_grid = F.interpolate(mask_low[None, None], size=(h_high, w_high), mode="nearest")[0, 0]
    full_res = F.interpolate(token_grid[None, None], size=(target_height, target_width), mode="nearest")[0, 0]
    num_hr = int(token_grid.sum().item())
    return BlockMask(mask_low, token_grid, full_res, num_hr + (h_high * w_high - num_hr) // (lr_factor ** 2))


def _top_k(score: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(score.size, dtype=np.float32)
    mask[np.argsort(-score.ravel(), kind="stable")[:k]] = 1.0
    return mask.reshape(score.shape)


def disc_mask(k: int, center_yx: Tuple[float, float], h_low: int, w_low: int) -> np.ndarray:
    """The k blocks closest to `center_yx` (block coordinates): a disc, clipped by the image border."""
    ys, xs = np.mgrid[0:h_low, 0:w_low].astype(np.float32)
    dist = (ys - center_yx[0]) ** 2 + (xs - center_yx[1]) ** 2
    return _top_k(-dist, k)


def detail_mask(draft_image: Image.Image, k: int, h_low: int, w_low: int, sigma_blocks: float = 1.5) -> np.ndarray:
    """Text-free saliency: the k blocks where the draft has the most local detail (smoothed Laplacian energy)."""
    gray = np.asarray(draft_image.convert("L"), dtype=np.float32)
    energy = cv2.Laplacian(gray, cv2.CV_32F, ksize=3) ** 2
    energy = cv2.resize(energy, (w_low, h_low), interpolation=cv2.INTER_AREA)
    return _top_k(cv2.GaussianBlur(energy, (0, 0), sigma_blocks), k)


def budget_matched_masks(draft_image: Image.Image, localizer, k: int, seed: int, h_low: int = 32, w_low: int = 32) -> Dict[str, np.ndarray]:
    """Block masks with exactly k blocks each: center disc, random disc, detail saliency, generic-prompt CLIPSeg."""
    rng = np.random.default_rng(seed)
    generic = localizer.infer_mask(
        draft_image, GENERIC_PROMPT, selection_mode="top_k", budget_ratio=k / (h_low * w_low), dilation_blocks=2
    ).mask_low.cpu().numpy()
    return {
        "center": disc_mask(k, ((h_low - 1) / 2, (w_low - 1) / 2), h_low, w_low),
        "random": disc_mask(k, (rng.uniform(0, h_low - 1), rng.uniform(0, w_low - 1)), h_low, w_low),
        "detail": detail_mask(draft_image, k, h_low, w_low),
        "generic": generic,
    }
