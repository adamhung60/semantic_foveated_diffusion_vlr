"""Focus-region evaluation metrics for Semantic Foveated Diffusion.

All focus metrics are computed on a square crop around the focus object (`focus_box_from_heatmap`):
  - focus_clip:     CLIP similarity between the crop and the focus prompt (semantic correctness)
  - focus_clip_iqa: CLIP-IQA style P("sharp photo" | crop) vs. "blurry photo" (perceived detail)
  - focus_hf:       mean squared Laplacian of the crop (high-frequency energy)
  - full_clip:      CLIP similarity between the full image and the scene prompt
"""

from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

Box = Tuple[int, int, int, int]


def focus_box_from_heatmap(heatmap: np.ndarray, threshold: float = 0.5, pad: float = 0.3, min_side: int = 160) -> Box:
    """Square box (x0, y0, x1, y1) around the largest connected component of `heatmap >= threshold`."""
    h, w = heatmap.shape
    binary = (heatmap >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        cy, cx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x0 = x1 = cx
        y0 = y1 = cy
    else:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x0, y0 = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP]
        x1, y1 = x0 + stats[best, cv2.CC_STAT_WIDTH], y0 + stats[best, cv2.CC_STAT_HEIGHT]
    side = int(min(h, w, max(min_side, (1 + pad) * max(x1 - x0, y1 - y0))))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bx = int(np.clip(cx - side // 2, 0, w - side))
    by = int(np.clip(cy - side // 2, 0, h - side))
    return bx, by, bx + side, by + side


def high_freq_energy(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    return float(np.mean(cv2.Laplacian(gray, cv2.CV_32F, ksize=3) ** 2))


class SemanticEvaluator:
    def __init__(self, clip_model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id, use_fast=False)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(self.device).eval()
        self.iqa_prompts = ["a sharp, detailed photo", "a blurry, low-resolution photo"]

    @torch.no_grad()
    def _clip(self, image: Image.Image, texts):
        inputs = self.clip_processor(text=texts, images=[image.convert("RGB")], return_tensors="pt", padding=True, truncation=True)
        out = self.clip_model(**{k: v.to(self.device) for k, v in inputs.items()})
        img = F.normalize(out.image_embeds.float(), dim=-1)
        txt = F.normalize(out.text_embeds.float(), dim=-1)
        return (img @ txt.T)[0]

    def clip_score(self, image: Image.Image, text: str) -> float:
        return float(max(0.0, 100.0 * self._clip(image, [text])[0].item()))

    def clip_iqa(self, image: Image.Image) -> float:
        sims = self._clip(image, self.iqa_prompts)
        return float(torch.softmax(100.0 * sims, dim=0)[0].item())

    def evaluate(self, image: Image.Image, scene_prompt: str, focus_prompt: str, box: Box) -> Dict[str, float]:
        crop = image.crop(box)
        return {
            "full_clip": self.clip_score(image, scene_prompt),
            "focus_clip": self.clip_score(crop, focus_prompt),
            "focus_clip_iqa": self.clip_iqa(crop),
            "focus_hf": high_freq_energy(crop),
        }
