"""CLI entry point for single-prompt or interactive Semantic Foveated Diffusion generation."""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic_foveation.pipeline import SemanticFoveatedPipeline


def save_heatmap_and_mask_vis(draft_img, heatmap_full, mask_full_res, out_dir):
    draft_np = np.asarray(draft_img.convert("RGB")).astype(np.float32)
    hm = (heatmap_full.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    hm_color = cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).astype(np.float32)
    Image.fromarray((0.55 * draft_np + 0.45 * hm_color).clip(0, 255).astype(np.uint8)).save(os.path.join(out_dir, "heatmap_overlay.png"))
    mask = (mask_full_res.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(mask).save(os.path.join(out_dir, "semantic_mask.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate an image using 2-Stage Semantic Foveated Diffusion")
    parser.add_argument("--prompt", type=str, required=True, help="Main scene prompt P")
    parser.add_argument("--focus_prompt", type=str, required=True, help="Semantic focus prompt P_focus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stage1_steps", type=int, default=14)
    parser.add_argument("--stage2_steps", type=int, default=14)
    parser.add_argument("--t_switch", type=float, default=0.85)
    parser.add_argument("--budget_ratio", type=float, default=0.25)
    parser.add_argument("--selection_mode", type=str, default="threshold", choices=["top_k", "threshold"])
    parser.add_argument("--lora_mode", type=str, default="random", choices=["random", "bbox", "saliency", "no_fov"])
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--output_dir", type=str, default="outputs/single_run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pipe = SemanticFoveatedPipeline(device=args.device, offload_text_encoder=True)
    out = pipe(
        prompt=args.prompt,
        focus_prompt=args.focus_prompt,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        t_switch=args.t_switch,
        budget_ratio=args.budget_ratio,
        selection_mode=args.selection_mode,
        lora_mode=args.lora_mode,
        cfg_scale=args.cfg_scale,
    )

    out.stage1.draft_image_upsampled.save(os.path.join(args.output_dir, "stage1_draft_upsampled.png"))
    save_heatmap_and_mask_vis(
        out.stage1.draft_image_upsampled,
        out.mask_output.heatmap_full,
        out.mask_output.mask_full_res,
        args.output_dir,
    )
    out.image.save(os.path.join(args.output_dir, "semantic_foveated.png"))

    print(
        f"Saved outputs to {args.output_dir} | "
        f"Stage 1: {out.stage1.stage1_time_s:.2f}s | "
        f"CLIPSeg Mask: {out.mask_output.inference_time_s:.3f}s ({out.mask_output.num_tokens}/4096 tokens) | "
        f"Stage 2: {out.stage2_time_s:.2f}s | "
        f"Total: {out.total_time_s:.2f}s"
    )


if __name__ == "__main__":
    main()
