"""Run the 60-prompt small-object benchmark.

Methods (latency always counts every step needed to produce the final image):
  full_hr      Full-resolution 1024^2 generation, 28 steps (main baseline)
  lr_draft     Stage 1 only: 256^2 draft, bicubic-upsampled to 1024^2
  ours         Stage 1 draft + CLIPSeg mask + foveated Stage 2 (fov_random LoRA)
  naive        Same as ours but Stage 2 without the foveated LoRA (ablation)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic_foveation.benchmark import BENCHMARK_PROMPTS, PROMPT_SETS
from src.semantic_foveation.metrics import SemanticEvaluator, focus_box_from_heatmap
from src.semantic_foveation.pipeline import SemanticFoveatedPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="/fast8tb/adam/semantic_foveated_diffusion_cache/eval_small_objects")
    parser.add_argument("--low_vram", action="store_true", help="Swap DiT to CPU during prompt encoding")
    parser.add_argument("--prompt_set", type=str, default="small", choices=list(PROMPT_SETS))
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument("--draft_factor", type=int, default=4)
    parser.add_argument("--stage1_steps", type=int, default=14)
    parser.add_argument("--stage2_steps", type=int, default=14)
    parser.add_argument("--full_hr_steps", type=int, default=28)
    parser.add_argument("--t_switch", type=float, default=0.85)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--dilation_blocks", type=int, default=2)
    parser.add_argument("--num_prompts", type=int, default=60)
    args = parser.parse_args()
    if args.results_csv is None:
        args.results_csv = "results/benchmark_metrics.csv" if args.prompt_set == "small" else f"results/{args.prompt_set}_subject_metrics.csv"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)

    pipe = SemanticFoveatedPipeline(device=args.device, offload_text_encoder=True, low_vram=args.low_vram)
    evaluator = SemanticEvaluator(device=args.device)
    for m in ["no_fov", "random"]:
        pipe._preload_lora_deltas(m)
    warm = BENCHMARK_PROMPTS[0]
    pipe(warm["prompt"], warm["focus_prompt"], stage1_steps=4, stage2_steps=4)
    pipe.run_single_stage_full_hr(warm["prompt"], num_steps=2)

    def mask_kwargs():
        return dict(selection_mode="threshold", threshold=args.mask_threshold, dilation_blocks=args.dilation_blocks)

    def stage2(prompt, stage1, mask, seed, lora, cached):
        return pipe.run_stage2_refinement(
            prompt, stage1,
            foveation_mask=mask.mask_token_grid,
            full_res_foveation_mask=mask.mask_full_res,
            seed=seed, stage2_steps=args.stage2_steps, t_switch=args.t_switch,
            lora_mode=lora, cached_prompts=cached,
        )

    records = []
    for idx, item in enumerate(PROMPT_SETS[args.prompt_set][: args.num_prompts]):
        pid, prompt, focus, seed = item["id"], item["prompt"], item["focus_prompt"], item["seed"]
        cached = pipe.encode_prompt_cached(prompt)

        img_full, t_full = pipe.run_single_stage_full_hr(prompt, seed=seed, num_steps=args.full_hr_steps, cached_prompts=cached)

        stage1 = pipe.run_stage1_draft(prompt, seed=seed, draft_factor=args.draft_factor,
                                       stage1_steps=args.stage1_steps, cached_prompts=cached)
        mask = pipe.localizer.infer_mask(stage1.draft_image_low, focus, **mask_kwargs())
        img_ours, t2_ours = stage2(prompt, stage1, mask, seed, "random", cached)
        img_naive, t2_naive = stage2(prompt, stage1, mask, seed, "no_fov", cached)

        t_s1, t_mask = stage1.stage1_time_s, mask.inference_time_s
        # Full HR has a different composition, so it gets its own focus box (evaluation only, not timed).
        box_draft = focus_box_from_heatmap(mask.heatmap_full.cpu().numpy())
        box_full = focus_box_from_heatmap(pipe.localizer.infer_mask(img_full, focus, **mask_kwargs()).heatmap_full.cpu().numpy())

        methods = {
            "full_hr": (img_full, t_full, 0.0, 0.0, t_full, 4096, box_full),
            "lr_draft": (stage1.draft_image_upsampled, t_s1, t_s1, 0.0, 0.0, 0, box_draft),
            "ours": (img_ours, t_s1 + t_mask + t2_ours, t_s1, t_mask, t2_ours, mask.num_tokens, box_draft),
            "naive": (img_naive, t_s1 + t_mask + t2_naive, t_s1, t_mask, t2_naive, mask.num_tokens, box_draft),
        }

        p_dir = os.path.join(args.output_dir, f"{pid:03d}")
        os.makedirs(p_dir, exist_ok=True)
        np.save(os.path.join(p_dir, "mask.npy"), mask.mask_full_res.cpu().numpy().astype(np.uint8))
        for name, (img, latency, ts1, tm, ts2, tokens, box) in methods.items():
            img.save(os.path.join(p_dir, f"{name}.png"))
            records.append({
                "id": pid, "category": item["category"], "method": name,
                "latency_s": latency, "stage1_s": ts1, "mask_s": tm, "stage2_s": ts2,
                "stage2_tokens": tokens, "hr_area": mask.hr_block_ratio, "box": json.dumps(box),
                **evaluator.evaluate(img, prompt, focus, box),
            })

        print(f"[{idx + 1:02d}] id={pid} full_hr={t_full:.2f}s ours={methods['ours'][1]:.2f}s "
              f"({t_full / methods['ours'][1]:.2f}x) hr_area={mask.hr_block_ratio:.3f}", flush=True)
        if (idx + 1) % 5 == 0:
            pd.DataFrame(records).to_csv(args.results_csv, index=False)

    df = pd.DataFrame(records)
    df.to_csv(args.results_csv, index=False)
    cols = ["latency_s", "stage1_s", "mask_s", "stage2_s", "stage2_tokens", "full_clip", "focus_clip", "focus_clip_iqa", "focus_hf"]
    print(df.groupby("method", sort=False)[cols].mean().to_string())


if __name__ == "__main__":
    main()
