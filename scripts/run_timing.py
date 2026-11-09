"""Repeated end-to-end latency measurement (the GPU is shared, so we report median and best of N runs)."""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic_foveation.benchmark import BENCHMARK_PROMPTS
from src.semantic_foveation.pipeline import SemanticFoveatedPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prompt_ids", type=int, nargs="+", default=[0, 15, 30, 45, 52])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out_csv", type=str, default="results/timing.csv")
    args = parser.parse_args()

    pipe = SemanticFoveatedPipeline(device=args.device, offload_text_encoder=True)
    for m in ["no_fov", "random"]:
        pipe._preload_lora_deltas(m)
    mask_kw = dict(selection_mode="threshold", threshold=0.5, dilation_blocks=2)

    rows = []
    for rep in range(args.repeats + 1):
        for pid in args.prompt_ids:
            item = BENCHMARK_PROMPTS[pid]
            prompt, seed = item["prompt"], item["seed"]
            cached = pipe.encode_prompt_cached(prompt)
            _, t_full = pipe.run_single_stage_full_hr(prompt, seed=seed, num_steps=28, cached_prompts=cached)
            s1 = pipe.run_stage1_draft(prompt, seed=seed, draft_factor=4, stage1_steps=14, cached_prompts=cached)
            mask = pipe.localizer.infer_mask(s1.draft_image_low, item["focus_prompt"], **mask_kw)
            _, t2 = pipe.run_stage2_refinement(
                prompt, s1, mask.mask_token_grid, mask.mask_full_res, seed=seed,
                stage2_steps=14, t_switch=0.85, lora_mode="random", cached_prompts=cached,
            )
            if rep == 0:
                continue  # warmup
            rows += [
                {"id": pid, "rep": rep, "method": "full_hr", "total_s": t_full},
                {"id": pid, "rep": rep, "method": "ours", "total_s": s1.stage1_time_s + mask.inference_time_s + t2,
                 "stage1_s": s1.stage1_time_s, "mask_s": mask.inference_time_s, "stage2_s": t2},
            ]
        print(f"rep {rep} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(df.groupby("method")[["total_s", "stage1_s", "mask_s", "stage2_s"]].agg(["median", "min"]).to_string())


if __name__ == "__main__":
    main()
