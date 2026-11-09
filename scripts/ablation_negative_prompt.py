"""Ablation on 4 prompts: negative prompt x t_switch, focus-crop high-frequency energy + crop sheets."""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.semantic_foveation.benchmark import BENCHMARK_PROMPTS as B
from src.semantic_foveation.metrics import focus_box_from_heatmap, high_freq_energy
from src.semantic_foveation.pipeline import DEFAULT_NEGATIVE_PROMPT as NEG, SemanticFoveatedPipeline

OUT = "/fast8tb/adam/semantic_foveated_diffusion_cache/exp_sharp"
os.makedirs(OUT, exist_ok=True)
pipe = SemanticFoveatedPipeline(device=sys.argv[1] if len(sys.argv) > 1 else "cuda:0", offload_text_encoder=True)
T = 256
for pid in [0, 15, 20, 47]:
    it = B[pid]
    rows = []
    for neg in ["", NEG]:
        cached = pipe.encode_prompt_cached(it["prompt"], negative_prompt=neg)
        s1 = pipe.run_stage1_draft(it["prompt"], seed=it["seed"], draft_factor=4, cached_prompts=cached)
        m = pipe.localizer.infer_mask(s1.draft_image_low, it["focus_prompt"], selection_mode="threshold", dilation_blocks=2)
        box = focus_box_from_heatmap(m.heatmap_full.cpu().numpy())
        imgs = [("draft", s1.draft_image_upsampled)]
        for ts in [0.85, 0.95]:
            imgs.append((f"ref{ts}", pipe.run_stage2_refinement(it["prompt"], s1, None, None, seed=it["seed"], t_switch=ts, lora_mode="no_fov", cached_prompts=cached)[0]))
            imgs.append((f"ours{ts}", pipe.run_stage2_refinement(it["prompt"], s1, m.mask_token_grid, m.mask_full_res, seed=it["seed"], t_switch=ts, lora_mode="random", cached_prompts=cached)[0]))
        full = pipe.run_single_stage_full_hr(it["prompt"], seed=it["seed"], cached_prompts=cached)[0]
        fbox = focus_box_from_heatmap(pipe.localizer.infer_mask(full, it["focus_prompt"]).heatmap_full.cpu().numpy())
        crops = [(n, im.crop(box)) for n, im in imgs] + [("full", full.crop(fbox))]
        print(pid, repr(neg[:5]), " ".join(f"{n}={high_freq_energy(c):.0f}" for n, c in crops), flush=True)
        rows.append(crops)
    sheet = Image.new("RGB", (T * len(rows[0]), T * 2))
    for r, crops in enumerate(rows):
        for c, (_, im) in enumerate(crops):
            sheet.paste(im.resize((T, T), Image.Resampling.LANCZOS), (c * T, r * T))
    sheet.save(f"{OUT}/sheet_{pid:03d}.png")
