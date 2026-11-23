"""Ablation benchmark: fair-compute baselines, budget-matched mask sources, and paired metrics.

Runs as separate processes, so the text encoder and the DiT are never in memory together and each
LoRA is merged exactly once (run each under scripts/run_capped.sh):

  --phase encode   text encoder only     cache prompt embeddings
  --phase base     DiT + no_fov LoRA     uniform baselines, draft, masks, all-token refinement, naive
  --phase fov      DiT + fov_random LoRA foveated Stage 2 for every mask source
  --phase metrics  evaluation models     focus-crop metrics for every saved image -> CSV
  --phase sweep    DiT + fov_random LoRA ours at other t_switch values and Stage-2 step counts (--sweep_t, --sweep_steps),
                                         on the cached drafts and masks
  --phase base_extra  DiT + no_fov LoRA  few-step uniform baselines (--extra_full_steps, --extra_768_steps)
  --phase sweep_metrics                  metrics for the sweep -> results/ablation_{set}_tswitch.csv

Methods
  full_hr_{N}    1024^2 generation with N steps
  uni{R}_{N}     uniform R^2 generation with N steps, bicubic-upsampled to 1024^2
  lr_draft       Stage 1 only (256^2, bicubic-upsampled)
  refine_all     Stage 2 on the same draft with all 4096 tokens at high resolution
  naive          CLIPSeg mask, Stage 2 without the foveated LoRA
  fov_clipseg    ours
  fov_{center,random,detail,generic}   Stage 2 with a mask that ignores the focus prompt, same token budget

Paired protocol (gt_*): the draft is the full_hr_28 image downsampled to 256^2, so full_hr_28 is a sharp
reference with exactly the same composition, and LPIPS to it measures how much true detail Stage 2 recovers.
  gt_lr_draft    the downsampled image, bicubic-upsampled (no refinement)
  gt_refine_all, gt_fov_clipseg, gt_fov_center   Stage 2 on that draft
"""

import argparse
import json
import os
import resource
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic_foveation.benchmark import PROMPT_SETS
from src.semantic_foveation.mask_baselines import budget_matched_masks, disc_mask, expand_block_mask

FULL_HR_STEPS = [28, 14, 8]
UNIFORM = [(768, 14), (512, 28), (512, 14)]
MASK_SOURCES = ["clipseg", "center", "random", "detail", "generic"]
GT_MASK_SOURCES = ["clipseg", "center"]
BICUBIC = Image.Resampling.BICUBIC


def image_draft(image_low):
    """A Stage-1 output built from an image instead of a generation (Stage 2 re-encodes it from pixels)."""
    from src.semantic_foveation.pipeline import Stage1Output
    return Stage1Output(image_low, image_low.resize((1024, 1024), BICUBIC), torch.zeros(1, 0, 128), 0.0)


def periphery_crop(img, box, size=512):
    """The image-corner crop farthest from the focus box: a region foveation leaves at low resolution."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    corners = [(x, y) for x in (0, img.width - size) for y in (0, img.height - size)]
    x, y = max(corners, key=lambda c: (c[0] + size / 2 - cx) ** 2 + (c[1] + size / 2 - cy) ** 2)
    return img.crop((x, y, x + size, y + size))


def whole_image_scores(paired, img, box):
    return {"full_musiq": paired.musiq(img), "periph_musiq": paired.musiq(periphery_crop(img, box))}


def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"{msg}  [peak RAM {rss:.1f} GB, peak VRAM {vram:.1f} GB]", flush=True)


def prompt_dir(args, pid):
    return os.path.join(args.output_dir, args.prompt_set, f"{pid:03d}")


def load_cached_prompt(args, pid, device, dtype):
    def to_dev(d):
        return {k: (v.to(device=device, dtype=dtype if v.is_floating_point() else None) if torch.is_tensor(v) else v) for k, v in d.items()}
    cache = os.path.join(args.output_dir, "prompt_cache")
    return to_dev(torch.load(os.path.join(cache, f"{pid:03d}.pt"))), to_dev(torch.load(os.path.join(cache, "negative.pt")))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def phase_encode(args, items):
    from src.semantic_foveation.pipeline import SemanticFoveatedPipeline

    cache = os.path.join(args.output_dir, "prompt_cache")
    os.makedirs(cache, exist_ok=True)
    pipe = SemanticFoveatedPipeline(device=args.device, offload_text_encoder=False, components=("text_encoder",), load_localizer=False)
    log("text encoder loaded")
    for item in items:
        path = os.path.join(cache, f"{item['id']:03d}.pt")
        if os.path.exists(path):
            continue
        posi, nega = pipe.encode_prompt_cached(item["prompt"])
        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in posi.items()}, path)
        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in nega.items()}, os.path.join(cache, "negative.pt"))
    log(f"encoded {len(items)} prompts")


def load_generator(args, lora_mode):
    from src.semantic_foveation.pipeline import SemanticFoveatedPipeline

    pipe = SemanticFoveatedPipeline(device=args.device, components=("transformer", "vae"))
    log("DiT + VAE loaded")
    pipe.set_lora_mode(lora_mode)
    pipe._lora_deltas.clear()  # the LoRA stays merged for the whole process; free the cached deltas
    log(f"LoRA '{lora_mode}' merged")
    return pipe


def phase_base(args, items):
    pipe = load_generator(args, "no_fov")
    mask_kw = dict(selection_mode="threshold", threshold=args.mask_threshold, dilation_blocks=args.dilation_blocks)

    def uniform(prompt, seed, cached, res, steps):
        img, t = pipe.run_single_stage_full_hr(prompt, seed=seed, target_height=res, target_width=res,
                                               num_steps=steps, cached_prompts=cached, lora_mode="no_fov")
        return (img if res == 1024 else img.resize((1024, 1024), BICUBIC)), t

    def refine(prompt, stage1, mask, seed, cached):
        return pipe.run_stage2_refinement(
            prompt, stage1, None if mask is None else mask.mask_token_grid, None if mask is None else mask.mask_full_res,
            seed=seed, stage2_steps=args.stage2_steps, t_switch=args.t_switch, lora_mode="no_fov", cached_prompts=cached)

    # Warm up every shape once so the first prompt's timings are clean
    warm = items[0]
    cached = load_cached_prompt(args, warm["id"], args.device, pipe.torch_dtype)
    for res in (1024, 768, 512):
        uniform(warm["prompt"], 0, cached, res, 2)
    s1 = pipe.run_stage1_draft(warm["prompt"], seed=0, draft_factor=args.draft_factor, stage1_steps=2, cached_prompts=cached)
    m = pipe.localizer.infer_mask(s1.draft_image_low, warm["focus_prompt"], **mask_kw)
    refine(warm["prompt"], s1, m, 0, cached)
    log("warm-up done")

    for idx, item in enumerate(items):
        pid, prompt, focus, seed = item["id"], item["prompt"], item["focus_prompt"], item["seed"]
        out = prompt_dir(args, pid)
        if os.path.exists(os.path.join(out, "base.json")):
            continue
        os.makedirs(out, exist_ok=True)
        cached = load_cached_prompt(args, pid, args.device, pipe.torch_dtype)
        latency = {}

        for steps in FULL_HR_STEPS:
            img, t = uniform(prompt, seed, cached, 1024, steps)
            img.save(os.path.join(out, f"full_hr_{steps}.png"))
            latency[f"full_hr_{steps}"] = t
            if steps == max(FULL_HR_STEPS):
                gt_low = img.resize((1024 // args.draft_factor,) * 2, Image.Resampling.LANCZOS)
        for res, steps in UNIFORM:
            img, t = uniform(prompt, seed, cached, res, steps)
            img.save(os.path.join(out, f"uni{res}_{steps}.png"))
            latency[f"uni{res}_{steps}"] = t

        stage1 = pipe.run_stage1_draft(prompt, seed=seed, draft_factor=args.draft_factor,
                                       stage1_steps=args.stage1_steps, cached_prompts=cached)
        stage1.draft_image_low.save(os.path.join(out, "draft_low.png"))
        stage1.draft_image_upsampled.save(os.path.join(out, "lr_draft.png"))
        torch.save(stage1.latents_low.cpu(), os.path.join(out, "latents_low.pt"))
        latency["lr_draft"] = stage1.stage1_time_s

        mask = pipe.localizer.infer_mask(stage1.draft_image_low, focus, **mask_kw)
        k = int(mask.mask_low.sum().item())
        masks = {"clipseg": mask.mask_low.cpu().numpy(),
                 **budget_matched_masks(stage1.draft_image_low, pipe.localizer, k, seed, *mask.mask_low.shape)}
        object_low = (mask.heatmap_low >= args.mask_threshold).float().cpu().numpy()
        np.savez_compressed(os.path.join(out, "masks.npz"), object=object_low,
                            heatmap_full=mask.heatmap_full.cpu().numpy().astype(np.float16), **masks)

        img, t2 = refine(prompt, stage1, None, seed, cached)
        img.save(os.path.join(out, "refine_all.png"))
        latency["refine_all"] = stage1.stage1_time_s + t2
        img, t2 = refine(prompt, stage1, mask, seed, cached)
        img.save(os.path.join(out, "naive.png"))
        latency["naive"] = stage1.stage1_time_s + mask.inference_time_s + t2

        # Paired protocol: the draft is the downsampled full-HR image
        gt_stage1 = image_draft(gt_low)
        gt_low.save(os.path.join(out, "gt_draft_low.png"))
        gt_stage1.draft_image_upsampled.save(os.path.join(out, "gt_lr_draft.png"))
        gt_mask = pipe.localizer.infer_mask(gt_low, focus, **mask_kw)
        gt_k = int(gt_mask.mask_low.sum().item())
        np.savez_compressed(os.path.join(out, "masks_gt.npz"), clipseg=gt_mask.mask_low.cpu().numpy(),
                            center=disc_mask(gt_k, ((gt_mask.mask_low.shape[0] - 1) / 2, (gt_mask.mask_low.shape[1] - 1) / 2), *gt_mask.mask_low.shape),
                            heatmap_full=gt_mask.heatmap_full.cpu().numpy().astype(np.float16))
        refine(prompt, gt_stage1, None, seed, cached)[0].save(os.path.join(out, "gt_refine_all.png"))

        save_json(os.path.join(out, "base.json"), {
            "id": pid, "category": item["category"], "prompt": prompt, "focus_prompt": focus, "seed": seed,
            "t_switch": args.t_switch, "k_blocks": k, "hr_area": mask.hr_block_ratio, "stage2_tokens": mask.num_tokens,
            "stage1_s": stage1.stage1_time_s, "mask_s": mask.inference_time_s, "latency_s": latency,
            "gt_hr_area": gt_mask.hr_block_ratio,
        })
        log(f"[base {idx + 1}/{len(items)}] id={pid} full_hr_28={latency['full_hr_28']:.1f}s uni512_14={latency['uni512_14']:.1f}s "
            f"refine_all={latency['refine_all']:.1f}s naive={latency['naive']:.1f}s hr_area={mask.hr_block_ratio:.3f}")


def phase_base_extra(args, items):
    """Few-step uniform baselines that match the latency of the shortened foveated schedules."""
    pipe = load_generator(args, "no_fov")
    specs = [(1024, n) for n in args.extra_full_steps] + [(768, n) for n in args.extra_768_steps]
    name = lambda res, n: f"full_hr_{n}" if res == 1024 else f"uni{res}_{n}"

    def uniform(prompt, seed, cached, res, steps):
        img, t = pipe.run_single_stage_full_hr(prompt, seed=seed, target_height=res, target_width=res,
                                               num_steps=steps, cached_prompts=cached, lora_mode="no_fov")
        return (img if res == 1024 else img.resize((1024, 1024), BICUBIC)), t

    todo = [it for it in items if os.path.exists(os.path.join(prompt_dir(args, it["id"]), "base.json"))]
    cached = load_cached_prompt(args, todo[0]["id"], args.device, pipe.torch_dtype)
    for res in sorted({r for r, _ in specs}):
        uniform(todo[0]["prompt"], 0, cached, res, 2)
    log("warm-up done")
    for idx, item in enumerate(todo):
        out = prompt_dir(args, item["id"])
        path = os.path.join(out, "extra.json")
        done = load_json(path) if os.path.exists(path) else {}
        if all(name(r, n) in done for r, n in specs):
            continue
        cached = load_cached_prompt(args, item["id"], args.device, pipe.torch_dtype)
        for res, n in specs:
            if name(res, n) in done:
                continue
            img, t = uniform(item["prompt"], item["seed"], cached, res, n)
            img.save(os.path.join(out, f"{name(res, n)}.png"))
            done[name(res, n)] = {"steps": n, "latency_s": t}
            save_json(path, done)
        log(f"[base_extra {idx + 1}/{len(todo)}] id={item['id']} " + " ".join(f"{k}={v['latency_s']:.2f}s" for k, v in done.items()))


def load_stage1(out, base):
    from src.semantic_foveation.pipeline import Stage1Output
    low = Image.open(os.path.join(out, "draft_low.png")).convert("RGB")
    latents = torch.load(os.path.join(out, "latents_low.pt"))
    return Stage1Output(low, low.resize((1024, 1024), BICUBIC), latents, base["stage1_s"])


def foveated_stage2(pipe, args, prompt, stage1, block_mask, seed, cached, steps, t_switch):
    dev = lambda t: t.to(args.device)
    return pipe.run_stage2_refinement(
        prompt, stage1, dev(block_mask.mask_token_grid), dev(block_mask.mask_full_res), seed=seed,
        stage2_steps=steps, t_switch=t_switch, lora_mode="random", cached_prompts=cached)


def t_tag(t_switch, steps=14):
    return f"t{round(t_switch * 100):03d}" + ("" if steps == 14 else f"_s{steps:02d}")


def phase_fov(args, items):
    pipe = load_generator(args, "random")

    def run(prompt, stage1, block_mask, seed, cached, steps):
        return foveated_stage2(pipe, args, prompt, stage1, block_mask, seed, cached, steps, args.t_switch)

    todo = [it for it in items if os.path.exists(os.path.join(prompt_dir(args, it["id"]), "base.json"))]
    if todo:
        out = prompt_dir(args, todo[0]["id"])
        masks = np.load(os.path.join(out, "masks.npz"))
        cached = load_cached_prompt(args, todo[0]["id"], args.device, pipe.torch_dtype)
        run(todo[0]["prompt"], load_stage1(out, load_json(os.path.join(out, "base.json"))),
            expand_block_mask(torch.from_numpy(masks["clipseg"])), 0, cached, 2)
        log("warm-up done")

    for idx, item in enumerate(todo):
        pid, prompt, seed = item["id"], item["prompt"], item["seed"]
        out = prompt_dir(args, pid)
        if os.path.exists(os.path.join(out, "fov.json")):
            continue
        base = load_json(os.path.join(out, "base.json"))
        stage1 = load_stage1(out, base)
        masks = np.load(os.path.join(out, "masks.npz"))
        cached = load_cached_prompt(args, pid, args.device, pipe.torch_dtype)
        latency, tokens = {}, {}
        for source in MASK_SOURCES:
            block_mask = expand_block_mask(torch.from_numpy(masks[source]))
            img, t2 = run(prompt, stage1, block_mask, seed, cached, args.stage2_steps)
            img.save(os.path.join(out, f"fov_{source}.png"))
            latency[f"fov_{source}"] = base["stage1_s"] + base["mask_s"] + t2
            tokens[f"fov_{source}"] = block_mask.num_tokens
        gt_stage1 = image_draft(Image.open(os.path.join(out, "gt_draft_low.png")).convert("RGB"))
        gt_masks = np.load(os.path.join(out, "masks_gt.npz"))
        for source in GT_MASK_SOURCES:
            img, _ = run(prompt, gt_stage1, expand_block_mask(torch.from_numpy(gt_masks[source])), seed, cached, args.stage2_steps)
            img.save(os.path.join(out, f"gt_fov_{source}.png"))
        save_json(os.path.join(out, "fov.json"), {"latency_s": latency, "tokens": tokens})
        log(f"[fov {idx + 1}/{len(todo)}] id={pid} ours={latency['fov_clipseg']:.2f}s tokens={tokens['fov_clipseg']}")


def phase_sweep(args, items):
    """Ours (CLIPSeg mask, foveated LoRA) at other re-noising levels, on the cached drafts and masks."""
    pipe = load_generator(args, "random")
    todo = [it for it in items if os.path.exists(os.path.join(prompt_dir(args, it["id"]), "fov.json"))]
    warm = False
    for idx, item in enumerate(todo):
        pid, prompt, seed = item["id"], item["prompt"], item["seed"]
        out = prompt_dir(args, pid)
        path = os.path.join(out, "sweep.json")
        done = load_json(path) if os.path.exists(path) else {}
        runs = [(t, n) for t in args.sweep_t for n in args.sweep_steps]
        if all(t_tag(t, n) in done for t, n in runs):
            continue
        base = load_json(os.path.join(out, "base.json"))
        cached = load_cached_prompt(args, pid, args.device, pipe.torch_dtype)
        stage1 = load_stage1(out, base)
        mask = expand_block_mask(torch.from_numpy(np.load(os.path.join(out, "masks.npz"))["clipseg"]))
        gt_stage1 = image_draft(Image.open(os.path.join(out, "gt_draft_low.png")).convert("RGB"))
        gt_mask = expand_block_mask(torch.from_numpy(np.load(os.path.join(out, "masks_gt.npz"))["clipseg"]))
        if not warm:
            foveated_stage2(pipe, args, prompt, stage1, mask, 0, cached, 2, args.sweep_t[0])
            warm = True
        for t, n in runs:
            tag = t_tag(t, n)
            if tag in done:
                continue
            img, t2 = foveated_stage2(pipe, args, prompt, stage1, mask, seed, cached, n, t)
            img.save(os.path.join(out, f"fov_clipseg_{tag}.png"))
            if n == 14:  # the paired protocol is only run at the default step count
                foveated_stage2(pipe, args, prompt, gt_stage1, gt_mask, seed, cached, n, t)[0].save(
                    os.path.join(out, f"gt_fov_clipseg_{tag}.png"))
            done[tag] = {"t_switch": t, "steps": n, "latency_s": base["stage1_s"] + base["mask_s"] + t2}
            save_json(path, done)
        log(f"[sweep {idx + 1}/{len(todo)}] id={pid} " + " ".join(f"{k}={v['latency_s']:.2f}s" for k, v in done.items()))


def phase_sweep_metrics(args, items):
    """One row per (prompt, t_switch, method); the base t_switch images come from the fov phase."""
    import pandas as pd

    from src.semantic_foveation.localizer import CLIPSegLocalizer
    from src.semantic_foveation.metrics import PairedEvaluator, SemanticEvaluator, focus_box_from_heatmap

    evaluator = SemanticEvaluator(device=args.device)
    paired = PairedEvaluator(device=args.device)
    localizer = CLIPSegLocalizer(device=args.device)
    records = []
    for item in items:
        out = prompt_dir(args, item["id"])
        if not os.path.exists(os.path.join(out, "sweep.json")):
            continue
        base, fov, sweep = (load_json(os.path.join(out, f"{n}.json")) for n in ("base", "fov", "sweep"))
        runs = [(base["t_switch"], 14, "", fov["latency_s"]["fov_clipseg"])] + [
            (v["t_switch"], v.get("steps", 14), f"_{k}", v["latency_s"]) for k, v in sweep.items()]
        box = focus_box_from_heatmap(np.load(os.path.join(out, "masks.npz"))["heatmap_full"].astype(np.float32))
        gt_box = focus_box_from_heatmap(np.load(os.path.join(out, "masks_gt.npz"))["heatmap_full"].astype(np.float32))
        draft = Image.open(os.path.join(out, "lr_draft.png")).convert("RGB")
        reference = Image.open(os.path.join(out, "full_hr_28.png")).convert("RGB")
        for t, steps, suffix, latency in sorted(runs):
            for method, b in (("fov_clipseg", box), ("gt_fov_clipseg", gt_box)):
                if not os.path.exists(os.path.join(out, f"{method}{suffix}.png")):
                    continue
                img = Image.open(os.path.join(out, f"{method}{suffix}.png")).convert("RGB")
                row = {"id": item["id"], "category": item["category"], "t_switch": t, "steps": steps, "method": method,
                       **evaluator.evaluate(img, item["prompt"], item["focus_prompt"], b), "focus_musiq": paired.musiq(img.crop(b)),
                       **whole_image_scores(paired, img, b)}
                if method == "fov_clipseg":  # how far Stage 2 moved the image away from its own draft
                    row.update(latency_s=latency, focus_lpips_draft=paired.lpips(img.crop(b), draft.crop(b)), full_lpips_draft=paired.lpips(img, draft))
                else:
                    row.update(focus_lpips_gt=paired.lpips(img.crop(b), reference.crop(b)), full_lpips_gt=paired.lpips(img, reference))
                records.append(row)
        extra_path = os.path.join(out, "extra.json")
        for method, info in (load_json(extra_path) if os.path.exists(extra_path) else {}).items():
            img = Image.open(os.path.join(out, f"{method}.png")).convert("RGB")
            b = focus_box_from_heatmap(localizer.infer_mask(img, item["focus_prompt"]).heatmap_full.cpu().numpy())
            records.append({"id": item["id"], "category": item["category"], "steps": info["steps"], "method": method,
                            **evaluator.evaluate(img, item["prompt"], item["focus_prompt"], b),
                            "focus_musiq": paired.musiq(img.crop(b)), **whole_image_scores(paired, img, b),
                            "latency_s": info["latency_s"]})
        print(f"[sweep_metrics] id={item['id']}", flush=True)

    df = pd.DataFrame(records)
    csv_path = f"results/ablation_{args.prompt_set}_tswitch.csv"
    df.to_csv(csv_path, index=False)
    print(df.groupby(["method", "t_switch", "steps"], dropna=False).median(numeric_only=True).drop(columns="id").round(3).to_string())
    log(f"wrote {csv_path}")


def phase_metrics(args, items):
    import pandas as pd

    from src.semantic_foveation.localizer import CLIPSegLocalizer
    from src.semantic_foveation.metrics import PairedEvaluator, SemanticEvaluator, focus_box_from_heatmap

    evaluator = SemanticEvaluator(device=args.device)
    paired = PairedEvaluator(device=args.device)
    localizer = CLIPSegLocalizer(device=args.device)
    log("evaluation models loaded")

    records = []
    for item in items:
        out = prompt_dir(args, item["id"])
        if not (os.path.exists(os.path.join(out, "base.json")) and os.path.exists(os.path.join(out, "fov.json"))):
            continue
        base, fov = load_json(os.path.join(out, "base.json")), load_json(os.path.join(out, "fov.json"))
        masks = np.load(os.path.join(out, "masks.npz"))
        latency = {**base["latency_s"], **fov["latency_s"]}
        box_draft = focus_box_from_heatmap(masks["heatmap_full"].astype(np.float32))
        obj = masks["object"] if masks["object"].sum() > 0 else masks["clipseg"]

        def common(method, img, box):
            return {"id": item["id"], "category": item["category"], "method": method, "box": json.dumps([int(v) for v in box]),
                    "box_side": int(box[2] - box[0]), **evaluator.evaluate(img, item["prompt"], item["focus_prompt"], box),
                    "focus_musiq": paired.musiq(img.crop(box)), **whole_image_scores(paired, img, box)}

        for method in latency:
            img = Image.open(os.path.join(out, f"{method}.png")).convert("RGB")
            if method.startswith(("full_hr", "uni")):
                # independently generated image: different composition, so it gets its own focus box
                box = focus_box_from_heatmap(localizer.infer_mask(img, item["focus_prompt"]).heatmap_full.cpu().numpy())
            else:
                box = box_draft
            row = {**common(method, img, box), "latency_s": latency[method], "hr_area": base["hr_area"]}
            source = method[4:] if method.startswith("fov_") else ("clipseg" if method == "naive" else None)
            if source is not None:
                m = masks[source]
                row["stage2_tokens"] = int(m.sum() * 4 + (m.size - m.sum()))
                row["object_cover"] = float((m * obj).sum() / obj.sum())
            elif method.startswith(("full_hr", "refine_all")):
                row["stage2_tokens"] = 4096
            records.append(row)

        # Paired protocol: every gt_* image shares the composition of full_hr_28, which is the reference
        gt_masks = np.load(os.path.join(out, "masks_gt.npz"))
        gt_box = focus_box_from_heatmap(gt_masks["heatmap_full"].astype(np.float32))
        reference = Image.open(os.path.join(out, "full_hr_28.png")).convert("RGB")
        records.append({**common("gt_reference", reference, gt_box), "hr_area": base["gt_hr_area"]})
        for method in ["gt_lr_draft", "gt_refine_all"] + [f"gt_fov_{s}" for s in GT_MASK_SOURCES]:
            img = Image.open(os.path.join(out, f"{method}.png")).convert("RGB")
            records.append({**common(method, img, gt_box), "hr_area": base["gt_hr_area"],
                            "focus_lpips_gt": paired.lpips(img.crop(gt_box), reference.crop(gt_box)),
                            "full_lpips_gt": paired.lpips(img, reference)})
        print(f"[metrics] id={item['id']}", flush=True)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)
    df.to_csv(args.results_csv, index=False)
    cols = ["latency_s", "focus_musiq", "focus_clip_iqa", "focus_hf", "focus_clip", "focus_lpips_gt", "full_lpips_gt", "object_cover"]
    print(df.groupby("method", sort=False)[cols].median().round(3).to_string())
    log(f"wrote {args.results_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["encode", "base", "fov", "metrics", "sweep", "sweep_metrics", "base_extra"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation")
    parser.add_argument("--prompt_set", type=str, default="small", choices=list(PROMPT_SETS))
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--draft_factor", type=int, default=4)
    parser.add_argument("--stage1_steps", type=int, default=14)
    parser.add_argument("--stage2_steps", type=int, default=14)
    parser.add_argument("--t_switch", type=float, default=0.85)
    parser.add_argument("--sweep_t", type=float, nargs="+", default=[0.90, 0.95, 1.00], help="Re-noising levels for --phase sweep")
    parser.add_argument("--sweep_steps", type=int, nargs="+", default=[14], help="Stage-2 step counts for --phase sweep")
    parser.add_argument("--extra_full_steps", type=int, nargs="+", default=[6, 5, 4], help="1024^2 step counts for --phase base_extra")
    parser.add_argument("--extra_768_steps", type=int, nargs="+", default=[8], help="768^2 step counts for --phase base_extra")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--dilation_blocks", type=int, default=2)
    parser.add_argument("--vram_fraction", type=float, default=0.9, help="Hard cap on this process's share of GPU memory")
    args = parser.parse_args()
    if args.results_csv is None:
        args.results_csv = f"results/ablation_{args.prompt_set}_metrics.csv"

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction, torch.device(args.device))
    items = PROMPT_SETS[args.prompt_set][: args.num_prompts]
    with torch.no_grad():
        {"encode": phase_encode, "base": phase_base, "fov": phase_fov, "metrics": phase_metrics,
         "sweep": phase_sweep, "sweep_metrics": phase_sweep_metrics, "base_extra": phase_base_extra}[args.phase](args, items)


if __name__ == "__main__":
    main()
