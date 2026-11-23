"""Resumable ablation runner (T4 / fp16). One PNG + JSON per (variant, prompt) under --out/<variant>/.

Latency always counts every stage the variant needs (draft, mask, optional SR, Stage 2 incl. VAE encode/decode).
Drafts are cached per (prompt, draft_steps) and reused across variants; their measured time is added back.

Usage:
  python scripts/run_ablations.py --variants fhr7 ours_t95_r7 --ids 0 2 4
  python scripts/run_ablations.py --group core --subset 30
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic_foveation.ablation import AblationPipeline
from src.semantic_foveation.benchmark import PROMPT_SETS


# ---------------------------------------------------------------------------------------------- variants
def S(**kw):
    """Stage-2 variant spec with the session-1 recommended defaults (t 0.95, 7 steps, 14-step 256^2 draft)."""
    base = dict(kind="stage2", draft_steps=14, t=0.95, steps=7, lora="random", mask="clipseg", init="bicubic",
                noise="independent", shift=None, cfg_off_last=0, reuse=())
    base.update(kw)
    return base


def FH(steps, **kw):
    return dict(kind="full_hr", steps=steps, cfg_off_last=kw.get("cfg_off_last", 0), reuse=kw.get("reuse", ()))


VARIANTS = {
    # baselines / latency-quality curve
    "fhr5": FH(5), "fhr7": FH(7), "fhr10": FH(10), "fhr14": FH(14), "fhr28": FH(28),
    "draft": dict(kind="draft", draft_steps=14, init="bicubic"),
    "draft_esr": dict(kind="draft", draft_steps=14, init="esrgan"),
    "ours_t85_r14": S(t=0.85, steps=14),  # repo default
    "ours_t95_r7": S(),                   # session-1 recommended (reference for most paired tests)
    "ours_t95_r5": S(steps=5),
    "ours_t95_r10": S(steps=10),
    "ours_t95_r14": S(steps=14),
    "ours_d5_t95_r7": S(draft_steps=5),
    "naive_t95_r7": S(lora="no_fov"),
    # (1) why is every-token-full-res worse?
    "allhr_fov_t95_r7": S(mask="all"),
    "allhr_nofov_t95_r7": S(mask="all", lora="no_fov"),
    "allhr_base_t95_r7": S(mask="all", lora=None),
    "allhr_fov_esr_t95_r7": S(mask="all", init="esrgan"),
    "ours_shiftL_t95_r7": S(shift="tokens"),
    # (2) correlated LR/HR noise (null check)
    "ours_corr_t95_r7": S(noise="correlated"),
    # (3) speed: CFG truncation and residual caching (also applied to Full HR)
    "ours_cfgoff2_t95_r7": S(cfg_off_last=2),
    "ours_cache_t95_r7": S(reuse=(3, 5)),
    "ours_cache_t95_r10": S(steps=10, reuse=(3, 5, 7)),
    "ours_cache_cfgoff2_t95_r10": S(steps=10, reuse=(3, 5, 7), cfg_off_last=2),
    "fhr7_cfgoff2": FH(7, cfg_off_last=2),
    "fhr10_cache": FH(10, reuse=(3, 5, 7)),
    "fhr14_cache": FH(14, reuse=(4, 6, 8, 10)),
    # (4) super-resolve the draft before re-encoding
    "ours_esr_t95_r7": S(init="esrgan"),
    "ours_esr_cache_cfgoff2_t95_r10": S(init="esrgan", steps=10, reuse=(3, 5, 7), cfg_off_last=2),
    "ours_esr_t85_r7": S(init="esrgan", t=0.85),
    "ours_esr_t90_r7": S(init="esrgan", t=0.90),
    "ours_esr_t85_r14": S(init="esrgan", t=0.85, steps=14),
    # (5) mask source at equal token budget, and token-budget sweep
    "mask_random_t95_r7": S(mask="random_box"),
    "mask_center_t95_r7": S(mask="center_box"),
    "mask_none_t95_r7": S(mask="none"),
    "topk05_t95_r7": S(mask="topk:0.05"),
    "topk25_t95_r7": S(mask="topk:0.25"),
    "topk50_t95_r7": S(mask="topk:0.50"),
}

GROUPS = {
    "core": ["draft", "ours_t95_r7", "fhr7", "allhr_fov_t95_r7", "allhr_nofov_t95_r7", "allhr_base_t95_r7",
             "naive_t95_r7", "ours_esr_t95_r7", "ours_esr_t85_r7", "mask_random_t95_r7", "mask_center_t95_r7",
             "mask_none_t95_r7", "ours_corr_t95_r7", "ours_shiftL_t95_r7", "ours_cfgoff2_t95_r7",
             "ours_cache_t95_r7", "ours_t85_r14", "draft_esr"],
    "curve": ["fhr5", "fhr10", "fhr14", "ours_t95_r5", "ours_t95_r10", "ours_t95_r14", "ours_d5_t95_r7",
              "ours_cache_t95_r10", "ours_cache_cfgoff2_t95_r10", "fhr7_cfgoff2", "fhr10_cache", "fhr14_cache",
              "topk05_t95_r7", "topk25_t95_r7", "topk50_t95_r7", "ours_esr_t90_r7", "allhr_fov_esr_t95_r7",
              "ours_esr_t85_r14"],
    "fhr28": ["fhr28"],
}

MASK_KW = dict(selection_mode="threshold", threshold=0.5, dilation_blocks=2)


# ---------------------------------------------------------------------------------------------- masks
def low_to_token_grid(mask_low: torch.Tensor) -> torch.Tensor:
    return F.interpolate(mask_low[None, None].float(), size=(64, 64), mode="nearest")[0, 0]


def box_mask(k: int, cy: float, cx: float, n: int = 32) -> torch.Tensor:
    a = max(1, int(round(math.sqrt(k))))
    b = max(1, int(round(k / a)))
    y0 = int(np.clip(round(cy - a / 2), 0, n - a))
    x0 = int(np.clip(round(cx - b / 2), 0, n - b))
    m = torch.zeros(n, n)
    m[y0:y0 + a, x0:x0 + b] = 1
    return m


def build_mask(pipe, spec_mask, draft_low, focus, pid):
    """Returns (mask_low [32x32] or None for all-HR, info). Equal-budget controls match CLIPSeg's block count."""
    t0 = time.perf_counter()
    ref = pipe.localizer.infer_mask(draft_low, focus, **MASK_KW)
    t_mask = time.perf_counter() - t0
    ref_low = ref.mask_low.cpu()
    k = int(ref_low.sum())
    if spec_mask == "clipseg":
        m = ref_low
    elif spec_mask == "all":
        return None, {"mask_s": 0.0, "k_ref": k}
    elif spec_mask == "none":
        m = torch.zeros(32, 32)
    elif spec_mask == "center_box":
        m = box_mask(k, 16, 16)
    elif spec_mask == "random_box":
        rng = np.random.default_rng(1234 + pid)
        dil = F.max_pool2d(ref_low[None, None], 5, 1, 2)[0, 0]
        best, best_ov = None, 1e9
        for _ in range(300):  # random placement that avoids the object (dilated CLIPSeg mask) as much as possible
            cand = box_mask(k, rng.uniform(0, 32), rng.uniform(0, 32))
            ov = float((cand * dil).sum())
            if ov < best_ov:
                best, best_ov = cand, ov
            if ov == 0:
                break
        m = best
    elif spec_mask.startswith("topk:"):
        t0 = time.perf_counter()
        r = pipe.localizer.infer_mask(draft_low, focus, selection_mode="top_k", budget_ratio=float(spec_mask[5:]),
                                      dilation_blocks=2)
        t_mask = time.perf_counter() - t0
        m = r.mask_low.cpu()
    else:
        raise ValueError(spec_mask)
    return m, {"mask_s": t_mask, "k_ref": k}


def seam_disagreement(hr: Image.Image, lr: Image.Image, mask_low: torch.Tensor, band_px: int = 24) -> float:
    """Mean |blur(HR decode) - blur(LR decode)| (0-255) within band_px of the HR/LR boundary.

    Both decodes come from the same final latent sequence; where they disagree at LR scale the blend shows a seam.
    """
    m = cv2.resize(mask_low.numpy().astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    if m.min() == m.max():
        return float("nan")
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    band = (np.maximum(dist_in, dist_out) <= band_px)
    a = cv2.GaussianBlur(np.asarray(hr, dtype=np.float32), (0, 0), 3)
    b = cv2.GaussianBlur(np.asarray(lr, dtype=np.float32), (0, 0), 3)
    return float(np.abs(a - b).mean(axis=2)[band].mean())


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--group", nargs="*", default=[])
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--subset", type=int, default=30, help="use every (60/subset)-th small-object prompt")
    ap.add_argument("--prompt_set", default="small")
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--out", default="outputs/abl")
    args = ap.parse_args()

    variants = list(args.variants)
    for g in args.group:
        variants += [v for v in GROUPS[g] if v not in variants]
    items = PROMPT_SETS[args.prompt_set]
    if args.ids is not None:
        items = [p for p in items if p["id"] in args.ids]
    elif args.prompt_set == "small":
        stride = max(1, len(items) // args.subset)
        items = items[::stride][: args.subset]
    print(f"{len(variants)} variants x {len(items)} prompts -> {args.out}", flush=True)

    pipe = AblationPipeline()
    for m in ["no_fov", "random"]:
        pipe._preload_lora_deltas(m)
    # warmup (CUDA kernels, cuDNN autotune for VAE)
    it0 = items[0]
    cp = pipe.encode_prompt_cached(it0["prompt"])
    s1 = pipe.run_stage1_draft(it0["prompt"], seed=0, draft_factor=4, stage1_steps=2, cached_prompts=cp)
    mk = pipe.localizer.infer_mask(s1.draft_image_low, it0["focus_prompt"], **MASK_KW)
    pipe.stage2_ext(cp, s1.draft_image_low, mk.mask_token_grid, steps=2)
    pipe.full_hr_ext(cp, seed=0, steps=1)

    draft_dir = os.path.join(args.out, "_drafts")
    os.makedirs(draft_dir, exist_ok=True)

    def get_draft(item, steps, cp, seed):
        tag = f"d{steps}_s{seed}_{item['id']:03d}"
        png, js = os.path.join(draft_dir, tag + ".png"), os.path.join(draft_dir, tag + ".json")
        if os.path.exists(png) and os.path.exists(js):
            return Image.open(png).convert("RGB"), json.load(open(js))["stage1_s"]
        s1 = pipe.run_stage1_draft(item["prompt"], seed=seed, draft_factor=4, stage1_steps=steps, cached_prompts=cp)
        s1.draft_image_low.save(png)
        json.dump({"stage1_s": s1.stage1_time_s}, open(js, "w"))
        return s1.draft_image_low, s1.stage1_time_s

    sr_cache = {}

    def get_sr(key, draft_low):
        if key not in sr_cache:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hr = pipe.sr_upscale(draft_low)
            torch.cuda.synchronize()
            sr_cache.clear()
            sr_cache[key] = (hr, time.perf_counter() - t0)
        return sr_cache[key]

    t_start = time.time()
    n_done = 0
    for item in items:
        pid, prompt, focus = item["id"], item["prompt"], item["focus_prompt"]
        seed = item["seed"] + args.seed_offset
        cp = pipe.encode_prompt_cached(prompt)
        for v in variants:
            spec = VARIANTS[v]
            vdir = os.path.join(args.out, v)
            os.makedirs(vdir, exist_ok=True)
            png, js = os.path.join(vdir, f"{pid:03d}.png"), os.path.join(vdir, f"{pid:03d}.json")
            if os.path.exists(png) and os.path.exists(js):
                continue
            meta = {"id": pid, "variant": v, "seed": seed, "category": item["category"]}
            if spec["kind"] == "full_hr":
                img, t, info = pipe.full_hr_ext(cp, seed=seed, steps=spec["steps"], cfg_off_last=spec["cfg_off_last"],
                                                reuse_steps=spec["reuse"])
                meta.update(latency_s=t, stage2_s=t, stage1_s=0.0, mask_s=0.0, sr_s=0.0, **info)
            elif spec["kind"] == "draft":
                draft_low, t1 = get_draft(item, spec["draft_steps"], cp, seed)
                if spec["init"] == "esrgan":
                    img, tsr = get_sr((pid, spec["draft_steps"], seed), draft_low)
                else:
                    img, tsr = draft_low.resize((1024, 1024), Image.Resampling.BICUBIC), 0.0
                meta.update(latency_s=t1 + tsr, stage1_s=t1, sr_s=tsr, mask_s=0.0, stage2_s=0.0, tokens=0, nfe=0)
            else:
                draft_low, t1 = get_draft(item, spec["draft_steps"], cp, seed)
                mask_low, minfo = build_mask(pipe, spec["mask"], draft_low, focus, pid)
                mask_tok = None if mask_low is None else low_to_token_grid(mask_low).to(pipe.device)
                init_hr = init_lr = None
                tsr = 0.0
                if spec["init"] == "esrgan":
                    init_hr, tsr = get_sr((pid, spec["draft_steps"], seed), draft_low)
                    init_lr = init_hr.resize((512, 512), Image.Resampling.LANCZOS)
                shift = None
                if spec["shift"] == "tokens" and mask_tok is not None:
                    hr = int(mask_tok.sum().item())
                    shift = hr + (4096 - hr) // 4
                img, t2, info = pipe.stage2_ext(
                    cp, draft_low, mask_tok, seed=seed, steps=spec["steps"], t_switch=spec["t"],
                    lora_mode=spec["lora"], init_hr=init_hr, init_lr=init_lr, noise_mode=spec["noise"],
                    shift_len=shift, cfg_off_last=spec["cfg_off_last"], reuse_steps=spec["reuse"],
                    return_parts=mask_tok is not None)
                seam = float("nan")
                if "hr_decode" in info:
                    seam = seam_disagreement(info.pop("hr_decode"), info.pop("lr_decode"), mask_low)
                hr_area = 1.0 if mask_low is None else float(mask_low.mean())
                meta.update(latency_s=t1 + minfo["mask_s"] + tsr + t2, stage1_s=t1, mask_s=minfo["mask_s"],
                            sr_s=tsr, stage2_s=t2, hr_area=hr_area, k_ref=minfo["k_ref"], seam=seam, **info)
                if mask_low is not None:
                    np.save(os.path.join(vdir, f"{pid:03d}_mask.npy"), mask_low.numpy().astype(np.uint8))
            img.save(png)
            json.dump(meta, open(js, "w"))
            n_done += 1
            print(f"[{time.time() - t_start:7.0f}s] id={pid:03d} {v:28s} lat={meta['latency_s']:6.2f}s "
                  f"tok={meta.get('tokens', 0)} nfe={meta.get('nfe', 0)}", flush=True)
    print(f"RUN_DONE {n_done} new outputs in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
