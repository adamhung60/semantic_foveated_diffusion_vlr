# Results

**Semantic Foveated Diffusion generates a 1024² image 3.4× faster than full-resolution generation for small subjects (3.7 s vs 12.6 s, all stages included), and about 2.3× faster for large foreground subjects. The focus region is clearly sharper than in the low-res draft, but still less detailed than in full-resolution generation.**

## Setup

- **Model:** FLUX.2-klein-base-4B with the pretrained Foveated Diffusion `fov_random` LoRA (no training). CFG 4, with the negative prompt "blurry, out of focus, low resolution, …" for all methods.
- **Benchmarks:**
  - *Small objects (main):* 60 wide scenes, each with one small object that matters (15 each of animals, objects, people, structures), e.g. *"a wide snowy mountain landscape, a skier in a red jacket…"* with focus prompt *"a skier in a red jacket"*. See `data/small_object_benchmark_60.csv`.
  - *Large subjects:* 12 close-ups where the subject fills much of the frame against a plain background (portrait, cat, macaw, …). See `data/large_subject_12.csv`.
- **Methods:**

| Method | What it does |
|---|---|
| **Full HR** (main baseline) | Standard 1024² generation, 28 steps, 4096 tokens |
| **Low-res draft** | Stage 1 only: 256² draft (14 steps, 256 tokens), bicubic-upsampled |
| **Ours** | Draft → CLIPSeg mask (threshold 0.5, dilated 2 blocks) → foveated Stage 2 from t=0.85, or 0.95 for large subjects (14 steps): full-res tokens in the mask, ½-res tokens elsewhere |
| Naive mixed-res (ablation) | Ours without the Foveated LoRA, i.e. tests the original Foveated Diffusion training |

- **Focus-region metrics.** Each image gets a square crop around the focus object, found by CLIPSeg on that image. Full HR needs its own crop because it has a different composition.
  - *CLIP-IQA*: probability that CLIP prefers "a sharp, detailed photo" over "a blurry, low-resolution photo".
  - *HF energy*: mean squared Laplacian, a measure of fine detail.
  - *Focus CLIP*: similarity between the crop and the focus prompt, i.e. whether the object is still correct.
- **Latency:** wall-clock on one idle RTX 5090, median of 5 runs × 5 prompts, counting every stage (draft, mask, Stage 2, VAE encode and decode).

## Small objects (main benchmark)

| Method | Latency | Speedup | Stage-2 tokens | Focus CLIP-IQA ↑ | Focus HF energy ↑ | Focus CLIP ↑ |
|---|---|---|---|---|---|---|
| Full HR | 12.57 s | 1.0× | 4096 | **0.57** | **16.6k** | 29.9 |
| Low-res draft | 1.01 s | 12.4× | – | 0.38 | 0.6k | 29.3 |
| **Ours** | **3.73 s** | **3.4×** | 1374 (median) | 0.46 | 2.1k | 29.9 |
| Naive mixed-res | ≈3.7 s | ≈3.4× | 1374 | 0.40 | 2.1k | 30.2 |

Ours' latency breaks down as a 1.01 s draft, a 0.01 s mask, and a 2.71 s Stage 2. The mask covers a median of 11% of the image.

**Paired comparisons (fraction of the 60 prompts where ours scores higher):**

| Ours vs. | CLIP-IQA | HF energy |
|---|---|---|
| Low-res draft | 70% | 95% |
| Naive mixed-res | 75% | 48% |
| Full HR | 35% | 0% |

Each example shows a row of full images, then a row of zoomed crops of the focus object. Yellow boxes mark the crop; the cyan outline on *Ours* marks the full-resolution token region.

![Qualitative comparison](assets/qualitative_comparison.png)

## Large foreground subjects

The mask now covers a median of 54% of the image (2695 Stage-2 tokens), so less is saved. These runs use `t_switch = 0.95`; at 0.85, Stage 2 added no detail to large subjects (focus HF energy 0.20k vs 0.20k for the draft).

| Method | Latency (est.) | Speedup | Focus HF energy ↑ | Focus CLIP-IQA |
|---|---|---|---|---|
| Full HR | 12.57 s | 1.0× | **6.7k** | 0.58 |
| Low-res draft | 1.01 s | 12.4× | 0.2k | 0.64 |
| **Ours** | **≈5.6 s** | **≈2.3×** | 2.4k | 0.57 |
| Naive mixed-res | ≈5.6 s | ≈2.3× | 1.8k | 0.53 |

- Ours has more fine detail than the draft on 12/12 prompts and more than naive on 9/12.
- Latency is estimated, because the GPU was shared during these runs. Per-step cost is linear in token count across the three clean measurements (256, 1374 and 4096 tokens, within 6%), and the estimate applies that to the median token count. It ranges from ≈2.0× (76% mask) to ≈3.0× (24% mask).
- CLIP-IQA is not reliable on these large, smooth crops: it rates the blurry draft highest.

Zoomed crops here are 320 px detail crops at the subject center, since the whole subject spans most of the image.

![Large-subject comparison](assets/large_subject_comparison.png)

## Takeaways

- **Speed:** 3.4× faster than full HR for small objects, ≈2.3× for large subjects. The saving scales with how much of the image the background takes up. Stage 2 dominates the cost; CLIPSeg is negligible.
- **Quality vs. the draft:** Stage 2 adds real detail to the focus region: the skier's limbs and poles, the watch's numerals, cat fur, and macaw feathers. Plain backgrounds look fine at ½ resolution.
- **Quality vs. full HR:** full HR is still clearly sharper (about 8× higher HF energy). Stage 2 starts from a blurry 256² draft and only partially re-noises it, so it cannot fully recover high-frequency detail. The object is semantically just as correct (focus CLIP is equal).
- **Foveated LoRA vs. naive:** similar fine detail, but crops from the LoRA look less blurry to CLIP-IQA on 75% of prompts.
- **Failure mode:** when CLIPSeg misses the object (e.g. a mountain goat in a rocky slope), the mask spreads over a large area. That example took 75% of the image, which costs speed and doesn't help the object.

## Design choices

- **`t_switch` depends on subject size:**
  - *Small objects use 0.85.* At 0.9 and above, the object's content drifts (e.g. the pocket watch turned into a different object).
  - *Large subjects need 0.95.* Their structure is already fixed in the draft, so lower values only reproduce its blur.
  - A simple automatic rule would pick `t_switch` from the mask area.
- **Negative prompt:** without it, Stage 2 keeps the draft's blur. On 4 test prompts, it raised ours' focus HF energy from 0.8k to 4.9k (`scripts/ablation_negative_prompt.py`).

Raw numbers: `results/benchmark_metrics.csv`, `results/large_subject_metrics.csv` (per prompt), and `results/timing.csv`.
