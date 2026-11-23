# Ablations (Joon)

Running log of ablations on Semantic Foveated Diffusion. Newest experiment at the bottom.

**Setup.** FLUX.2-klein-base-4B, 1024² output, one seed per prompt, RTX 5080. Small-object set: 60 prompts, `t_switch` 0.85. Large-subject set: 12 prompts, `t_switch` 0.95. Quality is median MUSIQ on the focus crop (higher is sharper). "Ours wins" is the share of prompts where our focus crop scores higher.

## Conclusion so far

- **Mask:** at equal tokens, focus-prompt placement beats text-free masks on 70–85% of small-object prompts.
- **Settings:** use `t_switch` 0.95 and 6 Stage 2 steps. 0.85 regenerates too little, 1.00 loses the layout, and steps beyond 6 add nothing.
- **Speed, small objects:** 3.8 s for the focus quality of 8-step full resolution (6.5 s). 5.9× faster than 28 steps, which stays about 2 MUSIQ points ahead.
- **The gain is in the focus region only.** Our periphery scores 52 against 64 for 8-step full resolution, and whole-image quality equals 768² at 8 steps (3.6 s). Foveation moves quality to the focus; it does not add any overall.
- **Baselines:** full resolution collapses below 8 steps on small objects, so it cannot match our focus quality by cutting steps. On large subjects it holds up down to 4–5 steps.
- **Large subjects:** no advantage. Ours with 6 steps takes 5.3 s; 768² at 8 steps reaches the same quality in 3.6 s.

## 1. Fair compute

| Method | Latency (s) | MUSIQ small | Ours wins (small) | MUSIQ large |
|---|---|---|---|---|
| Full resolution, 28 steps | 22.4 | 71.3 | 7% | 75.3 |
| Full resolution, 14 steps | 11.3 | 70.8 | 8% | 75.2 |
| Full resolution, 8 steps | 6.5 | 67.5 | 7% | 76.2 |
| 768² upsampled, 14 steps | 6.2 | 59.9 | 10% | 74.9 |
| 512² upsampled, 28 steps | 6.2 | 51.0 | 28% | 66.9 |
| 512² upsampled, 14 steps | 3.2 | 50.3 | 37% | 68.4 |
| **Ours** | 6.3 small / 9.8 large | **45.1** | – | **74.2** |
| Ours, standard LoRA | same as ours | 44.9 | 42% | 71.3 |
| Draft + Stage 2 on all 4,096 tokens | 13.0 | 31.7 | 95% | 70.9 |
| 256² draft only | 1.6 | 28.3 | 93% | 54.2 |

- Full resolution loses only 4 MUSIQ points from 28 to 8 steps, so few-step full resolution is the baseline to beat.
- All-token refinement at `t_switch` 0.85 barely changes the draft; at 0.95 it works.
- The foveation LoRA has no measurable effect on small objects and adds about 3 points on large subjects.

Figures: `assets/ablation_pareto_small.png`, `assets/ablation_pareto_large.png`.

## 2. Mask source at equal token budget

| Mask | Object coverage (small) | MUSIQ small | Focus prompt wins (small) | MUSIQ large |
|---|---|---|---|---|
| **Focus prompt (ours)** | 100% | **45.1** | – | 74.2 |
| Generic prompt ("the main subject of the photo") | 70% | 37.3 | 70% | 73.8 |
| Draft detail (Laplacian energy) | 32% | 36.7 | 85% | 73.4 |
| Center disc | 33% | 38.9 | 80% | 73.8 |
| Random disc | 17% | 34.7 | 85% | 73.9 |

- When the generic mask happens to cover the object, it scores the same as ours (median gap 0.5). The gain comes from coverage.
- On large subjects every mask covers most of the subject and the differences vanish.

Figures: `assets/ablation_masks_small.png`, `assets/ablation_masks_large.png`.

## 3. Paired metric

Draft = the 28-step full-resolution image downsampled to 256², so that image is a same-composition reference. LPIPS to the reference, focus crop, lower is closer. Small set.

| Image | Focus LPIPS | Focus MUSIQ |
|---|---|---|
| Downsampled reference, upsampled back | 0.439 | 27.6 |
| + Stage 2 on all tokens | 0.527 | 29.3 |
| + foveated Stage 2, center mask | 0.498 | 39.5 |
| + foveated Stage 2, focus-prompt mask (ours) | 0.448 | 41.8 |

- With the right mask Stage 2 adds detail without drifting from the reference; ours is closer than the center mask on 92% of prompts.
- On the large set (`t_switch` 0.95) every refinement lands further from the reference than the draft (0.22 → 0.30). Stage 2 regenerates there, so LPIPS measures drift, not quality.

## 4. `t_switch` sweep on the small set

Ours on the same 60 drafts and masks, changing only the re-noising level. Latency is 6.3 s at every setting. Baselines at that latency: 768² upsampled 59.9 (6.2 s), full resolution at 8 steps 67.5 (6.5 s).

| `t_switch` | Focus MUSIQ | Beats 768² upsampled | Beats full res, 8 steps | Focus CLIP | Focus LPIPS to own draft |
|---|---|---|---|---|---|
| 0.85 | 45.1 | 10% | 7% | 30.2 | 0.31 |
| 0.90 | 54.1 | 28% | 18% | 30.5 | 0.42 |
| **0.95** | **67.7** | **70%** | **52%** | 29.9 | 0.65 |
| 1.00 | 65.0 | 60% | 47% | 26.9 | 0.85 |

- 0.95 beats 0.85 on all 60 prompts. Against 8-step full resolution it is a tie (52% wins, p = 0.8); against 768² upsampled it wins (70%, p = 0.001). 28-step full resolution is still ahead at 71.3.
- At 1.00 Stage 2 starts from pure noise and the object often lands outside the mask: focus CLIP falls by more than 3 points on 28 of 60 prompts, against 4 of 60 at 0.95.
- At 0.95 Stage 2 keeps the layout but regenerates the object (LPIPS to the draft 0.65), so the draft fixes where things are, not what they look like. In the paired protocol, focus LPIPS to the reference rises from 0.448 to 0.513.

Summary table: `scripts/summarize_ablations.py` prints it when `results/ablation_small_tswitch.csv` exists.

## 5. Stage 2 step sweep at `t_switch` 0.95

Ours with 6, 8 and 10 Stage 2 steps on the same drafts and masks (draft still 14 steps), against few-step baselines of similar latency. Small set, 60 prompts.

| Method | Latency (s) | Focus MUSIQ | Focus CLIP | Ours at similar latency wins |
|---|---|---|---|---|
| 512² upsampled, 14 steps | 3.2 | 50.3 | 30.5 | 95% |
| Full resolution, 4 steps | 3.4 | 41.6 | 25.9 | 82% |
| 768² upsampled, 8 steps | 3.6 | 60.4 | 30.2 | 77% |
| **Ours, 6 Stage 2 steps** | **3.8** | **69.1** | 30.1 | – |
| Full resolution, 5 steps | 4.2 | 53.9 | 28.5 | 75% |
| Ours, 8 Stage 2 steps | 4.4 | 69.0 | 29.8 | – |
| Full resolution, 6 steps | 5.0 | 61.8 | 29.3 | 72% |
| Ours, 10 Stage 2 steps | 5.0 | 67.5 | 29.9 | – |
| 768² upsampled, 14 steps | 6.2 | 59.9 | 30.6 | 70% |
| Ours, 14 Stage 2 steps | 6.3 | 67.7 | 29.9 | – |
| Full resolution, 8 steps | 6.5 | 67.5 | 30.7 | 52% (p = 0.8) |
| Full resolution, 28 steps | 22.4 | 71.3 | 29.9 | – |

- Cutting Stage 2 from 14 to 6 steps costs nothing in the focus crop (69.1 against 67.7) and brings total latency to 3.8 s.
- Full resolution degrades fast below 8 steps: 61.8, 53.9 and 41.6 at 6, 5 and 4 steps. At 4 steps the object is often not formed (focus CLIP 25.9).
- Ours at 3.8 s is level with 8-step full resolution at 6.5 s (wins 45%, p = 0.9). Against 28-step full resolution it wins 40% of prompts (p = 0.24).
- Under 5 s, ours is the best method tested; every comparison in the last column except the 8-step one is significant at p < 0.01.

## 6. Whole-image quality, and large subjects with shorter Stage 2

Ours at `t_switch` 0.95. Whole image = MUSIQ on the full 1024² output. Periphery = MUSIQ on the 512² corner crop farthest from the focus box.

Small objects, 60 prompts:

| Method | Latency (s) | Focus | Whole image | Periphery |
|---|---|---|---|---|
| 512² upsampled, 14 steps | 3.2 | 50.3 | 62.4 | 54.8 |
| 768² upsampled, 8 steps | 3.6 | 60.4 | 68.3 | 58.4 |
| **Ours, 6 Stage 2 steps** | **3.8** | **69.1** | 67.8 | 52.2 |
| Full resolution, 6 steps | 5.0 | 61.8 | 68.2 | 59.3 |
| Full resolution, 8 steps | 6.5 | 67.5 | 70.8 | 63.7 |
| Full resolution, 28 steps | 22.4 | 71.3 | 72.6 | 66.9 |

Large subjects, 12 prompts:

| Method | Latency (s) | Focus | Whole image | Periphery |
|---|---|---|---|---|
| Full resolution, 4 steps | 3.4 | 71.0 | 73.2 | 64.6 |
| 768² upsampled, 8 steps | 3.6 | 74.3 | 74.1 | 70.5 |
| Full resolution, 6 steps | 5.0 | 75.2 | 75.5 | 69.2 |
| **Ours, 6 Stage 2 steps** | 5.3 | 75.6 | 75.4 | 66.6 |
| Full resolution, 8 steps | 6.5 | 76.2 | 75.9 | 71.6 |
| Ours, 14 Stage 2 steps | 9.8 | 74.2 | 74.2 | 65.3 |
| Full resolution, 28 steps | 22.4 | 75.3 | 75.4 | 73.8 |

- Small objects: ours has the best focus score under 6 s and the lowest periphery score of the sharp methods, about level with uniform 512². On the whole image it ties 768² at 8 steps (wins 40%, p = 0.1) and loses to 8-step full resolution (wins 27%, p < 0.001).
- Large subjects: 6 Stage 2 steps are again enough (75.6 against 74.2 at 14 steps) and cut latency from 9.8 to 5.3 s, but every method from 3.6 s up scores 74–76, so there is nothing left for foveation to win.
- Few-step full resolution fails on small objects (41.6 at 4 steps) but not on large subjects (71.0 at 4 steps). Small objects are the case that needs more steps at full resolution, and the case where foveation pays off.
- On the large set the focus box is usually the whole image, so "periphery" there overlaps the subject.

## Caveats

- MUSIQ is a proxy; no human ratings. Focus crops are ~300 px wide on the small set. CLIP-IQA is unreliable here (it ranks the bare draft at the top on the large set).
- One seed per prompt; one timed run per image. The large set has only 12 prompts.
- The two sets differ in both subject size and `t_switch`.

## Reproduce

```bash
# once per phase: encode, base, fov, metrics; then sweep, base_extra, sweep_metrics for experiments 4 and 5
scripts/run_capped.sh .venv/bin/python scripts/run_ablation_benchmark.py --phase base --prompt_set small
scripts/run_capped.sh .venv/bin/python scripts/run_ablation_benchmark.py --phase base --prompt_set large --t_switch 0.95
.venv/bin/python scripts/summarize_ablations.py --csv results/ablation_small_metrics.csv --tag small
```

Per-prompt rows: `results/ablation_{small,large}_metrics.csv` (experiments 1–3), `results/ablation_{small,large}_tswitch.csv` (experiments 4–6).
