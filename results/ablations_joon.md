# Ablations (Joon)

Running log of ablations on Semantic Foveated Diffusion. Newest experiment at the bottom.

**Setup.** FLUX.2-klein-base-4B, 1024² output, one seed per prompt, RTX 5080. Small-object set: 60 prompts, `t_switch` 0.85. Large-subject set: 12 prompts, `t_switch` 0.95. Quality is median MUSIQ on the focus crop (higher is sharper). "Ours wins" is the share of prompts where our focus crop scores higher.

## Conclusion so far

- **The semantic mask is justified.** At a fixed token budget, placing high-resolution tokens with the focus prompt beats generic, detail, center and random masks on 70–85% of small-object prompts.
- **The speed claim does not hold against equal-compute baselines.** Ours is 3.6× faster than 28-step full resolution, but full resolution at 8 steps or a 768² image upsampled takes the same 6.3 s and is sharper on about 90% of small-object prompts.
- **On large subjects quality matches full resolution, latency does not.** The mask covers 58% of the image, so ours takes 9.8 s; full resolution at 8 steps takes 6.5 s at the same score.
- **The limit looks like Stage 2 strength, not token count.** At `t_switch` 0.85 Stage 2 adds little detail to a 256² draft. Experiment 4 tests this.

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

In progress: ours at `t_switch` 0.90, 0.95 and 1.00, reusing the cached drafts and masks.

## Caveats

- MUSIQ on ~300 px crops is a proxy; no human ratings. CLIP-IQA is unreliable here (it ranks the bare draft at the top on the large set).
- One seed per prompt; one timed run per image. The large set has only 12 prompts.
- The two sets differ in both subject size and `t_switch`.

## Reproduce

```bash
# once per phase: encode, base, fov, metrics (then sweep, sweep_metrics for experiment 4)
scripts/run_capped.sh .venv/bin/python scripts/run_ablation_benchmark.py --phase base --prompt_set small
scripts/run_capped.sh .venv/bin/python scripts/run_ablation_benchmark.py --phase base --prompt_set large --t_switch 0.95
.venv/bin/python scripts/summarize_ablations.py --csv results/ablation_small_metrics.csv --tag small
```

Per-prompt rows: `results/ablation_{small,large}_metrics.csv`, `results/ablation_small_tswitch.csv`.
