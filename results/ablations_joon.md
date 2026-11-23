# Ablations (Joon)

Running log of ablations on Semantic Foveated Diffusion. Newest experiment at the bottom.

**Setup.** FLUX.2-klein-base-4B, 1024² output, one seed per prompt, RTX 5080. Small-object set: 60 prompts, `t_switch` 0.85. Large-subject set: 12 prompts, `t_switch` 0.95. Quality is median MUSIQ on the focus crop (higher is sharper). "Ours wins" is the share of prompts where our focus crop scores higher.

## Conclusion so far

- **The semantic mask is justified.** At a fixed token budget, placing high-resolution tokens with the focus prompt beats generic, detail, center and random masks on 70–85% of small-object prompts.
- **With the right settings, foveation is faster at equal focus quality on small objects.** At `t_switch` 0.95 and 6 Stage 2 steps, ours takes 3.8 s and matches 8-step full resolution (6.5 s) in the focus crop. That is 1.7× faster than the best equal-quality baseline and 5.9× faster than 28-step full resolution, which stays about 2 MUSIQ points ahead.
- **The original settings hid this.** At `t_switch` 0.85 and 14 Stage 2 steps, ours lost to every equal-latency baseline. 0.85 regenerates too little; 1.00 loses the layout; 14 steps is more than Stage 2 needs.
- **Full resolution cannot simply use fewer steps.** It holds up down to 8 steps and collapses below that, while Stage 2 is unchanged at 6.
- **No advantage on large subjects so far.** The mask covers 58% of the image, so ours took 9.8 s at 14 Stage 2 steps against 6.5 s for 8-step full resolution. Not yet re-run with 6 steps.
- **Open:** all scores are on the focus crop. Our periphery is generated at 512²-equivalent resolution and is not scored.

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

## Caveats

- MUSIQ on ~300 px crops is a proxy; no human ratings. CLIP-IQA is unreliable here (it ranks the bare draft at the top on the large set).
- One seed per prompt; one timed run per image. The large set has only 12 prompts.
- The two sets differ in both subject size and `t_switch`.

## Reproduce

```bash
# once per phase: encode, base, fov, metrics; then sweep, base_extra, sweep_metrics for experiments 4 and 5
scripts/run_capped.sh .venv/bin/python scripts/run_ablation_benchmark.py --phase base --prompt_set small
scripts/run_capped.sh .venv/bin/python scripts/run_ablation_benchmark.py --phase base --prompt_set large --t_switch 0.95
.venv/bin/python scripts/summarize_ablations.py --csv results/ablation_small_metrics.csv --tag small
```

Per-prompt rows: `results/ablation_{small,large}_metrics.csv`, `results/ablation_small_tswitch.csv`.
