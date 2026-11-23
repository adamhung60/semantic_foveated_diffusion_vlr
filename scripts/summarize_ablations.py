"""Summarize scripts/run_ablation_benchmark.py results: three tables, two figures, and the t_switch sweep if present.

  python scripts/summarize_ablations.py --csv results/ablation_small_metrics.csv --tag small
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
QUALITY = ["focus_musiq", "focus_clip_iqa", "focus_hf", "focus_clip"]

FAMILIES = [  # (legend label, color, [(method, point label)], connect points)
    ("Full resolution, 1024²", BLUE, [("full_hr_8", "8 steps"), ("full_hr_14", "14 steps"), ("full_hr_28", "28 steps")], True),
    ("Uniform low resolution, upsampled", ORANGE, [("uni512_14", "512², 14"), ("uni512_28", "512², 28"), ("uni768_14", "768², 14")], False),
    ("Draft, no foveation", AQUA, [("lr_draft", "256² draft"), ("refine_all", "refine all tokens")], False),
    ("Draft + foveated Stage 2", YELLOW, [("fov_clipseg", "ours"), ("naive", "no LoRA")], False),
]
MASK_ROWS = [("fov_clipseg", "Focus prompt (ours)"), ("fov_generic", "Generic prompt"), ("fov_detail", "Draft detail"),
             ("fov_center", "Center disc"), ("fov_random", "Random disc")]
GT_ROWS = ["gt_reference", "gt_lr_draft", "gt_refine_all", "gt_fov_center", "gt_fov_clipseg"]


def paired(df, a, b, col, higher_is_better=True):
    """Share of prompts where method a beats method b on `col`, with a Wilcoxon signed-rank p-value."""
    w = df[df.method.isin([a, b])].pivot(index="id", columns="method", values=col).dropna()
    diff = (w[a] - w[b]) * (1 if higher_is_better else -1)
    p = wilcoxon(diff).pvalue if len(diff) >= 6 and (diff != 0).any() else float("nan")
    return f"{(diff > 0).mean():.0%} (p={p:.1g})"


def tswitch_table(df, csv_path):
    """Ours at each re-noising level, against the equal-latency baselines of the main benchmark."""
    sw = pd.read_csv(csv_path)
    ours = sw[sw.method == "fov_clipseg"]
    wide = ours.pivot(index="id", columns="t_switch", values="focus_musiq")
    base = df.pivot(index="id", columns="method", values="focus_musiq")
    t = ours.groupby("t_switch")[["latency_s", "focus_musiq", "focus_hf", "focus_clip", "focus_lpips_draft"]].median()
    t["gt_focus_lpips"] = sw[sw.method == "gt_fov_clipseg"].groupby("t_switch").focus_lpips_gt.median()
    for other in ["lr_draft", "uni512_14", "uni768_14", "full_hr_8"]:
        t[f"beats {other}"] = [f"{(wide[c] > base[other].reindex(wide.index)).mean():.0%}" for c in t.index]
    print(f"\n== 4. t_switch sweep, ours ({wide.shape[0]} prompts, medians) ==")
    print(t.round(3).to_string())
    print("baselines (median MUSIQ):", base[["lr_draft", "uni512_14", "uni768_14", "full_hr_8", "full_hr_28"]].median().round(1).to_dict())


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/ablation_small_metrics.csv")
    parser.add_argument("--tag", default="small")
    parser.add_argument("--fig_dir", default="assets")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    med = df.groupby("method", sort=False).median(numeric_only=True)
    n = df.id.nunique()
    pd.set_option("display.width", 200)
    full = med.loc["full_hr_28", "latency_s"]

    print(f"\n== 1. Fair-compute comparison ({n} prompts, medians) ==")
    t1 = med.loc[[m for _, _, pts, _ in FAMILIES for m, _ in pts], ["latency_s"] + QUALITY].copy()
    t1.insert(1, "speedup", full / t1.latency_s)
    t1["ours wins on MUSIQ"] = [paired(df, "fov_clipseg", m, "focus_musiq") if m != "fov_clipseg" else "-" for m in t1.index]
    print(t1.round(2).to_string())

    print(f"\n== 2. Mask source at equal token budget ({n} prompts, medians) ==")
    t2 = med.loc[[m for m, _ in MASK_ROWS], ["stage2_tokens", "object_cover"] + QUALITY].copy()
    t2["ours wins on MUSIQ"] = [paired(df, "fov_clipseg", m, "focus_musiq") if m != "fov_clipseg" else "-" for m in t2.index]
    print(t2.round(2).to_string())
    print("draft (no Stage 2):", med.loc["lr_draft", QUALITY].round(2).to_dict())

    print(f"\n== 3. Paired protocol: draft = downsampled full-HR image, reference = that image ({n} prompts, medians) ==")
    t3 = med.loc[GT_ROWS, ["focus_lpips_gt", "full_lpips_gt"] + QUALITY].copy()
    t3["closer to reference than draft"] = [paired(df, m, "gt_lr_draft", "focus_lpips_gt", higher_is_better=False)
                                            if m not in ("gt_reference", "gt_lr_draft") else "-" for m in t3.index]
    print(t3.round(3).to_string())
    print("ours closer than all-token refinement:", paired(df, "gt_fov_clipseg", "gt_refine_all", "focus_lpips_gt", False))
    print("ours closer than center mask:", paired(df, "gt_fov_clipseg", "gt_fov_center", "focus_lpips_gt", False))

    sweep_csv = args.csv.replace("_metrics.csv", "_tswitch.csv")
    if os.path.exists(sweep_csv):
        tswitch_table(df, sweep_csv)

    # Figure 1: latency vs focus quality
    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
    style(ax)
    for label, color, pts, connect in FAMILIES:
        xs, ys = [med.loc[m, "latency_s"] for m, _ in pts], [med.loc[m, "focus_musiq"] for m, _ in pts]
        ax.plot(xs, ys, color=color, linewidth=2 if connect else 0, marker="o", markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        for (m, text), x, y in zip(pts, xs, ys):
            ax.annotate(text, (x, y), xytext=(8, -3), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("End-to-end latency on an RTX 5080 (s)", color=INK, fontsize=10)
    ax.set_ylabel("Focus-crop quality (MUSIQ, median)", color=INK, fontsize=10)
    ax.set_title(f"Latency vs. focus-region quality, {n} prompts ({args.tag} set)", color=INK, fontsize=11, loc="left")
    ax.set_xlim(0, full * 1.18)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{args.fig_dir}/ablation_pareto_{args.tag}.png", dpi=170)

    # Figure 2: mask sources at equal token budget
    fig, ax = plt.subplots(figsize=(7.6, 3.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    vals = [med.loc[m, "focus_musiq"] for m, _ in MASK_ROWS][::-1]
    ax.barh([name for _, name in MASK_ROWS][::-1], vals, color=BLUE, height=0.55, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9, color=INK)
    draft = med.loc["lr_draft", "focus_musiq"]
    ax.axvline(draft, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.text(draft, len(vals) - 0.42, " draft, no Stage 2", fontsize=9, color=MUTED, va="bottom")
    ax.set_xlabel("Focus-crop quality (MUSIQ, median)", color=INK, fontsize=10)
    ax.set_title(f"Where the high-resolution tokens go, same token budget ({args.tag} set)", color=INK, fontsize=11, loc="left")
    ax.set_ylim(-0.6, len(vals) - 0.05)
    ax.tick_params(axis="y", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{args.fig_dir}/ablation_masks_{args.tag}.png", dpi=170)
    print(f"\nfigures: {args.fig_dir}/ablation_pareto_{args.tag}.png, {args.fig_dir}/ablation_masks_{args.tag}.png")


if __name__ == "__main__":
    main()
