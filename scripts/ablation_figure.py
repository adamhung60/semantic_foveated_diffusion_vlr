"""Figures for Ablations.md / RESULTS.md (called by summarize_ablations.py, or run directly).

  assets/ablation_latency_quality.png  latency vs. object quality / PickScore: Full HR step sweep vs. ours, n = 30 shared prompts
  assets/ablation_effects.png          paired effects (Δ object TOPIQ, 95% bootstrap CI) for every ablation question
  assets/ablation_qualitative.jpg      same prompts, same draft: full image + OWLv2 object crop (the crop TOPIQ is scored on)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CSV = "results/ablations/metrics.csv"
CSV1 = "results/ablations/metrics_seed1.csv"
ROOT = "outputs/abl"

# palette: reference categorical slots 1-2 (validated all-pairs, light + dark); neutral ink for everything else
OURS, FHR = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})
rng = np.random.default_rng(0)


def load():
    df = pd.read_csv(CSV)
    base = df.groupby("id").seed.transform("min")
    d0 = df[df.seed == base]
    d1 = pd.read_csv(CSV1) if os.path.exists(CSV1) else None
    return d0, d1


# ---------------------------------------------------------------- 1. latency vs quality
def latency_quality(d0, out="assets/ablation_latency_quality.png"):
    fhr = [("fhr5", 5), ("fhr7", 7), ("fhr10", 10), ("fhr14", 14), ("fhr28", 28)]
    ours = [("ours_t95_r5", 5), ("ours_t95_r7", 7), ("ours_t95_r10", 10), ("ours_t95_r14", 14)]
    extra = [("ours_cache_cfgoff2_t95_r10", "Ours, recommended\n(10 steps, cache + CFG trunc.)"),
             ("draft", "Draft only"), ("draft_esr", "Draft + Real-ESRGAN")]
    allv = [v for v, _ in fhr + ours] + [v for v, _ in extra]
    allv = [v for v in allv if v in set(d0.variant)]
    sets = [set(d0[d0.variant == v].id) for v in allv]
    ids = sorted(set.intersection(*sets))
    d = d0[d0.id.isin(ids)]
    g = d.groupby("variant").agg(lat=("latency_s", "median"), q=("obj_topiq", "mean"), p=("pick", "mean"))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, col, ylab in [(axes[0], "q", "Object quality (TOPIQ, mean) ↑"), (axes[1], "p", "Whole-image PickScore (mean) ↑")]:
        for series, color, mk, name in [(fhr, FHR, "o", "Full HR"), (ours, OURS, "s", "Ours (foveated)")]:
            s = [(g.loc[v, "lat"], g.loc[v, col], k) for v, k in series if v in g.index]
            x, y, k = zip(*s)
            ax.plot(x, y, "-", color=color, lw=2, zorder=2)
            ax.scatter(x, y, s=64, marker=mk, color=color, edgecolor=SURF, linewidth=2, zorder=3, label=name)
            for xi, yi, ki in s:
                ax.annotate(f"{ki}", (xi, yi), xytext=(0, -15 if color == FHR else 9), textcoords="offset points",
                            ha="center", fontsize=8, color=INK2)
        for v, lab in extra:
            if v not in g.index:
                continue
            rec = v.startswith("ours")
            ax.scatter(g.loc[v, "lat"], g.loc[v, col], s=150 if rec else 50, marker="*" if rec else "D",
                       color=OURS if rec else MUTED, edgecolor=SURF, linewidth=1.5, zorder=4)
            if rec:
                ax.annotate(lab.replace("\n", " "), (g.loc[v, "lat"], g.loc[v, col]), xytext=(-6, 30), textcoords="offset points",
                            fontsize=8, color=INK2, ha="center", va="bottom",
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=6))
            else:
                ax.annotate(lab, (g.loc[v, "lat"], g.loc[v, col]), xytext=(7, 0), textcoords="offset points",
                            fontsize=8, color=INK2, va="center")
        ax.set_xscale("log")
        ax.set_xticks([8, 15, 20, 30, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Median end-to-end latency per image, Tesla T4 (s, log scale)")
        ax.set_ylabel(ylab)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.12)  # room for step labels + recommended-config label
    axes[0].legend(loc="lower right", frameon=False, fontsize=9, title="Numbers = denoising steps", title_fontsize=8)
    fig.suptitle(f"Latency vs. quality  (n = {len(ids)} small-object prompts, same prompts for every point)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. paired effects (forest plot)
EFFECTS = [
    ("Mask source (equal token budget)", [
        ("ours_t95_r7", "mask_random_t95_r7", "CLIPSeg mask vs. same-size box, wrong place", 0),
        ("ours_t95_r7", "mask_center_t95_r7", "CLIPSeg mask vs. same-size box, centered", 0),
        ("ours_t95_r7", "mask_none_t95_r7", "CLIPSeg mask vs. no full-res tokens", 0),
        ("ours_t95_r7", "mask_random_t95_r7", "CLIPSeg vs. wrong-place box (seed 2)", 1)]),
    ("Every token full-res (same draft)", [
        ("allhr_fov_t95_r7", "ours_t95_r7", "All full-res, foveated LoRA vs. ours", 0),
        ("allhr_nofov_t95_r7", "ours_t95_r7", "All full-res, no-foveation LoRA vs. ours", 0),
        ("allhr_base_t95_r7", "ours_t95_r7", "All full-res, no LoRA vs. ours", 0),
        ("allhr_fov_esr_t95_r7", "allhr_fov_t95_r7", "All full-res: Real-ESRGAN vs. bicubic init", 0),
        ("naive_t95_r7", "ours_t95_r7", "No foveated attention vs. ours", 0),
        ("ours_shiftL_t95_r7", "ours_t95_r7", "True-length schedule shift vs. ours", 0)]),
    ("Init and noise", [
        ("ours_corr_t95_r7", "ours_t95_r7", "Correlated low/high-res noise", 0),
        ("ours_esr_t95_r7", "ours_t95_r7", "Real-ESRGAN init, t=0.95", 0),
        ("ours_esr_t95_r7", "ours_t95_r7", "Real-ESRGAN init, t=0.95 (seed 2)", 1),
        ("ours_esr_t85_r14", "ours_t85_r14", "Real-ESRGAN init, t=0.85, 14 steps", 0)]),
    ("Speedups (vs. ours, 7 steps)", [
        ("ours_cfgoff2_t95_r7", "ours_t95_r7", "No CFG on last 2 steps  (−1.4 s)", 0),
        ("ours_cache_t95_r7", "ours_t95_r7", "Cache 2 of 7 steps  (−2.8 s)", 0),
        ("ours_cache_t95_r10", "ours_t95_r7", "10 steps, 3 cached  (±0 s)", 0),
        ("ours_cache_cfgoff2_t95_r10", "ours_t95_r7", "Recommended: 10 steps, cache + CFG trunc.  (−1.4 s)", 0)]),
    ("Ours vs. Full HR", [
        ("ours_t95_r5", "fhr5", "Ours vs. Full HR, 5 steps each  (3.3 s faster)", 0),
        ("ours_t95_r7", "fhr7", "Ours vs. Full HR, 7 steps each  (1.4× faster)", 0),
        ("ours_cache_cfgoff2_t95_r10", "fhr7", "Recommended vs. Full HR 7 steps  (1.5× faster)", 0),
        ("ours_cache_cfgoff2_t95_r10", "fhr10", "Recommended vs. Full HR 10 steps  (2.2× faster)", 0),
        ("ours_cache_cfgoff2_t95_r10", "fhr10", "Recommended vs. Full HR 10 steps (seed 2)", 1),
        ("ours_t95_r5", "fhr28", "Ours 5 steps vs. Full HR 28 steps  (6.6× faster)", 0)]),
]


def paired(df, a, b, m="obj_topiq"):
    if df is None:
        return None
    x = df[df.variant == a].set_index("id")[m]
    y = df[df.variant == b].set_index("id")[m]
    ids = x.index.intersection(y.index)
    dd = (x[ids] - y[ids]).dropna().values
    if len(dd) < 3:
        return None
    boots = rng.choice(dd, (4000, len(dd))).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dd.mean(), lo, hi, len(dd)


def effects(d0, d1, out="assets/ablation_effects.png"):
    rows = []  # (label, mean, lo, hi, n) or (group header,)
    for grp, items in EFFECTS:
        rr = []
        for a, b, lab, s in items:
            r = paired(d0 if s == 0 else d1, a, b)
            if r:
                rr.append((lab, *r))
        if rr:
            rows.append((grp,))
            rows += rr
    H = 0.32 * len(rows) + 1.0
    fig, ax = plt.subplots(figsize=(10, H))
    y = 0
    yt, yl = [], []
    for r in rows:
        y -= 1
        if len(r) == 1:
            ax.text(-0.015, y, r[0], transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=9.5, fontweight="bold", color=INK)
            continue
        lab, m, lo, hi, n = r
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [y, y], color=OURS if sig else MUTED, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([m], [y], s=46, color=OURS if sig else SURF, edgecolor=OURS if sig else MUTED, linewidth=1.8, zorder=3)
        ax.text(1.01, y, f"{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={n}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=INK2, family="DejaVu Sans Mono")
        yt.append(y)
        yl.append(lab)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.set_yticks(yt, yl, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.7, 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ object quality (TOPIQ), paired per prompt: first − second.  Filled = 95% CI excludes 0")
    ax.set_title("Every ablation in one view: paired differences with 95% bootstrap CIs", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. qualitative grid
COLS = [("draft", "Draft (256², bicubic ×4)"), ("fhr7", "Full HR, 7 steps"), ("allhr_fov_t95_r7", "All tokens full-res"),
        ("mask_random_t95_r7", "Mask in wrong place"), ("ours_t95_r7", "Ours, 7 steps"),
        ("ours_cache_cfgoff2_t95_r10", "Ours, recommended")]


def qualitative(d0, out="assets/ablation_qualitative.jpg", n=3, tile=300):
    from src.semantic_foveation.benchmark import PROMPT_SETS
    prompts = {p["id"]: p for s in PROMPT_SETS.values() for p in s}
    cols = [(v, t) for v, t in COLS if v in set(d0.variant)]
    ref = "allhr_nofov_t95_r7"
    have = d0[d0.variant.isin([v for v, _ in cols] + [ref])].groupby("id").variant.nunique()
    cand = have[have == len(cols) + 1].index
    # prompts chosen without looking at images: 25th/50th/75th percentile of the ours − all-full-res TOPIQ gap
    diff = (d0[d0.variant == "ours_t95_r7"].set_index("id").obj_topiq - d0[d0.variant == "allhr_fov_t95_r7"].set_index("id").obj_topiq)
    diff = diff[diff.index.isin(cand)].sort_values()
    ids = [diff.index[int(q * (len(diff) - 1))] for q in (0.25, 0.5, 0.75)][:n]

    fig, axes = plt.subplots(2 * len(ids), len(cols), figsize=(2.2 * len(cols), 2.2 * 2 * len(ids) + 0.6),
                             gridspec_kw=dict(wspace=0.03, hspace=0.06))
    for r, pid in enumerate(ids):
        for j, (v, title) in enumerate(cols):
            img = Image.open(os.path.join(ROOT, v, f"{pid:03d}.png")).convert("RGB")
            row = d0[(d0.variant == v) & (d0.id == pid)].iloc[0]
            box = json.loads(row["box"])  # this image's own OWLv2 box = the crop its TOPIQ is computed on
            a_full, a_crop = axes[2 * r, j], axes[2 * r + 1, j]
            a_full.imshow(img.resize((tile, tile), Image.Resampling.LANCZOS))
            s = tile / img.width
            a_full.add_patch(plt.Rectangle((box[0] * s, box[1] * s), (box[2] - box[0]) * s, (box[3] - box[1]) * s,
                                           fill=False, ec="#ffd400", lw=1.5))
            a_crop.imshow(img.crop(box).resize((tile, tile), Image.Resampling.LANCZOS))
            a_crop.text(0.03, 0.03, f"TOPIQ {row['obj_topiq']:.2f}", transform=a_crop.transAxes, fontsize=7.5,
                        color="white", bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.5))
            for a in (a_full, a_crop):
                a.set_xticks([]); a.set_yticks([]); a.grid(False)
                for sp in a.spines.values():
                    sp.set_visible(v.startswith("ours")); sp.set_color(OURS); sp.set_linewidth(2)
            if r == 0:
                a_full.set_title(title, fontsize=9, color=INK)
        axes[2 * r, 0].set_ylabel(f"“{prompts[pid]['focus_prompt']}”", fontsize=8.5, color=INK2)
        axes[2 * r + 1, 0].set_ylabel("object crop", fontsize=8, color=MUTED)
    fig.savefig(out, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d0, d1 = load()
    latency_quality(d0)
    effects(d0, d1)
    qualitative(d0)


if __name__ == "__main__":
    main()
