"""Summarize results/ablations/metrics.csv into markdown tables (+ latency/quality plot).

Every comparison is paired over the prompts both variants share, with 95% bootstrap CIs (2000 resamples).
Writes results/ablations/SUMMARY.md and assets/ablation_latency_quality.png, and splices SUMMARY.md into
RESULTS.md between the ABLATIONS markers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = "results/ablations/metrics.csv"
OUT = "results/ablations/SUMMARY.md"
PLOT = "assets/ablation_latency_quality.png"
METRICS = ["latency_s", "tokens", "obj_topiq", "obj_musiq", "obj_clipiqa", "obj_hf", "obj_clip", "owl_score", "pick",
           "drift_lpips", "ref_lpips_obj", "seam"]
rng = np.random.default_rng(0)


def paired(df, a, b, metric):
    x = df[df.variant == a].set_index("id")[metric]
    y = df[df.variant == b].set_index("id")[metric]
    ids = x.index.intersection(y.index)
    d = (x[ids] - y[ids]).dropna().values
    if len(d) < 3:
        return None
    boots = rng.choice(d, (2000, len(d))).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, len(d), (d > 0).mean()


def fmt_ci(r, digits=3):
    if r is None:
        return "–"
    m, lo, hi, n, _ = r
    star = " ✱" if (lo > 0 or hi < 0) else ""
    return f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{star}"


def table(df, variants, cols, title, note=""):
    present = [v for v in variants if v in set(df.variant)]
    if not present:
        return ""
    g = df[df.variant.isin(present)].groupby("variant")
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| Variant | n | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "---|" * len(cols))
    for v in present:
        s = g.get_group(v)
        cells = []
        for c in cols:
            if c not in s or s[c].isna().all():
                cells.append("–")
            elif c == "tokens":
                cells.append(f"{s[c].median():.0f}")
            elif c in ("obj_hf",):
                cells.append(f"{s[c].mean() / 1000:.1f}k")
            elif c in ("obj_musiq", "pick", "obj_clip", "seam"):
                cells.append(f"{s[c].mean():.2f}")
            elif c == "latency_s":
                cells.append(f"{s[c].median():.1f}")
            else:
                cells.append(f"{s[c].mean():.3f}")
        lines.append(f"| `{v}` | {len(s)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def paired_table(df, pairs, metrics, title, note=""):
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| A − B | n | " + " | ".join(f"Δ {m}" for m in metrics) + " |")
    lines.append("|---|---|" + "---|" * len(metrics))
    any_row = False
    for a, b in pairs:
        rs = [paired(df, a, b, m) for m in metrics]
        n = max([r[3] for r in rs if r] or [0])
        if n == 0:
            continue
        any_row = True
        lines.append(f"| `{a}` − `{b}` | {n} | " + " | ".join(fmt_ci(r, 2 if m in ('pick', 'obj_musiq', 'seam') else 3) for r, m in zip(rs, metrics)) + " |")
    return ("\n".join(lines) + "\n") if any_row else ""


def plot(df):
    g = df.groupby("variant").agg(lat=("latency_s", "median"), q=("obj_topiq", "mean"), p=("pick", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fams = [("Full HR (steps)", lambda v: v.startswith("fhr") and "_" not in v, "#444444", "o-"),
            ("Ours t0.95, refine steps", lambda v: v in ("ours_t95_r5", "ours_t95_r7", "ours_t95_r10", "ours_t95_r14"), "#d62728", "s-"),
            ("Full HR + speedups", lambda v: v.startswith("fhr") and "_" in v, "#7f7f7f", "x"),
            ("Ours + speedups / variants", lambda v: v.startswith("ours") and v not in ("ours_t95_r5", "ours_t95_r7", "ours_t95_r10", "ours_t95_r14"), "#ff7f0e", "^"),
            ("Draft only", lambda v: v.startswith("draft"), "#2ca02c", "D")]
    for ax, col, lab in [(axes[0], "q", "object TOPIQ ↑"), (axes[1], "p", "PickScore ↑")]:
        for name, f, c, mk in fams:
            s = g[[f(v) for v in g.index]].sort_values("lat")
            if len(s):
                ax.plot(s.lat, s[col], mk, color=c, label=name, ms=6)
                for v, r in s.iterrows():
                    ax.annotate(v.replace("ours_", "").replace("_t95", ""), (r.lat, r[col]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("median end-to-end latency on Tesla T4 (s)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=130)


def main():
    if not os.path.exists(CSV):
        print("no metrics yet")
        return
    df = pd.read_csv(CSV)
    seed1_csv = CSV.replace("metrics.csv", "metrics_seed1.csv")  # seed1 phase writes a separate csv
    if os.path.exists(seed1_csv):
        df = pd.concat([df, pd.read_csv(seed1_csv)], ignore_index=True)
    df_all = df
    df = df[df.seed == df.groupby("id").seed.transform("min")] if "seed" in df else df  # main seed only
    Q = ["latency_s", "tokens", "obj_topiq", "obj_musiq", "obj_clipiqa", "obj_hf", "owl_score", "pick"]
    parts = ["<!-- generated by scripts/summarize_ablations.py; do not edit by hand -->", ""]
    parts.append(table(df, ["draft", "draft_esr", "fhr5", "fhr7", "fhr10", "fhr14", "fhr28", "fhr7_cfgoff2", "fhr10_cache",
                            "fhr14_cache", "ours_t85_r14", "ours_t95_r5", "ours_t95_r7", "ours_t95_r10", "ours_t95_r14",
                            "ours_d5_t95_r7", "ours_cfgoff2_t95_r7", "ours_cache_t95_r7", "ours_cache_t95_r10",
                            "ours_cache_cfgoff2_t95_r10", "ours_esr_t95_r7", "ours_esr_t90_r7", "ours_esr_t85_r7",
                            "ours_esr_t85_r14", "ours_esr_cache_cfgoff2_t95_r10"],
                       Q + ["drift_lpips"], "A. Latency vs. quality (all variants, means; latency and tokens are medians)"))
    parts.append(table(df, ["ours_t95_r7", "naive_t95_r7", "allhr_fov_t95_r7", "allhr_nofov_t95_r7", "allhr_base_t95_r7",
                            "allhr_fov_esr_t95_r7", "ours_shiftL_t95_r7", "fhr7"],
                       Q + ["drift_lpips"], "B. Why is every-token-full-res worse?"))
    parts.append(paired_table(df, [("allhr_fov_t95_r7", "ours_t95_r7"), ("allhr_nofov_t95_r7", "ours_t95_r7"),
                                   ("allhr_base_t95_r7", "ours_t95_r7"), ("allhr_nofov_t95_r7", "allhr_fov_t95_r7"),
                                   ("allhr_fov_esr_t95_r7", "allhr_fov_t95_r7"), ("naive_t95_r7", "ours_t95_r7"),
                                   ("ours_shiftL_t95_r7", "ours_t95_r7")],
                              ["obj_topiq", "obj_musiq", "pick"], "B2. Paired differences for B"))
    parts.append(table(df, ["ours_t95_r7", "mask_random_t95_r7", "mask_center_t95_r7", "mask_none_t95_r7",
                            "topk05_t95_r7", "topk25_t95_r7", "topk50_t95_r7", "allhr_fov_t95_r7"],
                       Q + ["ref_lpips_obj"], "C. Mask source at equal token budget, and token-budget sweep"))
    parts.append(paired_table(df, [("ours_t95_r7", "mask_random_t95_r7"), ("ours_t95_r7", "mask_center_t95_r7"),
                                   ("ours_t95_r7", "mask_none_t95_r7"), ("mask_random_t95_r7", "mask_none_t95_r7")],
                              ["obj_topiq", "obj_musiq", "obj_hf", "ref_lpips_obj"],
                              "C2. Paired: CLIPSeg mask vs. same-size masks in the wrong place",
                              "`ref_lpips_obj` is LPIPS to the same-draft, no-foveation reference inside a shared object box (lower = closer)."))
    parts.append(paired_table(df, [("ours_corr_t95_r7", "ours_t95_r7"), ("ours_esr_t95_r7", "ours_t95_r7"),
                                   ("ours_esr_t85_r7", "ours_t95_r7"), ("ours_esr_t90_r7", "ours_t95_r7"),
                                   ("ours_esr_t85_r14", "ours_t85_r14"),
                                   ("ours_cfgoff2_t95_r7", "ours_t95_r7"), ("ours_cache_t95_r7", "ours_t95_r7"),
                                   ("ours_cache_t95_r10", "ours_t95_r7"), ("ours_cache_cfgoff2_t95_r10", "ours_t95_r7"),
                                   ("ours_d5_t95_r7", "ours_t95_r7"),
                                   ("ours_esr_cache_cfgoff2_t95_r10", "ours_t95_r7"), ("ours_esr_cache_cfgoff2_t95_r10", "ours_esr_t95_r7")],
                              ["latency_s", "obj_topiq", "obj_musiq", "pick", "drift_lpips", "seam"],
                              "D. Paired: noise, super-resolved init, speedups vs. the reference (`ours_t95_r7`)"))
    parts.append(paired_table(df, [("ours_t95_r5", "fhr5"), ("ours_t95_r7", "fhr7"), ("ours_cache_t95_r10", "fhr7"),
                                   ("ours_esr_t95_r7", "fhr7"), ("ours_t95_r10", "fhr10"), ("ours_t95_r14", "fhr14"),
                                   ("ours_cache_t95_r7", "fhr7_cfgoff2"), ("ours_cache_t95_r10", "fhr10_cache"),
                                   ("ours_d5_t95_r7", "fhr5"), ("ours_t85_r14", "fhr28"), ("ours_t95_r5", "fhr28"), ("ours_t95_r10", "fhr28"),
                                   ("ours_esr_cache_cfgoff2_t95_r10", "fhr7"), ("ours_esr_cache_cfgoff2_t95_r10", "fhr10"), ("ours_t95_r5", "fhr10"),
                                   ("ours_cache_cfgoff2_t95_r10", "fhr7"), ("ours_cache_cfgoff2_t95_r10", "fhr10"),
                                   ("ours_esr_cache_cfgoff2_t95_r10", "ours_cache_cfgoff2_t95_r10")],
                              ["latency_s", "obj_topiq", "obj_musiq", "pick"],
                              "E. Paired: ours vs. Full HR at comparable cost"))
    base_seed = df_all.groupby("id").seed.transform("min") if "seed" in df_all else None
    if base_seed is not None and (df_all.seed == base_seed + 1).any():
        d1 = df_all[df_all.seed == base_seed + 1]
        prs = [(a, b) for a, b in [("ours_t95_r7", "fhr7"), ("ours_t95_r7", "allhr_fov_t95_r7"), ("ours_t95_r7", "mask_random_t95_r7"),
                                   ("ours_esr_t95_r7", "ours_t95_r7"), ("ours_cache_t95_r10", "ours_t95_r7"), ("ours_t95_r7", "draft"),
                                   ("ours_cache_cfgoff2_t95_r10", "fhr10"), ("ours_cache_cfgoff2_t95_r10", "fhr7"),
                                   ("ours_cache_cfgoff2_t95_r10", "ours_t95_r7"), ("ours_t95_r7", "fhr10")]]
        parts.append(paired_table(d1, prs, ["latency_s", "obj_topiq", "obj_musiq", "pick"],
                                  "F. Second seed (robustness): same comparisons with a different noise seed"))
    parts.append("✱ = 95% bootstrap CI excludes 0. Positive Δ means A is higher. Latency: median wall-clock on one Tesla T4 (fp16).\n")
    md = "\n".join(p for p in parts if p)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(md)
    plot(df)
    try:
        import ablation_figure
        ablation_figure.main()
    except Exception as e:  # figure is optional
        print('figure failed:', e)
    # splice into RESULTS.md
    r = open("RESULTS.md").read()
    b, e = "<!-- ABLATIONS:BEGIN -->", "<!-- ABLATIONS:END -->"
    if b in r and e in r:
        r = r[: r.index(b) + len(b)] + "\n" + md + "\n" + r[r.index(e):]
        open("RESULTS.md", "w").write(r)
    print(md)


if __name__ == "__main__":
    main()
