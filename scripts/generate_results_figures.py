"""Generate the report figures from the benchmark outputs."""

import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

EVAL_DIR = "/fast8tb/adam/semantic_foveated_diffusion_cache/eval_small_objects"
COLUMNS = [("full_hr", "Full HR (28 steps)"), ("lr_draft", "256² draft (upsampled)"), ("ours", "Ours")]


def draw_box(img: np.ndarray, box, color=(255, 220, 0), width=8) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, width)
    return out


def draw_mask_contour(img: np.ndarray, mask: np.ndarray, color=(0, 230, 255), width=4) -> np.ndarray:
    out = img.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, width)
    return out


def qualitative_figure(df: pd.DataFrame, ids, out_path: str):
    fig, axes = plt.subplots(2 * len(ids), len(COLUMNS), figsize=(3.2 * len(COLUMNS), 3.2 * 2 * len(ids)))
    for r, pid in enumerate(ids):
        p_dir = os.path.join(EVAL_DIR, f"{pid:03d}")
        mask = np.load(os.path.join(p_dir, "mask.npy"))
        for c, (method, title) in enumerate(COLUMNS):
            img = np.asarray(Image.open(os.path.join(p_dir, f"{method}.png")).convert("RGB"))
            box = json.loads(df[(df.id == pid) & (df.method == method)].box.iloc[0])
            full = draw_mask_contour(img, mask) if method == "ours" else img
            x0, y0, x1, y1 = box
            crop = cv2.resize(img[y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_LANCZOS4)
            axes[2 * r, c].imshow(draw_box(full, box))
            axes[2 * r + 1, c].imshow(crop)
            if r == 0:
                axes[0, c].set_title(title, fontsize=12)
        focus = df[df.id == pid].focus_prompt.iloc[0] if "focus_prompt" in df else ""
        axes[2 * r, 0].set_ylabel(f'"{focus}"', fontsize=10)
        axes[2 * r + 1, 0].set_ylabel("focus crop", fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0.03, hspace=0.06, left=0.06, right=0.99, top=0.97, bottom=0.01)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def contact_sheet(df: pd.DataFrame, out_path: str, tile: int = 160):
    """All prompts: full_hr crop | draft crop | ours crop, for picking figure examples."""
    ids = sorted(df.id.unique())
    sheet = Image.new("RGB", (tile * 3 * 6, tile * ((len(ids) + 5) // 6)), "white")
    for k, pid in enumerate(ids):
        for c, (method, _) in enumerate(COLUMNS):
            img = Image.open(os.path.join(EVAL_DIR, f"{pid:03d}", f"{method}.png"))
            box = json.loads(df[(df.id == pid) & (df.method == method)].box.iloc[0])
            sheet.paste(img.crop(box).resize((tile, tile)), ((k % 6) * 3 * tile + c * tile, (k // 6) * tile))
    sheet.save(out_path)


if __name__ == "__main__":
    from src.semantic_foveation.benchmark import BENCHMARK_PROMPTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", default="results/benchmark_metrics.csv")
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 15, 45])
    parser.add_argument("--contact_sheet", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    focus = {p["id"]: p["focus_prompt"] for p in BENCHMARK_PROMPTS}
    df["focus_prompt"] = df.id.map(focus)
    os.makedirs("assets", exist_ok=True)
    if args.contact_sheet:
        contact_sheet(df, os.path.join(EVAL_DIR, "contact_sheet.png"))
    qualitative_figure(df, args.ids, "assets/qualitative_comparison.png")
