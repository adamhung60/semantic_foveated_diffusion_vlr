"""Qualitative grid for the ablations: for a few prompts, full images (top) and OWLv2 object crops (bottom)."""
import json, os, sys
import pandas as pd
from PIL import Image, ImageDraw

VARS = ["draft", "ours_t95_r7", "allhr_fov_t95_r7", "allhr_nofov_t95_r7", "mask_random_t95_r7", "ours_esr_t95_r7",
        "ours_cache_t95_r10", "fhr7"]


def main(csv="results/ablations/metrics.csv", root="outputs/abl", out="assets/ablation_qualitative.png", n=3, tile=256):
    df = pd.read_csv(csv)
    vars_ = [v for v in VARS if v in set(df.variant)]
    have = df[df.variant.isin(vars_)].groupby("id").variant.nunique()
    ids = [i for i in have[have == len(vars_)].index][:: max(1, len(have) // n)][:n]
    if not ids:
        return
    W = tile * len(vars_)
    canvas = Image.new("RGB", (W, (2 * tile + 18) * len(ids) + 18), "white")
    d = ImageDraw.Draw(canvas)
    for j, v in enumerate(vars_):
        d.text((j * tile + 4, 2), v, fill="black")
    for r, pid in enumerate(ids):
        y = 18 + r * (2 * tile + 18)
        for j, v in enumerate(vars_):
            row = df[(df.variant == v) & (df.id == pid)].iloc[0]
            img = Image.open(os.path.join(root, v, f"{pid:03d}.png")).convert("RGB")
            box = json.loads(row["box"])
            full = img.resize((tile, tile))
            dd = ImageDraw.Draw(full)
            s = tile / img.width
            dd.rectangle([box[0] * s, box[1] * s, box[2] * s, box[3] * s], outline="yellow", width=2)
            canvas.paste(full, (j * tile, y))
            canvas.paste(img.crop(box).resize((tile, tile), Image.Resampling.LANCZOS), (j * tile, y + tile))
            d.text((j * tile + 4, y + 2 * tile + 2), f"id {pid} topiq {row['obj_topiq']:.2f}", fill="black")
    canvas.save(out)


if __name__ == "__main__":
    main()
