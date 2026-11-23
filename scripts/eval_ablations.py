"""Evaluate every image under --root/<variant>/<id>.png (incremental: rows already in --csv are skipped).

Per image:
  owl_score, box      OWLv2 detection of the focus prompt on *this* image (independent of CLIPSeg, which makes masks)
  obj_topiq/obj_musiq no-reference quality (pyiqa TOPIQ-NR, MUSIQ) on the square object crop
  obj_clipiqa, obj_hf CLIP-IQA style sharp-vs-blurry probability, mean squared Laplacian (repo metrics)
  obj_clip            CLIP similarity crop <-> focus prompt
  pick                PickScore of the whole image vs the scene prompt
  drift_lpips         LPIPS(output downsized to 256^2, Stage-1 draft) -- faithfulness to the draft (Stage-2 variants)
  ref_lpips_obj       LPIPS on the object box shared with the paired reference variant (same draft, same noise,
                      every token full-res, no_fov LoRA): what foveation costs relative to not foveating
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic_foveation.benchmark import PROMPT_SETS
from src.semantic_foveation.metrics import SemanticEvaluator, high_freq_energy

REF_VARIANT = "allhr_nofov_t95_r7"


def square_box(b, W=1024, H=1024, pad=0.3, min_side=160):
    x0, y0, x1, y1 = b
    side = int(min(W, H, max(min_side, (1 + pad) * max(x1 - x0, y1 - y0))))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx = int(np.clip(cx - side / 2, 0, W - side))
    by = int(np.clip(cy - side / 2, 0, H - side))
    return [bx, by, bx + side, by + side]


class Evaluator:
    def __init__(self, device="cuda:0"):
        import lpips
        import pyiqa
        from transformers import AutoModel, AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor
        self.device = device
        self.owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
        self.topiq = pyiqa.create_metric("topiq_nr", device=device)
        self.musiq = pyiqa.create_metric("musiq", device=device)
        self.sem = SemanticEvaluator(device=device)
        self.pick_proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pick = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1", torch_dtype=torch.float16).to(device).eval()
        self.lp = lpips.LPIPS(net="alex", verbose=False).to(device)

    @torch.no_grad()
    def owl_box(self, img, text):
        inp = self.owl_proc(text=[[text]], images=img, return_tensors="pt").to(self.device)
        out = self.owl(**inp)
        # OWLv2 pads to square; image is square already
        res = self.owl_proc.post_process_grounded_object_detection(out, threshold=0.0, target_sizes=torch.tensor([[img.height, img.width]]))[0]
        i = int(res["scores"].argmax())
        return [float(v) for v in res["boxes"][i].tolist()], float(res["scores"][i])

    def _t(self, img):
        return torch.from_numpy(np.asarray(img.convert("RGB"))).permute(2, 0, 1)[None].float().div(255).to(self.device)

    @torch.no_grad()
    def iqa(self, crop):
        x = self._t(crop.resize((384, 384), Image.Resampling.BICUBIC) if crop.width < 224 else crop)
        return float(self.topiq(x)), float(self.musiq(x))

    @torch.no_grad()
    def pickscore(self, img, text):
        ii = self.pick_proc(images=img, return_tensors="pt").to(self.device)
        ti = self.pick_proc(text=text, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        ie = self.pick.get_image_features(pixel_values=ii["pixel_values"].half())
        te = self.pick.get_text_features(**ti)
        ie = getattr(ie, "pooler_output", ie)
        te = getattr(te, "pooler_output", te)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        return float(self.pick.logit_scale.exp() * (ie * te).sum())

    @torch.no_grad()
    def lpips(self, a, b):
        return float(self.lp(self._t(a) * 2 - 1, self._t(b) * 2 - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/abl")
    ap.add_argument("--csv", default="results/ablations/metrics.csv")
    ap.add_argument("--prompt_set", default="small")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    prompts = {p["id"]: p for s in PROMPT_SETS.values() for p in s}

    done = set()
    rows = []
    if os.path.exists(args.csv):
        old = pd.read_csv(args.csv)
        rows = old.to_dict("records")
        done = {(r["variant"], int(r["id"]), int(r["seed"])) for r in rows}

    todo = []
    for js in sorted(glob.glob(os.path.join(args.root, "*", "*.json"))):
        if "/_drafts/" in js:
            continue
        meta = json.load(open(js))
        if (meta["variant"], int(meta["id"]), int(meta["seed"])) not in done:
            todo.append((js, meta))
    # reference variant first so its boxes exist for the paired metric
    todo.sort(key=lambda x: x[1]["variant"] != REF_VARIANT)
    print(f"{len(todo)} images to evaluate", flush=True)
    if not todo:
        return
    ev = Evaluator()
    ref_box = {(int(r["id"]), int(r["seed"])): json.loads(r["box"]) for r in rows if r["variant"] == REF_VARIANT}
    root_of = lambda m: os.path.join(os.path.dirname(os.path.dirname(m)))

    for n, (js, meta) in enumerate(todo):
        pid, seed = int(meta["id"]), int(meta["seed"])
        p = prompts[pid]
        img = Image.open(js.replace(".json", ".png")).convert("RGB")
        raw, score = ev.owl_box(img, p["focus_prompt"])
        box = square_box(raw)
        crop = img.crop(box)
        topiq, musiq = ev.iqa(crop)
        row = dict(meta)
        row.update(box=json.dumps(box), owl_score=score, obj_topiq=topiq, obj_musiq=musiq,
                   obj_clipiqa=ev.sem.clip_iqa(crop), obj_hf=high_freq_energy(crop),
                   obj_clip=ev.sem.clip_score(crop, p["focus_prompt"]), pick=ev.pickscore(img, p["prompt"]))
        if meta["variant"].startswith(("ours", "naive", "allhr", "mask", "topk")):
            dsteps = 5 if "_d5_" in meta["variant"] else 14
            dpath = os.path.join(args.root, "_drafts", f"d{dsteps}_s{seed}_{pid:03d}.png")
            if os.path.exists(dpath):
                d = Image.open(dpath).convert("RGB")
                row["drift_lpips"] = ev.lpips(img.resize(d.size, Image.Resampling.LANCZOS), d)
                refp = os.path.join(args.root, REF_VARIANT, f"{pid:03d}.png")
                if (pid, seed) in ref_box and os.path.exists(refp) and dsteps == 14:
                    rb = ref_box[(pid, seed)]
                    ref = Image.open(refp).convert("RGB")
                    row["ref_lpips_obj"] = ev.lpips(img.crop(rb).resize((256, 256)), ref.crop(rb).resize((256, 256)))
        if meta["variant"] == REF_VARIANT:
            ref_box[(pid, seed)] = box
        rows.append(row)
        if (n + 1) % 25 == 0 or n + 1 == len(todo):
            pd.DataFrame(rows).to_csv(args.csv, index=False)
            print(f"evaluated {n + 1}/{len(todo)}", flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
