"""Build the colourisation dataset from WIDER FACE.

Colourisation predicts ab from L in Lab space, so what the model needs is a prior over
what colour things usually are. Scene variety decides that, not resolution - which is
why WIDER FACE is the better source here despite DIV2K having far larger images. It is
3,226 crowded real-world photographs (streets, parades, markets, sports) at a median
1024x741, against DIV2K's 100 landscapes.

Two filters:

  saturation  a near-grey photograph teaches the model to predict grey, and predicting
              grey is the exact failure mode of an undertrained colouriser. Images whose
              mean saturation is below the cutoff are dropped.

  colourfulness  a photo can be saturated in one hue and still teach nothing. The spread
              of the ab channels is what carries usable signal, so that is measured too.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
from dataroot import path as ds, require  # noqa: E402

SRC = require("widerface")
DST = ds("colour")
SIZE = 512


def stats(im: Image.Image) -> tuple[float, float]:
    a = np.asarray(im.resize((128, 128), Image.BILINEAR), dtype=np.float64) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())
    # ab spread, the part a colouriser can actually learn from
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    spread = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    return sat, spread


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sat", type=float, default=0.15)
    ap.add_argument("--min-spread", type=float, default=0.06)
    ap.add_argument("--max", type=int, default=3000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SRC, "**", "*.jpg"), recursive=True))
    if not files:
        raise SystemExit(f"  no images under {SRC}")
    print(f"  {len(files)} candidates in widerface")

    gt = os.path.join(DST, "gt")
    os.makedirs(gt, exist_ok=True)
    rng = random.Random(3407)
    rng.shuffle(files)

    kept, grey, flat = 0, 0, 0
    for p in files:
        if kept >= args.max:
            break
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        if min(im.size) < SIZE // 2:
            continue
        sat, spread = stats(im)
        if sat < args.min_sat:
            grey += 1
            continue
        if spread < args.min_spread:
            flat += 1
            continue
        w, h = im.size
        s = SIZE / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        w, h = im.size
        x, y = (w - SIZE) // 2, (h - SIZE) // 2
        im.crop((x, y, x + SIZE, y + SIZE)).save(
            os.path.join(gt, f"{kept:05d}.png"))
        kept += 1

    print(f"  kept {kept} at {SIZE}x{SIZE}   dropped {grey} near-grey, {flat} low-spread")

    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(gt, "*.png")))
    rng.shuffle(names)
    n = len(names)
    n_test = max(1, round(n * .05))
    n_val = max(1, round(n * .15))
    parts = {"test": names[:n_test], "val": names[n_test:n_test + n_val],
             "train": names[n_test + n_val:]}
    mi = os.path.join(DST, "meta_info")
    os.makedirs(mi, exist_ok=True)
    for k, v in parts.items():
        with open(os.path.join(mi, f"{k}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(v)))
    print("  splits: " + "  ".join(f"{k} {len(v)}" for k, v in parts.items()))

    with open(os.path.join(DST, "dataset.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": "WIDER FACE", "size": SIZE, "n": n,
                   "splits": {k: len(v) for k, v in parts.items()},
                   "min_sat": args.min_sat, "min_spread": args.min_spread,
                   "dropped": {"near_grey": grey, "low_spread": flat},
                   "note": "colourisation predicts ab from L; scene variety matters more "
                           "than resolution, so WIDER FACE beats DIV2K here"},
                  fh, indent=2)


if __name__ == "__main__":
    main()
