"""What the lab's committed outputs are, read from the project folders.

The repo ships no trained weights (the checkpoints are 109-294 MB, over GitHub's file
limit), so nothing here runs a model. Every image is a file a training run wrote into a
project's ``samples/`` folder, and every number is either read from that project's
``results.json`` or computed here from the saved PNGs.

The ``*_before_*`` files need care. The training scripts write them when a run starts
*or resumes*, so for a run that resumed they are a mid-run checkpoint, not the starting
weights. Each task below says which one it is, and ``tests/test_app.py`` checks the
label against the run's own history.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

EARLIER = "Earlier checkpoint"
EARLIER_WHY = ("This run resumed from a checkpoint, and the training script writes its "
               "'before' images at that moment, so this column is a mid-run checkpoint "
               "of the same model, not the starting weights.")

# kind "restore": input -> output, scored against a ground truth.
# kind "sr":      as restore, plus a bicubic column.
# kind "prompt":  text prompt -> image, no ground truth.
# kind "epochs":  one progress grid per saved epoch (input | output [| ground truth]).
TASKS = [
    {"id": "cloud-removal", "title": "Satellite cloud removal", "kind": "restore",
     "folder": "cloud-removal", "input": "input_cloudy",
     "model": "SpA-GAN generator (0.22 M parameters), fine-tuned",
     "data": "RICE, 2,000 aligned cloudy/clear pairs at 256x256, split 1600/300/100",
     "before": EARLIER, "before_note": EARLIER_WHY},
    {"id": "colourisation", "title": "Colourisation", "kind": "restore",
     "folder": "colourisation", "input": "input_grey",
     "model": "ColourNet, 4.38 M parameters, trained from scratch (L to ab in Lab space)",
     "data": "WIDER FACE scenes at 512x512, split 2280/428/142",
     "before": EARLIER, "before_note": EARLIER_WHY},
    {"id": "face-restoration", "title": "Face restoration", "kind": "restore",
     "folder": "face-restoration", "input": "input_degraded",
     "model": "GFPGANv1 generator, fine-tuned with L1",
     "data": "FFHQ, 1,000 faces at 512x512, split 800/150/50",
     "before": EARLIER, "before_note": EARLIER_WHY},
    {"id": "document-shadow-removal", "title": "Document shadow removal", "kind": "restore",
     "folder": "document-shadow-removal", "input": "input_shadow",
     "model": "DocShadow, fine-tuned from the published SD7K weights",
     "data": "ISTD shadow / shadow-free pairs, split 800/150/50",
     "before": "Published SD7K weights",
     "before_note": "The published weights before any fine-tuning (val PSNR 27.148)."},
]
for _m, _label in (("esrgan", "ESRGAN"), ("msrgan", "MSRGAN"),
                   ("realesrgan", "Real-ESRGAN"), ("realesrnet", "Real-ESRNet")):
    TASKS.append({
        "id": f"sr-{_m}", "title": f"Super-resolution x6 - {_label}", "kind": "sr",
        "folder": f"super-resolution/{_m}", "input": "input_lr",
        "model": f"{_label}, fine-tuned from the published x4 checkpoint into an x6 network",
        "data": "720 training frames at 2K, held-out val and test",
        "before": "Before fine-tuning",
        "before_note": "The network at the start of the run, before any fine-tuning step."})
TASKS.append({
    "id": "text-to-image", "title": "Text to image", "kind": "prompt", "folder": "text-to-image",
    "model": "tiny-SD (segmind/tiny-sd), UNet fine-tuned; VAE and text encoder frozen",
    "data": "833 Pokemon caption/image pairs at 512x512",
    "before": EARLIER, "before_note": EARLIER_WHY})
for _m, _title in (("pix2pix-maps", "pix2pix: map tile to aerial photo"),
                   ("pix2pix-facades", "pix2pix: label map to building facade"),
                   ("pix2pix-facades-l1", "pix2pix: facades, L1 only"),
                   ("cyclegan-a2o", "CycleGAN: apple to orange"),
                   ("cyclegan-s2w", "CycleGAN: summer to winter"),
                   ("cut-h2z", "CUT: horse to zebra")):
    TASKS.append({"id": _m, "title": _title, "kind": "epochs",
                  "folder": f"gan-translation/{_m}"})

BY_ID = {t["id"]: t for t in TASKS}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def split_scores(task: dict) -> list[dict]:
    """The run's own numbers, as rows of {split, metric, value}."""
    res = _read_json(ROOT / task["folder"] / "results.json")
    rows = []
    final = res.get("final") or {}
    for split in ("train", "val", "test"):
        for metric in ("psnr", "ssim"):
            v = (final.get(split) or {}).get(metric)
            if isinstance(v, (int, float)):
                label = "PSNR (dB)" if metric == "psnr" else "SSIM"
                if res.get("metric", "").startswith("negative cycle"):
                    label, v = "cycle-consistency L1", -v
                rows.append({"split": split, "metric": label, "value": round(float(v), 4)})
    best = res.get("best") or {}
    if isinstance(best.get("loss"), (int, float)):
        rows.append({"split": "val (best)", "metric": "diffusion loss",
                     "value": round(float(best["loss"]), 4)})
    return rows


def samples(task: dict) -> list[dict]:
    folder = ROOT / task["folder"] / "samples"
    out = []
    if task["kind"] in ("restore", "sr"):
        for gt in sorted(folder.glob("*_ground_truth.png")):
            stem = gt.name[: -len("_ground_truth.png")]
            item = {"id": stem,
                    "input": rel(folder / f"{stem}_{task['input']}.png"),
                    "before": rel(folder / f"{stem}_before_output.png"),
                    "after": rel(folder / f"{stem}_after_output.png"),
                    "truth": rel(gt)}
            if task["kind"] == "sr":
                item["bicubic"] = rel(folder / f"{stem}_bicubic.png")
            out.append(item)
    elif task["kind"] == "prompt":
        prompts = _read_json(folder / "prompts.json")
        prompts = prompts if isinstance(prompts, list) else []
        for i, p in enumerate(prompts):
            out.append({"id": f"prompt{i}", "prompt": p,
                        "before": rel(folder / f"prompt{i}_before.png"),
                        "after": rel(folder / f"prompt{i}_after.png")})
    elif task["kind"] == "epochs":
        for f in sorted(folder.glob("e*.png")):
            m = re.fullmatch(r"e(\d+)\.png", f.name)
            if m:
                out.append({"id": f.stem, "epoch": int(m.group(1)), "grid": rel(f)})
    return out


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def public_file(relpath: str) -> Path | None:
    """Resolve a requested file, allowing only PNG/JSON inside a task's samples folder."""
    if not relpath or "\\" in relpath or relpath.startswith("/"):
        return None
    p = (ROOT / relpath).resolve()
    for t in TASKS:
        base = (ROOT / t["folder"] / "samples").resolve()
        if p.parent == base and p.suffix.lower() in (".png", ".json") and p.is_file():
            return p
    return None


@lru_cache(maxsize=512)
def thumbnail(path: Path, size: int = 200) -> bytes:
    """A small JPEG of a sample for the picker; the SR frames are 2K PNGs of several MB."""
    import io

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse <= 1e-12 else 10.0 * float(np.log10(255.0 ** 2 / mse))


@lru_cache(maxsize=256)
def sample_scores(task_id: str, sample_id: str) -> dict:
    """RGB PSNR of every column against the ground truth, from the saved PNGs."""
    task = BY_ID[task_id]
    item = next((s for s in samples(task) if s["id"] == sample_id), None)
    if item is None or "truth" not in item:
        return {}
    truth = _rgb(ROOT / item["truth"])
    out = {}
    for key in ("input", "bicubic", "before", "after"):
        if key in item:
            img = _rgb(ROOT / item[key])
            if img.shape == truth.shape:
                out[key] = round(psnr(img, truth), 2)
    return out


def catalogue() -> dict:
    tasks = []
    for t in TASKS:
        tasks.append({**{k: v for k, v in t.items() if k != "folder"},
                      "folder": t["folder"], "scores": split_scores(t), "samples": samples(t)})
    return {"tasks": tasks}
