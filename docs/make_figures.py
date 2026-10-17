"""Build the README comparison figures from the committed sample outputs.

Nothing is re-run: every panel is a PNG a training run saved under a project's
``samples/`` folder, and each score is RGB PSNR against the ground truth computed from
those PNGs (app/catalogue.py), so it can differ from the run's own protocol.

    python docs/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import catalogue as C  # noqa: E402

OUT = ROOT / "docs" / "figures"


def load(rel: str) -> np.ndarray:
    with Image.open(ROOT / rel) as im:
        return np.asarray(im.convert("RGB"))


def detail_crop(truth: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of the size x size window with the most gradient energy."""
    g = truth.astype(np.float64).mean(axis=2)
    e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]
    s = e.cumsum(0).cumsum(1)
    best, at = -1.0, (0, 0)
    step = size // 4
    for y in range(0, e.shape[0] - size, step):
        for x in range(0, e.shape[1] - size, step):
            v = s[y + size - 1, x + size - 1] - s[y, x + size - 1] - s[y + size - 1, x] + s[y, x]
            if v > best:
                best, at = v, (y, x)
    return at


def grid(rows, columns, title, name, cell=2.6):
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(cell * len(columns), cell * 1.05 * len(rows)))
    for r, (label, cells) in enumerate(rows):
        for c, (img, caption) in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            top = columns[c] if r == 0 else ""
            ax.set_title("\n".join(x for x in (top, caption) if x), fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"Sr {r + 1}\n{label}", fontsize=8.5)
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def restore_figure(task_id: str, name: str) -> None:
    t = C.BY_ID[task_id]
    rows = []
    for s in C.samples(t):
        sc = C.sample_scores(task_id, s["id"])
        rows.append((s["id"], [
            (load(s["input"]), f"PSNR {sc['input']:.2f}"),
            (load(s["before"]), f"PSNR {sc['before']:.2f}"),
            (load(s["after"]), f"PSNR {sc['after']:.2f}"),
            (load(s["truth"]), ""),
        ]))
    grid(rows, ["Input", t["before"], "Fine-tuned model", "Ground truth"],
         f"{t['title']}: four held-out images (RGB PSNR in dB against the ground truth)", name)


def sr_figure(name: str) -> None:
    models = [("sr-msrgan", "MSRGAN"), ("sr-realesrnet", "Real-ESRNet"),
              ("sr-realesrgan", "Real-ESRGAN"), ("sr-esrgan", "ESRGAN")]
    first = C.BY_ID["sr-esrgan"]
    rows = []
    for s in C.samples(first):
        truth = load(s["truth"])
        size = 192
        y, x = detail_crop(truth, size)
        cut = lambda a: a[y:y + size, x:x + size]  # noqa: E731
        base = C.sample_scores("sr-esrgan", s["id"])
        cells = [(cut(load(s["input"])), f"PSNR {base['input']:.2f}"),
                 (cut(load(s["bicubic"])), f"PSNR {base['bicubic']:.2f}")]
        for tid, _ in models:
            item = next(i for i in C.samples(C.BY_ID[tid]) if i["id"] == s["id"])
            sc = C.sample_scores(tid, s["id"])
            cells.append((cut(load(item["after"])), f"PSNR {sc['after']:.2f}"))
        cells.append((cut(truth), f"{truth.shape[1]}x{truth.shape[0]} frame"))
        rows.append((f"frame {s['id']}", cells))
    grid(rows, ["Input (x6 down)", "Bicubic"] + [m for _, m in models] + ["Ground truth"],
         "x6 super-resolution, four held-out 2K frames, each fine-tuned model. A 192 px detail crop is "
         "shown; PSNR is over the whole frame", name, cell=2.2)


def prompt_figure(name: str) -> None:
    t = C.BY_ID["text-to-image"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.2))
    for c, s in enumerate(C.samples(t)):
        for r, key in enumerate(("before", "after")):
            ax = axes[r, c]
            ax.imshow(load(s[key]))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Sr {c + 1}: \"{s['prompt']}\"", fontsize=8.5)
            if c == 0:
                ax.set_ylabel("Earlier checkpoint" if key == "before" else "Final model", fontsize=9)
    fig.suptitle("Text to image: tiny-SD fine-tuned on Pokemon captions, same seed for every prompt",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


if __name__ == "__main__":
    restore_figure("cloud-removal", "cloud_removal.png")
    restore_figure("colourisation", "colourisation.png")
    restore_figure("face-restoration", "face_restoration.png")
    restore_figure("document-shadow-removal", "document_shadow_removal.png")
    sr_figure("super_resolution.png")
    prompt_figure("text_to_image.png")
