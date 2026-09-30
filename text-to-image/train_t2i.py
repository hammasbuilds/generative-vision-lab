"""Fine-tune tiny-SD (distilled Stable Diffusion) on Pokemon BLIP captions.

Model: `segmind/tiny-sd` - 0.99 GiB total, UNet only 617 MiB, so it fits a 16 GB card
comfortably where full SD 1.5 would be tight. Only the UNet trains; the VAE and text
encoder are frozen, which is the standard recipe and cuts memory further.

Data: `reach-vb/pokemon-blip-captions`, ~833 image/caption pairs. Small enough to finish,
large enough to learn a style - training a text-to-image model from scratch on 833
images would produce noise, so fine-tuning is the only honest option here.

Parameters derive from the dataset:  steps = epochs * N_train / batch.
N_train ~666 at batch 4 gives 166 steps/epoch, so 15,000 steps is about 90 epochs.
LR 1e-5 is the usual SD fine-tune rate; higher diverges and the damage shows up as
colour drift long before the loss moves.

There is no single scalar metric for text-to-image. Validation loss on held-out pairs is
tracked because it is the only automatic signal available, and fixed-prompt sample grids
are saved every chunk so the result can be judged by eye - which is how this task is
actually assessed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'tools'))
from autotune import LRAutoTune  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "distill-sd", "tiny-sd")
from dataroot import require as _ds  # noqa: E402
DATA = _ds("text-to-image", "pokemon", "data")
SIZE = 512

PROMPTS = [
    "a cartoon pikachu with a red hat",
    "a blue dragon pokemon with wings",
    "a green plant pokemon with big eyes",
    "a fire pokemon standing on rocks",
]


class PokemonSet(Dataset):
    """Reads the parquet once into memory - 833 rows at 512px is a few hundred MB."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        img, cap = self.rows[i]
        im = Image.open(io.BytesIO(img)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 127.5 - 1.0)
        return x, cap


def load_rows():
    import pyarrow.parquet as pq
    files = [f for f in sorted(os.listdir(DATA)) if f.endswith(".parquet")]
    rows = []
    for f in files:
        t = pq.ParquetFile(os.path.join(DATA, f)).read()
        cols = t.column_names
        icol = "image" if "image" in cols else cols[0]
        ccol = "text" if "text" in cols else ("caption" if "caption" in cols else cols[-1])
        for im, cap in zip(t.column(icol).to_pylist(), t.column(ccol).to_pylist()):
            b = im.get("bytes") if isinstance(im, dict) else im
            if b:
                rows.append((b, str(cap)))
    return rows


def save_rolling(path, payload):
    tmp = path + ".tmp"; torch.save(payload, tmp); os.replace(tmp, path)


def _require_cuda():
    # Abort rather than silently train on CPU. Installing anything that depends on
    # torch (facexlib, diffusers, transformers) pulls a CPU build into the workspace
    # venv, and PYTHONPATH is searched before the interpreter's own site-packages, so
    # that CPU build shadows the CUDA one. Training then runs ~50x slower with no
    # error - just a "pin_memory ... no accelerator found" warning in the log. That
    # cost two runs before it was spotted, so it is a hard failure now.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "  REFUSING TO TRAIN ON CPU: torch " + torch.__version__ +
            " reports no CUDA. A CPU torch is probably shadowing the CUDA build on "
            "PYTHONPATH. Check: python -c \"import torch;print(torch.__file__)\"")


def main() -> None:
    _require_cuda()
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    # The queue passes --total to every non-GAN model so the schedule spans the
    # whole run rather than each chunk. This trainer was missed when the flag
    # was added to the other six, and argparse rejected the unknown argument:
    # "unrecognized arguments: --total 15000", failing the chunk instantly.
    ap.add_argument("--total", type=int, default=0,
                    help="whole budget, so the schedule spans the run not the chunk")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every-steps", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    n0 = sum(len(f) for _, _, f in os.walk(DATA)); time.sleep(10)
    if sum(len(f) for _, _, f in os.walk(DATA)) != n0:
        raise SystemExit("  REFUSING TO TRAIN: dataset still downloading")

    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407); np.random.seed(3407); random.seed(3407)

    rows = load_rows()
    rng = random.Random(3407); rng.shuffle(rows)
    n = len(rows); n_test = max(1, round(n*.05)); n_val = max(1, round(n*.15))
    te_r, va_r, tr_r = rows[:n_test], rows[n_test:n_test+n_val], rows[n_test+n_val:]
    print(f"  {n} pairs -> train {len(tr_r)} val {len(va_r)} test {len(te_r)}", flush=True)

    tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    # tiny-sd ships text_encoder/config.json with torch_dtype: float16 while the
    # unet and vae carry no dtype and load as float32. The fp16 hidden states then
    # hit the unet's fp32 attention projection:
    #     RuntimeError: expected mat1 and mat2 to have the same dtype,
    #                   but got struct c10::Half != float
    # Everything is forced to float32; mixed precision comes from autocast and
    # GradScaler below, which is the supported way to get it.
    txt = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder",
                                        torch_dtype=torch.float32).to(device).eval()
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float32).to(device)
    sched = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
    for p in txt.parameters(): p.requires_grad_(False)
    for p in vae.parameters(): p.requires_grad_(False)
    print(f"  UNet {sum(p.numel() for p in unet.parameters())/1e6:.1f} M (training), "
          f"VAE + text encoder frozen", flush=True)

    # pin_memory=False, deliberately. Pinned pages are page-locked and cannot be paged
    # out, so under memory pressure they are the last thing that should be held: this run
    # died with `Unable to allocate 3.00 MiB for an array with shape (3, 512, 512)` while
    # another workload on the box held 14 GB. Losing the pinned-transfer speedup costs a
    # few percent; failing the chunk costs the whole chunk.
    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    tr = DataLoader(PokemonSet(tr_r), batch_size=args.bs, shuffle=True, drop_last=True,
                    pin_memory=False, **kw)
    va = DataLoader(PokemonSet(va_r), batch_size=args.bs, **kw)
    spe = max(1, len(tr_r) // args.bs)
    print(f"  batch {args.bs}  {spe} steps/epoch -> {args.steps} steps "
          f"= {args.steps/spe:.0f} epochs  lr {args.lr:.2e}", flush=True)

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    last = os.path.join(args.out, "t2i_last.pth")
    step, best = 0, {"loss": 1e9, "step": 0}
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        unet.load_state_dict(ck["unet"]); opt.load_state_dict(ck["optimizer"])
        step = ck["step"]; best = ck["best"]
        print(f"  RESUMED from step {step}", flush=True)

    def encode(caps):
        ids = tok(list(caps), padding="max_length", truncation=True,
                  max_length=tok.model_max_length, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            return txt(ids)[0]

    def batch_loss(x, caps, train=True):
        with torch.no_grad():
            lat = vae.encode(x.to(device)).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=device).long()
        noisy = sched.add_noise(lat, noise, t)
        emb = encode(caps)
        pred = unet(noisy, t, encoder_hidden_states=emb).sample
        return F.mse_loss(pred.float(), noise.float())

    @torch.no_grad()
    def val_loss():
        unet.eval(); ls = []
        for x, caps in va:
            ls.append(float(batch_loss(x, caps, False)))
        unet.train()
        return float(np.mean(ls)) if ls else float("nan")

    @torch.no_grad()
    def sample(tag):
        unet.eval()
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL, unet=unet, vae=vae, text_encoder=txt, tokenizer=tok,
            safety_checker=None, requires_safety_checker=False).to(device)
        d = os.path.join(args.out, "inference_t2i"); os.makedirs(d, exist_ok=True)
        g = torch.Generator(device=device).manual_seed(3407)   # fixed seed: comparable
        for i, p in enumerate(PROMPTS):
            im = pipe(p, num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
            im.save(os.path.join(d, f"prompt{i}_{tag}.png"))
        with open(os.path.join(d, "prompts.json"), "w", encoding="utf-8") as fh:
            json.dump(PROMPTS, fh, indent=2)
        unet.train()

    print(f"  BEFORE: val loss {val_loss():.4f}", flush=True)
    sample("before")

    # val loss is the metric here and LOWER is better, so it is negated before
    # being fed to the tuner, which expects higher-is-better.
    tuner = LRAutoTune(opt, good=0.0005, max_lr=5e-5)
    hist, t0 = [], time.perf_counter()
    unet.train()
    while step < args.steps:
        for x, caps in tr:
            if step >= args.steps: break
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = batch_loss(x, caps)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            step += 1
            if step % args.save_every_steps == 0 or step == args.steps:
                vl = val_loss(); hist.append({"step": step, "train": float(loss.detach()), "val": vl})
                flag = ""
                if vl < best["loss"]:
                    best = {"loss": vl, "step": step}
                    torch.save({"unet": unet.state_dict()},
                               os.path.join(args.out, "t2i_best.pth")); flag = "  *best"
                print(f"  step {step:6d}/{args.steps}  train {float(loss.detach()):.4f}  "
                      f"val {vl:.4f}{flag}", flush=True)
                print(f"      autotune: {tuner.step(-vl)}", flush=True)
                save_rolling(last, {"unet": unet.state_dict(), "optimizer": opt.state_dict(),
                                    "step": step, "best": best, "history": hist,
                                    "args": vars(args)})

    mins = (time.perf_counter() - t0) / 60
    sample("after")
    print(f"\n  trained {args.steps} steps in {mins:.1f} min; best val {best['loss']:.4f} "
          f"@ {best['step']}", flush=True)
    with open(os.path.join(args.out, "results_t2i.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": "segmind/tiny-sd", "n_pairs": n, "batch": args.bs,
                   "steps": args.steps, "lr": args.lr, "minutes": round(mins, 2),
                   "best": best, "history": hist,
                   "note": "no single scalar metric for text-to-image; val loss is the "
                           "only automatic signal. Judge the fixed-prompt samples."},
                  fh, indent=2)


if __name__ == "__main__":
    main()
