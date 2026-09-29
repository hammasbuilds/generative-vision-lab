"""Colourisation: greyscale -> colour, trained from scratch on images already on disk.

Built rather than adapted. DDColor's shipped config is a 400,000-iteration from-scratch
run on full ImageNet across 4 GPUs with `pretrain_network_g: ~`; its `pretrain/` holds
only a placeholder. Copying that onto one 16 GB card is a multi-week run that ends worse
than the checkpoint they published, so the repo's training path is not usable here.

This is a compact U-Net doing the standard formulation: predict the **ab** chrominance
channels from the **L** luminance channel in Lab space. That is the right shape for the
task - the input already carries all the luminance, so a model that predicts RGB wastes
capacity relearning brightness it was given.

Data: any colour photographs. The 1,000 sharp LIP-ATR images prepared for
super-resolution serve directly, so this needs no download at all - greyscale inputs are
made by conversion, exactly as the degradation-synthesis projects do.

Metric note: PSNR against the original colours is reported, but it is a weak judge here.
A plausible recolouring that differs from the original scores badly, which is why
DDColor's own config measures FID and colourfulness and leaves PSNR commented out. The
saved before/after images matter more than the number.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..']*2), 'tools'))
from autotune import LRAutoTune  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
from dataroot import require as _ds  # noqa: E402
SRC = _ds("colour", "gt")
SIZE = 256


def rgb_to_lab(x: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> Lab. Written out rather than pulling in skimage."""
    m = x > 0.04045
    lin = np.where(m, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505
    X /= 0.95047; Z /= 1.08883
    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return np.stack([L, a, bb], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    def inv(t):
        return np.where(t ** 3 > 0.008856, t ** 3, (t - 16 / 116) / 7.787)
    X = inv(fx) * 0.95047; Y = inv(fy); Z = inv(fz) * 1.08883
    r = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    g = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    bl = X * 0.0557 + Y * -0.2040 + Z * 1.0570
    out = np.stack([r, g, bl], axis=-1)
    m = out > 0.0031308
    out = np.where(m, 1.055 * np.clip(out, 0, None) ** (1 / 2.4) - 0.055, 12.92 * out)
    return np.clip(out, 0, 1)


class ColourSet(Dataset):
    def __init__(self, files, train=True):
        self.files, self.train = files, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        if self.train and random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        lab = rgb_to_lab(np.asarray(im, np.float32) / 255.0)
        L = lab[..., :1] / 100.0                      # [0,1]
        ab = lab[..., 1:] / 110.0                     # roughly [-1,1]
        return (torch.from_numpy(L.transpose(2, 0, 1).astype(np.float32)),
                torch.from_numpy(ab.transpose(2, 0, 1).astype(np.float32)))


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))


class ColourNet(nn.Module):
    """Small U-Net: 1 channel in (L), 2 out (ab)."""

    def __init__(self, w=48):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = blk(1, w), blk(w, w*2), blk(w*2, w*4), blk(w*4, w*8)
        self.d3 = blk(w*8 + w*4, w*4); self.d2 = blk(w*4 + w*2, w*2); self.d1 = blk(w*2 + w, w)
        self.out = nn.Conv2d(w, 2, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        u = F.interpolate(e4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u, e3], 1))
        u = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u, e2], 1))
        u = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u, e1], 1))
        return torch.tanh(self.out(d1))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ps = []
    for L, ab in loader:
        L, ab = L.to(device), ab.to(device)
        p = model(L)
        for i in range(L.shape[0]):
            lab_t = np.concatenate([(L[i].cpu().numpy().transpose(1,2,0))*100,
                                    (ab[i].cpu().numpy().transpose(1,2,0))*110], axis=-1)
            lab_p = np.concatenate([(L[i].cpu().numpy().transpose(1,2,0))*100,
                                    (p[i].cpu().numpy().transpose(1,2,0))*110], axis=-1)
            a = lab_to_rgb(lab_t); b = lab_to_rgb(lab_p)
            mse = float(np.mean((a-b)**2))
            ps.append(10*np.log10(1.0/max(mse,1e-12)))
    return {"psnr": float(np.mean(ps)), "n": len(ps)}


@torch.no_grad()
def save_inference(model, files, device, out_dir, tag, k=4):
    os.makedirs(out_dir, exist_ok=True); model.eval()
    for p in files[:k]:
        stem = os.path.splitext(os.path.basename(p))[0]
        im = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        lab = rgb_to_lab(np.asarray(im, np.float32)/255.0)
        L = lab[..., :1]/100.0
        x = torch.from_numpy(L.transpose(2,0,1).astype(np.float32))[None].to(device)
        ab = model(x)[0].cpu().numpy().transpose(1,2,0)*110
        rgb = lab_to_rgb(np.concatenate([L*100, ab], axis=-1))
        im.convert("L").save(os.path.join(out_dir, f"{stem}_input_grey.png"))
        im.save(os.path.join(out_dir, f"{stem}_ground_truth.png"))
        Image.fromarray((rgb*255).astype("uint8")).save(
            os.path.join(out_dir, f"{stem}_{tag}_output.png"))


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


def main():
    _require_cuda()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--total", type=int, default=0,
                    help="whole budget, so the cosine spans the run not the chunk")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every-epochs", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--decay-lr", action="store_true",
                    help="opt back in to cosine decay; OFF by default - the rate is held")
    ap.add_argument("--out", default=os.path.join(HERE, "results_colour"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SRC, "*.png")))
    if len(files) < 100:
        raise SystemExit(f"  only {len(files)} images in {SRC}")
    rng = random.Random(3407); rng.shuffle(files)
    n = len(files); n_test = max(1, round(n*.05)); n_val = max(1, round(n*.15))
    te_f, va_f, tr_f = files[:n_test], files[n_test:n_test+n_val], files[n_test+n_val:]

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407); np.random.seed(3407); random.seed(3407)
    model = ColourNet().to(device)
    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    tr = DataLoader(ColourSet(tr_f), batch_size=args.bs, shuffle=True, drop_last=True,
                    pin_memory=True, **kw)
    va = DataLoader(ColourSet(va_f, False), batch_size=args.bs, **kw)
    te = DataLoader(ColourSet(te_f, False), batch_size=args.bs, **kw)
    spe = max(1, len(tr_f)//args.bs)
    print(f"  ColourNet {sum(p.numel() for p in model.parameters())/1e6:.2f} M params (scratch)")
    print(f"  train {len(tr_f)} val {len(va_f)} test {len(te_f)}")
    print(f"  batch {args.bs}  {spe} steps/epoch -> {args.epochs} ep = {args.epochs*spe} steps"
          f"  lr {args.lr:.2e}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # T_max is the WHOLE budget, not this chunk's target. The queue trains in 5-epoch
    # chunks and passes the chunk end as --epochs, so a cosine built on that completed
    # inside every chunk and landed on eta_min at its end. The queue then read that
    # floor out of the checkpoint and passed it as the next chunk's --lr, so the rate
    # stuck at the floor permanently: gfpgan ran its last 7,500 steps - half its budget
    # - at 1e-7, its autotuner reporting "healthy, lr held" the whole time because the
    # gain was positive and it had no idea the schedule had bottomed out.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.total or args.epochs), eta_min=1e-5)
    last = os.path.join(args.out, "colour_last.pth"); best_p = os.path.join(args.out, "colour_best.pth")
    start, best = 0, {"psnr": -1, "epoch": -1}
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        try:
            sched.load_state_dict(ck["scheduler"])
        except KeyError:
            # The scheduler state is from a different scheduler type. Runs that
            # started under CosineAnnealingLR save no "lr_lambdas", so restoring
            # them into the constant LambdaLR raises KeyError and killed
            # sr_msrgan at 5,000 of 20,000 steps. The schedule is rebuilt below
            # anyway, so the stored state is not needed - only the weights and
            # the optimiser are.
            print("  scheduler state is from a different scheduler type; "
                  "rebuilding instead of restoring", flush=True); start = ck["epoch"]+1; best = ck["best"]
        print(f"  RESUMED from epoch {start}", flush=True)

        # A restored optimiser carries the DECAYED rate, not the base rate. The override
        # below compares sched.base_lrs, which still reads the original 2e-4 and
        # therefore matches --lr, so it never fires - while the optimiser is actually
        # running at whatever the cosine had decayed to. colourisation resumed and
        # trained at 1e-6, two hundred times below the floor, with the autotuner
        # reporting "healthy, lr held" because the metric happened to be rising.
        _cur = opt.param_groups[0]["lr"]
        if _cur < 1e-5:
            for _g in opt.param_groups:
                _g["lr"] = max(args.lr, 1e-5)
            print(f"  LR FLOOR: restored {_cur:.2e} is below the 1e-5 floor -> "
                  f"{opt.param_groups[0]['lr']:.2e}", flush=True)

        # A restored scheduler carries its own base_lr and rewrites the optimiser's lr
        # on every .step(), so a --lr given on a resume is silently discarded. That is
        # what kept MSRGAN on an old 1e-4 schedule (decayed to 5e-5) after asking for
        # 2e-4, and left it flat for 6,000 steps looking converged. Honour the caller.
        _restored = sched.base_lrs[0] if getattr(sched, "base_lrs", None) else None
        if _restored and abs(_restored - args.lr) / max(args.lr, 1e-12) > 0.01:
            for _g in opt.param_groups:
                _g["lr"] = args.lr
            sched =         torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, max(1, args.epochs - start), eta_min=1e-5)
            print(f"  LR OVERRIDE: restored base {_restored:.2e} -> using {args.lr:.2e}",
                  flush=True)


    before = evaluate(model, va, device)
    print(f"  BEFORE (random init): val PSNR {before['psnr']:.3f}", flush=True)
    save_inference(model, te_f, device, os.path.join(args.out, "inference_colour"), "before")

    tuner = LRAutoTune(opt, good=0.002)

    if not args.decay_lr:
        # Measured on colourisation: the cosine restarts at the chunk's high rate every
        # time the queue resumes, which COST 0.25 dB per chunk before the decay brought
        # it back. Mean gain per epoch-block by rate: 3.63e-4 -0.2484, 2e-4 +0.0997,
        # 1e-5 +0.2832, 1e-6 +0.0225. Almost all of each chunk was spent recovering from
        # damage the high rate had just done. Holding the best measured rate removes the
        # restart entirely.
        for _g in opt.param_groups:
            _g["lr"] = args.lr
            # A previous CosineAnnealingLR leaves "initial_lr" behind in every param
            # group, and LambdaLR takes THAT as its base rather than the current lr. So
            # the constant schedule multiplied the old 2e-4 base by 1.0 and pinned the
            # run at 2e-4 while printing "LR HELD AT 1.00e-05" - true when printed, false
            # one line later. Clear it so the base is the rate actually asked for.
            _g.pop("initial_lr", None)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        tuner = None
        print(f"  LR HELD AT {args.lr:.2e} for the whole chunk "
              f"(no cosine, no per-epoch autotune; the queue changes it between chunks only when the metric says so)",
              flush=True)
    hist, t0 = [], time.perf_counter()
    for ep in range(start, args.epochs):
        model.train(); tot = nb = 0
        for L, ab in tr:
            L, ab = L.to(device, non_blocking=True), ab.to(device, non_blocking=True)
            loss = F.l1_loss(model(L), ab)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        v = evaluate(model, va, device); hist.append({"epoch": ep, "loss": tot/max(nb,1), **v})
        flag = ""
        if v["psnr"] > best["psnr"]:
            best = {"psnr": v["psnr"], "epoch": ep}
            torch.save({"model": model.state_dict()}, best_p); flag = "  *best"
        if ep % 5 == 0 or ep == args.epochs-1:
            print(f"  epoch {ep:4d}/{args.epochs}  loss {tot/max(nb,1):.4f}  "
                  f"val PSNR {v['psnr']:.3f}  lr {opt.param_groups[0]['lr']:.2e}{flag}", flush=True)
            if tuner is not None:
                print(f"      autotune: {tuner.step(v['psnr'])}", flush=True)
        if (ep+1) % args.save_every_epochs == 0 or ep == args.epochs-1:
            save_rolling(last, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                "scheduler": sched.state_dict(), "epoch": ep, "best": best,
                                "history": hist, "args": vars(args)})

    mins = (time.perf_counter()-t0)/60
    if os.path.exists(best_p):
        model.load_state_dict(torch.load(best_p, map_location=device)["model"])
    res = {"train": evaluate(model, DataLoader(ColourSet(tr_f[:len(va_f)], False), batch_size=args.bs), device),
           "val": evaluate(model, va, device), "test": evaluate(model, te, device)}
    save_inference(model, te_f, device, os.path.join(args.out, "inference_colour"), "after")
    print(f"\n  trained in {mins:.1f} min; best val {best['psnr']:.3f} @ {best['epoch']}")
    for s in ("train","val","test"):
        print(f"  {s:7s} PSNR {res[s]['psnr']:.3f}")
    with open(os.path.join(args.out, "results_colour.json"), "w", encoding="utf-8") as fh:
        json.dump({"n": n, "batch": args.bs, "epochs": args.epochs, "lr": args.lr,
                   "minutes": round(mins,2), "before_val": before, "final": res,
                   "history": hist,
                   "note": "scratch U-Net, L->ab in Lab space. PSNR is a weak judge of "
                           "colourisation - a plausible recolouring that differs from the "
                           "original scores badly. Judge the images."}, fh, indent=2)


if __name__ == "__main__":
    main()
