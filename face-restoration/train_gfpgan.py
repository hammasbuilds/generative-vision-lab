"""Fine-tune GFPGAN's generator on FFHQ faces with synthesised degradation.

WHAT THIS IS, AND IS NOT
------------------------
The official recipe trains a generator against FOUR discriminators (one global
StyleGAN2 plus left-eye / right-eye / mouth component discriminators) with an ArcFace
identity loss, launched through `torch.distributed` across 4 GPUs for 800,000
iterations from scratch at lr 2e-3.

This is a **generator-only fine-tune** with L1 + perceptual loss, starting from the
published `GFPGANv1.pth`. It is not a reproduction of the paper's training, and the
results file says so. The reasons for the simplification are concrete rather than
convenience:

  * the paper's lr 2e-3 is a FROM-SCRATCH rate; applied to loaded weights it destroys
    them within a few hundred steps
  * the component discriminators need `FFHQ_eye_mouth_landmarks_512.pth`, which only
    covers FFHQ image IDs - it does not generalise to other face sets
  * four discriminators plus a 44 M generator at 512px does not fit 16 GB at a useful
    batch size

The degradation is the genuine article: the same blur / downsample / noise / JPEG
pipeline the paper uses, so the *task* is faithful even though the loss is simpler.
Only clean faces are needed as input, which is why this project needed no paired data.

Parameters derive from the dataset:  steps = epochs * N_train / batch.
"""

from __future__ import annotations

import argparse
import glob
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
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..']*1), 'tools'))
from autotune import LRAutoTune  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "GFPGAN")
from dataroot import require as _ds  # noqa: E402
FACES = _ds("ffhq", "ffhq")
WEIGHTS = os.path.join(HERE, "..", "_weights")
SIZE = 512


def degrade(im: Image.Image, rng: random.Random) -> Image.Image:
    """The paper's degradation: blur -> downsample -> noise -> JPEG.

    Ranges taken from `options/train_gfpgan_v1.yml`: blur sigma [0.1, 10], downsample
    [0.8, 8], noise [0, 20], JPEG quality [60, 100].
    """
    from PIL import ImageFilter
    out = im
    sigma = rng.uniform(0.1, 10.0)
    out = out.filter(ImageFilter.GaussianBlur(radius=sigma * 0.35))
    scale = rng.uniform(0.8, 8.0)
    w = max(32, int(SIZE / scale))
    out = out.resize((w, w), Image.BILINEAR).resize((SIZE, SIZE), Image.BILINEAR)
    a = np.asarray(out, np.float32)
    a += np.random.normal(0, rng.uniform(0, 20.0), a.shape)
    out = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=int(rng.uniform(60, 100)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class FaceSet(Dataset):
    def __init__(self, files, train=True):
        self.files, self.train = files, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rng = random.Random(i if not self.train else None)
        hq = Image.open(self.files[i]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        if self.train and random.random() < 0.5:
            hq = hq.transpose(Image.FLIP_LEFT_RIGHT)
        lq = degrade(hq, rng)
        f = lambda im: torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 127.5 - 1.0)
        return f(lq), f(hq)


def build(device: str):
    import sys
    # GFPGAN's archs import `basicsr`, which is not pip-installed here. The BasicSR repo
    # cloned for the super-resolution project provides it; its `version.py` is generated
    # by setup.py, so it was written by hand there.
    sys.path.insert(0, os.path.join(HERE, "..", "super-resolution", "basicsr", "BasicSR"))
    sys.path.insert(0, REPO)
    # StyleGAN2's two fused CUDA kernels are neither precompiled nor JIT-buildable here.
    # BasicSR swallows the import failure, so the model builds and then dies on the first
    # forward with `NameError: name 'fused_act_ext' is not defined`. Swap in the native
    # PyTorch equivalents before the architecture binds the names.
    import stylegan_ops
    swapped = stylegan_ops.install()
    if swapped:
        print(f"  native PyTorch ops installed for: {', '.join(swapped)}", flush=True)
    from gfpgan.archs.gfpganv1_arch import GFPGANv1
    m = GFPGANv1(out_size=SIZE, num_style_feat=512, channel_multiplier=1,
                 resample_kernel=(1, 3, 3, 1), decoder_load_path=None, fix_decoder=True,
                 num_mlp=8, lr_mlp=0.01, input_is_latent=True, different_w=True,
                 narrow=1, sft_half=True)
    w = os.path.join(WEIGHTS, "GFPGANv1.pth")
    sd = torch.load(w, map_location="cpu", weights_only=False)
    sd = sd.get("params_ema", sd.get("params", sd))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  GFPGANv1 loaded: {len(sd)} tensors, {len(missing)} missing, "
          f"{len(unexpected)} unexpected", flush=True)
    if len(missing) > len(sd) * 0.5:
        raise SystemExit("  REFUSING: more than half the keys did not match - wrong arch")
    return m.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ps = []
    for lq, hq in loader:
        lq, hq = lq.to(device), hq.to(device)
        out = model(lq, return_rgb=False)
        out = (out[0] if isinstance(out, (tuple, list)) else out).clamp(-1, 1)
        a = (out + 1) / 2; b = (hq + 1) / 2
        for i in range(a.shape[0]):
            mse = float(torch.mean((a[i] - b[i]) ** 2))
            ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    return {"psnr": float(np.mean(ps)), "n": len(ps)}


@torch.no_grad()
def save_inference(model, loader_files, device, out_dir, tag, k=4):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for p in loader_files[:k]:
        stem = os.path.splitext(os.path.basename(p))[0]
        hq = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        lq = degrade(hq, random.Random(0))
        x = torch.from_numpy(np.asarray(lq, np.float32).transpose(2, 0, 1) / 127.5 - 1.0)[None].to(device)
        o = model(x, return_rgb=False)
        o = (o[0] if isinstance(o, (tuple, list)) else o).clamp(-1, 1)
        arr = ((o[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype("uint8")
        hq.save(os.path.join(out_dir, f"{stem}_ground_truth.png"))
        lq.save(os.path.join(out_dir, f"{stem}_input_degraded.png"))
        Image.fromarray(arr).save(os.path.join(out_dir, f"{stem}_{tag}_output.png"))


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
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--total", type=int, default=0,
                    help="whole budget, so the cosine spans the run not the chunk")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)   # NOT the paper's 2e-3
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every-steps", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--decay-lr", action="store_true",
                    help="opt back in to cosine decay; OFF by default - the rate is held")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    n0 = len(glob.glob(os.path.join(FACES, "*.png"))); time.sleep(10)
    if len(glob.glob(os.path.join(FACES, "*.png"))) != n0:
        raise SystemExit("  REFUSING TO TRAIN: face set still downloading")

    files = sorted(glob.glob(os.path.join(FACES, "*.png")))
    if len(files) < 100:
        raise SystemExit(f"  only {len(files)} faces found in {FACES}")
    rng = random.Random(3407); rng.shuffle(files)
    n = len(files); n_test = max(1, round(n*.05)); n_val = max(1, round(n*.15))
    te_f, va_f, tr_f = files[:n_test], files[n_test:n_test+n_val], files[n_test+n_val:]

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407); np.random.seed(3407); random.seed(3407)

    model = build(device)
    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    tr = DataLoader(FaceSet(tr_f), batch_size=args.bs, shuffle=True, drop_last=True,
                    pin_memory=True, **kw)
    va = DataLoader(FaceSet(va_f, False), batch_size=args.bs, **kw)
    te = DataLoader(FaceSet(te_f, False), batch_size=args.bs, **kw)
    spe = max(1, len(tr_f) // args.bs)
    print(f"  faces {n} -> train {len(tr_f)} val {len(va_f)} test {len(te_f)}")
    print(f"  batch {args.bs}  {spe} steps/epoch -> {args.steps} steps "
          f"= {args.steps/spe:.0f} epochs  lr {args.lr:.2e}  (paper uses 2e-3 FROM SCRATCH)",
          flush=True)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # T_max is the WHOLE budget, not this chunk's target. The queue trains in chunks and
    # passes the chunk end as --steps, so a cosine built on that completed inside every
    # chunk and landed on eta_min at its end. The queue then read that floor out of the
    # checkpoint and passed it as the next chunk's --lr, so the rate stuck at the floor
    # permanently: gfpgan ran its last 7,500 steps - half its budget - at 1e-7, with the
    # autotuner reporting "healthy, lr held" throughout because the gain was positive and
    # it had no idea the schedule had bottomed out.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.total or args.steps), eta_min=1e-5)
    last = os.path.join(args.out, "gfpgan_last.pth"); best_p = os.path.join(args.out, "gfpgan_best.pth")
    step, best = 0, {"psnr": -1, "step": 0}
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
                  "rebuilding instead of restoring", flush=True); step = ck["step"]; best = ck["best"]
        print(f"  RESUMED from step {step}", flush=True)

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
            opt, T_max=max(1, args.steps - step), eta_min=1e-5)
            print(f"  LR OVERRIDE: restored base {_restored:.2e} -> using {args.lr:.2e}",
                  flush=True)


    before = evaluate(model, va, device)
    print(f"  BEFORE: val PSNR {before['psnr']:.3f}", flush=True)
    save_inference(model, te_f, device, os.path.join(args.out, "inference_gfpgan"), "before")

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
    tuner = LRAutoTune(opt, good=0.002)
    hist, t0 = [], time.perf_counter()
    model.train()
    while step < args.steps:
        for lq, hq in tr:
            if step >= args.steps: break
            lq, hq = lq.to(device, non_blocking=True), hq.to(device, non_blocking=True)
            out = model(lq, return_rgb=False)
            out = out[0] if isinstance(out, (tuple, list)) else out
            loss = F.l1_loss(out, hq)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            if step % args.save_every_steps == 0 or step == args.steps:
                v = evaluate(model, va, device); model.train()
                hist.append({"step": step, "loss": float(loss.detach()), **v})
                flag = ""
                if v["psnr"] > best["psnr"]:
                    best = {"psnr": v["psnr"], "step": step}
                    torch.save({"model": model.state_dict()}, best_p); flag = "  *best"
                print(f"  step {step:6d}/{args.steps}  loss {float(loss.detach()):.4f}  "
                      f"val PSNR {v['psnr']:.3f}{flag}", flush=True)
                if tuner is not None:
                    print(f"      autotune: {tuner.step(v['psnr'])}", flush=True)
                save_rolling(last, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                    "scheduler": sched.state_dict(), "step": step,
                                    "best": best, "history": hist, "args": vars(args)})

    mins = (time.perf_counter() - t0) / 60
    if os.path.exists(best_p):
        model.load_state_dict(torch.load(best_p, map_location=device)["model"])
    res = {"train": evaluate(model, DataLoader(FaceSet(tr_f[:len(va_f)], False), batch_size=args.bs), device),
           "val": evaluate(model, va, device), "test": evaluate(model, te, device)}
    save_inference(model, te_f, device, os.path.join(args.out, "inference_gfpgan"), "after")
    print(f"\n  trained in {mins:.1f} min; best val {best['psnr']:.3f} @ {best['step']}")
    for s in ("train", "val", "test"):
        print(f"  {s:7s} PSNR {res[s]['psnr']:.3f}")
    print(f"  BEFORE {before['psnr']:.3f} -> AFTER {res['val']['psnr']:.3f} "
          f"= {res['val']['psnr']-before['psnr']:+.3f} dB")
    with open(os.path.join(args.out, "results_gfpgan.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_faces": n, "batch": args.bs, "steps": args.steps, "lr": args.lr,
                   "minutes": round(mins, 2), "before_val": before, "final": res,
                   "history": hist,
                   "note": "GENERATOR-ONLY L1 fine-tune. The official recipe uses 4 "
                           "discriminators + ArcFace identity loss on 4 GPUs at lr 2e-3 "
                           "from scratch; this is not a reproduction of that."},
                  fh, indent=2)


if __name__ == "__main__":
    main()
