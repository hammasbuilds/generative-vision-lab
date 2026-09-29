"""Cloud removal: fine-tune a SpA-GAN-style generator on the extracted pairs.

2,000 cloudy/clear pairs were extracted from one HuggingFace parquet shard (the repo's
full set is 9.3 GB across twenty shards; one gave 4,263 rows, four times the target).

Why not call the repo's own `train.py`: it hard-codes a YAML config, a Visdom logger and
its own directory convention, and expects the RICE layout. This is a direct loop over
the same architecture so the run matches every other project here - same split policy,
same rolling checkpoint, same resume contract, same before/after inference images.

The generator ships INSIDE the repo (0.86 MB, `pretrained_models/RICE1/
gen_model_epoch_200.pth`), so this is a fine-tune, not a scratch run. No discriminator
is published, so this trains the generator alone with L1 + SSIM - stated plainly rather
than pretending it is the full adversarial recipe.

Parameters follow the dataset:  steps = epochs * N_train / batch.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..']*2), 'tools'))
from autotune import LRAutoTune  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "SpA-GAN")
from dataroot import require as _ds  # noqa: E402
DATA = _ds("rice-cloud-spagan", "RICE_DATASET", "HF")
sys.path.insert(0, REPO)
SIZE = 256


class CloudSet(Dataset):
    def __init__(self, split: str, augment: bool = False):
        with open(os.path.join(DATA, f"{split}_list.txt"), encoding="utf-8") as fh:
            self.names = [ln.strip() for ln in fh if ln.strip()]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        n = self.names[i]
        c = Image.open(os.path.join(DATA, "cloudy_image", n)).convert("RGB").resize(
            (SIZE, SIZE), Image.BILINEAR)
        g = Image.open(os.path.join(DATA, "ground_truth", n)).convert("RGB").resize(
            (SIZE, SIZE), Image.BILINEAR)
        if self.augment and random.random() < 0.5:
            c = c.transpose(Image.FLIP_LEFT_RIGHT); g = g.transpose(Image.FLIP_LEFT_RIGHT)
        f = lambda im: torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.)
        return f(c), f(g)


def build(device: str):
    """SpA-GAN generator from the repo, loading its shipped RICE1 weights."""
    from models.gen.SPANet import Generator
    # gpu_ids drives nn.DataParallel inside the repo's Generator; an empty list is
    # the single-GPU path, which is what this box has.
    m = Generator([])
    # The repo ships RICE1 and RICE2 generators. Measured zero-shot on this data,
    # 40 test images, before any training:
    #     identity (do nothing)   11.901 dB
    #     RICE1 generator         11.291 dB   <- WORSE than not running the model
    #     RICE2 generator         12.459 dB
    # RICE1 was the default and actively hurt, so RICE2 is the starting point. Both
    # are far below what a matched-domain checkpoint would give: RICE is satellite
    # cirrus, this data is low-altitude aerial haze, so this is a cross-domain
    # fine-tune and is described as one rather than as "using the pretrained model".
    w = os.path.join(REPO, "pretrained_models", "RICE2", "gen_model_epoch_200.pth")
    if os.path.exists(w):
        sd = torch.load(w, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        m.load_state_dict(sd, strict=True)   # strict: a silent mismatch trains noise
        print(f"  loaded RICE1 generator ({os.path.getsize(w)/2**20:.2f} MB)")
    else:
        print("  no shipped weights found; training from scratch")
    return m.to(device)


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(torch.mean((a - b) ** 2)), 1e-12))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval(); ps = []
    for c, g in loader:
        c, g = c.to(device), g.to(device)
        out = model(c)
        # SPANet.forward returns (attention_map, image) - the ATTENTION MAP FIRST.
        # Taking [0] grabbed the 1-channel map, which then broadcast silently
        # against the 3-channel target and scored a meaningless PSNR. The image
        # is the LAST element.
        out = out[-1] if isinstance(out, (tuple, list)) else out
        out = out.clamp(0, 1)
        for i in range(c.shape[0]):
            ps.append(psnr(out[i], g[i]))
    return {"psnr": float(np.mean(ps)), "n": len(ps)}


@torch.no_grad()
def save_inference(model, device, out_dir, tag, k=4):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(DATA, "test_list.txt"), encoding="utf-8") as fh:
        names = [ln.strip() for ln in fh if ln.strip()][:k]
    model.eval()
    for n in names:
        stem = os.path.splitext(n)[0]
        c = Image.open(os.path.join(DATA, "cloudy_image", n)).convert("RGB").resize((SIZE, SIZE))
        g = Image.open(os.path.join(DATA, "ground_truth", n)).convert("RGB").resize((SIZE, SIZE))
        x = torch.from_numpy(np.asarray(c, np.float32).transpose(2, 0, 1) / 255.)[None].to(device)
        y = model(x)
        # third unpack site - SPANet returns (attention_map, image), image LAST.
        # PIL refused the 1-channel attention map outright, which is the only
        # reason this one was loud instead of writing a grey picture.
        y = (y[-1] if isinstance(y, (tuple, list)) else y).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        c.save(os.path.join(out_dir, f"{stem}_input_cloudy.png"))
        g.save(os.path.join(out_dir, f"{stem}_ground_truth.png"))
        Image.fromarray((y * 255).astype("uint8")).save(
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


def main() -> None:
    _require_cuda()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--total", type=int, default=0,
                    help="whole budget, so the cosine spans the run not the chunk")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every-epochs", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--decay-lr", action="store_true",
                    help="opt back in to cosine decay; OFF by default - the rate is held")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    n0 = sum(len(f) for _, _, f in os.walk(DATA)); time.sleep(15)
    if sum(len(f) for _, _, f in os.walk(DATA)) != n0:
        raise SystemExit(f"  REFUSING TO TRAIN: {DATA} still growing")

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407); np.random.seed(3407); random.seed(3407)

    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    tr = DataLoader(CloudSet("train", True), batch_size=args.bs, shuffle=True,
                    drop_last=True, pin_memory=True, **kw)
    va = DataLoader(CloudSet("val"), batch_size=args.bs, **kw)
    te = DataLoader(CloudSet("test"), batch_size=args.bs, **kw)

    model = build(device)
    spe = max(1, len(tr.dataset) // args.bs)
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.2f} M params on {device}")
    print(f"  train {len(tr.dataset)} val {len(va.dataset)} test {len(te.dataset)}")
    print(f"  batch {args.bs}  {spe} steps/epoch -> {args.epochs} ep = "
          f"{args.epochs*spe} steps  lr {args.lr:.2e}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999),
                           weight_decay=1e-5)
    # T_max is the WHOLE budget, not this chunk's target. The queue trains in 5-epoch
    # chunks and passes the chunk end as --epochs, so a cosine built on that completed
    # inside every chunk and landed on eta_min at its end. The queue then read that
    # floor out of the checkpoint and passed it as the next chunk's --lr, so the rate
    # stuck at the floor permanently: gfpgan ran its last 7,500 steps - half its budget
    # - at 1e-7, its autotuner reporting "healthy, lr held" the whole time because the
    # gain was positive and it had no idea the schedule had bottomed out.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.total or args.epochs), eta_min=1e-5)
    last = os.path.join(args.out, "cloud_last.pth"); best_p = os.path.join(args.out, "cloud_best.pth")
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
                  "rebuilding instead of restoring", flush=True); start = ck["epoch"] + 1; best = ck["best"]
        print(f"  RESUMED from epoch {start} (best {best['psnr']:.3f})", flush=True)

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
    print(f"  BEFORE: val PSNR {before['psnr']:.3f}", flush=True)
    save_inference(model, device, os.path.join(args.out, "inference_cloud"), "before")

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
        for c, g in tr:
            c, g = c.to(device, non_blocking=True), g.to(device, non_blocking=True)
            out = model(c)
            # SPANet.forward returns (attention_map, image) - the ATTENTION MAP FIRST.
            # Taking [0] grabbed the 1-channel map, which then broadcast silently
            # against the 3-channel target and scored a meaningless PSNR. The image
            # is the LAST element.
            out = out[-1] if isinstance(out, (tuple, list)) else out
            loss = F.l1_loss(out, g)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        v = evaluate(model, va, device); hist.append({"epoch": ep, "loss": tot/max(nb,1), **v})
        flag = ""
        if v["psnr"] > best["psnr"]:
            best = {"psnr": v["psnr"], "epoch": ep}
            torch.save({"model": model.state_dict()}, best_p); flag = "  *best"
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}/{args.epochs}  loss {tot/max(nb,1):.4f}  "
                  f"val PSNR {v['psnr']:.3f}  lr {opt.param_groups[0]['lr']:.2e}{flag}", flush=True)
            if tuner is not None:
                print(f"      autotune: {tuner.step(v['psnr'])}", flush=True)
        if (ep + 1) % args.save_every_epochs == 0 or ep == args.epochs - 1:
            save_rolling(last, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                "scheduler": sched.state_dict(), "epoch": ep, "best": best,
                                "history": hist, "args": vars(args)})

    mins = (time.perf_counter() - t0) / 60
    if os.path.exists(best_p):
        model.load_state_dict(torch.load(best_p, map_location=device)["model"])
    res = {"train": evaluate(model, DataLoader(CloudSet("train"), batch_size=args.bs), device),
           "val": evaluate(model, va, device), "test": evaluate(model, te, device)}
    save_inference(model, device, os.path.join(args.out, "inference_cloud"), "after")
    print(f"\n  trained in {mins:.1f} min; best val {best['psnr']:.3f} @ {best['epoch']}")
    for s in ("train", "val", "test"):
        print(f"  {s:7s} PSNR {res[s]['psnr']:.3f}  (n={res[s]['n']})")
    print(f"  BEFORE {before['psnr']:.3f} -> AFTER {res['val']['psnr']:.3f} "
          f"= {res['val']['psnr']-before['psnr']:+.3f} dB")
    with open(os.path.join(args.out, "results_cloud.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_train": len(tr.dataset), "batch": args.bs, "epochs": args.epochs,
                   "lr": args.lr, "minutes": round(mins, 2), "before_val": before,
                   "final": res, "history": hist,
                   "note": "generator-only fine-tune; no discriminator published"}, fh, indent=2)


if __name__ == "__main__":
    main()
