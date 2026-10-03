"""Fine-tune DocShadow on ISTD shadow / shadow-free pairs.

ISTD layout is `<split>/<split>_A` (shadow), `_B` (mask), `_C` (shadow-free). Only A and
C are used here: A is the input, C is the target.

Honest scope note. DocShadow was trained on SD7K, which is **photographed documents**.
ISTD is **natural scenes** - shadows cast on ground and objects. Fine-tuning on ISTD
therefore adapts the model to a different domain, and the result should be described as
general shadow removal, not document shadow removal. No document-shadow dataset exists
on any source reachable from this machine (searched HF for document shadow, docshadow,
Jung, Kligler - all empty; SD7K itself is OneDrive/Kaggle only).

The repo's own training path needs `accelerate` and `yacs` and wraps everything in
HuggingFace Accelerate; this is a direct loop instead, matching how the SR models here
are trained so the two are comparable.

Parameters derive from the dataset:  steps = epochs * N_train / batch.
"""

from __future__ import annotations

import argparse
import glob
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
sys.path.insert(0, os.path.join(HERE, "DocShadow-SD7K"))
from dataroot import require as _ds  # noqa: E402
DATA = _ds("istd-shadow")
CKPT = os.path.join(HERE, "..", "..", "_weights", "docshadow_sd7k.pth")
PATCH = 256


def pairs(split: str) -> list[tuple[str, str]]:
    a = sorted(glob.glob(os.path.join(DATA, split, f"{split}_A", "*.png")))
    out = []
    for p in a:
        c = p.replace(f"{split}_A", f"{split}_C")
        if os.path.exists(c):
            out.append((p, c))
    return out


class ShadowSet(Dataset):
    def __init__(self, items, train=True):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pa, pc = self.items[i]
        a = Image.open(pa).convert("RGB")
        c = Image.open(pc).convert("RGB")
        if self.train:
            w, h = a.size
            x = random.randint(0, max(0, w - PATCH)); y = random.randint(0, max(0, h - PATCH))
            a = a.crop((x, y, x + PATCH, y + PATCH)); c = c.crop((x, y, x + PATCH, y + PATCH))
            if random.random() < 0.5:
                a = a.transpose(Image.FLIP_LEFT_RIGHT); c = c.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            a = a.resize((PATCH, PATCH), Image.BILINEAR)
            c = c.resize((PATCH, PATCH), Image.BILINEAR)
        f = lambda im: torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.)
        return f(a), f(c)


def psnr_t(x, y):
    mse = torch.mean((x - y) ** 2).item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ps = []
    for a, c in loader:
        a, c = a.to(device), c.to(device)
        out = model(a).clamp(0, 1)
        for i in range(a.shape[0]):
            ps.append(psnr_t(out[i], c[i]))
    return {"psnr": float(np.mean(ps)), "n": len(ps)}


@torch.no_grad()
def save_inference(model, device, out_dir, tag, items, k=4):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for pa, pc in items[:k]:
        stem = os.path.splitext(os.path.basename(pa))[0]
        a = Image.open(pa).convert("RGB").resize((PATCH, PATCH), Image.BILINEAR)
        c = Image.open(pc).convert("RGB").resize((PATCH, PATCH), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(a, np.float32).transpose(2, 0, 1) / 255.)[None].to(device)
        y = model(x).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        a.save(os.path.join(out_dir, f"{stem}_input_shadow.png"))
        c.save(os.path.join(out_dir, f"{stem}_ground_truth.png"))
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


def main():
    _require_cuda()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
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

    # Refuse to train on a dataset that is still being written.
    n0 = sum(len(f) for _, _, f in os.walk(DATA)); time.sleep(20)
    n1 = sum(len(f) for _, _, f in os.walk(DATA))
    if n1 != n0:
        raise SystemExit(f"  REFUSING TO TRAIN: {DATA} still growing ({n0} -> {n1})")

    all_tr = pairs("train") or pairs("test")     # ISTD test split is the larger mirror
    if len(all_tr) < 50:
        raise SystemExit(f"  only {len(all_tr)} pairs found under {DATA}; not enough")
    rng = random.Random(3407); rng.shuffle(all_tr)
    n = len(all_tr); n_test = max(1, round(n * .05)); n_val = max(1, round(n * .15))
    te_i, va_i, tr_i = all_tr[:n_test], all_tr[n_test:n_test+n_val], all_tr[n_test+n_val:]

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407); np.random.seed(3407); random.seed(3407)

    from models import Model
    model = Model()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    # Saved from a DataParallel wrapper: every key is `module.*`. strict=False would
    # report success having loaded nothing at all.
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model = model.to(device)

    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    tr = DataLoader(ShadowSet(tr_i, True), batch_size=args.bs, shuffle=True,
                    drop_last=True, pin_memory=True, **kw)
    va = DataLoader(ShadowSet(va_i, False), batch_size=args.bs, **kw)
    te = DataLoader(ShadowSet(te_i, False), batch_size=args.bs, **kw)
    spe = max(1, len(tr_i) // args.bs)
    print(f"  DocShadow {sum(p.numel() for p in model.parameters())/1e6:.2f} M params on {device}")
    print(f"  pairs {n}  train {len(tr_i)} val {len(va_i)} test {len(te_i)}")
    print(f"  batch {args.bs}  {spe} steps/epoch -> {args.epochs} epochs = "
          f"{args.epochs*spe} steps  lr {args.lr}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    # T_max is the WHOLE budget, not this chunk's target. The queue trains in 5-epoch
    # chunks and passes the chunk end as --epochs, so a cosine built on that completed
    # inside every chunk and landed on eta_min at its end. The queue then read that
    # floor out of the checkpoint and passed it as the next chunk's --lr, so the rate
    # stuck at the floor permanently: gfpgan ran its last 7,500 steps - half its budget
    # - at 1e-7, its autotuner reporting "healthy, lr held" the whole time because the
    # gain was positive and it had no idea the schedule had bottomed out.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.total or args.epochs), eta_min=1e-5)
    last = os.path.join(args.out, "docshadow_last.pth")
    best_p = os.path.join(args.out, "docshadow_best.pth")
    start, best = 0, {"psnr": -1, "epoch": -1}
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        # Restore the epoch counter and the best score UNCONDITIONALLY.
        #
        # These two assignments used to sit inside the `except KeyError` below, so they
        # only happened when the scheduler state failed to load. When it loaded cleanly
        # the weights and optimiser resumed but `start` stayed 0 and `best` stayed
        # {"psnr": -1} - so the run reported "RESUMED from epoch 0" having actually
        # resumed epoch-29 weights, and the first epoch became "*best" automatically and
        # overwrote docshadow_best.pth with a model that had trained for one epoch.
        start, best = ck["epoch"] + 1, ck["best"]
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
                  "rebuilding instead of restoring", flush=True)
        print(f"  RESUMED from epoch {start} (best {best['psnr']:.3f} "
              f"at epoch {best['epoch']})", flush=True)

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

    # Re-baseline `best` against THIS validation set.
    #
    # A stored best is only comparable while the val set is unchanged. The checkpoint
    # here carried best=30.640 from a run on 540 triples; this run's val set is 150
    # images, where the very same weights score 27.148. Carrying 30.640 forward makes an
    # unbeatable target: no epoch would ever save a new best, docshadow_best.pth would
    # keep whatever it already held, and the final `load_state_dict(best_p)` would throw
    # away every epoch of this run.
    #
    # The resumed model's score on the current val set is the only honest starting point.
    if best["psnr"] > before["psnr"]:
        print(f"  stored best {best['psnr']:.3f} (epoch {best['epoch']}) is not "
              f"comparable with this val set - re-baselining to {before['psnr']:.3f}",
              flush=True)
        best = {"psnr": before["psnr"], "epoch": start - 1}
    save_inference(model, device, os.path.join(args.out, "inference_docshadow"), "before", te_i)

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
        for a, c in tr:
            a, c = a.to(device, non_blocking=True), c.to(device, non_blocking=True)
            loss = F.l1_loss(model(a), c)
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
    res = {"train": evaluate(model, DataLoader(ShadowSet(tr_i, False), batch_size=args.bs), device),
           "val": evaluate(model, va, device), "test": evaluate(model, te, device)}
    save_inference(model, device, os.path.join(args.out, "inference_docshadow"), "after", te_i)
    print(f"\n  trained in {mins:.1f} min; best val PSNR {best['psnr']:.3f} @ {best['epoch']}")
    for s in ("train","val","test"):
        print(f"  {s:7s} PSNR {res[s]['psnr']:.3f}  (n={res[s]['n']})")
    print(f"  BEFORE {before['psnr']:.3f} -> AFTER {res['val']['psnr']:.3f} "
          f"= {res['val']['psnr']-before['psnr']:+.3f} dB")
    with open(os.path.join(args.out, "results_docshadow.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_pairs": n, "batch": args.bs, "epochs": args.epochs, "lr": args.lr,
                   "minutes": round(mins,2), "before_val": before, "final": res,
                   "domain_note": "fine-tuned on ISTD natural scenes, not documents",
                   "history": hist}, fh, indent=2)


if __name__ == "__main__":
    main()
