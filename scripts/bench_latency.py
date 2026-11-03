#!/usr/bin/env python3
"""Inference-latency benchmark for the SonoSPADE texture stage.

Measures single-frame latency (the RL / interactive regime: one frame per step, batch 1) on CPU
(no GPU) and on the available accelerator, plus a diffusion-cost reference (number of network
evaluations). Reports ms/frame and frames/s. This is the evidence for the low-cost, real-time,
RL-oriented framing: a diffusion synthesizer needs tens to hundreds of denoising steps per frame,
SonoSPADE needs one generator pass.

Checkpoint and input frames (env USBG_SEAT): canonical keeps the v1/SASHIMI measurement
(data/dataset/exp_real_sonospade_curvi.pt on data/dataset/pairs_med_curvi.npz). Anatomical (default)
times the v2 SonoSPADE stage from --tex-manifest (outputs/v2/textures/manifest.json, key "sonospade")
on the v2 curvilinear pairs, refuses the v1 checkpoints, and writes outputs/v2/bench_latency.json
with a provenance block unless --report says otherwise.

  USBG_SEAT=anatomical PYTHONPATH=src .venv/bin/python scripts/bench_latency.py \
      --tex-manifest outputs/v2/textures/manifest.json --report outputs/v2/bench_latency.json

--stages (default: sonospade only, the behavior and report layout above, unchanged) times more texture
stages from the same manifest on the same frames, e.g. all three learned stages of the v2 comparison:

  USBG_SEAT=anatomical PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python \
      scripts/bench_latency.py --tex-manifest outputs/v2/textures/manifest.json \
      --stages cut cyclegan sonospade --report outputs/v2/bench_latency.json

CUT and CycleGAN are single-input (rendered sector frame -> textured frame); SonoSPADE also takes the
aligned label. With --stages the report keeps the SonoSPADE "ckpt"/"per_device" keys (when SonoSPADE is
among the stages) and adds "per_stage": {stage: {"ckpt", "md5", "per_device": {dev: {mean_ms, std_ms,
median_ms, p25_ms, p75_ms, fps}}}}. Physics has no texture stage (identity), so it is not timed here.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

# ---- texture checkpoints: the v2 manifest under the anatomical seat; the v1 paths only under canonical
TEX_MANIFEST = "outputs/v2/textures/manifest.json"
LEGACY_TEX = {"physics": None, "cut": "data/dataset/exp_real_cut_curvi.pt",       # v1 / SASHIMI, trained
              "cyclegan": "data/dataset/exp_real_cyclegan_curvi.pt",              # on pairs that include
              "sonospade": "data/dataset/exp_real_sonospade_curvi.pt"}            # held-out patients (D3)


def texture_ckpts(manifest=None):
    """{stage: checkpoint path or None (physics)}. An explicit manifest always wins; otherwise the
    anatomical seat reads TEX_MANIFEST and the canonical seat keeps the v1 checkpoints."""
    import json
    from usbg.seatmode import seat_mode
    if manifest is None and seat_mode() == "canonical":
        return dict(LEGACY_TEX)
    path = manifest or TEX_MANIFEST
    if not os.path.exists(path):
        raise SystemExit(f"texture manifest {path} not found (written by the v2 texture runs); pass "
                         f"--tex-manifest / an explicit checkpoint, or set USBG_SEAT=canonical for v1")
    with open(path) as fh:
        m = json.load(fh)
    return {k: m.get(k) for k in ("physics", "cut", "cyclegan", "sonospade")}


LEGACY_TEX_MD5 = {"d2f6e37d1b5dbd53f3203e7d5241e518", "ea9e275f4a6974eb1c3a6279656d0253",  # curvi CUT, CycleGAN
                  "997c9e4f84a10dcdc2b7da1470bad381", "3edb0ee0ff32d5b72fb4e4a8c5bda693",  # curvi CycleGAN-SC, SonoSPADE
                  "d9d6f7c85c43b5529d7358e653121a09"}                                        # linear SonoSPADE


def check_texture_ckpt(path):
    """Under the anatomical seat, refuse a v1 texture checkpoint (by name or md5, so a renamed copy
    is caught) or a private-pool descendant, and warn when the provenance sidecar is missing."""
    from usbg.seatmode import seat_mode
    if not path or seat_mode() == "canonical":
        return path
    if not os.path.exists(path):
        raise SystemExit(f"texture checkpoint {path} not found")
    from usbg.provenance import is_private_descendant, md5_file
    md5 = md5_file(path)
    legacy = md5 in LEGACY_TEX_MD5 or (os.path.basename(path).startswith("exp_real_") and "v2" not in path)
    why = is_private_descendant(path, md5) or ("a v1 texture checkpoint (canonical seat only)" if legacy else None)
    if why:
        raise SystemExit(f"USBG_SEAT=anatomical refuses texture checkpoint {path}: {why}")
    if not any(os.path.exists(p) for p in (path + ".provenance.json", os.path.splitext(path)[0] + ".provenance.json")):
        print(f"[warn] texture checkpoint {path} has no provenance sidecar")
    return path


V2_PAIRS = ("data/dataset/v2/pairs_med_curvi.npz", "data/dataset/v2/pairs_med_curvi_v2.npz")


def resolve_v2_pairs(explicit=None, manifest=None):
    """Curvilinear pairs for the anatomical seat: --pairs, else the manifest's "pairs_curvi" key, else
    the first existing V2_PAIRS path."""
    import json
    if explicit:
        return explicit
    mp = manifest or TEX_MANIFEST
    if os.path.exists(mp):
        with open(mp) as fh:
            p = json.load(fh).get("pairs_curvi")
        if p:
            return p
    for p in V2_PAIRS:
        if os.path.exists(p):
            return p
    raise SystemExit(f"no v2 curvilinear pairs found (tried {list(V2_PAIRS)}); pass --pairs")


def bench(device_str, ckpt, pairs, n=100, warmup=15, stage="sonospade", return_ts=False):
    """Single-frame (batch 1) latency of one texture stage. SonoSPADE: translate_aligned(img, label);
    CUT / CycleGAN: the single-input generator, translator(img). Default call = the original SonoSPADE
    measurement; return_ts=True also returns the per-frame times (ms) for percentiles."""
    import torch
    from usbg import texture_gan as TG
    from usbg.dataset import load_pairs_npz
    dev = torch.device(device_str)
    if stage == "sonospade":
        tr = TG.SonoSPADETranslator(ckpt, device=dev)
        run = lambda im, lb: tr.translate_aligned(im, lb)
    else:
        tr = TG.load_translator(ckpt, device=dev)
        if getattr(tr, "wants_label", False):
            raise SystemExit(f"{stage} checkpoint {ckpt} is a dual-input (label-conditioned) generator")
        run = lambda im, lb: tr(im)
    imgs, labs, _s, _m = load_pairs_npz(pairs)
    imgs = imgs.astype(np.float32) / 255.0
    labs = labs.astype(np.int64)
    idx = list(range(min(n + warmup, len(imgs))))
    # warmup
    for i in idx[:warmup]:
        run(imgs[i % len(imgs)], labs[i % len(imgs)])
    if device_str == "mps":
        torch.mps.synchronize()
    # timed, single-frame
    ts = []
    for i in idx[warmup:]:
        t0 = time.perf_counter()
        run(imgs[i % len(imgs)], labs[i % len(imgs)])
        if device_str == "mps":
            torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    ts = np.array(ts)
    if return_ts:
        return float(ts.mean()), float(ts.std()), float(np.median(ts)), ts
    return float(ts.mean()), float(ts.std()), float(np.median(ts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="SonoSPADE checkpoint (default: manifest 'sonospade' under the anatomical seat, "
                         "data/dataset/exp_real_sonospade_curvi.pt under canonical)")
    ap.add_argument("--tex-manifest", default=None,
                    help=f"texture manifest (default {TEX_MANIFEST} under the anatomical seat)")
    ap.add_argument("--pairs", default=None,
                    help="input frames (default: v2 curvilinear pairs under the anatomical seat, "
                         "data/dataset/pairs_med_curvi.npz under canonical)")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=15, help="untimed frames before timing (default 15)")
    ap.add_argument("--stages", nargs="+", default=["sonospade"], choices=["cut", "cyclegan", "sonospade"],
                    help="texture stages to time (default: sonospade only, the original measurement). "
                         "Other stages read their checkpoint from the texture manifest.")
    ap.add_argument("--report", default=None,
                    help="write the measured latencies to JSON. Without this the numbers exist only "
                         "on stdout, so nothing in the paper's timing claims is reproducible or "
                         "checkable after the fact. Anatomical default: outputs/v2/bench_latency.json")
    args = ap.parse_args(argv)
    from usbg.seatmode import seat_mode
    seat = seat_mode()
    if args.ckpt is None:
        args.ckpt = texture_ckpts(args.tex_manifest)["sonospade"]     # canonical: the v1 checkpoint
        if not args.ckpt:
            raise SystemExit("the texture manifest has no 'sonospade' checkpoint; pass --ckpt")
    if args.pairs is None:
        args.pairs = ("data/dataset/pairs_med_curvi.npz" if seat == "canonical"
                      else resolve_v2_pairs(None, args.tex_manifest))
    if seat == "anatomical" and args.report is None:
        args.report = "outputs/v2/bench_latency.json"
    check_texture_ckpt(args.ckpt)
    print(f"seat {seat}; ckpt {args.ckpt}; pairs {args.pairs}")

    import torch
    print(f"torch {torch.__version__}; threads {torch.get_num_threads()}")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    multi = list(args.stages) != ["sonospade"]
    per_stage = {}
    if multi:                                     # opt-in: every requested stage, same frames, same devices
        ckpts = texture_ckpts(args.tex_manifest)
        from usbg import provenance as _P
        for st in args.stages:
            ck = args.ckpt if st == "sonospade" else ckpts.get(st)
            if not ck:
                raise SystemExit(f"the texture manifest has no {st!r} checkpoint")
            check_texture_ckpt(ck)
            per_stage[st] = {"ckpt": ck, "md5": _P.md5(ck), "per_device": {}}
            for dev in devices:
                mean, std, med, ts = bench(dev, ck, args.pairs, n=args.n, warmup=args.warmup, stage=st,
                                            return_ts=True)
                per_stage[st]["per_device"][dev] = dict(
                    mean_ms=mean, std_ms=std, median_ms=med, p25_ms=float(np.percentile(ts, 25)),
                    p75_ms=float(np.percentile(ts, 75)), fps=1000.0 / mean, fps_from_median=1000.0 / med,
                    n_timed=int(len(ts)))
                print(f"{st:9s} {dev:4s}  {mean:7.2f} ms/frame (median {med:6.2f}, IQR "
                      f"{np.percentile(ts, 25):6.2f}-{np.percentile(ts, 75):6.2f}, sd {std:5.2f})  ->  "
                      f"{1000.0 / mean:7.1f} fps")
    results = {}
    if "sonospade" in per_stage:                  # reuse the SonoSPADE run for the legacy keys
        results = {d: {k: v[k] for k in ("mean_ms", "std_ms", "median_ms", "fps")}
                   for d, v in per_stage["sonospade"]["per_device"].items()}
    elif not multi:
        for dev in devices:
            mean, std, med = bench(dev, args.ckpt, args.pairs, n=args.n, warmup=args.warmup)
            fps = 1000.0 / mean
            results[dev] = dict(mean_ms=mean, std_ms=std, median_ms=med, fps=fps)
            print(f"{dev:4s}  {mean:7.2f} ms/frame (median {med:6.2f}, sd {std:5.2f})  ->  {fps:7.1f} fps")
    # diffusion reference: latency scales with the number of denoising steps (one net eval each)
    print("\nreference: a semantic-diffusion synthesizer runs T denoising steps per frame")
    for T in ([50, 250, 1000] if "cpu" in results else []):
        eq = results["cpu"]["mean_ms"] * T
        print(f"  T={T:4d} steps -> ~{eq/1000:6.2f} s/frame at SonoSPADE's per-pass CPU cost "
              f"({1000.0/eq:.3f} fps)")
    if args.report:
        import torch as _t
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        payload = {
            "ckpt": args.ckpt, "pairs": args.pairs, "n": args.n,
            "torch": _t.__version__, "threads": _t.get_num_threads(),
            "machine": f"{platform.system()} {platform.machine()}",
            "per_device": results,
        }
        if multi:
            payload["stages"] = list(args.stages)
            payload["per_stage"] = per_stage
            payload["pairs_frames_timed"] = int(args.n)
            payload["warmup_frames"] = int(args.warmup)
            payload["batch"] = 1
        if seat == "anatomical":                 # the canonical report keeps its v1 layout
            from usbg import provenance as P
            payload["provenance"] = P.collect(stage="sonospade", texture_ckpt=args.ckpt,
                                              pairs=args.pairs, pairs_md5=P.md5(args.pairs),
                                              tex_manifest=args.tex_manifest or TEX_MANIFEST)
            if multi:
                payload["provenance"]["stages_timed"] = {k: {"ckpt": v["ckpt"], "md5": v["md5"]}
                                                         for k, v in per_stage.items()}
        with open(args.report, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[report] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
