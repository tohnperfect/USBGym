#!/usr/bin/env python3
"""Time the physics B-mode render on each available device and write the timings to a JSON report.

It times the stages of the render chain of usbg.view_match.curvi_render one by one, and always writes the
report:

  PYTHONPATH=src .venv/bin/python scripts/bench_render.py --report=outputs/bench_render.json

Stages timed (matching usbg.view_match.curvi_render's chain):
  slice          slice_at_pose        label slice through the inverse affine
  render         render(..,"physics") the analytic single-pass forward model
  warp           warp_to_sector       linear -> curvilinear sector geometry
  render+warp    render + warp        one physics frame

SonoSPADE's per-tissue stage is timed separately by scripts/bench_latency.py; the two together are
the fully textured loop.

Seat (env USBG_SEAT): canonical times the v1 liver seat (depth 0.05, standoff 5 mm, the defaults the
v1 number used) and writes outputs/bench_render.json. Anatomical (default) times the probe on the
skin at screened liver window --window (standoff 0, gel-coupled render) and writes
outputs/v2/bench_render.json with a provenance block.

  USBG_SEAT=anatomical PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python \
      scripts/bench_render.py --n 200 --warmup 20 --report outputs/v2/bench_render.json

RenderConfig has no device field, so the renderer's device is passed explicitly and each row renders on the
device it names. In the v1 report outputs/bench_render.json the "cpu" row was rendered on MPS (an earlier
version set a device field that RenderConfig does not have, and get_device() chose MPS)
(its cpu and mps rows agree to 0.05 ms).
Each stage also reports p25_ms / p75_ms (interquartile range).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _timeit(fn, n, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": float(np.mean(ts)), "median_ms": float(np.median(ts)),
            "std_ms": float(np.std(ts)), "p25_ms": float(np.percentile(ts, 25)),
            "p75_ms": float(np.percentile(ts, 75)), "fps_from_median": float(1000.0 / np.median(ts))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="s0011")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--window", type=int, default=0,
                    help="anatomical seat: screened liver window index (modulo the count)")
    ap.add_argument("--report", default=None,
                    help="default outputs/bench_render.json (canonical), outputs/v2/bench_render.json "
                         "(anatomical)")
    args = ap.parse_args(argv)
    from usbg.seatmode import seat_mode
    seat = seat_mode()
    if args.report is None:
        args.report = "outputs/bench_render.json" if seat == "canonical" else "outputs/v2/bench_render.json"

    import torch
    from usbg import render_lotus as R, slicer as S, view_match as VM, volume as V
    from usbg.placement import seat_volume_under_probe
    from usbg.segmenter import align_label_to_image

    lv = V.load_cache(os.path.join("data/cache", f"{args.case}.npz"))
    # same seating the view-match task uses, so the timed pose is a pose the gym actually visits
    if seat == "canonical":
        wfv, T = seat_volume_under_probe(lv, V.LIVER, depth_m=0.05)        # v1: standoff default 5 mm
    else:
        wfv, T = seat_volume_under_probe(lv, V.LIVER, standoff_m=0.0, window=args.window)

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    print(f"torch {torch.__version__}; threads {torch.get_num_threads()}; devices {devices}")
    print(f"case {args.case}; size {args.size}; n={args.n} (warmup {args.warmup}); seat {seat}"
          + (f" window {args.window}" if seat == "anatomical" else "") + "\n")

    results = {}
    for dev in devices:
        cfg = R.default_config()
        # the device goes to render() itself: RenderConfig is frozen and has no device field, so the old
        # `cfg.device = dev` (inside a bare try/except) silently left every row on get_device() (MPS)

        lab_lin = S.slice_at_pose(lv, np.asarray(T, np.float64), wfv)
        stages = {}
        stages["slice"] = _timeit(lambda: S.slice_at_pose(lv, np.asarray(T, np.float64), wfv),
                                  args.n, args.warmup)
        stages["render"] = _timeit(lambda: R.render(lab_lin, "physics", device=dev, cfg=cfg), args.n, args.warmup)

        img_lin = R.render(lab_lin, "physics", device=dev, cfg=cfg)
        img_sq = VM._resize01(img_lin, args.size)
        lab_sq = align_label_to_image(lab_lin, (args.size, args.size))
        stages["warp"] = _timeit(lambda: VM.warp_to_sector(img_sq, lab_sq), args.n, args.warmup)

        def render_and_warp():
            il = R.render(lab_lin, "physics", device=dev, cfg=cfg)
            i2 = VM._resize01(il, args.size)
            l2 = align_label_to_image(lab_lin, (args.size, args.size))
            return VM.warp_to_sector(i2, l2)

        stages["render+warp"] = _timeit(render_and_warp, args.n, args.warmup)

        # A full physics frame the env produces per step: slice the volume at the pose, render, warp.
        # Timed as ONE block so the paper's "a physics frame" number is not a sum of separately-timed
        # stages (which overcounts by each stage's call overhead).
        def full_frame():
            ll = S.slice_at_pose(lv, np.asarray(T, np.float64), wfv)
            il = R.render(ll, "physics", device=dev, cfg=cfg)
            i2 = VM._resize01(il, args.size)
            l2 = align_label_to_image(ll, (args.size, args.size))
            return VM.warp_to_sector(i2, l2)

        stages["frame"] = _timeit(full_frame, args.n, args.warmup)
        results[dev] = stages

        print(f"[{dev}]")
        for k, v in stages.items():
            print(f"  {k:12s} {v['median_ms']:7.2f} ms (median)  mean {v['mean_ms']:6.2f} "
                  f"sd {v['std_ms']:5.2f}  -> {v['fps_from_median']:7.1f} fps")
        print()

    payload = {
        "case": args.case, "size": args.size, "n": args.n, "warmup": args.warmup,
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "machine": f"{platform.system()} {platform.machine()}",
        "note": "render+warp is what the paper calls 'a physics frame'; the physics-loop fps in the "
                "paper is derived from its median. frame = slice + render + resize + warp, timed as one "
                "block. Each row renders on the device it names (render(device=dev)).",
        "render_device_explicit": True,
        "per_device": results,
    }
    if seat == "anatomical":                     # the canonical report keeps its v1 layout
        from usbg import provenance as P
        payload["seat"] = {"mode": seat, "target": "liver", "window": args.window, "standoff_m": 0.0}
        payload["provenance"] = P.collect(case=args.case)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[report] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
