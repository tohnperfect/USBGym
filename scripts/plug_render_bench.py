#!/usr/bin/env python3
"""plug_render_bench.py: per-frame cost, determinism and preview frames of the plugged renderers.

v2.1 pluggability study, renderer slot (exploratory). For each renderer (physics reference, the in-repo LOTUS
reimplementation, the ORIGINAL LOTUS forward model through usbg.plug.renderers, SIMUS through PyMUST) on each
device it supports (torch renderers: cpu and mps; SIMUS: cpu, NumPy plus a process pool), it times the
stages of scripts/bench_render.py at the same pose (TRAIN patient --case, liver window --window, anatomical
seat, probe on the skin):

  render   render_lotus.render(label slice, <renderer>)          the forward model alone
  frame    slice + render + resize + sector warp, as ONE block    what the paper calls "a frame"

It also checks that a renderer is a deterministic function of (slice, seed) (two renders, same seed), records
the CPU-vs-MPS maximum difference of the torch renderers, and writes --frames preview frames per renderer
(jittered poses at the case's screened liver and kidney windows, eval_realism.window_views) in the measured
128 px sector, as a PNG contact sheet (rows: renderers, last row: CT label) plus an .npz of the frames.

The machine load average before and after each renderer is recorded: timings under contention are upper
bounds. Writes a JSON with a provenance block (analysis_status "exploratory, v2.1 pluggability").
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _timeit(fn, n, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"n": n, "warmup": warmup, "mean_ms": float(np.mean(ts)), "median_ms": float(np.median(ts)),
            "std_ms": float(np.std(ts)), "p25_ms": float(np.percentile(ts, 25)),
            "p75_ms": float(np.percentile(ts, 75)), "fps_from_median": float(1000.0 / np.median(ts))}


def _load():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):                                      # pragma: no cover
        return None


def _sheet(rows, labels, path, cell=128, pad=2):
    """rows: list of lists of (H,W) float [0,1] (or int label maps, colored); write a grayscale/RGB PNG."""
    from PIL import Image, ImageDraw
    pal = np.array([[0, 0, 0], [180, 80, 60], [60, 180, 80], [60, 90, 200], [200, 180, 60], [140, 140, 140],
                    [200, 120, 200], [80, 200, 200], [160, 60, 60], [230, 210, 150], [255, 255, 255],
                    [120, 120, 220], [100, 100, 100]] + [[255, 0, 255]] * 243, np.uint8)
    nr, nc = len(rows), max(len(r) for r in rows)
    lw = 110
    W = lw + nc * (cell + pad) + pad
    H = nr * (cell + pad) + pad
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    dr = ImageDraw.Draw(canvas)
    for i, (r, name) in enumerate(zip(rows, labels)):
        y = pad + i * (cell + pad)
        dr.text((4, y + cell // 2 - 6), name, fill=(230, 230, 230))
        for j, a in enumerate(r):
            a = np.asarray(a)
            if a.dtype.kind in "iu":
                rgb = pal[np.clip(a, 0, 255)]
            else:
                g = (np.clip(a, 0, 1) * 255).astype(np.uint8)
                rgb = np.repeat(g[..., None], 3, axis=2)
            im = Image.fromarray(rgb).resize((cell, cell), Image.NEAREST)
            canvas.paste(im, (lw + pad + j * (cell + pad), y))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    canvas.save(path)


def lotus_parity(previews, names, sigma=3.0):
    """How close the in-repo LOTUS reimplementation is to the ORIGINAL LOTUS code on the same views: Pearson r of
    the speckle-smoothed sector frames (Gaussian sigma px; the two draw different noise, so raw frames cannot
    match), mean absolute difference of the smoothed frames, and per-label mean intensity of each (CT label of
    the sector), with the correlation of those per-label means. Physics is reported the same way as a yardstick."""
    from scipy.ndimage import gaussian_filter
    lab = np.asarray(previews["_label"])
    out = {"sigma_px": sigma, "n_views": int(len(lab)), "pairs": {}}
    pairs = [("lotus_reimpl", "lotus_orig")] + [("physics", k) for k in ("lotus_orig", "lotus_reimpl") if "physics" in names]
    for a, b in pairs:
        rs, mads = [], []
        for fa, fb, lb in zip(previews[a], previews[b], lab):
            m = lb > 0
            sa, sb = gaussian_filter(np.asarray(fa, float), sigma)[m], gaussian_filter(np.asarray(fb, float), sigma)[m]
            rs.append(float(np.corrcoef(sa, sb)[0, 1]) if sa.std() > 1e-9 and sb.std() > 1e-9 else float("nan"))
            mads.append(float(np.abs(sa - sb).mean()))
        labels = sorted(int(x) for x in np.unique(lab) if x > 0)
        ma = {L: float(np.mean([np.asarray(f)[lb == L].mean() for f, lb in zip(previews[a], lab) if (lb == L).any()]))
              for L in labels}
        mb = {L: float(np.mean([np.asarray(f)[lb == L].mean() for f, lb in zip(previews[b], lab) if (lb == L).any()]))
              for L in labels}
        va, vb = np.array([ma[L] for L in labels]), np.array([mb[L] for L in labels])
        out["pairs"][f"{a}~{b}"] = {
            "smoothed_pearson_r_mean": float(np.nanmean(rs)), "smoothed_pearson_r": rs,
            "smoothed_mad_mean": float(np.mean(mads)),
            "per_label_mean": {str(L): [ma[L], mb[L]] for L in labels},
            "per_label_mean_corr": float(np.corrcoef(va, vb)[0, 1]) if len(labels) > 2 and va.std() > 1e-9
            and vb.std() > 1e-9 else float("nan")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renderers", nargs="+", default=["physics", "lotus_reimpl", "lotus_orig", "simus"])
    ap.add_argument("--case", default="s0011", help="a TRAIN patient (benchmark.TRAIN_PATIENTS)")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--n-simus", type=int, default=3)
    ap.add_argument("--warmup-simus", type=int, default=1)
    ap.add_argument("--frames", type=int, default=5, help="preview frames per renderer")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-test-case", action="store_true", help="permit a held-out patient as --case")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")

    import torch
    from usbg import render_lotus as R, slicer as S, view_match as VM, volume as V
    from usbg import provenance as P
    from usbg.benchmark import TEST_PATIENTS
    from usbg.placement import seat_volume_under_probe
    from usbg.segmenter import align_label_to_image
    from usbg.seatmode import seat_mode
    from usbg.plug import renderers as PR
    if args.case in TEST_PATIENTS and not args.allow_test_case:
        raise SystemExit(f"{args.case} is a held-out patient; the bench uses a TRAIN patient")
    PR.register_all()
    seat = seat_mode()
    lv = V.load_cache(os.path.join("data/cache", f"{args.case}.npz"))
    wfv, T = seat_volume_under_probe(lv, V.LIVER, standoff_m=0.0, window=args.window) if seat == "anatomical" \
        else seat_volume_under_probe(lv, V.LIVER, depth_m=0.05)
    T = np.asarray(T, np.float64)
    lab_lin = S.slice_at_pose(lv, T, wfv)
    cfg = R.RenderConfig(seed=args.seed)
    have_mps = torch.backends.mps.is_available()
    names = [PR.canonical_name(r) for r in args.renderers]

    def call(name, lab, dev, seed):
        if name == "physics":
            return R.render(lab, "physics", device=dev, cfg=cfg.with_seed(seed))
        return R.render(lab, name, device=dev, cfg=cfg, seed=seed)

    results, previews = {}, {}
    sys.path.insert(0, HERE)
    import eval_realism as ER
    views, skipped, counts = ER.window_views(lv, args.case, ["liver", "kidney"], args.frames, args.seed,
                                             dict(slide=0.02, tilt=15.0, roll=10.0), 0.005)
    for name in names:
        devs = ["cpu"] if name == "simus" else [d for d in args.devices if d != "mps" or have_mps]
        n, w = (args.n_simus, args.warmup_simus) if name == "simus" else (args.n, args.warmup)
        res = {"devices": {}, "load_before": _load()}
        t_first = time.perf_counter()
        first = call(name, lab_lin, devs[0], args.seed)     # first call: module load / pool / DAS / TGC setup
        res["first_call_ms"] = round((time.perf_counter() - t_first) * 1000.0, 1)
        outs = {}
        for dev in devs:
            st = {"render": _timeit(lambda: call(name, lab_lin, dev, args.seed), n, w)}

            def frame():
                ll = S.slice_at_pose(lv, T, wfv)
                il = call(name, ll, dev, args.seed)
                i2 = VM._resize01(il, args.size)
                l2 = align_label_to_image(ll, (args.size, args.size))
                return VM.warp_to_sector(i2, l2)
            st["frame"] = _timeit(frame, n, 0)
            a = call(name, lab_lin, dev, args.seed); b = call(name, lab_lin, dev, args.seed)
            c = call(name, lab_lin, dev, args.seed + 1)
            st["deterministic_same_seed"] = bool(np.array_equal(a, b))
            st["differs_other_seed"] = bool(not np.array_equal(a, c))
            st["out_shape"] = list(a.shape); st["out_dtype"] = str(a.dtype)
            st["out_range"] = [float(a.min()), float(a.max())]; st["out_mean"] = float(a.mean())
            outs[dev] = a
            res["devices"][dev] = st
            print(f"[{name}/{dev}] render {st['render']['median_ms']:.2f} ms | frame {st['frame']['median_ms']:.2f} ms "
                  f"| deterministic {st['deterministic_same_seed']} | mean {st['out_mean']:.3f}", flush=True)
        if "cpu" in outs and "mps" in outs:
            res["cpu_vs_mps_max_abs"] = float(np.abs(outs["cpu"] - outs["mps"]).max())
        res["matches_first_call"] = bool(np.array_equal(first, outs[devs[0]]))
        if name == "simus":
            res["simus"] = PR.simus_info()
        res["load_after"] = _load()
        results[name] = res
        frames = []
        for v in views:
            il = call(name, v["lab"], devs[-1] if name != "simus" else "cpu", v["vseed"])
            wi, wl = VM.warp_to_sector(VM._resize01(il, args.size), align_label_to_image(v["lab"], (args.size, args.size)))
            frames.append(wi)
            if name == names[0]:
                previews.setdefault("_label", []).append(wl)
        previews[name] = frames
    parity = lotus_parity(previews, names) if {"lotus_reimpl", "lotus_orig"} <= set(names) else None
    stem = args.out[:-5] if args.out.endswith(".json") else args.out
    png = stem + "_frames.png"
    _sheet([previews[n] for n in names] + [previews["_label"]], names + ["CT label"], png)
    np.savez_compressed(stem + "_frames.npz", **{n: np.asarray(previews[n], np.float32) for n in names},
                        label=np.asarray(previews["_label"], np.int64),
                        view_ids=np.asarray([f"{v['case']}:{v['window']}:{v['vseed']}" for v in views]))
    out = {"case": args.case, "window": args.window, "seat": seat, "size": args.size, "seed": args.seed,
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "machine": f"{platform.system()} {platform.machine()}", "cpu_count": os.cpu_count(),
           "note": "frame = slice + render + resize + sector warp, timed as one block (scripts/bench_render.py "
                   "protocol). Load averages recorded: timings under contention are upper bounds. SIMUS runs on "
                   "CPU (NumPy) with a process pool of workers_resolved processes.",
           "renderers": results, "lotus_parity": parity, "preview": {"png": os.path.abspath(png), "npz": os.path.abspath(stem + "_frames.npz"),
                                              "views": [{k: v[k] for k in ("case", "organ", "window", "vseed")}
                                                        for v in views], "skipped": skipped, "window_counts": counts},
           "analysis_status": PR.PLUG_STATUS}
    prov = P.collect(case=args.case)
    prov["analysis_status"] = PR.PLUG_STATUS
    prov["plug"] = {n: PR.plug_info(n) for n in names}
    out["provenance"] = prov
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {args.out} and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
