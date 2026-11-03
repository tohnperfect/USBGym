#!/usr/bin/env python3
"""Precompute the screened acoustic windows of the anatomical seat (usbg.windows) for every cached
case, write the versioned JSON every seat lookup reads, and a label-slice contact sheet for QA.

Usage:
  USBG_SEAT=anatomical PYTHONPATH=src .venv/bin/python scripts/build_windows.py --workers 5 \
      --contact-sheet outputs/windows/usbg-windows-v1_contact.png
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _one(case):
    from usbg import volume as V, windows as W
    lv = V.load_cache(os.path.join("data", "cache", f"{case}.npz"))
    return W.build_case(lv)


def _contact_sheet(path, data, cases, per_organ=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from usbg import volume as V, windows as W, slicer as S
    from usbg.placement import seat_from_window
    from usbg.segmenter import align_label_to_image
    from usbg.view_match import warp_to_sector
    organs = ("liver", "kidney_right", "kidney_left", "gallbladder")
    cols = per_organ * len(organs)
    fig, axs = plt.subplots(len(cases), cols, figsize=(1.6 * cols, 1.7 * len(cases)), squeeze=False)
    cmap = matplotlib.colormaps["tab20"].resampled(13)
    for r, c in enumerate(cases):
        lv = V.load_cache(os.path.join("data", "cache", f"{c}.npz"))
        rec = data["cases"][c]["windows"]
        for oi, o in enumerate(organs):
            for j in range(per_organ):
                ax = axs[r, oi * per_organ + j]
                ax.axis("off")
                if j >= len(rec.get(o, [])):
                    continue
                m = rec[o][j]
                Wfv, T0 = seat_from_window(np.asarray(m["T_vp"]))
                lab = S.slice_at_pose(lv, T0, Wfv)
                lab_sq = align_label_to_image(lab, (128, 128))
                _, lab_sec = warp_to_sector(np.zeros((128, 128), np.float32), lab_sq)
                ax.imshow(lab_sec, cmap=cmap, vmin=0, vmax=12, interpolation="nearest")
                ax.set_title(f"{c} {o.replace('kidney_', 'K')}{j} phi{int(m['phi'])}", fontsize=5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"contact sheet -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=None, help="default usbg.windows.WINDOWS_PATH")
    ap.add_argument("--contact-sheet", default=None)
    a = ap.parse_args(argv)
    from usbg import windows as W
    from usbg.seatmode import seat_mode
    if seat_mode() != "anatomical":
        raise SystemExit("run with USBG_SEAT=anatomical (windows are screened on the bladder-free cache)")
    out = a.out or W.WINDOWS_PATH
    cases = a.cases or sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob("data/cache/s*.npz"))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(_one, cases))
    data = {"params": W.params(), "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "cases": {r["case"]: r for r in recs}}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    json.dump(data, open(tmp, "w"))
    os.replace(tmp, out)
    print(f"windows -> {out} (md5 {W.file_md5(out)})")
    print(f"{'case':7s} " + " ".join(f"{o:>12s}" for o in W.ORGANS_BUILT))
    for c in cases:
        w = data["cases"][c]["windows"]
        print(f"{c:7s} " + " ".join(f"{len(w.get(o, [])):12d}" for o in W.ORGANS_BUILT))
    if a.contact_sheet:
        _contact_sheet(a.contact_sheet, data, cases)


if __name__ == "__main__":
    main()
