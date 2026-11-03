#!/usr/bin/env python3
"""Build the bladder-free label caches the anatomical seat reads (data/cache_anat/<id>.npz).

The v1 caches (data/cache/<id>.npz) map urinary_bladder to KIDNEY, which drags the KIDNEY centroid
to the midline and makes 'kidney' views image the bladder. This rebuilds each case from the raw
TotalSegmentator masks with urinary_bladder left out (its voxels keep the HU fill), and verifies that
the ONLY voxels that change are former KIDNEY voxels. data/cache/ is never modified.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/build_anat_cache.py [--cases s0011 ...] [--workers 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _one(args):
    case, raw_root, old_dir, out_dir = args
    from usbg import volume as V
    d = os.path.join(raw_root, case)
    lv = V.load_labeled_ct(os.path.join(d, "ct.nii.gz"), os.path.join(d, "segmentations"),
                           exclude_structures=V.ANAT_EXCLUDE)
    old = np.load(os.path.join(old_dir, f"{case}.npz"), allow_pickle=False)
    o = old["labels"]
    assert o.shape == lv.labels.shape, (case, o.shape, lv.labels.shape)
    assert np.allclose(old["affine_m"], lv.affine_m), case
    diff = o != lv.labels
    bad = diff & (o != V.KIDNEY)
    rep = {
        "case": case,
        "changed_voxels": int(diff.sum()),
        "changed_outside_old_kidney": int(bad.sum()),
        "old_kidney": int((o == V.KIDNEY).sum()),
        "new_kidney": int((lv.labels == V.KIDNEY).sum()),
        "bladder_relabelled_to": {V.USBG_LABEL_NAMES[int(k)]: int(c) for k, c in
                                  zip(*np.unique(lv.labels[diff], return_counts=True))},
    }
    if rep["changed_outside_old_kidney"]:
        rep["error"] = "labels changed outside the old KIDNEY label; cache code drifted since July"
        return rep
    lv.meta["case"] = case
    lv.meta["anat_cache"] = {"excluded": list(V.ANAT_EXCLUDE), "source_cache": os.path.join(old_dir, f"{case}.npz")}
    V.save_cache(lv, os.path.join(out_dir, f"{case}.npz"))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None, help="default: every data/cache/*.npz")
    ap.add_argument("--raw-root", default="data/raw/totalseg_small")
    ap.add_argument("--old-dir", default="data/cache")
    ap.add_argument("--out-dir", default="data/cache_anat")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--report", default="data/cache_anat/build_report.json")
    a = ap.parse_args(argv)
    cases = a.cases or sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(a.old_dir, "s*.npz")))
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(c, a.raw_root, a.old_dir, a.out_dir) for c in cases]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        reps = list(ex.map(_one, jobs))
    bad = [r for r in reps if "error" in r]
    for r in reps:
        print(f"{r['case']}: kidney {r['old_kidney']} -> {r['new_kidney']} voxels, changed {r['changed_voxels']}"
              f" (outside old KIDNEY: {r['changed_outside_old_kidney']}) {r['bladder_relabelled_to']}"
              + (f"  ERROR {r['error']}" if 'error' in r else ""))
    json.dump({"cases": reps, "excluded": ["urinary_bladder"]}, open(a.report, "w"), indent=1)
    print(f"report -> {a.report}")
    if bad:
        raise SystemExit(f"{len(bad)} case(s) failed verification: {[r['case'] for r in bad]}")


if __name__ == "__main__":
    main()
