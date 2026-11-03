#!/usr/bin/env python3
"""Build and cache a canonical label volume from a TotalSegmentator case, print its label
counts, and (optionally) save a contact sheet of colored LABEL slices at several probe poses.

This is the Phase 1 human-eyeball artifact: it shows that different poses cut different,
anatomically plausible label slices through a real abdominal CT (no renderer involved yet;
LOTUS B-mode rendering is Phase 2).

Seat (env USBG_SEAT): canonical builds the v1 cache (urinary_bladder merged into KIDNEY) and aims
the sheet with the legacy base_pose_over (identity world_from_volume). Anatomical (default) builds the
bladder-free volume (volume.ANAT_EXCLUDE, as scripts/build_anat_cache.py does), refuses to write into
a directory named cache/ (the v1 caches), and aims the sheet from screened window --window of the
target on the patient's skin (seat_volume_under_probe); a case or organ with no screened window gets
no sheet.

Usage:
  python scripts/build_ct_volume.py <case_dir> --out data/cache/<id>.npz \\
      --contact-sheet outputs/<id>_label_slices.png --target gallbladder     # USBG_SEAT=canonical
  USBG_SEAT=anatomical python scripts/build_ct_volume.py data/raw/totalseg_small/<id> \\
      --out outputs/v2/ct/<id>.npz --contact-sheet outputs/v2/ct/<id>_label_slices.png --target liver
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from usbg import volume as V
from usbg import slicer as S
from usbg.contracts_bridge import pose_from_rpy, make_world_from_volume
from usbg.placement import base_pose_over, seat_mode

# a fixed, readable color per canonical label (RGB 0..1)
_LABEL_RGB = {
    V.BACKGROUND: (0.05, 0.05, 0.08), V.LIVER: (0.75, 0.35, 0.30),
    V.GALLBLADDER: (0.20, 0.85, 0.55), V.VESSEL: (0.85, 0.20, 0.20),
    V.KIDNEY: (0.55, 0.40, 0.75), V.GI: (0.90, 0.75, 0.30), V.PANCREAS: (0.95, 0.55, 0.20),
    V.SPLEEN: (0.60, 0.25, 0.45), V.MUSCLE: (0.55, 0.55, 0.55), V.FAT: (0.90, 0.85, 0.60),
    V.BONE: (0.98, 0.98, 0.98), V.LUNG: (0.30, 0.55, 0.85), V.SOFT_TISSUE: (0.35, 0.30, 0.28),
}


def _colorize(lab: np.ndarray) -> np.ndarray:
    """(L,S) labels -> (S,L,3) RGB image (depth down rows, lateral across cols)."""
    rgb = np.zeros((*lab.shape, 3), dtype=np.float32)
    for k, c in _LABEL_RGB.items():
        rgb[lab == k] = c
    return np.transpose(rgb, (1, 0, 2))   # (S, L, 3): depth rows, lateral cols


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--out", required=True, help="npz cache path for the built label volume")
    ap.add_argument("--contact-sheet", default=None, help="PNG path for the label-slice sheet")
    ap.add_argument("--target", default=None,
                    help="organ to aim the probe at (default: gallbladder canonical, liver anatomical)")
    ap.add_argument("--window", type=int, default=0,
                    help="anatomical seat: screened window index for the contact sheet")
    args = ap.parse_args(argv)
    seat = seat_mode()
    if args.target is None:
        args.target = "gallbladder" if seat == "canonical" else "liver"
    if seat == "anatomical" and os.path.basename(os.path.dirname(os.path.abspath(args.out))) == "cache":
        raise SystemExit(f"USBG_SEAT=anatomical builds the bladder-free volume; refusing to write it into "
                         f"the v1 cache dir ({args.out}). Use data/cache_anat/ via "
                         f"scripts/build_anat_cache.py, or another --out")

    ct = os.path.join(args.case_dir, "ct.nii.gz")
    seg = os.path.join(args.case_dir, "segmentations")
    print(f"building label volume from {args.case_dir} (seat {seat}) ...")
    if seat == "canonical":
        lv = V.load_labeled_ct(ct, seg)
    else:
        lv = V.load_labeled_ct(ct, seg, exclude_structures=V.ANAT_EXCLUDE)
        lv.meta["case"] = os.path.basename(os.path.normpath(args.case_dir))
    V.save_cache(lv, args.out)
    print(f"  cached -> {args.out}")
    print(f"  shape {lv.labels.shape}, spacing(m) {np.round(lv.spacing_m, 5)}")
    print("  label counts:")
    for name, n in sorted(lv.label_counts().items(), key=lambda kv: -kv[1]):
        print(f"    {name:12s} {n}")

    if not args.contact_sheet:
        return 0

    name_to_label = {v: k for k, v in V.USBG_LABEL_NAMES.items()}
    if seat == "canonical":
        target = name_to_label.get(args.target, V.GALLBLADDER)
        if not lv.has(target):
            target = V.LIVER if lv.has(V.LIVER) else V.SOFT_TISSUE
            print(f"  target '{args.target}' absent; aiming at {V.USBG_LABEL_NAMES[target]} instead")
        W = make_world_from_volume()
        base = base_pose_over(lv, target)
    else:
        if args.target not in name_to_label:
            raise SystemExit(f"unknown --target {args.target!r}; choose from {sorted(name_to_label)}")
        target = name_to_label[args.target]
        from usbg.placement import seat_volume_under_probe, NoAcceptedWindow
        try:   # probe face on the skin at the window; W carries the anatomy
            W, base = seat_volume_under_probe(lv, target, standoff_m=0.0, window=args.window)
        except (NoAcceptedWindow, RuntimeError) as e:
            print(f"  no contact sheet: {e}")
            return 0
    poses = {
        "straight": base,
        "fan -15": base @ pose_from_rpy(rx=-15),
        "fan +15": base @ pose_from_rpy(rx=15),
        "slide -2cm": base @ pose_from_rpy(tx=-0.02),
        "slide +2cm": base @ pose_from_rpy(tx=0.02),
        "rotate 20": base @ pose_from_rpy(rz=20),
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(poses), figsize=(2.6 * len(poses), 3.4))
    for ax, (nm, T) in zip(axes, poses.items()):
        lab = S.slice_at_pose(lv, T, W)
        ax.imshow(_colorize(lab), aspect="auto")
        ax.set_title(nm, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    where = f" (window {args.window})" if seat == "anatomical" else ""
    fig.suptitle(
        f"{os.path.basename(args.case_dir)}: label slices aimed at "
        f"{V.USBG_LABEL_NAMES[target]}{where} (green=gallbladder, red=vessel, salmon=liver)",
        fontsize=9,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.contact_sheet)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.contact_sheet, dpi=130)
    plt.close(fig)
    print(f"  wrote contact sheet -> {args.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
