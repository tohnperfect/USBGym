"""Ingest the Open Kidney (kidneyUS) dataset into a real-kidney test npz.

kidneyUS (github.com/rsingla92/kidneyUS, CC-BY-NC-SA) is real 2D convex B-mode with expert kidney
annotations. MASKS are public in the repo (labels/reviewed_masks_{1,2}/{capsule,regions}); IMAGES are
gated behind a UBC registration form and are NOT redistributable. So:

  1. Register with the KidneyUS maintainers (Open Kidney Ultrasound Data Set) and download the images.
  2. Drop the PNGs into --images-dir (filenames match the mask filenames, e.g.
     100_IM-0246-0030_anon.png).
  3. Run this script.

Whole-kidney mask = union(capsule, regions>0) -> class KIDNEY (6-class id 4). Single-organ set, so the
output is PARTIAL-LABEL: only {background, kidney} are valid; all other classes are UNLABELED.

  ./run.sh scripts/ingest_kidneyus.py \
      --images-dir data/real/kidneyus/images \
      --masks-dir data/real/kidneyus_repo/labels/reviewed_masks_1 \
      --out data/real/kidneyus_kidney_test.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from us_seg import ontology as ONT  # noqa: E402

KIDNEY6 = ONT.NNUNET_LABELS["kidney"]  # == 4


def _norm01(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((g - lo) / (hi - lo), 0.0, 1.0)


def _whole_kidney(masks_dir: str, stem: str):
    """Union of capsule + regions for one case -> binary whole-kidney mask, or None if absent."""
    cap_p = os.path.join(masks_dir, "capsule", f"{stem}.png")
    reg_p = os.path.join(masks_dir, "regions", f"{stem}.png")
    m = None
    for p in (cap_p, reg_p):
        if os.path.exists(p):
            a = np.array(Image.open(p))
            b = a > 0
            m = b if m is None else (m | b)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", default="data/real/kidneyus/images",
                    help="registered kidneyUS PNG images (you download after approval)")
    ap.add_argument("--masks-dir", default="data/real/kidneyus_repo/labels/reviewed_masks_1")
    ap.add_argument("--out", default="data/real/kidneyus_kidney_test.npz")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--preview", default="outputs/kidneyus_preview.png")
    args = ap.parse_args(argv)

    mask_stems = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(args.masks_dir, "capsule", "*.png"))
    )
    if not os.path.isdir(args.images_dir) or not os.listdir(args.images_dir):
        print(f"[kidneyus] no images in {args.images_dir!r}.")
        print(f"[kidneyus] masks are ready ({len(mask_stems)} cases in {args.masks_dir}).")
        print("[kidneyus] Register + download images (Open Kidney Ultrasound Data Set registration), "
              "drop PNGs there, and re-run. Filenames must match the mask stems, e.g. "
              f"{mask_stems[0]}.png")
        return 0

    # index available images by stem (tolerate .png/.jpg)
    img_by_stem = {}
    for p in glob.glob(os.path.join(args.images_dir, "*")):
        img_by_stem.setdefault(os.path.splitext(os.path.basename(p))[0], p)

    images, labels, stems = [], [], []
    missing = 0
    for stem in mask_stems:
        ip = img_by_stem.get(stem)
        km = _whole_kidney(args.masks_dir, stem)
        if ip is None or km is None:
            missing += 1
            continue
        img = _norm01(np.array(Image.open(ip).convert("L")))
        img128 = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize(
            (args.size, args.size), Image.BILINEAR), np.float32) / 255.0
        m128 = np.asarray(Image.fromarray(km.astype(np.uint8)).resize(
            (args.size, args.size), Image.NEAREST), np.uint8)
        lab6 = (m128 > 0).astype(np.uint8) * KIDNEY6
        images.append(img128); labels.append(lab6); stems.append(stem)

    if not images:
        print(f"[kidneyus] found images dir but 0 matched the mask stems (missing={missing}). "
              "Check filenames match, e.g. " + f"{mask_stems[0]}.png")
        return 1

    images = np.stack(images).astype(np.float32)
    labels = np.stack(labels).astype(np.uint8)
    valid_classes = np.array([ONT.NNUNET_LABELS["background"], KIDNEY6], dtype=np.int64)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, images=images, label6=labels, stems=np.array(stems),
                        valid_classes=valid_classes)
    print(f"[kidneyus] {len(images)} matched frames (missing/unmatched {missing}); wrote {args.out}")
    print("[kidneyus] valid (annotated) classes: background, kidney (partial-label)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(6, len(images))
        idx = np.linspace(0, len(images) - 1, n).astype(int)
        fig, ax = plt.subplots(2, n, figsize=(2 * n, 4))
        for j, i in enumerate(idx):
            ax[0, j].imshow(images[i], cmap="gray"); ax[0, j].axis("off")
            ax[1, j].imshow(images[i], cmap="gray")
            ax[1, j].imshow(np.ma.masked_where(labels[i] == 0, labels[i]), cmap="winter", alpha=0.5)
            ax[1, j].axis("off")
        fig.suptitle("kidneyUS real kidney frames (resized 128)", fontsize=10)
        fig.tight_layout(); os.makedirs(os.path.dirname(args.preview) or ".", exist_ok=True)
        fig.savefig(args.preview, dpi=120); print(f"[kidneyus] preview {args.preview}")
    except Exception as e:  # noqa: BLE001
        print(f"[kidneyus] preview skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
