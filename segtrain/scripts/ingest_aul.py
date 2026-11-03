"""Ingest the AUL (Annotated Ultrasound Liver) dataset into a real-liver test npz.

AUL (Zenodo 7272660, CC-BY-4.0) is real B-mode with liver-outline polygon annotations (JSON, in
original-image pixel coords). It is a SINGLE-ORGAN set: only liver is labeled. We rasterize the liver
polygon, normalize + resize the image to 128x128, and emit a partial-label test set (only class
liver=1 is valid; all other classes are UNLABELED, not background).

CAVEAT: AUL frames are raw clinical screenshots, NOT in the fixed 128x128 convex-sector geometry the
segmenter trained on. Absolute Dice is therefore a lower bound; the useful signal is class-assignment
consistency (is a true-liver region predicted liver?), reported liver-only.

  ./run.sh scripts/ingest_aul.py --src data/real/aul --out data/real/aul_liver_test.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from us_seg import ontology as ONT  # noqa: E402

LIVER6 = ONT.NNUNET_LABELS["liver"]  # == 1


def _norm01(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((g - lo) / (hi - lo), 0.0, 1.0)


def _rasterize(poly, hw) -> np.ndarray:
    """Polygon (list of [x,y], or list of such rings) -> binary mask at hw=(H,W)."""
    H, W = hw
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    rings = poly if (poly and isinstance(poly[0][0], (list, tuple))) else [poly]
    for ring in rings:
        pts = [(float(x), float(y)) for x, y in ring]
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
    return np.array(m, dtype=np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/real/aul")
    ap.add_argument("--out", default="data/real/aul_liver_test.npz")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--preview", default="outputs/aul_preview.png")
    args = ap.parse_args(argv)

    images, labels, groups, stems = [], [], [], []
    for cls in ("Benign", "Malignant", "Normal"):
        idir = os.path.join(args.src, cls, "image")
        ldir = os.path.join(args.src, cls, "segmentation", "liver")
        if not os.path.isdir(idir):
            continue
        for ip in sorted(glob.glob(os.path.join(idir, "*"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            jp = os.path.join(ldir, f"{stem}.json")
            if not os.path.exists(jp):
                continue
            im = Image.open(ip).convert("L")
            W, H = im.size
            poly = json.load(open(jp))
            if not poly:
                continue
            mask = _rasterize(poly, (H, W))
            img = _norm01(np.array(im))
            img128 = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize(
                (args.size, args.size), Image.BILINEAR), np.float32) / 255.0
            m128 = np.asarray(Image.fromarray(mask).resize(
                (args.size, args.size), Image.NEAREST), np.uint8)
            lab6 = (m128 > 0).astype(np.uint8) * LIVER6
            images.append(img128)
            labels.append(lab6)
            groups.append(cls.lower())
            stems.append(f"{cls[0].lower()}{stem}")

    images = np.stack(images).astype(np.float32)
    labels = np.stack(labels).astype(np.uint8)
    # partial-label mask: which 6-class ids are annotated in this dataset (only liver + background)
    valid_classes = np.array([ONT.NNUNET_LABELS["background"], LIVER6], dtype=np.int64)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, images=images, label6=labels,
                        groups=np.array(groups), stems=np.array(stems),
                        valid_classes=valid_classes)
    liver_frames = int((labels == LIVER6).any(axis=(1, 2)).sum())
    print(f"[aul] {len(images)} frames ({dict(zip(*np.unique(groups, return_counts=True)))}); "
          f"liver present in {liver_frames}; wrote {args.out}")
    print(f"[aul] valid (annotated) classes: background, liver  (partial-label: others UNLABELED)")

    # quick preview
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = 6
        idx = np.linspace(0, len(images) - 1, n).astype(int)
        fig, ax = plt.subplots(2, n, figsize=(2 * n, 4))
        for j, i in enumerate(idx):
            ax[0, j].imshow(images[i], cmap="gray"); ax[0, j].axis("off")
            ax[1, j].imshow(images[i], cmap="gray")
            ax[1, j].imshow(np.ma.masked_where(labels[i] == 0, labels[i]), cmap="autumn", alpha=0.5)
            ax[1, j].axis("off"); ax[1, j].set_title(stems[i], fontsize=7)
        ax[0, 0].set_ylabel("image"); ax[1, 0].set_ylabel("liver GT")
        fig.suptitle("AUL real liver frames (resized to 128; not sector-warped)", fontsize=10)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.preview) or ".", exist_ok=True)
        fig.savefig(args.preview, dpi=120)
        print(f"[aul] preview {args.preview}")
    except Exception as e:  # noqa: BLE001
        print(f"[aul] preview skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
