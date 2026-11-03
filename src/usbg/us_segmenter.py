"""Cross-domain US organ segmenter for USB Gym: the mask source the view-match reward's
segmentation anchor uses at deployment (no CT at real inference).

    from usbg import us_segmenter
    seg = us_segmenter.load_segmenter()          # MPS/CPU, CUDA-free; reads US_SEG_CKPT or the default
    mask = seg.segment(img01)                    # (128,128) float[0,1] -> (128,128) int {0,1,2,3,4,7}

It loads the gym's own `usbg.segmenter.Segmenter` (trained on that exact architecture, n_classes=6),
with no runtime dependency on the segmenter training project (only `usbg` + numpy). The checkpoint
defaults to `data/dataset/us_seg_sim_real.pt`; override with `US_SEG_CKPT` or `load_segmenter(ckpt_path=)`.

The model outputs a contiguous 6-class scheme {0 bg,1 liver,2 gallbladder,3 vessel,4 kidney,5 spleen};
this wrapper remaps spleen 5->7 (see `_SIX_TO_REPO` below) so the emitted labels are the repo ontology
{0,1,2,3,4,7} that `view_match.MATCH_ORGANS` / `seg_organ_iou` consume by exact integer equality.
"""
from __future__ import annotations

import os

import numpy as np

from usbg.segmenter import Segmenter
from usbg.render_lotus import get_device

# 6-class training id (index) -> repo/canonical id.  spleen 5 -> 7; others identity.
_SIX_TO_REPO = np.array([0, 1, 2, 3, 4, 7], dtype=np.int64)

# default checkpoint location inside the gym repo (override via arg or $US_SEG_CKPT)
_DEFAULT_CKPT = os.environ.get("US_SEG_CKPT", "data/dataset/us_seg_sim_real.pt")


class USOrganSegmenter:
    """Wraps a 6-class usbg Segmenter; emits repo-ontology {0,1,2,3,4,7} masks."""

    def __init__(self, seg: "Segmenter", ckpt_path: str | None = None):
        self._seg = seg
        self.ckpt_path = ckpt_path            # absolute path of the loaded checkpoint (provenance)

    @property
    def device(self):
        return self._seg.device

    def segment(self, img01: np.ndarray) -> np.ndarray:
        """(H,W) float32 in [0,1] curvilinear B-mode -> (H,W) int64 mask in {0,1,2,3,4,7}."""
        lab6 = self._seg.predict(np.asarray(img01, np.float32))  # (H,W) int64 in {0..5}
        return _SIX_TO_REPO[lab6].astype(np.int64)

    def segment_batch(self, imgs: np.ndarray) -> np.ndarray:
        return np.stack([self.segment(im) for im in np.asarray(imgs, np.float32)])


def load_segmenter(ckpt_path: str | None = None, device=None) -> USOrganSegmenter:
    """Load the trained cross-domain segmenter. device defaults to usbg get_device() (MPS).

    The returned object records the absolute checkpoint path it was built from (``.ckpt_path``), so a
    segmenter bound via ``view_match.set_mask_source`` is still identifiable in a run's provenance.
    With ``USBG_REQUIRE_SEG_CKPT=1`` an implicit default (no ``ckpt_path`` and ``US_SEG_CKPT`` unset)
    raises instead of silently loading ``data/dataset/us_seg_sim_real.pt`` (withdrawn, D2)."""
    if ckpt_path is None and not os.environ.get("US_SEG_CKPT") \
            and os.environ.get("USBG_REQUIRE_SEG_CKPT", "0").strip() == "1":
        raise RuntimeError("USBG_REQUIRE_SEG_CKPT=1 but neither ckpt_path nor $US_SEG_CKPT is set; "
                           f"refusing the implicit default {_DEFAULT_CKPT!r}")
    path = ckpt_path or _DEFAULT_CKPT
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"segmenter checkpoint not found at {path!r}. Copy it from the training repo, e.g.\n"
            "  cp segtrain/outputs/us_seg_6cls.pt data/dataset/us_seg_sim_real.pt\n"
            "or set $US_SEG_CKPT / pass ckpt_path=."
        )
    seg = Segmenter.load(path, device=device or get_device())
    ncls = getattr(seg, "_cfg", {}).get("n_classes", None)
    if ncls not in (None, 6):
        raise ValueError(f"expected a 6-class checkpoint, got n_classes={ncls}")
    return USOrganSegmenter(seg, ckpt_path=os.path.abspath(path))


# module-level singleton so curvi_render can call it every frame without reloading
_SINGLETON: USOrganSegmenter | None = None


def segment(img01: np.ndarray) -> np.ndarray:
    """Convenience: segment with a lazily-loaded shared model (repo ontology {0,1,2,3,4,7})."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = load_segmenter()
    return _SINGLETON.segment(img01)
