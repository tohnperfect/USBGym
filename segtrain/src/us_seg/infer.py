"""Deployment inference wrapper: the integration contract with usbg.us_segmenter.

Exposes `load_segmenter(ckpt, device=None)` and `Segmenter6.segment(img01) -> label` where the output
is in the **repo ontology** `{0,1,2,3,4,7}` (spleen=7), a literal drop-in for the CT-derived label
returned by `usbg.view_match.curvi_render`. CUDA-free: defaults to `usbg`'s MPS device.

Reuses the existing MPS-native U-Net (`usbg.segmenter.Segmenter`, trained here with n_classes=6);
only the 6-class -> repo remap is added on top.
"""
from __future__ import annotations

import numpy as np

from . import gym_bridge
from . import ontology as ONT


class Segmenter6:
    """Wraps a 6-class `usbg.segmenter.Segmenter`; emits repo-ontology labels."""

    def __init__(self, seg):
        self._seg = seg  # usbg.segmenter.Segmenter instance (n_classes=6)

    @property
    def device(self):
        return self._seg.device

    def segment(self, img01: np.ndarray) -> np.ndarray:
        """(H,W) float32 [0,1]  ->  (H,W) int64 label in repo ontology {0,1,2,3,4,7}."""
        lab6 = self._seg.predict(np.asarray(img01, np.float32))  # (H,W) int64 in {0..5}
        return ONT.six_to_repo(lab6).astype(np.int64)

    def segment6(self, img01: np.ndarray) -> np.ndarray:
        """(H,W) float32 [0,1] -> (H,W) int64 label in the 6-class scheme {0..5} (for eval)."""
        return self._seg.predict(np.asarray(img01, np.float32)).astype(np.int64)

    def segment_batch(self, imgs: np.ndarray) -> np.ndarray:
        """(N,H,W) float32 [0,1] -> (N,H,W) int64 repo-ontology labels."""
        return np.stack([self.segment(im) for im in np.asarray(imgs, np.float32)])


def load_segmenter(ckpt_path: str, device=None) -> Segmenter6:
    """Load a trained checkpoint into the §4.5 wrapper. device defaults to usbg get_device() (MPS)."""
    gym_bridge.ensure_usbg_importable()
    from usbg import segmenter as SEG

    seg = SEG.Segmenter.load(ckpt_path, device=device)
    if getattr(seg, "_cfg", {}).get("n_classes", ONT.N_CLASSES) != ONT.N_CLASSES:
        raise ValueError(
            f"checkpoint has n_classes={seg._cfg.get('n_classes')}, expected {ONT.N_CLASSES}; "
            "this wrapper assumes the 6-class scheme."
        )
    return Segmenter6(seg)
