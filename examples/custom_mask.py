"""Plug a custom cross-domain mask source into the view-match reward's segmentation anchor, via the
public setter (no private-global monkeypatch).

Run:  python examples/custom_mask.py
"""
import numpy as np

import usbg.view_match as VM


class ThresholdMask:
    """A toy mask source: label pixels brighter than 0.5 as 'liver' (id 1), else background. A real
    one returns the repo ontology {0 bg, 1 liver, 2 gallbladder, 3 vessel, 4 kidney, 7 spleen}."""

    def segment(self, img01: np.ndarray) -> np.ndarray:
        m = np.zeros(np.asarray(img01).shape, np.int64)
        m[np.asarray(img01) > 0.5] = 1
        return m


if __name__ == "__main__":
    VM.set_mask_source(ThresholdMask())          # public hook; call BEFORE the first rollout
    img = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    print("custom mask ids:", np.unique(VM._US.segment(img)))
    VM.set_mask_source(None)                      # None -> fall back to the CT-oracle label
    print("mask source reset to CT-oracle (view_match._US is None:", VM._US is None, ")")
