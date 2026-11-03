"""Template plug-in for the mask-source slot (usbg.interfaces.MaskSource).

The factory ``make(device)`` returns an object with ``segment(img01) -> int64 mask`` in the repository ontology
{0 background, 1 liver, 2 gallbladder, 3 vessel, 4 kidney, 7 spleen}, on 128 x 128 frames in [0, 1]. A network
runs with batch normalization in eval mode and is trained on the training patients only.

The slot condition (SUBMITTING.md): SonoSPADE frames; mask from this source; servo base fitted under this mask,
residual off; seeds 0 to 2 (seed 0 for a source without training). The maintainers run this slot.

This toy source labels the brightest third of the sector as liver.
"""
from __future__ import annotations

import numpy as np


class ThresholdMask:
    privileged = False

    def __init__(self, device="cpu"):
        self.device = device

    def segment(self, img01):
        x = np.asarray(img01, np.float32)
        t = np.quantile(x[x > 0], 2.0 / 3.0) if np.any(x > 0) else 1.0
        return np.where(x >= t, 1, 0).astype(np.int64)


def make(device="cpu"):
    return ThresholdMask(device)
