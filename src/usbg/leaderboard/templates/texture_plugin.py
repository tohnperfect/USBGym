"""Template plug-in for the texture-stage slot (usbg.interfaces.TextureTranslator).

The factory ``make(device)`` returns a callable from an H x W float32 frame in [0, 1] (the physics frame in the
measured sector) to a frame of the same shape in [0, 1]. A label-conditioned stage sets ``wants_label = True`` and
is then called as ``stage(img01, label_slice)``; the gate marks it as reading the CT label slice.

The slot condition (SUBMITTING.md): frames from this stage; mask from this stage's own matched U-Net, trained by
the runner with the reference recipe (segtrain) on frames of this stage; servo base, residual off; seeds 0 to 2.

    python -m usbg.leaderboard check --slot texture --plugin my_stage_module:make
    python -m usbg.leaderboard run --slot texture --plugin my_stage_module:make --name ... --doi ...
"""
from __future__ import annotations

import numpy as np


def my_stage(img01):
    """The toy stage of the quickstart: a gamma curve."""
    return np.clip(img01, 0.0, 1.0) ** 1.5


class MyStage:
    """A stateful stage would load its weights here (on `device`) and keep batch normalization in eval mode."""
    wants_label = False

    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, img01):
        return my_stage(np.asarray(img01, np.float32)).astype(np.float32)


def make(device="cpu"):
    return MyStage(device)
