"""Template plug-in for the renderer slot (usbg.interfaces.Renderer).

The factory ``make(device)`` returns ``render(label_slice, *, device=None, cfg=None, seed=None, **kw)``: an int64
CT label slice in the physics renderer's convex geometry in, a float32 B-mode frame in [0, 1] out. It must be
deterministic for a given seed (cfg.seed is passed as ``seed``; target and observation share it).

The slot condition (SUBMITTING.md): frames from this renderer, no texture stage; CT mask; the physics-trained
reference policy of each seed (release assets, ``python -m usbg.leaderboard fetch``), residual on; seeds 0 to 2.

This toy renderer is the physics renderer followed by a gamma curve.
"""
from __future__ import annotations

import numpy as np


def make(device="cpu"):
    from usbg import render_lotus as R

    def render(label_slice, *, device=None, cfg=None, seed=None, **kw):
        c = cfg if cfg is not None else R.RenderConfig()
        if seed is not None:
            c = c.with_seed(int(seed))
        img = R.render_bmode_physics(label_slice, c, device=device, warp=kw.get("warp", False))
        return (np.clip(np.asarray(img, np.float32), 0.0, 1.0) ** 1.5).astype(np.float32)

    return render
