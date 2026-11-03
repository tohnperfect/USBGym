"""_reuse.py: the B-mode scan geometry and NumPy appearance model shared with the probe-action recognizer code.

The probe imaging plane, its scan-point grid, the analytic phantom and the NumPy appearance model come
from the probe-action recognizer's scripts/us_render.py. Up to v2.0.0 this module located that file through
the installed sibling `par` package and executed it from the sibling tree; it is now vendored verbatim
(minus its plotting demo) as usbg._par.us_render, so usbg runs from a single clone with no sibling repo.
tests/test_vendor_regression.py pins that the switch changed no number.

Exposes the module as `us_render`, plus its geometry constants for convenience.
"""
from __future__ import annotations

from ._par import us_render

# geometry constants (probe imaging plane), re-exported so downstream never hardcodes them
N_LINES = us_render.N_LINES            # scanlines (lateral)
N_SAMPLES = us_render.N_SAMPLES        # samples per scanline (axial/depth)
IMG_DEPTH_M = us_render.IMG_DEPTH_M    # axial depth imaged (m)
IMG_WIDTH_M = us_render.IMG_WIDTH_M    # lateral extent at the face (m)
PROBE = us_render.PROBE                # "convex" | "linear"

__all__ = ["us_render", "N_LINES", "N_SAMPLES", "IMG_DEPTH_M", "IMG_WIDTH_M", "PROBE"]
