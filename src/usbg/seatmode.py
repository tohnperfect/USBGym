"""seatmode.py: the ONE switch between the v1 seat and the anatomical seat.

USBG_SEAT=anatomical (default): the CT is placed so that the probe rests on the skin of the patient at
a screened acoustic window and images into the body (see placement.py, windows.py), the KIDNEY label
excludes the urinary bladder (volume.load_cache reads data/cache_anat/), and the renderers couple the
probe face to the skin with gel.

USBG_SEAT=canonical: the v1 seat, unchanged (the CT unrotated, the beam pointing inferior across a
coronal plane, the probe face within the trunk). Kept only so that the v1 numbers stay reproducible.

This module has no usbg imports so volume, placement and render_lotus can all read it.
"""
from __future__ import annotations

import os

SEAT_MODES = ("anatomical", "canonical")


def seat_mode() -> str:
    m = os.environ.get("USBG_SEAT", "anatomical").strip().lower()
    if m not in SEAT_MODES:
        raise ValueError(f"USBG_SEAT={m!r}; expected one of {SEAT_MODES}")
    return m


def gel_coupling() -> bool:
    """Anechoic gel between the probe face and the skin (anatomical seat only). USBG_GEL=0/1 overrides."""
    v = os.environ.get("USBG_GEL")
    if v is not None and v.strip() != "":
        return v.strip().lower() not in ("0", "false", "no", "off")
    return seat_mode() == "anatomical"
