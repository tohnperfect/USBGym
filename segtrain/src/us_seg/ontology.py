"""Label ontology for the cross-domain US organ segmenter.

Three label spaces are in play:

1. **Canonical** `usbg.volume` ids, 0..12 (what `curvi_render` and `load_labeled_ct` emit):
   background=0, liver=1, gallbladder=2, vessel=3, kidney=4, gi=5, pancreas=6, spleen=7,
   muscle=8, fat=9, bone=10, lung=11, soft_tissue=12.  (NOTE spleen is **7**, not 5.)

2. **Six-class** training scheme, contiguous 0..5 (what nnU-Net / the U-Net head use):
   0 background, 1 liver, 2 gallbladder, 3 vessel, 4 kidney, 5 spleen.
   The five target organs; every other canonical class collapses to background, which matches the
   real Kaggle labels, where only these five organs are annotated and everything else is background.

3. **Repo ontology** = the target subset of canonical, `{0,1,2,3,4,7}`, consumed by the controller's
   `MATCH_ORGANS` via exact integer equality. The shipped `segment()` MUST emit this (spleen=7).

Keep this module the single source of truth; do not hardcode these integers elsewhere.
"""
from __future__ import annotations

import numpy as np

# ---- canonical (usbg.volume) ids we care about --------------------------------------------------
BACKGROUND = 0
LIVER = 1
GALLBLADDER = 2
VESSEL = 3
KIDNEY = 4
SPLEEN = 7  # <-- not 5; gi=5, pancreas=6 sit between kidney and spleen

# The five foreground organs in canonical ids (order == 6-class order 1..5).
TARGET_CANONICAL = (LIVER, GALLBLADDER, VESSEL, KIDNEY, SPLEEN)

# ---- six-class training scheme ------------------------------------------------------------------
N_CLASSES = 6
CLASS_NAMES = ("background", "liver", "gallbladder", "vessel", "kidney", "spleen")

# nnU-Net dataset.json "labels" block: name -> integer.
NNUNET_LABELS = {name: i for i, name in enumerate(CLASS_NAMES)}

# forward: canonical id (0..12) -> 6-class id (0..5). Non-target canonical -> background.
# index by canonical id; length 13 covers the full canonical ontology.
_CANONICAL_TO_SIX = np.zeros(13, dtype=np.int64)
for six_id, canon in enumerate(TARGET_CANONICAL, start=1):
    _CANONICAL_TO_SIX[canon] = six_id

# inverse: 6-class id (0..5) -> repo/canonical id. background 0->0, spleen 5->7, else identity.
SIX_TO_REPO = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: SPLEEN}
_SIX_TO_REPO_LUT = np.array([0, 1, 2, 3, 4, SPLEEN], dtype=np.int64)


def canonical_to_six(label: np.ndarray) -> np.ndarray:
    """Map a canonical (0..12) label array to the contiguous 6-class scheme (0..5).

    Applied identically to sim renders and to the real Kaggle test labels so both share one label
    space. Values outside 0..12 raise (they should never occur; the volume never emits them).
    """
    lab = np.asarray(label)
    lab = lab.astype(np.int64, copy=False)
    if lab.size and (lab.min() < 0 or lab.max() > 12):
        raise ValueError(f"label out of canonical range 0..12: [{lab.min()},{lab.max()}]")
    return _CANONICAL_TO_SIX[lab]


def six_to_repo(label6: np.ndarray) -> np.ndarray:
    """Map a 6-class prediction (0..5) back to the repo ontology {0,1,2,3,4,7} for the controller."""
    lab = np.asarray(label6).astype(np.int64, copy=False)
    if lab.size and (lab.min() < 0 or lab.max() > 5):
        raise ValueError(f"6-class label out of range 0..5: [{lab.min()},{lab.max()}]")
    return _SIX_TO_REPO_LUT[lab]
