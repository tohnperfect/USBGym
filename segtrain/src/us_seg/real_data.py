"""Load the real Kaggle (Vitale) abdominal-US frames.

The only real+labeled data is the 60-frame TEST split (`kaggle_vitale_test.npz`): images
`(60,128,128)` uint8, labels `(60,128,128)` uint8 with canonical ids `{0,1,2,3,4,7}`. Used as the
zero-shot cross-domain test set (train-on-sim, test-on-real). There is no real labeled train split.

The 404-frame unlabeled real pool (`data/real/kaggle_vitale/pool/*.png`) is available for optional
unsupervised domain adaptation; it has no masks.
"""
from __future__ import annotations

import os

import numpy as np

from . import gym_bridge
from . import ontology as ONT


def load_real_test() -> dict:
    """Return the 60 real labeled test frames.

    keys: images (N,H,W) float32 [0,1]; label6 (N,H,W) uint8 {0..5};
          label_repo (N,H,W) uint8 {0,1,2,3,4,7} (controller ontology);
          sessions (N,) str; n int.
    """
    path = gym_bridge.gym_path(gym_bridge.REAL_TEST_NPZ)
    d = np.load(path, allow_pickle=True)
    images = np.asarray(d["images"], np.float32) / 255.0
    label_repo = np.asarray(d["labels"]).astype(np.uint8)  # canonical {0,1,2,3,4,7}
    label6 = ONT.canonical_to_six(label_repo).astype(np.uint8)
    sessions = np.asarray(d["sessions"]) if "sessions" in d else np.arange(len(images)).astype(str)
    return {
        "images": images,
        "label6": label6,
        "label_repo": label_repo,
        "sessions": sessions,
        "n": int(len(images)),
    }


def real_pool_pngs() -> list[str]:
    """Sorted list of unlabeled real pool PNG paths (for optional domain adaptation)."""
    d = gym_bridge.gym_path(gym_bridge.REAL_POOL_DIR)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))


# --- KidneyUS: single-organ (kidney) partial-label real set, leakage-free train/test split ------
_KID_NPZ = "data/real/kidneyus_kidney_test.npz"
_KID_SPLIT = "data/real/kidneyus_split.npz"


def _seg_path(p: str) -> str:
    """Resolve a path relative to THIS (segmenter) repo root. The KidneyUS files live here, not in
    the gym repo (unlike the Vitale set), so gym_bridge.gym_path is wrong for them."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return p if os.path.isabs(p) else os.path.join(root, p)


def load_kidneyus_train(npz: str = _KID_NPZ, split: str = _KID_SPLIT) -> dict:
    """KidneyUS partial-label TRAIN frames (the split's held-out test frames are excluded).

    Images are already float32 in [0,1]. Non-kidney pixels are set to the ignore label 255 so
    training only supervises kidney-vs-not-kidney and never teaches other organs -> background.
    Returns empty if the files are absent. keys: images (N,H,W) float32; label6 (N,H,W) int64
    ({4 kidney, 255 ignore}); n int.
    """
    kid6 = ONT.NNUNET_LABELS["kidney"]                         # 6-class kidney id (4)
    p = _seg_path(npz)
    if not os.path.exists(p):
        return {"images": np.zeros((0, 128, 128), np.float32), "label6": np.zeros((0, 128, 128), np.int64), "n": 0}
    d = np.load(p, allow_pickle=True)
    imgs = np.asarray(d["images"], np.float32)
    lab6 = np.asarray(d["label6"]).astype(np.int64)           # {0 background, 4 kidney}, partial
    sp = _seg_path(split)
    if not os.path.exists(sp):
        # never train on the KidneyUS holdout by accident: without the split file there is no holdout
        raise FileNotFoundError(f"KidneyUS split {sp} missing; refusing to train on all {len(imgs)} frames")
    tr = ~np.asarray(np.load(sp, allow_pickle=True)["is_test"], bool)
    imgs, lab6 = imgs[tr], lab6[tr]
    y = np.full_like(lab6, 255)                               # 255 = ignore
    y[lab6 == kid6] = kid6                                    # supervise only kidney pixels
    return {"images": imgs, "label6": y, "n": int(len(imgs))}


def load_kidneyus_test(npz: str = _KID_NPZ, split: str = _KID_SPLIT) -> dict:
    """KidneyUS held-out TEST frames (partial label {0,4}) for the leakage-free kidney eval."""
    d = np.load(_seg_path(npz), allow_pickle=True)
    imgs = np.asarray(d["images"], np.float32)
    lab6 = np.asarray(d["label6"]).astype(np.uint8)
    te = np.ones(len(imgs), bool)
    sp = _seg_path(split)
    if os.path.exists(sp):
        te = np.asarray(np.load(sp, allow_pickle=True)["is_test"], bool)
    return {"images": imgs[te], "label6": lab6[te], "n": int(te.sum())}


# --- Vitale v2 split: fixed train / frame-disjoint holdout (never trained on by ANY v2 model) ----
VITALE_SPLIT_V2 = "data/real/vitale_split_v2.npz"
VITALE_HOLDOUT_V2 = "data/real/vitale_holdout_v2.npz"


def load_vitale_split(path: str = VITALE_SPLIT_V2) -> dict:
    """Load the fixed Vitale train/holdout split (made once by scripts/make_vitale_split.py).

    Returns {is_holdout (60,) bool, train_idx, holdout_idx, mode, sessions}. Validates that the split's
    session order matches load_real_test() so an index can never silently point at another frame.
    """
    d = np.load(_seg_path(path), allow_pickle=True)
    is_ho = np.asarray(d["is_holdout"], bool)
    rt = load_real_test()
    if len(is_ho) != rt["n"] or [str(x) for x in d["sessions"]] != [str(x) for x in rt["sessions"]]:
        raise ValueError(f"Vitale split {path} does not match kaggle_vitale_test.npz frame order")
    return {"is_holdout": is_ho,
            "train_idx": np.where(~is_ho)[0], "holdout_idx": np.where(is_ho)[0],
            "mode": str(d["mode"]) if "mode" in d else "?",
            "sessions": np.asarray(d["sessions"]).astype(str)}
