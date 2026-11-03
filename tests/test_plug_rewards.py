"""Fast checks of usbg.plug.rewards (v2.1 pluggability, exploratory): no patient data needed."""
import importlib.util

import numpy as np
import pytest

from usbg import view_match as VM
from usbg.plug import rewards as RW

HAVE_SKIMAGE = importlib.util.find_spec("skimage") is not None     # the SSIM term: pip install "usbg[plug]"


def _frames(seed=0):
    rng = np.random.default_rng(seed)
    H = W = 128
    img = np.clip(rng.random((H, W)).astype(np.float32), 0, 1)
    lab = rng.integers(0, 8, (H, W)).astype(np.int64)
    m = RW.sector_mask(H, W)
    return img * m, lab * m


def test_identity_and_range():
    a, _ = _frames(0)
    b, _ = _frames(1)
    for t in ("ncc_lit", "ncc_sector", "ncc_frame", "nmi") + (("ssim",) if HAVE_SKIMAGE else ()):
        f = RW.appearance_fn(t)
        assert f(a, a) == pytest.approx(1.0, abs=1e-6), t
        assert 0.0 <= f(a, b) <= 1.0, t
    assert 0.0 <= RW.app_mi(a, b) <= RW.app_mi(a, a) <= 1.0


def test_paper_equals_combined_score_bitwise():
    a, la = _frames(0)
    b, lb = _frames(1)
    plug = RW.RewardPlug("paper")
    for ws, wa in ((0.6, 0.4), (1.0, 0.0), (0.0, 1.0)):
        assert plug.potential(a, la, b, lb, w_seg=ws, w_app=wa) == VM.combined_score(a, la, b, lb, None, ws, wa)


def test_privileged_needs_pose():
    a, la = _frames(0)
    with pytest.raises(ValueError):
        RW.RewardPlug("pose").potential(a, la, a, la)
    T = np.eye(4)
    assert RW.RewardPlug("pose").potential(a, la, a, la, pose=T, tgt_pose=T) == 0.0


def test_similarity_registry():
    pytest.importorskip("skimage")
    from usbg.similarity import load_similarity
    a, _ = _frames(0)
    assert load_similarity("plugapp:ssim").score(a, a) == pytest.approx(1.0, abs=1e-6)
