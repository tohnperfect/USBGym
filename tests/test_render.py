"""Phase 2 acceptance: both renderers turn a label slice into a well-formed B-mode, the LOTUS
path runs on the resolved device (MPS/CPU) and is deterministic, and the image changes as the
probe fans (a structure enters/leaves the plane). Uses the phantom, so it needs no data."""
from __future__ import annotations

import numpy as np
import pytest

from usbg import volume as V
from usbg import slicer as S
from usbg import render_lotus as R
from usbg.contracts_bridge import pose_from_rpy, make_world_from_volume
from usbg.placement import base_pose_over


@pytest.fixture(scope="module")
def slices():
    lv = V.synthetic_phantom()
    W = make_world_from_volume()
    base = base_pose_over(lv, V.GALLBLADDER)
    return [S.slice_at_pose(lv, base @ pose_from_rpy(rx=a), W) for a in (-20, 0, 20)]


@pytest.mark.parametrize("renderer", ["lotus", "numpy"])
def test_render_shape_and_range(slices, renderer):
    img = R.render(slices[1], renderer)
    assert img.shape == (S.N_SAMPLES, S.N_LINES)
    assert img.dtype == np.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert float(img.std()) > 0.0                       # not a flat image


def test_lotus_runs_on_device(slices):
    dev = R.get_device()
    assert dev in ("mps", "cuda", "cpu")
    img = R.render(slices[1], "lotus", device=dev)
    assert np.isfinite(img).all()


def test_lotus_is_deterministic_with_seed(slices):
    a = R.render(slices[1], "lotus", seed=7)
    b = R.render(slices[1], "lotus", seed=7)
    assert np.allclose(a, b)


def test_render_changes_with_pose(slices):
    # fanning the probe should change what the image shows (structure enters/leaves)
    for renderer in ("lotus", "numpy"):
        imgs = [R.render(s, renderer, seed=0) for s in slices]
        assert not np.allclose(imgs[0], imgs[1])
        assert not np.allclose(imgs[1], imgs[2])


def test_warp_changes_shape(slices):
    warped = R.render(slices[1], "lotus", warp=True)
    assert warped.ndim == 2 and warped.shape != (S.N_SAMPLES, S.N_LINES)
    assert 0.0 <= float(warped.min()) and float(warped.max()) <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
