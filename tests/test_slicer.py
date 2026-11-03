"""Phase 1 acceptance: slicing at two poses yields different, plausible label slices, and the
slice is deterministic. Uses the synthetic phantom so it runs with no data download."""
from __future__ import annotations

import numpy as np
import pytest

from usbg import volume as V
from usbg import slicer as S
from usbg.contracts_bridge import pose_from_rpy, make_world_from_volume, volume_from_probe


@pytest.fixture(scope="module")
def phantom():
    return V.synthetic_phantom()


def test_phantom_has_expected_labels(phantom):
    counts = phantom.label_counts()
    # us_render's phantom is soft tissue everywhere with a vessel and a cyst overlaid; its
    # bone slab and most of the vessel sit at depths beyond the 11 cm volume extent, so the
    # in-volume labels are soft_tissue, vessel, and cyst (remapped to gallbladder).
    assert "soft_tissue" in counts
    assert phantom.has(V.VESSEL)
    assert phantom.has(V.GALLBLADDER)   # cyst remapped to the gallbladder label


def test_slice_shape_and_dtype(phantom):
    W = make_world_from_volume()
    T = pose_from_rpy()                          # identity probe pose
    lab = S.slice_at_pose(phantom, T, W)
    assert lab.shape == (S.N_LINES, S.N_SAMPLES)
    assert lab.dtype == np.uint8


def test_slice_is_deterministic(phantom):
    W = make_world_from_volume()
    T = pose_from_rpy(tz=0.0, rx=10)
    a = S.slice_at_pose(phantom, T, W)
    b = S.slice_at_pose(phantom, T, W)
    assert np.array_equal(a, b)


def test_two_poses_give_different_slices(phantom):
    W = make_world_from_volume()
    T0 = pose_from_rpy()                          # straight
    T1 = pose_from_rpy(rx=25)                     # fanned about lateral axis
    T2 = pose_from_rpy(tx=0.02)                   # slid laterally
    s0 = S.slice_at_pose(phantom, T0, W)
    s1 = S.slice_at_pose(phantom, T1, W)
    s2 = S.slice_at_pose(phantom, T2, W)
    assert not np.array_equal(s0, s1)
    assert not np.array_equal(s0, s2)


def test_slice_is_plausible(phantom):
    # the imaging plane should cut through more than one tissue and contain some anatomy
    W = make_world_from_volume()
    lab = S.slice_at_pose(phantom, pose_from_rpy(), W)
    uniq = set(np.unique(lab).tolist())
    assert len(uniq) >= 2                          # not a single flat label
    assert uniq - {V.BACKGROUND}                   # sees something besides background


def test_slice_at_pose_matches_manual_composition(phantom):
    W = make_world_from_volume(t=(0.01, 0.0, 0.005), rpy_deg=(0, 0, 10))
    T = pose_from_rpy(tx=0.01, rx=5)
    direct = S.slice_at_pose(phantom, T, W)
    manual = S.slice_labels(phantom, volume_from_probe(T, W))
    assert np.array_equal(direct, manual)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
