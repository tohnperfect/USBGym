"""Phase 0 acceptance: the bridge re-exports the par contract verbatim, geometry parity
holds, and the volume/probe frame transforms round-trip exactly."""
from __future__ import annotations

import numpy as np
import pytest

from usbg import contracts_bridge as cb
from usbg._par import contracts as C  # the vendored par contract (parity with par itself: tests/test_vendor_regression.py)


def test_import_and_geometry_parity():
    # must not raise: se3.py has not drifted from the pinned hash
    cb.assert_geometry_parity()


def test_contract_values_match_par_exactly():
    assert cb.CHANNELS_TWIST == C.CHANNELS_TWIST == ["vx", "vy", "vz", "wx", "wy", "wz"]
    assert cb.CHANNELS_REL6 == C.CHANNELS_REL6 == ["x", "y", "z", "rx", "ry", "rz"]
    assert cb.CHANNELS_INPUT == C.CHANNELS_INPUT
    assert cb.IN_CHANNELS == C.IN_CHANNELS == 12
    assert cb.TARGET_CHANNELS == C.TARGET_CHANNELS == 6
    assert cb.CLASSES_5 == C.CLASSES_5 == ["slide", "rock", "tilt", "rotate", "compress"]
    assert cb.SESSION_CLASSES_6 == C.SESSION_CLASSES_6 == cb.CLASSES_5 + ["hold"]
    assert cb.N_CLASSES == C.N_CLASSES == 6
    assert cb.HOLD_IDX == C.HOLD_IDX == 5
    assert cb.DT_C == C.DT_C == 0.1
    assert cb.OUT_RATE_HZ == C.OUT_RATE_HZ == 10.0
    assert cb.DOMINANT_AXIS == C.DOMINANT_AXIS


def test_summary_has_all_keys():
    info = cb.summary()
    for k in ("CHANNELS_INPUT", "CLASSES_5", "SESSION_CLASSES_6", "DT_C", "DOMINANT_AXIS"):
        assert k in info


def test_pose_from_rpy_is_valid_se3():
    T = cb.pose_from_rpy(tx=0.01, ty=-0.02, tz=0.03, rx=15, ry=-5, rz=30)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)          # orthonormal
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)        # proper rotation
    assert np.allclose(T[3], [0, 0, 0, 1])                     # homogeneous row
    assert np.allclose(T[:3, 3], [0.01, -0.02, 0.03])


def test_volume_from_probe_identity_is_probe_pose():
    T_wp = cb.pose_from_rpy(tx=0.05, tz=0.02, ry=10)
    W_identity = cb.make_world_from_volume()
    assert np.allclose(W_identity, np.eye(4))
    T_vp = cb.volume_from_probe(T_wp, W_identity)
    assert np.allclose(T_vp, T_wp)


def test_volume_from_probe_round_trip():
    # For any placement W and probe pose T:  W @ volume_from_probe(T, W) == T
    W = cb.make_world_from_volume(t=(0.1, -0.05, 0.2), rpy_deg=(5, 10, -15))
    T_wp = cb.pose_from_rpy(tx=0.03, ty=0.01, tz=0.08, rx=-8, ry=12, rz=45)
    T_vp = cb.volume_from_probe(T_wp, W)
    assert np.allclose(W @ T_vp, T_wp, atol=1e-9)


def test_invert_T_matches_numpy_inverse():
    W = cb.make_world_from_volume(t=(0.2, 0.3, -0.1), rpy_deg=(20, -30, 40))
    assert np.allclose(cb.invert_T(W), np.linalg.inv(W), atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
