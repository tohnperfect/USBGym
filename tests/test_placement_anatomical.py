"""D4 regression tests: the anatomical seat (probe on the patient's own skin at a screened window)
and the guarantee that USBG_SEAT=canonical still reproduces the v1 seat exactly."""
from __future__ import annotations

import os

import numpy as np
import pytest

from usbg import volume as V
from usbg import placement as P
from usbg.contracts_bridge import pose_from_rpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = "s0011"
HAVE_ANAT = os.path.exists(os.path.join(REPO, "data", "cache_anat", f"{CASE}.npz"))


def _windows_ready():
    from usbg import windows as W
    return HAVE_ANAT and os.path.exists(W.WINDOWS_PATH)


def _lv():
    return V.load_cache(os.path.join(REPO, "data", "cache", f"{CASE}.npz"))


def test_canonical_seat_is_the_v1_seat(monkeypatch):
    monkeypatch.setenv("USBG_SEAT", "canonical")
    lv = V.synthetic_phantom()
    Wfv, T0 = P.seat_volume_under_probe(lv, V.GALLBLADDER, standoff_m=0.0, depth_m=0.05)
    T0_ref = pose_from_rpy(tz=0.0, rx=180.0)
    p_world = T0_ref[:3, :3] @ np.array([0.0, 0.0, 0.05]) + T0_ref[:3, 3]
    W_ref = np.eye(4); W_ref[:3, 3] = p_world - P.target_centroid_m(lv, V.GALLBLADDER)
    assert np.array_equal(T0, T0_ref) and np.array_equal(Wfv, W_ref)
    Wd, Td = P.seat_volume_under_probe(lv, V.GALLBLADDER)                 # canonical defaults kept
    assert np.isclose(Td[2, 3], 0.005)


def test_seat_from_window_puts_face_at_origin_beam_down():
    from usbg.windows import probe_rotation
    T_vp = np.eye(4); T_vp[:3, :3] = probe_rotation([0.3, -0.9, -0.1], -60.0); T_vp[:3, 3] = [0.01, 0.12, -0.3]
    Wfv, T0 = P.seat_from_window(T_vp)
    assert np.allclose(Wfv @ T_vp, pose_from_rpy(rx=180.0), atol=1e-12)
    assert np.allclose(T0[:3, 2], [0, 0, -1])                           # imaging along world -z


def test_anterior_window_matches_r_seat():
    from usbg.windows import probe_rotation
    R = probe_rotation([0.0, -1.0, 0.0], 90.0)                          # probe +x = patient right
    assert np.allclose(R[:, 0], [1, 0, 0])
    T_vp = np.eye(4); T_vp[:3, :3] = R
    Wfv, _ = P.seat_from_window(T_vp)
    assert np.allclose(Wfv[:3, :3], P.R_SEAT, atol=1e-12)


def test_clinical_display_convention():
    """phi=-90 on an anterior window: probe +x = patient LEFT, so patient right is on image left."""
    from usbg.windows import probe_rotation
    assert np.allclose(probe_rotation([0.0, -1.0, 0.0], -90.0)[:, 0], [-1, 0, 0])
    assert np.allclose(probe_rotation([0.0, -1.0, 0.0], 0.0)[:, 0], [0, 0, -1])   # caudal: cranial on image left


def test_clamp_lifts_every_face_corner_above_the_skin():
    rng = np.random.default_rng(0)
    for _ in range(200):
        T = pose_from_rpy(rx=180.0) @ pose_from_rpy(tx=rng.uniform(-.02, .02), ty=rng.uniform(-.02, .02),
                                                    tz=rng.uniform(-.01, .01), rx=rng.uniform(-18, 18),
                                                    ry=rng.uniform(-18, 18), rz=rng.uniform(-4, 4))
        Tc, lift = P.clamp_start_outside_skin(T)
        bx, by, _ = P.PROBE_HALF
        C = np.array([[0, 0, 0, 1.0]] + [[sx * bx, sy * by, 0, 1.0] for sx in (-1, 1) for sy in (-1, 1)])
        assert (Tc @ C.T)[2].min() >= -1e-12
        assert lift >= 0 and np.allclose(Tc[:3, :3], T[:3, :3])


@pytest.mark.anatomical
def test_anatomical_rejects_depth_m():
    lv = V.synthetic_phantom()
    with pytest.raises(ValueError):
        P.seat_volume_under_probe(lv, V.GALLBLADDER, standoff_m=0.0, depth_m=0.05)


@pytest.mark.anatomical
@pytest.mark.skipif(not HAVE_ANAT, reason="needs data/cache_anat (scripts/build_anat_cache.py)")
def test_anatomical_cache_is_bladder_free():
    lv = _lv()
    assert lv.meta["cache_variant"] == "anat" and lv.meta["case"] == CASE
    assert "urinary_bladder" in lv.meta.get("excluded_structures", [])


@pytest.mark.anatomical
@pytest.mark.skipif(not _windows_ready(), reason="needs the windows file (scripts/build_windows.py)")
def test_window_seat_slices_the_volume_frame_window():
    from usbg import slicer as S, windows as W
    lv = _lv()
    for label in (V.LIVER, V.KIDNEY):
        metas = W.windows_meta(lv, label)
        assert metas, f"{CASE} has no {label} window"
        for k in range(min(3, len(metas))):
            Wfv, T0 = P.seat_volume_under_probe(lv, label, window=k)
            T_vp = np.asarray(metas[k]["T_vp"])
            assert np.array_equal(S.slice_at_pose(lv, T0, Wfv), S.slice_labels(lv, T_vp))
            m = metas[k]
            assert m["face_lab"] in (V.FAT, V.SOFT_TISSUE) and m["tilt_deg"] <= W.TILT_MAX_DEG + 1e-3
            assert -W.FLAT_ABOVE_MM <= m["flat_gap_mm"][1] and m["flat_gap_mm"][0] <= W.FLAT_BELOW_MM


@pytest.mark.anatomical
@pytest.mark.skipif(not _windows_ready(), reason="needs the windows file (scripts/build_windows.py)")
def test_sample_setup_meta_and_no_start_inside_the_skin():
    from usbg import goal_rl as G
    lv = _lv()
    rng = np.random.default_rng(0)
    n_clamped = 0
    for _ in range(30):
        Wfv, T_tgt, T_start, meta = G.sample_setup(lv, V.LIVER, rng, lateral_m=0.02, axial_m=0.01,
                                                   fan_deg=18.0, return_meta=True)
        assert meta["seat"] == "anatomical" and 0 <= meta["window"] < meta["n_windows"]
        assert np.allclose(T_tgt, pose_from_rpy(rx=180.0))
        Tc, lift = P.clamp_start_outside_skin(T_start)
        assert lift <= 1e-12
        n_clamped += meta["start_clamped"]
    assert 0 < n_clamped < 30


@pytest.mark.anatomical
@pytest.mark.skipif(not _windows_ready(), reason="needs the windows file (scripts/build_windows.py)")
def test_no_spleen_target_by_default(monkeypatch):
    monkeypatch.delenv("USBG_ALLOW_SPLEEN", raising=False)
    assert P.n_windows(_lv(), V.SPLEEN) == 0
