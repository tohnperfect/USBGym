"""v2.1.2 (opt-in): the opt-in plane-general clamp, the opt-in contact plane of USProbeEnv and the
pre-lift start in sample_setup's meta. The defaults must be the v2.1.1 code paths."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from usbg import placement as P
from usbg import volume as V
from usbg.contracts_bridge import pose_from_rpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sha256 of mujoco_env._mjcf() at v2.1.1 (computed from `git show v2.1.1:src/usbg/mujoco_env.py`)
MJCF_SHA256_V211 = "1ecf2392cdd42cfd63f1022f9e139736773e6efb6416f8c89dcf361d342329e9"
HAVE_TRAIN = os.path.exists(os.path.join(REPO, "data", "cache_anat", "s0011.npz"))


def _random_poses(n, seed=0, spread=0.03, tilt=25.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(pose_from_rpy(tx=rng.uniform(-spread, spread), ty=rng.uniform(-spread, spread),
                                 tz=rng.uniform(-0.02, 0.02), rx=180.0 + rng.uniform(-tilt, tilt),
                                 ry=rng.uniform(-tilt, tilt), rz=rng.uniform(-180, 180)))
    return out


def _tilted_plane(seed=1):
    rng = np.random.default_rng(seed)
    n = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 1.0])
    return rng.uniform(-0.01, 0.01, 3), n / np.linalg.norm(n)


def test_plane_equals_z0_clamp():
    """With p = 0 and n = +z the plane clamp is the v2.1.1 clamp, pose and lift, bit for bit."""
    for T in _random_poses(200):
        for m in (0.0, -0.005, 0.002):
            a, la = P.clamp_start_outside_skin(T, min_clear_m=m)
            b, lb = P.clamp_start_outside_plane(T, np.zeros(3), np.array([0.0, 0.0, 1.0]), min_clear_m=m)
            assert np.array_equal(a, b) and la == lb


def test_lift_clears_corners():
    p, n = _tilted_plane()
    lifted = 0
    for T in _random_poses(200, seed=3):
        Tc, lift = P.clamp_start_outside_plane(T, p, n)
        gc, gm = P.plane_gap_m(Tc, p, n)
        assert gm >= -1e-12
        if lift > 0:
            lifted += 1
            assert abs(gm) < 1e-12                        # lifted exactly onto the plane
            assert np.allclose(Tc[:3, 3] - np.asarray(T)[:3, 3], lift * n, atol=1e-15)
            assert np.array_equal(Tc[:3, :3], np.asarray(T)[:3, :3])
        else:
            assert np.array_equal(Tc, T)
    assert lifted > 20


def _env(**kw):
    from usbg.mujoco_env import USProbeEnv
    lv = V.synthetic_phantom()
    return USProbeEnv(lv, np.eye(4), pose_from_rpy(tz=0.02, rx=180.0), device="cpu", observe=False, **kw)


def test_press_cap_5mm(monkeypatch):
    monkeypatch.setenv("USBG_SEAT", "anatomical")
    p, n = _tilted_plane(5)
    env = _env(contact_plane=(p, n))
    assert env.max_press_m == pytest.approx(0.005)
    for T in _random_poses(50, seed=7):
        deep = np.asarray(T, np.float64).copy()
        deep[:3, 3] = p - 0.02 * n + (deep[:3, 3] - p) - ((deep[:3, 3] - p) @ n) * n   # 20 mm below the plane
        Tl = env._press_limited(deep)
        assert P.plane_gap_m(Tl, p, n)[1] == pytest.approx(-0.005, abs=1e-12)
    # default env: the v2.1.1 press cap against z = 0
    env0 = _env()
    for T in _random_poses(20, seed=8):
        assert np.array_equal(env0._press_limited(T), P.clamp_start_outside_skin(T, min_clear_m=-0.005)[0])


def test_tilted_box_contact():
    """The re-posed box top is the plane; pressing a face into it gives force, hovering gives none."""
    import mujoco
    from usbg.mujoco_env import plane_rotation
    p, n = _tilted_plane(9)
    env = _env(contact_plane=(p, n), contact=True)
    gid = env._patient_gid
    mujoco.mj_forward(env.model, env.data)
    R = env.data.geom_xmat[gid].reshape(3, 3)
    top_centre = env.data.geom_xpos[gid] + R[:, 2] * env.model.geom_size[gid][2]
    assert np.allclose(top_centre, p, atol=1e-12) and np.allclose(R[:, 2], n, atol=1e-12)
    assert np.allclose(plane_rotation(n)[:, 2], n, atol=1e-15)
    # probe imaging along -n (its +z into the patient), face 3 mm into the box, then 5 mm above it
    Rp = plane_rotation(n) @ pose_from_rpy(rx=180.0)[:3, :3]
    T_in = np.eye(4); T_in[:3, :3] = Rp; T_in[:3, 3] = p - 0.003 * n
    T_out = T_in.copy(); T_out[:3, 3] = p + 0.005 * n
    assert env.force_at_pose(T_in) > 0.0
    assert env.force_at_pose(T_out) == 0.0


def test_mjcf_sha_equals_v211():
    from usbg.mujoco_env import _mjcf
    assert hashlib.sha256(_mjcf().encode()).hexdigest() == MJCF_SHA256_V211
    env = _env()
    gid = env._patient_gid
    assert env.contact_plane is None
    assert np.array_equal(env.model.geom_pos[gid], [0.0, 0.0, -0.08])
    assert np.array_equal(env.model.geom_quat[gid], [1.0, 0.0, 0.0, 0.0])


def test_contact_plane_refuses_zero_normal():
    with pytest.raises(ValueError):
        _env(contact_plane=(np.zeros(3), np.zeros(3)))


@pytest.mark.anatomical
@pytest.mark.skipif(not HAVE_TRAIN, reason="needs data/cache_anat (TRAIN case s0011)")
def test_sample_setup_unchanged_stream():
    """The pre-lift start joins the meta; the random stream, the returned triple and the other keys do not
    change (the returned start is the v2.1.1 clamp of the pre-lift start)."""
    from usbg import goal_rl as G
    lv = V.load_cache(os.path.join(REPO, "data", "cache", "s0011.npz"))
    for seed in range(6):
        r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
        a = G.sample_setup(lv, V.LIVER, r1, lateral_m=0.02, axial_m=0.01, fan_deg=18.0)
        b = G.sample_setup(lv, V.LIVER, r2, lateral_m=0.02, axial_m=0.01, fan_deg=18.0, return_meta=True)
        for x, y in zip(a, b[:3]):
            assert np.array_equal(x, y)
        assert r1.bit_generator.state == r2.bit_generator.state
        meta = b[3]
        assert set(meta) == {"seat", "window", "n_windows", "start_clamped", "clamp_mm", "start_prelift"}
        Tc, lift = P.clamp_start_outside_skin(meta["start_prelift"])
        assert np.array_equal(Tc, b[2]) and float(lift * 1000.0) == meta["clamp_mm"]
