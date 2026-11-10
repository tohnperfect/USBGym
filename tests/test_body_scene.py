"""usbg.body_scene: a human body as the opt-in contact surface of USProbeEnv and ViewMatchEnv, and the MuJoCo scene
file of make_body_scene.py.

Synthetic tests use an ellipsoid 'body' (no data). The anatomical test runs the gym on the training case s0011 when
its label cache is present."""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("skimage")

from usbg import body_scene as BS
from usbg import placement as P
from usbg import volume as V
from usbg.contracts_bridge import pose_from_rpy
from usbg.mujoco_env import USProbeEnv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_S0011 = os.path.exists(os.path.join(REPO, "data", "cache_anat", "s0011.npz"))
AX = np.array([0.15, 0.10, 0.18])          # ellipsoid semi-axes (m): left-right, anterior-posterior, superior-inferior


def _ellipsoid(case="synthetic_ellipsoid", sp=0.004):
    n = np.ceil(2 * AX / sp).astype(int) + 6
    origin = -(n - 1) / 2.0 * sp
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in n], indexing="ij")
    x, y, z = origin[0] + ii * sp, origin[1] + jj * sp, origin[2] + kk * sp
    body = (x / AX[0]) ** 2 + (y / AX[1]) ** 2 + (z / AX[2]) ** 2 <= 1.0
    lab = np.where(body, V.SOFT_TISSUE, V.BACKGROUND).astype(np.uint8)
    lab[body & ((x + 0.05) ** 2 + (y - 0.03) ** 2 + (z - 0.03) ** 2 <= 0.05 ** 2)] = V.LIVER
    aff = np.eye(4)
    aff[:3, :3] *= sp
    aff[:3, 3] = origin
    return V.LabeledVolume(labels=lab, affine_m=aff, spacing_m=np.full(3, sp), origin_m=origin, meta={"case": case})


def _seat():
    """world_from_volume with the anterior pole of the ellipsoid (0, b, 0) at the world origin, anterior up (+z)."""
    W = np.eye(4)
    W[:3, :3] = P.R_SEAT
    W[:3, 3] = -P.R_SEAT @ np.array([0.0, AX[1], 0.0])
    return W


def _probe_at(height_m):
    return pose_from_rpy(tz=height_m, rx=180.0)          # imaging along world -z (into the body)


@pytest.fixture(scope="module")
def ell():
    BS.clear_cache()
    lv = _ellipsoid()
    return lv, BS.ct_body(lv), _seat()


def test_ct_body_slabs(ell):
    lv, g, _W = ell
    assert g.kind == "ct_skin" and np.array_equal(g.volume_from_body, np.eye(4))
    assert len(g.slabs) == BS.SLAB_K == g.info["n_slabs"]
    zlo = min(s[:, 2].min() for s in g.slabs)
    zhi = max(s[:, 2].max() for s in g.slabs)
    assert zlo <= g.verts[:, 2].min() + 0.002 and zhi >= g.verts[:, 2].max() - 0.002   # slab vertices are subsampled
    assert BS.ct_body(lv) is g                               # cached per patient


def test_surface_plane_on_a_cube():
    cube = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)])
    g = BS.BodyGeometry("ct_skin", cube, np.zeros((0, 3), np.int64), [cube], np.eye(4))
    s = BS.place(g, np.eye(4))
    p, n = s.surface_plane(_probe_at(0.15))
    assert np.allclose(n, [0, 0, 1]) and np.allclose(p, [0, 0, 0.1])
    assert s.gap_mm(_probe_at(0.15)) == pytest.approx(50.0)
    T = pose_from_rpy(tx=0.06, tz=0.02, rx=180.0)            # inside, nearest the +x face
    p, n = s.surface_plane(T)
    assert np.allclose(n, [1, 0, 0]) and s.gap_mm(T) == pytest.approx(-40.0)
    Tc, lift = s.clamp(_probe_at(0.05))
    assert lift > 0 and P.plane_gap_m(Tc, *s.surface_plane(Tc))[1] == pytest.approx(0.0, abs=1e-12)


def test_window_face_on_the_ct_skin(ell):
    _lv, g, W = ell
    s = BS.place(g, W)
    assert abs(s.gap_mm(_probe_at(0.0))) < 3.0                 # slabs follow the skin at the anterior pole
    assert s.gap_mm(_probe_at(0.01)) == pytest.approx(s.gap_mm(_probe_at(0.0)) + 10.0, abs=0.05)   # facet may change


def test_default_env_keeps_the_box(ell):
    lv, _g, W = ell
    env = USProbeEnv(lv, W, _probe_at(0.005), renderer="numpy", observe=False)
    assert env.body is None and len(env._patient_gids) == 1 and env._patient_gid in env._patient_gids


def test_body_env_contact_and_press_cap(ell):
    lv, g, W = ell
    s = BS.place(g, W)
    env = USProbeEnv(lv, W, _probe_at(0.005), renderer="numpy", observe=False, body=s, max_press_m=0.005)
    assert len(env._patient_gids) == len(g.slabs)
    env.reset()
    assert env.contact_force() == 0.0                          # 5 mm above the skin
    for _ in range(15):
        env.step(np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0]))    # 3 mm per step along the beam, into the body
    assert env.contact_force() > 0.5
    assert s.gap_mm(env._target) >= -5.0 - 1e-6               # the commanded press is capped at 5 mm
    off = USProbeEnv(lv, W, _probe_at(0.005), renderer="numpy", observe=False, body=s, contact=False,
                     max_press_m=0.005)
    for _ in range(15):
        off.step(np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0]))
    assert off.contact_force() == 0.0


def test_body_and_contact_plane_are_exclusive(ell):
    lv, g, W = ell
    with pytest.raises(ValueError, match="cannot be combined"):
        USProbeEnv(lv, W, _probe_at(0.005), renderer="numpy", observe=False, body=BS.place(g, W),
                   contact_plane=(np.zeros(3), np.array([0.0, 0.0, 1.0])))


def test_scene_file(ell):
    import mujoco
    lv, g, W = ell
    xml, rec = BS.scene_xml(lv, W, _probe_at(0.0), "ct_skin", visual=True)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.ngeom == len(g.slabs) + 4                         # slabs, drawn body, floor, probe, mocap marker
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "probe")
    assert np.allclose(d.xpos[bid], np.asarray(rec["probe_start"])[:3, 3], atol=1e-6)
    assert rec["surface"] == "ct_skin" and rec["n_slabs"] == len(g.slabs) and abs(rec["window_gap_mm"]) < 3.0
    v, f = BS.place(g, W).visual_mesh()
    tri = v[f]
    assert np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() > 0   # faces wound outward


def test_packaged_avatar():
    v, f = BS.load_avatar()
    assert 15000 <= len(f) <= 20000
    ext = v.max(0) - v.min(0)
    assert 0.55 < ext[2] < 0.75 and 0.3 < ext[0] < 0.6 and 0.2 < ext[1] < 0.35
    # orientation pin of this asset: the neck and chin (top tenth) lie anterior (+y) of the torso's centroid
    top = v[v[:, 2] > v[:, 2].max() - 0.1 * ext[2]]
    assert top[:, 1].mean() - v[:, 1].mean() > 0.03


def test_avatar_fit_on_the_ellipsoid(ell):
    lv, _g, W = ell
    a = BS.avatar_body(lv)
    assert a.kind == "avatar" and a.info["mesh"] == BS.AVATAR_FILE and len(a.slabs) == BS.SLAB_K
    assert np.all(np.isfinite(a.volume_from_body)) and 0.3 < a.info["reg_scale"] < 3.0
    assert np.isfinite(a.info["reg_rms_mm"])
    s = BS.place(a, W)
    assert np.isfinite(s.gap_mm(_probe_at(0.0)))


def test_view_match_env_rejects_an_unknown_surface():
    pytest.importorskip("gymnasium")
    from usbg.gym_env import ViewMatchEnv
    with pytest.raises(ValueError, match="contact_surface"):
        ViewMatchEnv(contact_surface="sphere")


@pytest.mark.anatomical
@pytest.mark.skipif(not HAVE_S0011, reason="needs the s0011 label cache (README, Data preparation)")
def test_view_match_env_on_body_surfaces():
    pytest.importorskip("gymnasium")
    from usbg.gym_env import ViewMatchEnv
    env = ViewMatchEnv(patients=["s0011"], targets=("liver",), min_windows=1, device="cpu")
    for surface in ("ct_skin", "avatar"):
        obs, info = env.reset(seed=0, options={"contact_surface": surface, "window": 0})
        assert info["contact_surface"] == surface
        body = env._probe.body
        assert body is not None and body.kind == surface
        assert body.gap_mm(env._probe.world_from_probe) >= -1.0      # the start is lifted out of the body
        if surface == "ct_skin":
            assert abs(info["surface_gap_mm"]) < 3.0
        for _ in range(3):
            obs, r, term, trunc, info = env.step(np.zeros(6, np.float32))
        assert np.isfinite(r) and info["contact_surface"] == surface
    obs, info = env.reset(seed=0, options={"window": 0})
    assert info["contact_surface"] == "box" and info["surface_gap_mm"] == 0.0 and env._probe.body is None
    env.close()
