"""Vendoring regression for the self-contained release.

Up to v2.0.0, usbg imported three pieces of code from the authors' probe-action recognizer code
(package `par`): the SE(3) math (src/par/se3.py), the cross-repo contract (src/par/contracts.py) and the
fan-scan geometry with the NumPy appearance model (scripts/us_render.py, which usbg._reuse executed from
the sibling tree). They are now vendored in usbg._par. This file pins that the switch changed no number.

Layer 1, stored reference (always runs: no data, no sibling repo, CPU only).
  tests/data/vendor_regression_ref.json was captured BEFORE the switch, with `par` installed, by running
  synthetic_outputs() below on the platform it names. Per output it stores the sha256 of the exact bytes
  and a strided subsample. On the capture platform with the same numpy / torch / mujoco / scipy / pillow
  versions, every digest must match bit for bit. Anywhere else (CI on Linux, Windows or Intel, or other
  library versions) BLAS, SIMD and library versions may change the last bits, so the subsamples must
  agree within TOL (floats) and IMG_MEAN / IMG_P99 (images, 8-bit quantized on the way), and integer
  label maps may differ in at most INT_MISMATCH of the stored elements. USBG_REGRESSION_MODE=tolerance forces the tolerance check (a self-test of that path).
Layer 2, parity with the source (runs only where `par` and its scripts/us_render.py are present):
  every vendored function runs side by side with its original on the same inputs and must agree bit
  for bit; se3.py must be byte-identical (its sha256 is the cross-repo GEOMETRY_SHA256).
Layer 3, local data (skips cleanly without data/cache_anat/s0358.npz, the screened-windows file, the v2
  SonoSPADE checkpoint and the v2 physics CT-mask policy; data/ and outputs/ are not in git): an
  anatomical-seat physics and SonoSPADE frame of held-out patient s0358, and two policy evaluation
  episodes of outputs/v2/table4/ctmask/physics_deploy_s0.pt on CPU. They are episodes 0 and 1 of that
  policy's archived 60-episode evaluation (same setups, rng seeds 999 and 1000). Only digests, summary
  statistics and per-episode metrics are stored, never image content.

Every computation runs on the CPU (render_lotus.get_device is patched to "cpu") with one torch thread.

Re-capture (only for a deliberate numerics change; the stored file is the pre-switch record):
  PYTHONPATH=src .venv/bin/python tests/test_vendor_regression.py --capture <out_dir>
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import platform
import sys
import zlib

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PATH = os.path.join(REPO, "tests", "data", "vendor_regression_ref.json")
MAX_SUB = 2048                      # elements kept per float subsample
MAX_INT_FULL = 1 << 20              # integer arrays up to this size are stored whole (zlib)

# Tolerance mode (another platform or other library versions), by output-name prefix.
# Images in [0,1] pass through 8-bit quantization (view_match._resize01), so a last-bit difference can flip
# a pixel by one gray level (1/255): they need mean |diff| <= IMG_MEAN and 99th-percentile |diff| <= IMG_P99.
# A real change (geometry, constants, renderer) moves whole regions and fails both by orders of magnitude.
IMAGES = ("render_", "curvi_img", "texture_frame", "env_obs", "ep_obs", "ep_target")
IMG_MEAN, IMG_P99 = 1e-3, 2.0 / 255.0
TOL = {                             # (rtol, atol) for np.allclose on the other float outputs
    "se3_": (0.0, 1e-12),           # pure float64 NumPy math
    "scan_": (0.0, 1e-7),           # float32 grid (meters)
    "env_poses": (0.0, 1e-5),       # meters (10 micrometers)
    "ep_poses": (0.0, 1e-5),
    "env_forces": (1e-3, 1e-2),     # newtons
    "ep_forces": (1e-3, 1e-2),
    "": (1e-3, 2e-3),               # actions, pose errors, scores
}
INT_MISMATCH = 1e-3                 # max fraction of differing elements in an integer label map

# Layer 3 inputs (all git-ignored)
DATA_CASE = "s0358"
DATA_ANAT = os.path.join(REPO, "data", "cache_anat", f"{DATA_CASE}.npz")
DATA_TEX = os.path.join(REPO, "data", "dataset", "v2", "sonospade_curvi.pt")
DATA_POLICY = os.path.join(REPO, "outputs", "v2", "table4", "ctmask", "physics_deploy_s0.json")
EP_KEYS = ("start_pos", "start_ang", "clamp_mm", "static_seg", "static_match", "final_pos", "final_ang",
           "final_seg", "final_match", "keepbest_pos", "keepbest_ang", "keepbest_seg", "peak_force",
           "contact_frac")


# ========================================================================================== helpers
def _versions() -> dict:
    import mujoco
    import PIL
    import scipy
    import torch
    return {"system": platform.system(), "machine": platform.machine(),
            "python": ".".join(platform.python_version_tuple()[:2]), "numpy": np.__version__,
            "torch": torch.__version__, "mujoco": mujoco.__version__, "scipy": scipy.__version__,
            "pillow": PIL.__version__}


def _digest(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _enc(a) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 9)).decode("ascii")


def _dec(s, dtype) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype=np.dtype(dtype))


def _is_int(a) -> bool:
    return np.issubdtype(np.asarray(a).dtype, np.integer) or np.asarray(a).dtype == bool


def _subsample(a):
    flat = np.asarray(a).reshape(-1)
    if _is_int(a) and flat.size <= MAX_INT_FULL:
        return flat, 1
    stride = max(1, -(-flat.size // MAX_SUB))
    return flat[::stride], stride


def _stats(a) -> dict:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}


def record(outputs: dict, with_content=True) -> dict:
    """{name: {dtype, shape, sha256, stats[, sub, stride]}}; with_content=False keeps digests/stats only."""
    rec = {}
    for k, a in outputs.items():
        a = np.ascontiguousarray(np.asarray(a))
        r = {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": _digest(a), "stats": _stats(a)}
        if with_content:
            sub, stride = _subsample(a)
            r.update(sub=_enc(sub), stride=stride)
        rec[k] = r
    return rec


def _tol(name):
    for p in sorted(TOL, key=len, reverse=True):
        if name.startswith(p):
            return TOL[p]
    return TOL[""]


def exact_mode(ref_versions: dict) -> bool:
    if os.environ.get("USBG_REGRESSION_MODE", "").strip().lower() == "tolerance":
        return False
    return _versions() == ref_versions


def compare(ref: dict, outputs: dict, exact: bool, with_content=True) -> list:
    """Problems (empty = pass). exact: sha256 per output. Else: subsample / stats within tolerance."""
    bad = []
    for k, r in ref.items():
        if k not in outputs:
            bad.append(f"{k}: missing from the recomputed outputs")
            continue
        a = np.ascontiguousarray(np.asarray(outputs[k]))
        if a.dtype.str != r["dtype"] or list(a.shape) != r["shape"]:
            bad.append(f"{k}: dtype/shape {a.dtype.str}{list(a.shape)} != reference {r['dtype']}{r['shape']}")
            continue
        if exact:
            if _digest(a) != r["sha256"]:
                diff = ""
                if with_content:
                    sub, _ = _subsample(a)
                    diff = f" (max |diff| on the subsample {np.max(np.abs(sub.astype(np.float64) - _dec(r['sub'], r['dtype']).astype(np.float64))):.3g})"
                bad.append(f"{k}: sha256 differs from the pre-switch reference{diff}")
            continue
        rtol, atol = _tol(k)
        if with_content:
            got, _ = _subsample(a)
            want = _dec(r["sub"], r["dtype"])
            if _is_int(a):
                frac = float(np.mean(got != want)) if got.size else 0.0
                if frac > INT_MISMATCH:
                    bad.append(f"{k}: {frac:.2e} of the label elements differ (> {INT_MISMATCH})")
                continue
            d = np.abs(got.astype(np.float64) - want.astype(np.float64))
            if k.startswith(IMAGES):
                if d.mean() > IMG_MEAN or np.quantile(d, 0.99) > IMG_P99:
                    bad.append(f"{k}: mean |diff| {d.mean():.3g} (<= {IMG_MEAN}) / p99 {np.quantile(d, 0.99):.3g} "
                               f"(<= {IMG_P99:.3g}) out of tolerance")
            elif not np.allclose(got.astype(np.float64), want.astype(np.float64), rtol=rtol, atol=atol):
                bad.append(f"{k}: max |diff| {d.max():.3g} exceeds rtol {rtol} / atol {atol}")
        else:
            s = _stats(a)
            for sk, v in r["stats"].items():
                if not np.isclose(s[sk], v, rtol=max(rtol, 1e-3), atol=max(atol, 2e-3)):
                    bad.append(f"{k}: stats[{sk}] {s[sk]:.6g} != reference {v:.6g}")
    return bad


@contextlib.contextmanager
def cpu_only(seat="canonical"):
    """CPU device everywhere (curvi_render calls the renderer without a device), one torch thread, the CT
    mask (no learned segmenter), the given seat and no gel override; everything restored on exit."""
    import torch
    from usbg import render_lotus as R
    from usbg import view_match as VM
    saved_env = {k: os.environ.get(k) for k in ("USBG_SEAT", "USBG_GEL", "USBG_CONTACT", "USBG_MAX_PRESS_MM")}
    saved = (R.get_device, VM._US, torch.get_num_threads())
    os.environ["USBG_SEAT"] = seat
    for k in ("USBG_GEL", "USBG_CONTACT", "USBG_MAX_PRESS_MM"):
        os.environ.pop(k, None)
    R.get_device = lambda *a, **k: "cpu"
    VM._US = None
    torch.set_num_threads(1)
    try:
        yield
    finally:
        R.get_device, VM._US = saved[0], saved[1]
        torch.set_num_threads(saved[2])
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def contract_values() -> dict:
    from usbg import contracts_bridge as cb
    keys = ("CHANNELS_TWIST", "CHANNELS_REL6", "CHANNELS_INPUT", "IN_CHANNELS", "TARGET_CHANNELS", "CLASSES_5",
            "SESSION_CLASSES_6", "N_CLASSES", "HOLD_IDX", "DT_C", "OUT_RATE_HZ", "DOMINANT_AXIS",
            "GEOMETRY_SHA256", "CLIP_SIGMA", "NEXT_ID_NONE", "TTNB_UNITS", "LATENT_D")
    out = {k: getattr(cb, k) for k in keys}
    out["class_names"] = [cb.class_names("clip"), cb.class_names("session")]
    ns = cb.NormStats(center=np.arange(12.0), scale=np.r_[np.ones(11), 0.0], method="mad")
    out["NormStats_json"] = ns.to_json()
    out["NormStats_roundtrip"] = cb.NormStats.from_json(ns.to_json()).apply(np.arange(12.0) + 1.0).tolist()
    out["us_render_consts"] = _us_render_consts()
    return json.loads(json.dumps(out))


def _us_render_consts() -> dict:
    from usbg._reuse import us_render as U
    names = ("IMG_DEPTH_M", "IMG_WIDTH_M", "N_LINES", "N_SAMPLES", "PROBE", "FOV_DEG", "APEX_M", "FREQ_ATTEN",
             "SPECKLE", "BEAM_LAT_MM", "PULSE_AX_MM", "LABELS")
    d = {n: getattr(U, n) for n in names}
    d["TISSUES"] = {str(k): list(v) for k, v in U.TISSUES.items()}
    d["Z_OF"], d["A_OF"], d["S_OF"] = U.Z_OF.tolist(), U.A_OF.tolist(), U.S_OF.tolist()
    return d


# ===================================================================================== computations
def _se3_outputs(cb) -> dict:
    rng = np.random.default_rng(20261014)
    tw = rng.normal(size=(16, 6)) * np.array([0.02, 0.02, 0.02, 0.6, 0.6, 0.6])
    tw[0, 3:] = [1e-11, -2e-11, 3e-11]                         # small-angle branch
    tw[1, 3:] = [np.pi - 1e-8, 0.0, 0.0]                         # near-pi branch
    Ts = np.stack([cb.se3_exp(t) for t in tw])
    return {"se3_exp": Ts,
            "se3_log": np.stack([cb.se3_log(T) for T in Ts]),
            "se3_so3_exp": np.stack([cb.so3_exp(t[3:]) for t in tw]),
            "se3_so3_log": np.stack([cb.so3_log(T[:3, :3]) for T in Ts]),
            "se3_invert": np.stack([cb.invert_T(T) for T in Ts]),
            "se3_interp": np.stack([cb.se3_interpolate(Ts[i], Ts[i + 1], 0.37) for i in range(15)]),
            "se3_vol_from_probe": np.stack([cb.volume_from_probe(Ts[i], Ts[i + 1]) for i in range(15)])}


def _texture_translator(tmpdir, device="cpu"):
    """A seeded, untrained SonoSPADE generator (small) saved and reloaded through the standard loader, so
    the texture stage runs with no checkpoint download. Not the published model; a numerics probe only."""
    import torch
    from usbg import texture_gan as TG
    torch.manual_seed(0)
    G = TG._spade_generator(13, ngf=8, n_blocks=2, spade_hidden=16)
    path = os.path.join(tmpdir, "seeded_spade.pt")
    torch.save({"G_spade": G.state_dict(), "n_classes": 13, "ngf": 8, "n_blocks": 2, "spade_hidden": 16}, path)
    return TG.load_translator(path, device=device)


def synthetic_outputs(tmpdir) -> dict:
    """Layer 1: everything the vendored code touches, on the analytic phantom, CPU only."""
    import torch
    from usbg import contracts_bridge as cb, goal_rl as G, render_lotus as R, slicer as S, view_match as VM
    from usbg import volume as V
    from usbg.eval import _prep
    from usbg.mujoco_env import USProbeEnv
    from usbg.placement import base_pose_over, seat_volume_under_probe
    from usbg.policy import GoalPolicy

    out = _se3_outputs(cb)
    pts, depth = S.scan_grid_probe()
    out.update(scan_pts=pts, scan_depth=depth)

    lv = V.synthetic_phantom()
    out["phantom_labels"] = lv.labels
    out["phantom_counts"] = np.bincount(lv.labels.reshape(-1), minlength=16).astype(np.int64)

    # label slices and the three renderers (physics = the v2 renderer; numpy = us_render's appearance model)
    W = cb.make_world_from_volume()
    base = base_pose_over(lv, V.GALLBLADDER)
    poses = [base @ cb.pose_from_rpy(rx=a, tx=dx) for a, dx in ((-20, 0.0), (0, 0.0), (20, 0.005))]
    slices = np.stack([S.slice_at_pose(lv, T, W) for T in poses])
    out["slices"] = slices
    cfg = R.RenderConfig(seed=3)
    out["render_physics"] = np.stack([R.render(s, "physics", device="cpu", cfg=cfg) for s in slices])
    out["render_lotus"] = R.render(slices[1], "lotus", device="cpu", seed=7)
    out["render_numpy"] = R.render(slices[1], "numpy", rng=np.random.default_rng(5))
    out["render_warp"] = R.render(slices[1], "physics", device="cpu", cfg=cfg, warp=True)

    # the v2 observation: curvilinear physics frame + its CT label, and a texture-stage frame
    Wfv, T0 = seat_volume_under_probe(lv, V.GALLBLADDER, standoff_m=0.0, depth_m=0.04)
    T_view = T0 @ cb.pose_from_rpy(rx=6.0, ty=0.003)
    img, lab = VM.curvi_render(lv, T_view, Wfv, None, R.RenderConfig(seed=0), 128)
    out.update(curvi_img=img, curvi_lab=lab)
    tr = _texture_translator(tmpdir)
    out["texture_frame"], _ = VM.curvi_render(lv, T_view, Wfv, tr, R.RenderConfig(seed=0), 128)

    # MuJoCo env rollout (contact on, physics observation)
    Wfv2, T02 = seat_volume_under_probe(lv, V.GALLBLADDER, standoff_m=0.006, depth_m=0.04)
    env = USProbeEnv(lv, Wfv2, T02, renderer="physics", device="cpu", cfg=R.RenderConfig(seed=1),
                     rng=np.random.default_rng(0), contact=True)
    obs, info = env.reset()
    e_poses, e_forces, e_obs = [env.world_from_probe], [info["contact_force"]], [obs]
    for tw in ([0, 0, 0.012, 0, 0, 0],) * 3 + ([0.01, 0, 0.004, 0.1, 0, 0], [0, -0.01, 0, 0, 0.15, 0.05],
                                                [0, 0, -0.01, 0, 0, 0]):
        obs, info = env.step(tw)
        e_poses.append(env.world_from_probe); e_forces.append(info["contact_force"]); e_obs.append(obs)
    out.update(env_poses=np.stack(e_poses), env_forces=np.asarray(e_forces), env_obs=np.stack(e_obs))

    # one short policy evaluation episode, as train_view_match_rl.evaluate runs it: a seeded Gaussian goal
    # policy (BatchNorm in train mode, as in v2) adding a bounded residual to the analytic pose base
    torch.manual_seed(0)
    net = GoalPolicy.build(history=4)
    torch.nn.init.normal_(net.head.weight, std=0.05)            # the zero-init head would ignore the images
    pol = G.build_goal_gaussian_policy(net, max_twist=0.5 * G.MAX_TWIST)
    pol.train()
    rng = np.random.default_rng(999)
    Wfv3, T_tgt, T_start = G.sample_setup(lv, V.GALLBLADDER, rng, lateral_m=0.01, axial_m=0.004, fan_deg=8.0)
    env = USProbeEnv(lv, Wfv3, T_start, renderer="physics", device="cpu", rng=np.random.default_rng(999),
                     contact=True, observe=False)
    c = R.RenderConfig(seed=0)
    tgt_img, tgt_lab = VM.curvi_render(lv, T_tgt, Wfv3, None, c, 128)
    tgt_t = torch.from_numpy(_prep(tgt_img, 128)[None, None])
    env.reset()
    cur_img, cur_lab = VM.curvi_render(lv, env.world_from_probe, Wfv3, None, c, 128)
    hist = [_prep(cur_img, 128)] * 4
    p_poses, p_act, p_pe, p_match, p_forces, p_obs = [env.world_from_probe], [], [], [], [], [cur_img]
    p_pe.append(VM.pose_error(env.world_from_probe, T_tgt))
    p_match.append(VM.combined_score(cur_img, cur_lab, tgt_img, tgt_lab, None, 0.6, 0.4))
    with torch.no_grad():
        for _ in range(5):
            a, _ = pol.act(torch.from_numpy(np.stack(hist, 0)[None]), tgt_t, deterministic=True)
            a_exec = np.clip(a[0].numpy().astype(np.float64) + G.analytic_twist(env.world_from_probe, T_tgt),
                             -G.MAX_TWIST, G.MAX_TWIST)
            env.step(a_exec)
            cur_img, cur_lab = VM.curvi_render(lv, env.world_from_probe, Wfv3, None, c, 128)
            hist = hist[1:] + [_prep(cur_img, 128)]
            p_poses.append(env.world_from_probe); p_act.append(a_exec); p_forces.append(env.contact_force())
            p_pe.append(VM.pose_error(env.world_from_probe, T_tgt)); p_obs.append(cur_img)
            p_match.append(VM.combined_score(cur_img, cur_lab, tgt_img, tgt_lab, None, 0.6, 0.4))
    out.update(ep_poses=np.stack(p_poses), ep_actions=np.stack(p_act), ep_pose_err=np.asarray(p_pe),
               ep_match=np.asarray(p_match), ep_forces=np.asarray(p_forces), ep_obs=np.stack(p_obs),
               ep_target=tgt_img)
    return out


def data_available() -> bool:
    from usbg import windows as Wn
    return all(os.path.exists(p) for p in (DATA_ANAT, Wn.WINDOWS_PATH, DATA_TEX, DATA_POLICY,
                                           DATA_POLICY.replace(".json", ".pt")))


def data_outputs() -> tuple[dict, dict]:
    """Layer 3 (anatomical seat, CPU): frames of held-out s0358 at its first liver window and two policy
    evaluation episodes of the v2 physics CT-mask policy, exactly as eval_view_ladder --rung policy
    rebuilds it. Returns (arrays, per-episode records)."""
    import torch
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import eval_view_ladder as EVL
    import train_view_match_rl as T
    from usbg import goal_rl as G, render_lotus as R, slicer as S, texture_gan as TG, view_match as VM
    from usbg import volume as V

    out = {}
    lv = T.load_volumes([DATA_CASE])
    Wfv, T_tgt, T_start, meta = G.sample_setup(lv[0], V.LIVER, np.random.default_rng(999), window=0,
                                               return_meta=True, lateral_m=0.02, axial_m=0.01, fan_deg=18.0)
    out["anat_poses"] = np.stack([Wfv, T_tgt, T_start])
    out["anat_slice"] = S.slice_at_pose(lv[0], T_tgt, Wfv)
    cfg = R.RenderConfig(seed=0)
    out["anat_physics"], out["anat_lab"] = VM.curvi_render(lv[0], T_tgt, Wfv, None, cfg, 128)
    tr = TG.load_translator(DATA_TEX, device="cpu")
    out["anat_sonospade"], _ = VM.curvi_render(lv[0], T_tgt, Wfv, tr, cfg, 128)

    with open(DATA_POLICY) as fh:
        J = json.load(fh)
    c = J["config"]
    saved = {k: getattr(T, k) for k in ("SEAT_ORGANS", "_MIN_WINDOWS", "_CONTACT", "_RESIDUAL_MODE", "_IBVS")}
    try:
        T.SEAT_ORGANS = list(c["targets"])
        T._MIN_WINDOWS = int(c["min_windows"])
        T._CONTACT = True
        T._RESIDUAL_MODE = J.get("residual_mode", "none")
        ck = torch.load(DATA_POLICY.replace(".json", ".pt"), map_location="cpu", weights_only=False)
        ib = ck["ibvs"]
        T._IBVS = {"kind": "linear", "model": np.asarray(ib["model"]), "rich": ib["rich"]}
        pol, _ = EVL.load_policy(DATA_POLICY.replace(".json", ".pt"), c.get("residual_frac", 0.5),
                                 bool(c["fix_residual"]), "cpu")
        torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
        with torch.no_grad():
            ev = T.evaluate(lv, [DATA_CASE], None, R.RenderConfig(seed=c["seed"]), pol, None, c["history"],
                            c["image_size"], "cpu", c["w_seg"], c["w_app"],
                            dict(lateral_m=c["lateral_m"], axial_m=c["axial_m"], fan_deg=c["fan_deg"]),
                            c["n_steps"], 2, seed=c["eval_seed"], residual=bool(c["fix_residual"]),
                            design=c["eval_design"])
    finally:
        for k, v in saved.items():
            setattr(T, k, v)
    eps = ev["episodes"]
    out["policy_episode_metrics"] = np.array([[float(e[k]) for k in EP_KEYS] for e in eps])
    out["policy_pos_traj"] = np.array([e["pos_traj"] for e in eps])
    out["policy_match_traj"] = np.array([e["match_traj"] for e in eps])
    return out, {"episodes": [{k: e[k] for k in ("ep", "patient", "target", "window", *EP_KEYS)} for e in eps]}


def _load_ref():
    with open(REF_PATH) as fh:
        return json.load(fh)


# ======================================================================================== the tests
@pytest.fixture(scope="module")
def ref():
    if not os.path.exists(REF_PATH):
        pytest.fail(f"{REF_PATH} is missing (it is committed with this test)")
    return _load_ref()


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    with cpu_only("canonical"):
        return synthetic_outputs(str(tmp_path_factory.mktemp("vendor_regression")))


def test_contract_values_unchanged(ref):
    with cpu_only("canonical"):
        got = contract_values()
    assert got == ref["contract"], "the vendored contract / scan constants differ from the pre-switch values"


def test_synthetic_pipeline_matches_pre_switch_reference(ref, synthetic):
    exact = exact_mode(ref["captured"]["versions"])
    bad = compare(ref["synthetic"], synthetic, exact)
    mode = "bit-for-bit" if exact else f"tolerance ({_versions()} vs capture {ref['captured']['versions']})"
    assert not bad, f"[{mode}] vendoring regression:\n  " + "\n  ".join(bad)


def test_tolerance_path_accepts_the_reference_itself(ref, synthetic):
    """The tolerance comparison (used off the capture platform) must pass on the exact outputs too."""
    assert not compare(ref["synthetic"], synthetic, exact=False)


def test_no_v2_module_imports_the_sibling_repos():
    """The v2 import surface never reaches par / us_rl_sim / stable_baselines3 at import time."""
    import subprocess
    code = ("import sys, importlib\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('par', 'us_rl_sim', 'stable_baselines3'):\n"
            "            raise ModuleNotFoundError('blocked: ' + name)\n"
            "sys.meta_path.insert(0, _Block())\n"
            "for m in ('usbg.contracts_bridge', 'usbg.slicer', 'usbg.render_lotus', 'usbg.volume', 'usbg.windows',\n"
            "          'usbg.placement', 'usbg.mujoco_env', 'usbg.view_match', 'usbg.goal_rl', 'usbg.eval',\n"
            "          'usbg.texture_gan', 'usbg.us_segmenter', 'usbg.segmenter', 'usbg.policy', 'usbg.provenance',\n"
            "          'usbg.benchmark', 'usbg.similarity', 'usbg.gym_env', 'usbg.interfaces'):\n"
            "    importlib.import_module(m)\n"
            "sys.path.insert(0, 'scripts')\n"
            "for m in ('train_view_match_rl', 'eval_view_ladder', 'run_table4_v2', 'agg_table4_v2'):\n"
            "    importlib.import_module(m)\n"
            "import usbg.contracts_bridge as cb; cb.assert_geometry_parity()\n"
            "print('ok')\n")
    env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


# ==== layer 2: parity with the original sibling code (skips where par is not installed)
def _par_us_render():
    par = pytest.importorskip("par", reason="the sibling `par` package is not installed (parity layer)")
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(par.__file__))))
    path = os.path.join(root, "scripts", "us_render.py")
    if not os.path.exists(path):
        pytest.skip(f"{path} is absent (parity layer)")
    spec = importlib.util.spec_from_file_location("par_us_render_parity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vendored_se3_is_byte_identical_and_bitwise_equal():
    par_se3 = pytest.importorskip("par.se3", reason="the sibling `par` package is not installed (parity layer)")
    from usbg._par import se3 as V
    with open(par_se3.__file__, "rb") as a, open(V.__file__, "rb") as b:
        assert a.read() == b.read(), "vendored se3.py is not byte-identical to par/se3.py"
    rng = np.random.default_rng(7)
    for _ in range(200):
        tw = rng.normal(size=6) * rng.choice([1e-10, 1e-3, 0.3, 3.0])
        T = V.se3_exp(tw)
        assert np.array_equal(T, par_se3.se3_exp(tw))
        assert np.array_equal(V.se3_log(T), par_se3.se3_log(T))
        assert np.array_equal(V.so3_log(T[:3, :3]), par_se3.so3_log(T[:3, :3]))
        assert np.array_equal(V.invert_T(T), par_se3.invert_T(T))
        q = rng.normal(size=4)
        assert np.array_equal(V.quat_to_matrix(q), par_se3.quat_to_matrix(q))
        assert np.array_equal(V.matrix_to_quat(T), par_se3.matrix_to_quat(T))
        T2 = V.se3_exp(rng.normal(size=6))
        assert np.array_equal(V.se3_interpolate(T, T2, 0.3), par_se3.se3_interpolate(T, T2, 0.3))


def test_vendored_contract_equals_par():
    par_c = pytest.importorskip("par.contracts", reason="the sibling `par` package is not installed (parity layer)")
    from usbg._par import contracts as V
    shared = [n for n in dir(V) if n.isupper() and hasattr(par_c, n)]
    assert len(shared) >= 17
    for n in shared:
        assert getattr(V, n) == getattr(par_c, n), n
    V.assert_geometry_parity()
    assert V.se3_source_sha256() == par_c.se3_source_sha256() == par_c.GEOMETRY_SHA256
    for dom in ("clip", "session"):
        assert V.class_names(dom) == par_c.class_names(dom)
    d = {"median": list(range(12)), "mad": [1.0] * 12, "source": "x"}
    x = np.linspace(-3, 3, 12)
    assert np.array_equal(V.NormStats.from_json(d).apply(x), par_c.NormStats.from_json(d).apply(x))


def test_vendored_us_render_equals_original():
    U0 = _par_us_render()
    from usbg._par import us_render as U
    for n in ("IMG_DEPTH_M", "IMG_WIDTH_M", "N_LINES", "N_SAMPLES", "PROBE", "FOV_DEG", "APEX_M", "FREQ_ATTEN",
              "SPECKLE", "BEAM_LAT_MM", "PULSE_AX_MM", "TISSUES", "LABELS"):
        assert getattr(U, n) == getattr(U0, n), n
    for n in ("Z_OF", "A_OF", "S_OF"):
        assert np.array_equal(getattr(U, n), getattr(U0, n)) and getattr(U, n).dtype == getattr(U0, n).dtype
    for a, b in zip(U._scan_grid(), U0._scan_grid()):
        assert np.array_equal(a, b) and a.dtype == b.dtype
    vol, sp, org = U.make_phantom()
    vol0, sp0, org0 = U0.make_phantom()
    assert np.array_equal(vol, vol0) and sp == sp0 and np.array_equal(org, org0)
    pts = np.random.default_rng(1).uniform(-0.06, 0.06, size=(5000, 3))
    assert np.array_equal(U._sample_labels(vol, sp, org, pts), U0._sample_labels(vol0, sp0, org0, pts))
    assert np.array_equal(U.pose(rx=12, ty=0.01), U0.pose(rx=12, ty=0.01))
    for s in (0.4, 1.7, 3.2):
        assert np.array_equal(U._gauss1d(s), U0._gauss1d(s))
    T = U.pose(rx=15, tx=0.01)
    with np.errstate(all="ignore"):         # Accelerate's spurious matmul FP flags (see usbg/__init__.py)
        a, b = U.render_bmode(T, vol, sp, org, np.random.default_rng(3)), \
            U0.render_bmode(T, vol0, sp0, org0, np.random.default_rng(3))
    assert np.array_equal(a, b)


# ==== layer 3: local data (skips cleanly in a fresh clone)
@pytest.mark.anatomical
def test_local_data_frames_and_policy_episodes_match_pre_switch(ref):
    if not data_available():
        pytest.skip("local v2 data absent (data/cache_anat, the windows file, the v2 SonoSPADE checkpoint, "
                    "outputs/v2/table4/ctmask/physics_deploy_s0.*); none of it is in git")
    if "data" not in ref:
        pytest.skip("the reference has no local-data section")
    with cpu_only("anatomical"):
        got, _ = data_outputs()
    exact = exact_mode(ref["captured"]["versions"])
    bad = compare(ref["data"], got, exact, with_content=False)
    assert not bad, "vendoring regression (local data):\n  " + "\n  ".join(bad)


# ========================================================================================== capture
def _capture(out_dir):
    """Write <out_dir>/baseline_synthetic.npz, baseline_data.npz (if data), baseline_meta.json and the
    reference JSON (also copied to REF_PATH only with --write-ref)."""
    import datetime
    import subprocess
    import tempfile
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as td, cpu_only("canonical"):
        syn = synthetic_outputs(td)
        con = contract_values()
    np.savez_compressed(os.path.join(out_dir, "baseline_synthetic.npz"), **syn)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    try:
        import par
        par_file = par.__file__
    except ImportError:
        par_file = None
    ref = {"schema": "usbg-vendor-regression-1",
           "captured": {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                        "versions": _versions(), "usbg_git": git, "usbg_dirty_tracked": dirty.splitlines(),
                        "par_module": par_file, "torch_threads": 1, "device": "cpu",
                        "note": "captured with tests/test_vendor_regression.py --capture"},
           "contract": con, "synthetic": record(syn)}
    meta = {}
    if data_available():
        with cpu_only("anatomical"):
            dat, eps = data_outputs()
        np.savez_compressed(os.path.join(out_dir, "baseline_data.npz"), **dat)
        ref["data"] = record(dat, with_content=False)
        meta["data_episodes"] = eps
    with open(os.path.join(out_dir, "vendor_regression_ref.json"), "w") as fh:
        json.dump(ref, fh, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "baseline_meta.json"), "w") as fh:
        json.dump({"captured": ref["captured"], **meta}, fh, indent=1, default=float)
    print(f"wrote {out_dir} ({len(syn)} synthetic outputs, data layer: {'data' in ref})")
    return ref


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--capture":
        sys.path.insert(0, os.path.join(REPO, "src"))
        _capture(sys.argv[2])
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
