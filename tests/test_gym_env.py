"""USBGym/ViewMatch-v0 (usbg.gym_env.ViewMatchEnv): registration, the Gymnasium API checks, and parity
with the evaluation code path of scripts/train_view_match_rl.py.

Parity: on the CPU, a fixed action sequence gives the same probe poses and rewards as the trainer's
rollout() and the same pose errors and match scores as its evaluate() (the code path of every oracle-ladder
number), with and without the image-servo base, under the CT mask and under a stand-in mask source. The
servo calibration and its features match the trainer's. Needs the anatomical cache (data/cache_anat) and
the windows file; skipped otherwise."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

gym = pytest.importorskip("gymnasium")

import train_view_match_rl as T  # noqa: E402
import usbg.gym_env as GE  # noqa: E402
from usbg import goal_rl as G, render_lotus as R, view_match as VM  # noqa: E402

pytestmark = pytest.mark.anatomical

HELD = ["s0358", "s0513"]
TRAIN = ["s0011", "s0058", "s0344"]          # s0058 has < 2 kidney windows: its kidney cell is skipped
SETUP = dict(lateral_m=0.02, axial_m=0.01, fan_deg=18.0)


def _data_ready():
    from usbg import windows as W
    return os.path.exists(W.WINDOWS_PATH) and all(
        os.path.exists(os.path.join(REPO, "data", "cache_anat", f"{c}.npz")) for c in HELD + TRAIN)


@pytest.fixture(autouse=True)
def _cpu_and_trainer_globals(monkeypatch):
    """CPU only (MPS hidden, as on a machine without a GPU), the CT mask unless a test binds another
    source, and the trainer's module globals restored afterwards."""
    if not _data_ready():
        pytest.skip("anatomical cache / windows file not built")
    import torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(VM, "_US", None)
    saved = dict(SEAT_ORGANS=list(T.SEAT_ORGANS), _CELLS=T._CELLS, _CONTACT=T._CONTACT, _RANDOMIZE=T._RANDOMIZE,
                 _MIN_WINDOWS=T._MIN_WINDOWS, _RESIDUAL_MODE=T._RESIDUAL_MODE, _IBVS=T._IBVS)
    T.SEAT_ORGANS, T._MIN_WINDOWS, T._CONTACT, T._RANDOMIZE = ["liver", "kidney"], 2, True, "off"
    yield
    for k, v in saved.items():
        setattr(T, k, v)
    T._SKIPS.clear()


class ToyMask:
    """A stand-in learned mask source: bright pixels are liver, mid-gray pixels kidney."""

    def segment(self, img01):
        a = np.asarray(img01)
        m = np.zeros(a.shape, np.int64)
        m[a > 0.35] = VM.LIVER
        m[(a > 0.15) & (a <= 0.35)] = VM.KIDNEY
        return m


class FixedPolicy:
    """Replays a fixed sequence of normalized actions as the trainer's policy object; the executed
    residual is action * bound, the same float32 product the environment forms."""

    def __init__(self, actions, bound):
        self.actions, self.bound, self.k = actions, np.asarray(bound, np.float32), 0

    def reset(self):
        self.k = 0

    def act(self, cur, tgt, deterministic=True):
        import torch
        a = np.asarray(self.actions[self.k], np.float32) * self.bound
        self.k += 1
        return torch.from_numpy(a[None]), None


def _actions(n=12, seed=7):
    return np.random.default_rng(seed).uniform(-1, 1, (n, 6)).astype(np.float32)


def _servo_matrix():
    return np.random.default_rng(3).normal(0.0, 0.05, (36, 6))


def _set_trainer_base(base, servo):
    if base == "servo":
        T._RESIDUAL_MODE, T._IBVS = "image", {"kind": "linear", "model": servo, "rich": True}
        return True, 0.5 * G.MAX_TWIST
    T._RESIDUAL_MODE, T._IBVS = "oracle", None
    return False, G.MAX_TWIST


def test_registered_and_passes_the_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    assert GE.ENV_ID in gym.registry
    env = gym.make(GE.ENV_ID, patients=HELD, device="cpu", include_masks=True, base="servo",
                   servo_base=_servo_matrix(), render_mode="rgb_array")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env.unwrapped)
    gym_warnings = [str(w.message) for w in caught if "WARN" in str(w.message)]
    assert not gym_warnings, gym_warnings
    obs, info = env.reset(seed=1)
    assert obs in env.observation_space and set(obs) == {"current", "target", "current_mask", "target_mask"}
    assert env.render().shape == (128, 256, 3)
    assert {"pose_error_mm", "ct_mask", "probe_pose"} <= set(info["privileged"])
    env.close()


def test_stable_baselines3_checker_with_uint8_frames():
    sb3 = pytest.importorskip("stable_baselines3.common.env_checker")
    env = GE.ViewMatchEnv(patients=HELD, device="cpu", frame_dtype="uint8", privileged_info=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sb3.check_env(env, warn=True)
    assert not [str(w.message) for w in caught if "observation" in str(w.message)]
    env.close()


@pytest.mark.parametrize("base,mask", [(None, "ct"), ("servo", "ct"), ("servo", "toy")])
def test_fixed_actions_reproduce_trainer_rollout(monkeypatch, base, mask):
    """Same poses after every step and the same rewards as train_view_match_rl.rollout."""
    servo = _servo_matrix()
    toy = ToyMask()
    patient, organ, window, seed = "s0513", "kidney", 3, 999
    acts = _actions()
    # the existing code path (the mask source bound the way the trainer binds it)
    monkeypatch.setattr(VM, "_US", toy if mask == "toy" else None)
    residual, bound = _set_trainer_base(base, servo)
    lv = T.load_volumes([patient])[0]
    Wfv, T_tgt, T_start, meta = T.make_setup(lv, np.random.default_rng(seed), organ, SETUP, window=window, count=False)
    env_ref = T.make_env(lv, Wfv, T_start, "cpu", np.random.default_rng(seed))
    poses_ref, step0 = [], env_ref.step

    def recorded_step(tw):
        out = step0(tw)
        poses_ref.append(env_ref.world_from_probe.copy())
        return out
    env_ref.step = recorded_step
    ref = T.rollout(env_ref, lv, Wfv, T_tgt, None, R.RenderConfig(seed=0), FixedPolicy(acts, bound), None, 4, 128,
                    "cpu", 0.6, 0.4, 12, deterministic=True, residual=residual)
    monkeypatch.setattr(VM, "_US", None)
    # the environment
    env = GE.ViewMatchEnv(patients=HELD, device="cpu", base=base, servo_base=servo if base else None,
                          mask=toy if mask == "toy" else "ct")
    obs, info = env.reset(seed=seed, options={"patient": patient, "target": organ, "window": window})
    assert info["window"] == meta["window"] and info["clamp_mm"] == pytest.approx(meta["clamp_mm"])
    np.testing.assert_array_equal(info["privileged"]["target_pose"], T_tgt)
    rewards, poses = [], []
    for k in range(12):
        obs, r, term, trunc, info = env.step(acts[k])
        rewards.append(r)
        poses.append(info["privileged"]["probe_pose"])
        assert trunc == (k == 11) and not term
    np.testing.assert_allclose(np.asarray(poses), np.asarray(poses_ref), rtol=0, atol=1e-12)
    np.testing.assert_allclose(rewards, ref["rewards"], rtol=0, atol=1e-12)
    assert info["match"] == pytest.approx(ref["s_final"], abs=1e-12)
    env.close()


@pytest.mark.parametrize("base", [None, "servo"])
def test_eval_episode_option_reproduces_trainer_evaluate(base):
    """options={'eval_episode': k} gives the crossed-design episode k of train_view_match_rl.evaluate:
    same cell, window, start lift, per-step pose errors and match scores."""
    servo = _servo_matrix()
    residual, bound = _set_trainer_base(base, servo)
    acts = _actions(seed=11)
    lvs = T.load_volumes(HELD)
    ev = T.evaluate(lvs, HELD, None, R.RenderConfig(seed=0), FixedPolicy(acts, bound), None, 4, 128, "cpu",
                    0.6, 0.4, SETUP, 12, 3, seed=999, residual=residual, design="crossed")
    env = GE.ViewMatchEnv(patients=HELD, device="cpu", base=base, servo_base=servo if base else None)
    for rec in ev["episodes"]:
        obs, info = env.reset(options={"eval_episode": rec["ep"]})
        assert (info["patient"], info["target_organ"], info["window"]) == (rec["patient"], rec["target"], rec["window"])
        assert info["clamp_mm"] == pytest.approx(rec["clamp_mm"], abs=1e-12)
        assert info["privileged"]["pose_error_mm"] == pytest.approx(rec["start_pos"], abs=1e-12)
        assert info["match"] == pytest.approx(rec["static_match"], abs=1e-12)
        pos, match = [info["privileged"]["pose_error_mm"]], [info["match"]]
        for k in range(12):
            _, _, _, _, info = env.step(acts[k])
            pos.append(info["privileged"]["pose_error_mm"]); match.append(info["match"])
        assert pos[-1] == pytest.approx(rec["final_pos"], abs=1e-12)
        assert info["privileged"]["rotation_error_deg"] == pytest.approx(rec["final_ang"], abs=1e-12)
        assert match[-1] == pytest.approx(rec["final_match"], abs=1e-12)
        np.testing.assert_allclose(np.round(pos, 3), rec["pos_traj"], atol=1e-9)
        np.testing.assert_allclose(np.round(match, 4), rec["match_traj"], atol=1e-9)
    env.close()


def test_servo_features_and_calibration_match_trainer():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.choice([0, 1, 2, 3, 4, 7, 12], size=(128, 128), p=[.4, .25, .05, .1, .1, .05, .05])
        b = np.roll(a, (5, -3), axis=(0, 1))
        for rich in (True, False):
            np.testing.assert_array_equal(GE.servo_features(a, b, rich), T.ibvs_features(a, b, rich))
    lvs = T.load_volumes(TRAIN)
    T.init_train_cells(lvs, TRAIN, log=lambda *_: None)
    ref = T.calibrate_ibvs(lvs, np.random.default_rng(0), None, R.RenderConfig(seed=0), 128, SETUP, kind="linear",
                           rich=True, n=8, dr_rng=np.random.default_rng([0, T._DR_SALT]))
    got = GE.calibrate_servo_base(patients=TRAIN, n=8, seed=0, device="cpu")
    np.testing.assert_allclose(got["model"], ref["model"], rtol=0, atol=1e-12)


def test_seeded_resets_are_deterministic_and_options_are_checked():
    env = GE.ViewMatchEnv(patients=TRAIN, device="cpu")
    o1, i1 = env.reset(seed=5)
    o2, i2 = env.reset(seed=5)
    np.testing.assert_array_equal(o1["current"], o2["current"])
    assert (i1["patient"], i1["target_organ"], i1["window"]) == (i2["patient"], i2["target_organ"], i2["window"])
    for _ in range(10):                                   # random draws never pick a skipped cell
        _, info = env.reset()
        assert (info["patient"], info["target_organ"]) != ("s0058", "kidney")
    _, info = env.reset(seed=0, options={"contact": False})
    assert info["contact"] is False
    with pytest.raises(ValueError):
        env.reset(options={"patient": "s0358"})           # not in this environment's patients
    with pytest.raises(ValueError):
        env.reset(options={"speed": 2})
    env.close()


@pytest.mark.slow
def test_reproduces_a_reported_ladder_cell():
    """The CT-mask servo-base rung, physics stage, calibration seed 0 (outputs/v2/table4/ladder, run on MPS):
    on the CPU the environment refits the same base and gives the same final pose error in all 60 held-out
    episodes, with contact on and off."""
    import json
    import torch
    cell = os.path.join(REPO, "outputs", "v2", "table4", "ladder", "ibvs_base_physics_ct_s0.json")
    pt = os.path.join(REPO, "outputs", "v2", "table4", "ctmask", "physics_deploy_s0.pt")
    if not (os.path.exists(cell) and os.path.exists(pt)):
        pytest.skip("v2 ladder outputs not present")
    base = GE.calibrate_servo_base(n=500, seed=0, device="cpu")
    saved = torch.load(pt, map_location="cpu", weights_only=False)["ibvs"]["model"]
    np.testing.assert_allclose(base["model"], np.asarray(saved), rtol=0, atol=1e-12)
    for r in json.load(open(cell))["results"]:
        if r["eval_randomize"] != "off":
            continue
        env = GE.ViewMatchEnv(patients="test", base="servo", servo_base=base, contact=r["contact"] == "on",
                              device="cpu")
        for rec in r["episodes"]:
            env.reset(options={"eval_episode": rec["ep"]})
            for _ in range(12):
                _, _, _, _, info = env.step(np.zeros(6, np.float32))
            assert info["privileged"]["pose_error_mm"] == pytest.approx(rec["final_pos"], abs=1e-9)
        env.close()
