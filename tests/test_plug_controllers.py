"""usbg.plug.controllers (v2.1 pluggability, controller slot; exploratory).

Fast checks need no data. The anatomical checks (TRAIN patients, CPU, two episodes) need the anatomical cache and
the windows file and skip otherwise: the environment evaluator (ViewMatch-v0, reset(options={"eval_episode": k}))
gives exactly the episodes and numbers of the trainer's evaluate for the pose-oracle base, the servo base, static
and a v2 REINFORCE policy; the SB3 wrappers pass SB3's checker; the teacher label is the analytic twist."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from usbg.plug import controllers as C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = ["s0011", "s0344"]
V2PT = os.path.join(REPO, "outputs", "v2", "table4", "ctmask", "physics_deploy_s0.pt")


def test_discrete_table():
    t = C.discrete_table(1.0)
    assert t.shape == (13, 6) and not t[0].any()
    assert len({tuple(r) for r in t}) == 13
    assert np.allclose(np.abs(t[1:]).sum(1), 1.0)


def test_episode_store_rebuilds_the_env_history():
    s = C.EpisodeStore(history=4)
    frames = np.stack([np.full((2, 2), i, np.uint8) for i in range(4)])
    s.add(frames, np.zeros((2, 2), np.uint8), np.zeros((3, 6), np.float32))
    assert len(s) == 3
    assert [int(x[0, 0]) for x in s.stack(0, 0)] == [0, 0, 0, 0]
    assert [int(x[0, 0]) for x in s.stack(0, 2)] == [0, 0, 1, 2]


def _data_ready():
    from usbg import windows as W
    return os.path.exists(W.WINDOWS_PATH) and os.path.exists(V2PT) and all(
        os.path.exists(os.path.join(REPO, "data", "cache_anat", f"{c}.npz")) for c in TRAIN)


@pytest.fixture
def trainer(monkeypatch):
    if not _data_ready():
        pytest.skip("anatomical cache / windows file / v2 checkpoint not present")
    import torch
    from usbg import view_match as VM
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(VM, "_US", None)
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import train_view_match_rl as T
    saved = dict(SEAT_ORGANS=list(T.SEAT_ORGANS), _CELLS=T._CELLS, _CONTACT=T._CONTACT, _RANDOMIZE=T._RANDOMIZE,
                 _MIN_WINDOWS=T._MIN_WINDOWS, _RESIDUAL_MODE=T._RESIDUAL_MODE, _IBVS=T._IBVS)
    T.SEAT_ORGANS, T._MIN_WINDOWS, T._CONTACT, T._RANDOMIZE = ["liver", "kidney"], 2, True, "off"
    yield T
    for k, v in saved.items():
        setattr(T, k, v)


@pytest.mark.anatomical
def test_env_evaluator_equals_trainer_evaluate(trainer):
    import torch
    import eval_view_ladder as L
    from usbg import render_lotus as R
    from usbg.policy import GoalPolicy
    T = trainer
    lvs = T.load_volumes(TRAIN)
    ck = torch.load(V2PT, map_location="cpu", weights_only=False)
    setup = dict(lateral_m=0.02, axial_m=0.01, fan_deg=18.0)

    def tr_eval(pol, residual, mode):
        T._RESIDUAL_MODE = mode
        T._IBVS = {"kind": "linear", "model": np.asarray(ck["ibvs"]["model"]), "rich": ck["ibvs"]["rich"]} \
            if mode == "image" else None
        with torch.no_grad():
            return T.evaluate(lvs, TRAIN, None, R.RenderConfig(seed=0), pol, None, 4, 128, "cpu", 0.6, 0.4, setup, 12,
                              2, seed=999, residual=residual, design="crossed")

    def env_eval(base, ctrl):
        env = C.make_env(patients=TRAIN, base=base, servo_base=V2PT if base == "servo" else None, render_seed=0,
                         device="cpu")
        try:
            return C.evaluate(env, ctrl, 2)
        finally:
            env.close()

    net = GoalPolicy.build(history=4)
    net.load_state_dict(ck["state_dict"])
    pol, _ = L.load_policy(V2PT, 0.5, True, "cpu")
    cases = [(tr_eval(L.ZeroPolicy(), True, "oracle"), env_eval("pose_oracle", C.ZeroController())),
             (tr_eval(L.ZeroPolicy(), True, "image"), env_eval("servo", C.ZeroController())),
             (tr_eval(L.ZeroPolicy(), False, "oracle"), env_eval(None, C.ZeroController())),
             (tr_eval(pol, True, "image"), env_eval("servo", C.GoalNetController(net, "cpu", "v2", "train")))]
    for a, b in cases:
        for ea, eb in zip(a["episodes"], b["episodes"]):
            assert (ea["patient"], ea["target"], ea["window"]) == (eb["patient"], eb["target"], eb["window"])
            for k in ("start_pos", "final_pos", "final_ang", "final_match", "keepbest_pos", "peak_force"):
                assert ea[k] == eb[k], k


@pytest.mark.anatomical
def test_sb3_wrappers_and_teacher(trainer):
    pytest.importorskip("stable_baselines3")          # the controller slot's own package (README, Install, Notes)
    from stable_baselines3.common.env_checker import check_env
    from usbg import goal_rl as G
    env = C.make_env(patients=TRAIN, base=None, frame_dtype="uint8", privileged_info=True, device="cpu")
    check_env(C.FiniteHorizonTerminal(C.StackedGoalObs(env)), warn=True)
    obs, info = env.reset(seed=3)
    y = C.teacher_action(env)
    assert np.allclose(y, info["privileged"]["analytic_twist"] / G.MAX_TWIST, atol=1e-6)
    env.close()
    env = C.make_env(patients=TRAIN, base="pose_oracle", privileged_info=True, device="cpu")
    env.reset(seed=3)
    assert np.allclose(C.teacher_action(env), 0.0)
    env.close()
    d = C.DiscreteTwist(C.make_env(patients=TRAIN, base=None, frame_dtype="uint8", privileged_info=False, device="cpu"))
    check_env(C.FiniteHorizonTerminal(C.StackedGoalObs(d)), warn=True)
    d.close()
