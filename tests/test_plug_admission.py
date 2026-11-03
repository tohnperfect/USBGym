"""usbg.plug.admission: the adapter admission gate (v2.1 pluggability; exploratory).

The automated privileged-channel test: a component labeled pose-free must never read privileged state (true pose,
CT label slices, CT volume, simulator handles) through the gym environment or its adapters, and the test must have
power (deliberately leaky components are caught). Fast checks need no data; the anatomical checks run on TRAIN
patients on the CPU and skip without the anatomical cache and the windows file."""
from __future__ import annotations

import os

import numpy as np
import pytest

from usbg.plug import admission as AD

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = ["s0011"]


class _FakeEnv:
    n_steps = 12
    a_bound = np.ones(6, np.float32)
    _T_tgt = np.eye(4)

    @property
    def unwrapped(self):
        return self


def test_guarded_env_refuses_private_state_and_logs_it():
    log = []
    g = AD.GuardedEnv(_FakeEnv(), log)
    assert g.n_steps == 12 and g.unwrapped.n_steps == 12
    with pytest.raises(AD.PrivilegedAccess):
        g._T_tgt
    with pytest.raises(AD.PrivilegedAccess):
        g.unwrapped._T_tgt
    with pytest.raises(AD.PrivilegedAccess):
        g.pipeline
    with pytest.raises(AD.PrivilegedAccess):
        g.n_steps = 3
    assert log == ["_T_tgt", "_T_tgt", "pipeline", "set:n_steps"]


def test_strip_and_poison_privileged():
    info = {"match": 0.5, "privileged": {"pose_error_mm": 3.0, "probe_pose": np.eye(4),
                                         "ct_mask": np.zeros((2, 2), np.uint8)}}
    s = AD.strip_privileged(info)
    assert "privileged" not in s and s["match"] == 0.5 and "privileged" in info
    p = AD.poison_privileged(info, np.random.default_rng(0))
    assert p["match"] == 0.5
    assert p["privileged"]["pose_error_mm"] != 3.0
    assert not np.array_equal(p["privileged"]["probe_pose"], np.eye(4))
    assert p["privileged"]["ct_mask"].dtype == np.uint8 and p["privileged"]["ct_mask"].shape == (2, 2)
    assert info["privileged"]["pose_error_mm"] == 3.0                       # the original is untouched


def test_cuda_watch_counts_cuda_targets_only():
    import torch
    with AD.cuda_watch() as cw:
        torch.zeros(2).to("cpu")
        torch.zeros(2).to(torch.float64)
    assert cw["n"] == 0
    assert AD.cuda_status()["torch_cuda_available"] is False


def _data_ready():
    from usbg import windows as W
    return os.path.exists(W.WINDOWS_PATH) and all(
        os.path.exists(os.path.join(REPO, "data", "cache_anat", f"{c}.npz")) for c in TRAIN)


@pytest.fixture
def env(monkeypatch):
    if not _data_ready():
        pytest.skip("anatomical cache / windows file not present")
    from usbg import view_match as VM
    from usbg.plug import controllers as C
    monkeypatch.setattr(VM, "_US", None)
    e = C.make_env(patients=TRAIN, base=None, privileged_info=True, device="cpu")
    yield e
    e.close()


class _ReadsAttr:
    """A controller that pokes at the environment's true pose but acts on images only (a hidden read)."""
    privileged = False
    name = "hidden_reader"

    def reset(self, ep, env):
        self.env = env

    def act(self, obs, info):
        _ = self.env.unwrapped._probe                                     # never used, still a privileged read
        return np.zeros(6, np.float32)


@pytest.mark.anatomical
def test_deployable_controllers_pass_and_leaky_ones_are_caught(env):
    from usbg.plug import controllers as C
    ok = AD.check_controller(env, lambda: C.RandomController(0), label="random", declared_privileged=False,
                             episodes=(0,))
    assert ok["passed"] and not ok["read_detected"] and ok["max_abs_action_diff"]["poisoned"] == 0.0
    for probe in (AD.LeakyInfoProbe, AD.LeakyEnvProbe, _ReadsAttr):
        r = AD.check_controller(env, probe, label=probe.name, declared_privileged=False, episodes=(0,))
        assert r["read_detected"] and not r["passed"], probe.name
        r = AD.check_controller(env, probe, label=probe.name, declared_privileged=True, require_detect=True,
                                episodes=(0,))
        assert r["passed"], probe.name


@pytest.mark.anatomical
def test_goalnet_controller_is_pose_free(env):
    import torch
    from usbg.plug import controllers as C
    from usbg.policy import GoalPolicy
    torch.manual_seed(0)
    sd = GoalPolicy.build(history=4).state_dict()

    def fac():
        n = GoalPolicy.build(history=4)
        n.load_state_dict(sd)
        return C.GoalNetController(n, "cpu", "goalnet", bn_mode="eval")
    r = AD.check_controller(env, fac, label="goalnet", declared_privileged=False, episodes=(0,))
    assert r["passed"], r


@pytest.mark.anatomical
def test_reward_mask_and_texture_channels(env):
    pytest.importorskip("skimage")                    # the SSIM term (pip install "usbg[plug]")
    from usbg.plug import masks as PM
    from usbg.plug import rewards as RW
    views = AD.train_views(TRAIN[0], n=2)
    assert AD.check_reward(RW.RewardPlug("ssim"), views, label="ssim")["passed"]
    pose = AD.check_reward(RW.RewardPlug("pose"), views, label="pose")
    assert pose["passed"] and pose["read_detected"]

    class Thresh:                                       # non-privileged: reads the frame only
        privileged = False

        def segment(self, img01):
            return (np.asarray(img01) > 0.3).astype(np.int64)

    class CTReader:                                     # privileged: reads the CT label through the tap
        privileged = True

        def segment(self, img01):
            return np.asarray(PM.current_ct_label(img01), np.int64)

    assert AD.check_mask_source(Thresh(), views, label="thresh")["passed"]
    ct = AD.check_mask_source(CTReader(), views, label="ct_reader")
    assert ct["passed"] and ct["read_detected"]

    class Undeclared(CTReader):
        privileged = False
    bad = AD.check_mask_source(Undeclared(), views, label="undeclared")
    assert not bad["passed"]

    class Flat:                                         # label-free texture stage
        wants_label = False

        def __call__(self, img01):
            return np.clip(np.asarray(img01, np.float32) * 0.9, 0, 1)

    class LabelPainter:                                 # label-conditioned texture stage
        wants_label = True

        def translate_aligned(self, img01, lab):
            return np.clip(np.asarray(img01, np.float32) + 0.05 * (np.asarray(lab) > 0), 0, 1)
    assert AD.check_texture(Flat(), views, label="flat")["passed"]
    lp = AD.check_texture(LabelPainter(), views, label="painter", declared_label_conditioned=True)
    assert lp["passed"] and lp["read_detected"]
    assert not AD.check_texture(LabelPainter(), views, label="painter_undeclared")["passed"]


@pytest.mark.anatomical
def test_training_env_exposes_no_privileged_state(env):
    from usbg.plug import controllers as C

    def mk():
        e = C.make_env(patients=TRAIN, base=None, privileged_info=False, frame_dtype="uint8", device="cpu")
        return C.FiniteHorizonTerminal(C.StackedGoalObs(e))
    r = AD.check_training_env(mk, n_steps=2)
    assert r["passed"], r


def test_imports_without_the_resource_module():
    """usbg.plug.admission imports where the POSIX resource module is absent (Windows): peak memory is None."""
    import subprocess
    import sys
    code = ("import builtins, sys\n"
            "real = builtins.__import__\n"
            "def imp(name, *a, **k):\n"
            "    if name == 'resource':\n"
            "        raise ImportError('blocked')\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = imp\n"
            "sys.modules.pop('resource', None)\n"
            "from usbg.plug import admission as AD\n"
            "assert AD.resource is None and AD.peak_rss_mb() is None\n"
            "print('ok')\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(REPO, "src"), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]
