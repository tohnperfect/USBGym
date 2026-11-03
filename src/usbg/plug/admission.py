"""usbg.plug.admission: the adapter admission gate of the v2.1 pluggability study (EXPLORATORY, opt-in).

Nothing in the v2 or v2.1 queues imports this module. It is reached only through
``scripts/plug/admission_gate.py`` (one queue job per adapter, minutes each) and ``tests/test_plug_admission.py``.
The protocol is fixed before any run: no number
of an adapter is reported before its gate output exists, and each plug job behind a reported number waits
for the gate job of its adapter.

Three checks per adapter
  (i)   Privileged channel. A pose-free component runs once with every privileged channel it could reach taken
        away and once with each such channel filled with wrong values; both runs must reproduce its outputs bit
        for bit, and it must read none of them:
          controllers   the info["privileged"] block (the true pose error, the poses of probe and target, the
                        analytic twist and the CT label slices) is dropped in one run and filled with random
                        values in another; reset() receives a proxy of the environment, which refuses each
                        private attribute and logs the attempt (the render state; ``_probe`` and ``_T_tgt``,
                        the true poses; ``_lv``, the CT volume; ``_ct`` and ``_tgt_ct``, the CT label slices;
                        ``_Wfv``, the patient pose), and the render pipeline;
          mask sources  in one run a poisoned label, in another an exception, from the CT-label tap of the
                        process (current_ct_label of usbg.plug.masks);
          rewards       random poses for the probe and the target in one run, None in another;
          textures      the stage is offered a permuted label slice (view_match.curvi_render, its cond_label
                        hook); a stage with wants_label = True is reported as
                        label-conditioned, so the label-following Dice probe applies to it.
        A component labeled privileged must be caught reading a privileged channel (this checks the power
        of the test itself); it is admitted with the privileged mark. Deliberately leaky probes are run first
        in every controller gate and must be caught, otherwise the gate fails.
  (ii)  Pretraining corpus. The declared pretraining corpus of every pretrained weight file the adapter loads
        (PRETRAINING below), its overlap status with the public real sets (Vitale, AUL, KidneyUS) and with the
        held-out and cohort CT ids, and whether its real-frame Dice may be reported. An unverified corpus does
        not block simulated results; it withholds real-frame Dice until the check is recorded.
  (iii) No CUDA; the same outputs on each device. torch.cuda.is_available() has to return False, and the
        adapter may not try to move a tensor to a CUDA device; on identical inputs its CPU and MPS outputs
        agree within the declared tolerances (TOL); one TRAIN episode runs on each device and records the
        wall time, the peak RSS and the latency per step. An adapter without MPS support (SIMUS) is marked
        CPU-only, which also uses no CUDA.

Only TRAIN patients are touched (default s0011). Every output carries analysis_status
"exploratory, v2.1 pluggability".
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import sys
import time

try:                                     # POSIX only; on Windows the peak memory is not recorded
    import resource
except ImportError:                      # pragma: no cover  (Windows)
    resource = None

import numpy as np

ANALYSIS_STATUS = "exploratory, v2.1 pluggability"
GATE_VERSION = "usbg-plug-admission-1"
GATE_CASE = "s0011"            # a TRAIN patient (usbg.benchmark.TRAIN_PATIENTS)

# Declared tolerances (the pre-specified protocol). Fixed before any gate ran.
TOL = {
    "privileged_max_abs_diff": 0.0,            # non-privileged outputs must be bit-identical (i)
    "action_cpu_mps_max_abs": 1e-3,            # normalized action units, identical observations (iii)
    "mask_cpu_mps_min_agreement": 0.99,        # pixel agreement on identical frames (iii)
    "frame_cpu_mps_max_abs": 0.02,             # [0, 1] frames, identical inputs (iii)
    "frame_cpu_mps_mean_abs": 1e-3,
    "frame_stat_abs": 0.02,                    # per-frame mean and SD, device-RNG adapters only (lotus_reimpl)
    "episode_final_pos_mm": 1.0,               # one TRAIN episode on each device (CPU against MPS, per episode)
    "episode_match_max_abs": 0.05,             # max over steps of |S_cpu - S_mps|
}

# Environment attributes a non-privileged controller may read. Everything private (leading underscore) and the
# render pipeline are refused: they give the true poses, the CT, CT label slices or a simulator handle.
DENY_PUBLIC = frozenset({"pipeline"})

# (ii) Declared pretraining corpora. "trained here" artifacts are trained by the plug jobs on TRAIN data only.
PRETRAINING = {
    "sb3": {"corpus": "none: random initialization, trained here on the 13 TRAIN patients", "status": "documented",
            "real_frame_dice_reportable": None},
    "goalnet": {"corpus": "none: random initialization (BC, DAgger) or the paper's v2 policies (TRAIN patients)",
                "status": "documented", "real_frame_dice_reportable": None},
    "reference": {"corpus": "none (no learned weights; the servo base is calibrated on TRAIN pairs)",
                  "status": "documented", "real_frame_dice_reportable": None},
    "tv": {"corpus": "none: torchvision architectures trained from scratch (backbone_weights 'none'; the ImageNet "
                     "and COCO weights are not downloaded) on the v2 segmenter data: TRAIN-patient renders, the "
                     "Vitale TRAIN split and the KidneyUS training partial labels",
           "status": "documented", "real_frame_dice_reportable": True},
    "nnunet": {"corpus": "none: nnU-Net v2 trained from scratch on the same v2 segmenter data (fold 0 = the v2 "
                         "train/val split)", "status": "documented", "real_frame_dice_reportable": True},
    "sam": {"corpus": "SA-1B: about 11M licensed natural photographs with 1.1B automatic masks (Kirillov et al., "
                      "ICCV 2023); no medical images are described in the dataset paper",
            "overlap": {"vitale_aul_kidneyus": "not expected (natural photographs only)",
                        "heldout_and_cohort_ct": "not expected (no CT)"},
            "status": "documented from the dataset description", "real_frame_dice_reportable": True},
    "medsam": {"corpus": "SAM ViT-B fine-tuned on roughly 1.57M image-mask pairs from public medical sets, 10 "
                         "modalities, ultrasound and CT among them (Ma et al., Nat. Commun. 2024, Supplementary Table 1)",
               "overlap": {"vitale_aul_kidneyus": "UNVERIFIED: the per-dataset list must be checked against "
                                                  "Vitale, AUL and KidneyUS",
                           "heldout_and_cohort_ct": "UNVERIFIED: the public CT sets of the list are not yet "
                                                    "compared with the TotalSegmentator cases that serve as "
                                                    "held-out or cohort CTs"},
               "status": "unverified (training-data overlap not checked)", "real_frame_dice_reportable": False,
               "action": "withhold MedSAM real-frame Dice until the overlap check is recorded; simulated rows "
                         "are not affected (the frames are rendered from TotalSegmentator labels, not CT)"},
    "texture": {"corpus": "none: trained here on the v2 pairs (8 TRAIN patients) and the public Vitale TRAIN pool "
                          "(histogram matching: fitted on the same frames)", "status": "documented",
                "real_frame_dice_reportable": None},
    "reward": {"corpus": "none (closed-form image similarity terms; LPIPS, whose AlexNet trunk is ImageNet-trained, "
                         "is excluded from this launch because its weights are not downloaded)",
               "status": "documented", "real_frame_dice_reportable": None},
    "renderer": {"corpus": "none (LOTUS forward model with its published default parameters; SIMUS is a physical "
                           "simulator; no weights trained on our data)", "status": "documented",
                 "real_frame_dice_reportable": None},
}


# ===================================================================================== guards
class PrivilegedAccess(RuntimeError):
    """Raised when a component labeled non-privileged reaches a privileged channel during the gate."""


class GuardedEnv:
    """A proxy around a ViewMatchEnv that refuses (and logs) every private attribute and the render pipeline.
    ``unwrapped`` returns a guarded proxy of the unwrapped environment; public settings pass through."""

    def __init__(self, env, log):
        object.__setattr__(self, "_g_env", env)
        object.__setattr__(self, "_g_log", log)

    def __getattr__(self, name):
        env = object.__getattribute__(self, "_g_env")
        log = object.__getattribute__(self, "_g_log")
        if name == "unwrapped":
            return GuardedEnv(env.unwrapped, log)
        if (name.startswith("_") and not name.startswith("__")) or name in DENY_PUBLIC:
            log.append(name)
            raise PrivilegedAccess(f"non-privileged component read the environment's {name!r}")
        return getattr(env, name)

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_g_log").append("set:" + name)
        raise PrivilegedAccess(f"non-privileged component wrote the environment's {name!r}")


def strip_privileged(info):
    return {k: v for k, v in dict(info).items() if k != "privileged"}


def _poison_tree(x, rng):
    if isinstance(x, dict):
        return {k: _poison_tree(v, rng) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.integer):
            return rng.integers(0, 13, size=x.shape).astype(x.dtype)
        return (rng.normal(size=x.shape) * 1e3).astype(x.dtype)
    if isinstance(x, (float, np.floating)):
        return float(rng.normal() * 1e3)
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return int(rng.integers(0, 1000))
    return x


def poison_privileged(info, rng):
    out = dict(info)
    if "privileged" in out:
        out["privileged"] = _poison_tree(out["privileged"], rng)
    return out


def _maxdiff(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    d = np.abs(a - b)
    if not np.all(np.isfinite(d)):
        return float("inf")
    return float(d.max()) if d.size else 0.0


@contextlib.contextmanager
def _patched(obj, attr, value):
    saved = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, saved)


# ===================================================================================== (iii) CUDA watch
@contextlib.contextmanager
def cuda_watch():
    """Count every Tensor.to / Tensor.cuda / Module.to call whose target is a CUDA device while the body runs.
    The calls are passed through unchanged (on a machine without CUDA they would fail anyway)."""
    import torch
    seen = {"n": 0, "where": []}

    def _is_cuda(args, kw):
        for a in list(args) + list(kw.values()):
            if isinstance(a, torch.device) and a.type == "cuda":
                return True
            if isinstance(a, str) and a.startswith("cuda"):
                return True
        return False

    t_to, t_cuda, m_to = torch.Tensor.to, torch.Tensor.cuda, torch.nn.Module.to

    def to(self, *a, **kw):
        if _is_cuda(a, kw):
            seen["n"] += 1
            seen["where"].append("Tensor.to")
        return t_to(self, *a, **kw)

    def cuda(self, *a, **kw):
        seen["n"] += 1
        seen["where"].append("Tensor.cuda")
        return t_cuda(self, *a, **kw)

    def mto(self, *a, **kw):
        if _is_cuda(a, kw):
            seen["n"] += 1
            seen["where"].append("Module.to")
        return m_to(self, *a, **kw)

    torch.Tensor.to, torch.Tensor.cuda, torch.nn.Module.to = to, cuda, mto
    try:
        yield seen
    finally:
        torch.Tensor.to, torch.Tensor.cuda, torch.nn.Module.to = t_to, t_cuda, m_to


def cuda_status():
    import torch
    return {"torch_cuda_available": bool(torch.cuda.is_available()),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mps_available": bool(torch.backends.mps.is_available()), "torch": torch.__version__}


def peak_rss_mb():
    """Peak resident memory of this process in MB, or None where the resource module is absent (Windows)."""
    if resource is None:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS, KiB on Linux
    return round(r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0, 1)


def mps_ok():
    import torch
    return bool(torch.backends.mps.is_available())


def sync(device):
    import torch
    if str(device).startswith("mps"):
        torch.mps.synchronize()


# ===================================================================================== TRAIN views
def train_views(case=GATE_CASE, organ="liver", n=4, seed=0, translator=None, render_seed=0, image_size=128):
    """n (frame, CT label, probe pose, target pose) tuples on one TRAIN patient: the target view and a perturbed
    start view per draw (goal_rl.sample_setup with the paper's start distribution), rendered by
    view_match.curvi_render with the CT-label output (so a CT-label tap, if installed, sees each frame)."""
    from .. import goal_rl as G
    from .. import render_lotus as R
    from .. import view_match as VM
    from .viewmatch_env import ORGANS, load_patient, _ct_label_output
    from ..benchmark import TRAIN_PATIENTS
    if case not in TRAIN_PATIENTS:
        raise ValueError(f"the admission gate runs on TRAIN patients only, got {case}")
    lv = load_patient(case)
    rng = np.random.default_rng(seed)
    cfg = R.RenderConfig().with_seed(render_seed)
    out = []
    for i in range(int(n)):
        Wfv, T_tgt, T_cur = G.sample_setup(lv, ORGANS[organ], rng, lateral_m=0.02, axial_m=0.01, fan_deg=18.0)
        for T in (T_cur, T_tgt):
            with _ct_label_output():
                img, lab = VM.curvi_render(lv, T, Wfv, translator, cfg, image_size)
            out.append({"img": img, "lab": np.asarray(lab, np.int64), "T": np.asarray(T), "T_tgt": np.asarray(T_tgt),
                        "Wfv": Wfv, "lv": lv, "cfg": cfg, "case": case, "organ": organ})
            if len(out) >= n:
                return out
    return out


# ===================================================================================== (i) controllers
def check_controller(env, factory, *, label, declared_privileged, episodes=(0, 1), seed=12345,
                     require_detect=False):
    """Run ``factory()`` controllers on the environment's crossed episodes ``episodes`` three times in lockstep:
    reference (full info, the real environment), stripped (info["privileged"] removed, guarded environment) and
    poisoned (info["privileged"] randomized, guarded environment). The environment always steps with the
    reference action. -> dict with the denied reads, the largest action differences and the verdict.
    require_detect (detector-power probes and privileged teachers): the check passes only if a read is
    detected."""
    rng = np.random.default_rng(seed)
    u = env.unwrapped
    reads, errors = [], []
    diff = {"stripped": 0.0, "poisoned": 0.0}
    n_steps = 0
    for ep in episodes:
        obs, info = env.reset(options={"eval_episode": int(ep)})
        ref = factory()
        ref.reset(int(ep), env)
        variants = []
        for v in ("stripped", "poisoned"):
            log = []
            c = factory()
            try:
                c.reset(int(ep), GuardedEnv(env, log))
                variants.append([v, c, log, True])
            except PrivilegedAccess as e:
                reads.append(f"{v}: reset: {e}")
                variants.append([v, c, log, False])
        for t in range(u.n_steps):
            a_ref = np.asarray(ref.act(copy.deepcopy(obs), info), np.float32).reshape(-1)
            for rec in variants:
                v, c, log, alive = rec
                if not alive:
                    continue
                inf = strip_privileged(info) if v == "stripped" else poison_privileged(info, rng)
                try:
                    a = np.asarray(c.act(copy.deepcopy(obs), inf), np.float32).reshape(-1)
                    diff[v] = max(diff[v], _maxdiff(a, a_ref))
                except (PrivilegedAccess, KeyError, TypeError, ValueError) as e:
                    reads.append(f"{v}: step {t}: {type(e).__name__}: {e}")
                    rec[3] = False
                if log:
                    reads += [f"{v}: env.{x}" for x in log]
                    log.clear()
            obs, _r, _term, _trunc, info = env.step(a_ref)
            n_steps += 1
    read_detected = bool(reads) or any(d > TOL["privileged_max_abs_diff"] for d in diff.values())
    if require_detect:
        verdict = ("detected as reading privileged state (test power)" if read_detected else
                   "FAIL: a privileged reader was not detected (the test has no power)")
        passed = read_detected
    elif declared_privileged:
        verdict = "privileged (read detected)" if read_detected else "privileged (declared; no read detected)"
        passed = True
    else:
        verdict = "non-privileged (no privileged read, bit-identical)" if not read_detected else "FAIL: reads privileged state"
        passed = not read_detected
    return {"component": label, "declared_privileged": bool(declared_privileged or require_detect),
            "require_detect": bool(require_detect), "episodes": [int(e) for e in episodes],
            "steps": n_steps, "max_abs_action_diff": {k: (None if not np.isfinite(v) else v) for k, v in diff.items()},
            "privileged_reads": sorted(set(reads))[:20], "n_privileged_reads": len(reads),
            "read_detected": read_detected, "verdict": verdict, "passed": passed}


class LeakyInfoProbe:
    """Detector-power probe: acts on the analytic twist in info['privileged'] (must be caught)."""
    privileged = True
    name = "leaky_info_probe"

    def reset(self, ep, env):
        self.u = env.unwrapped
        self.bound = np.asarray(self.u.a_bound, np.float64)

    def act(self, obs, info):
        return np.clip(np.asarray(info["privileged"]["analytic_twist"], np.float64) / self.bound, -1, 1)


class LeakyEnvProbe:
    """Detector-power probe: reads the target pose from the environment (must be caught)."""
    privileged = True
    name = "leaky_env_probe"

    def reset(self, ep, env):
        self.env = env

    def act(self, obs, info):
        from .. import goal_rl as G
        u = self.env.unwrapped
        tw = G.analytic_twist(u._probe.world_from_probe, u._T_tgt)
        return np.clip(tw / np.asarray(u.a_bound, np.float64), -1, 1)


def check_training_env(make_train_env, n_steps=3):
    """The SB3 training environment (built as scripts/plug_train_sb3.py builds it, privileged_info=False) may
    not put a privileged key into info, and its observations are images only."""
    env = make_train_env()
    obs, info = env.reset(seed=0)
    keys = set(info)
    space = env.observation_space
    for _ in range(int(n_steps)):
        obs, _r, term, trunc, info = env.step(env.action_space.sample())
        keys |= set(info)
        if term or trunc:
            break
    env.close()
    shape = getattr(space, "shape", None)
    ok = "privileged" not in keys and getattr(space, "dtype", None) == np.uint8 and shape is not None and len(shape) == 3
    return {"info_keys": sorted(keys), "observation_space": str(space), "passed": bool(ok),
            "verdict": "training observations are images only; no privileged info key" if ok else
            "FAIL: the training environment exposes privileged state"}


# ===================================================================================== (i) mask sources
def check_mask_source(src, views, *, label):
    """A non-privileged mask source must not call the CT-label tap and must give identical masks when the tap
    returns a poisoned label; a CT-prompted (privileged) source must be detected."""
    from . import masks as PM
    from .. import view_match as VM
    from .viewmatch_env import _ct_label_output
    PM.install_ct_label_tap()
    rng = np.random.default_rng(7)
    calls = {"n": 0}
    diff_px, n_px, reads = 0, 0, []
    for v in views:
        lab = v["lab"]
        # a fresh render immediately before segmenting, so the tap's identity check sees this exact frame
        with _ct_label_output():
            img, _lab = VM.curvi_render(v["lv"], v["T"], v["Wfv"], None, v["cfg"], img_size(v))
        ref = np.asarray(src.segment(img), np.int64)
        perm = rng.permutation(13)

        def poisoned(img01=None, _lab=lab, _perm=perm):
            calls["n"] += 1
            return _perm[np.asarray(_lab, np.int64)]

        def refuse(img01=None):
            calls["n"] += 1
            raise PrivilegedAccess("mask source read the CT label of the frame (CT-label tap)")

        with _patched(PM, "current_ct_label", poisoned):
            p = np.asarray(src.segment(img), np.int64)
        diff_px += int((p != ref).sum())
        n_px += int(ref.size)
        try:
            with _patched(PM, "current_ct_label", refuse):
                src.segment(img)
        except PrivilegedAccess as e:
            reads.append(str(e))
    read_detected = calls["n"] > 0 or diff_px > 0 or bool(reads)
    declared = bool(getattr(src, "privileged", False))
    if declared:                                   # a CT-prompted source must be caught (power of the test)
        verdict = ("privileged (CT-label read detected)" if read_detected else
                   "FAIL: declared privileged but no CT-label read detected (the test has no power)")
        passed = read_detected
    else:
        verdict = "non-privileged (no CT-label read, identical masks)" if not read_detected else "FAIL: reads the CT label"
        passed = not read_detected
    return {"component": label, "declared_privileged": declared, "frames": len(views), "tap_calls": calls["n"],
            "pixels_changed_under_poison": diff_px, "pixels": n_px, "reads": sorted(set(reads))[:5],
            "read_detected": read_detected, "verdict": verdict, "passed": passed}


def img_size(v):
    return int(np.asarray(v["img"]).shape[0])


def mask_parity(make_src, views, *, label):
    """Masks on identical frames from the adapter built on CPU and on MPS: pixel agreement and per-frame cost."""
    out = {"component": label}
    if not mps_ok():
        return dict(out, skipped="MPS not available", passed=True)
    res = {}
    for dev in ("cpu", "mps"):
        src = make_src(dev)
        ms, ts = [], []
        for v in views:
            t0 = time.perf_counter()
            ms.append(np.asarray(src.segment(np.asarray(v["img"], np.float32)), np.int64))
            sync(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        res[dev] = (ms, ts)
    agree = float(np.mean([np.mean(a == b) for a, b in zip(res["cpu"][0], res["mps"][0])]))
    ok = agree >= TOL["mask_cpu_mps_min_agreement"]
    return dict(out, pixel_agreement=round(agree, 6), tolerance=TOL["mask_cpu_mps_min_agreement"],
                ms_per_frame={d: round(float(np.median(res[d][1])), 2) for d in res}, passed=bool(ok))


# ===================================================================================== (i) rewards
def check_reward(plug, views, *, label):
    """A non-privileged reward potential must not depend on the probe or target pose; a privileged one must."""
    rng = np.random.default_rng(11)
    diffs, reads = [], []
    pairs = [(views[i], views[i + 1]) for i in range(0, len(views) - 1, 2)]
    for cur, tgt in pairs:
        args = (cur["img"], cur["lab"], tgt["img"], tgt["lab"])
        ref = plug.potential(*args, pose=cur["T"], tgt_pose=tgt["T"])
        bad = np.eye(4)
        bad[:3, 3] = rng.normal(size=3)
        try:
            p = plug.potential(*args, pose=bad, tgt_pose=np.eye(4))
            diffs.append(abs(float(p) - float(ref)))
        except Exception as e:                          # noqa: BLE001
            reads.append(f"poisoned pose: {type(e).__name__}")
        try:
            n = plug.potential(*args, pose=None, tgt_pose=None)
            diffs.append(abs(float(n) - float(ref)))
        except ValueError as e:
            reads.append(f"pose None: {e}")
    read_detected = bool(reads) or (max(diffs) if diffs else 0.0) > 0.0
    declared = bool(plug.privileged)
    if declared:                                   # a pose reward must be caught (power of the test)
        verdict = ("privileged (pose read detected)" if read_detected else
                   "FAIL: declared privileged but no pose read detected (the test has no power)")
        passed = read_detected
    else:
        verdict = "non-privileged (pose-free potential, identical values)" if not read_detected else "FAIL: reads the pose"
        passed = not read_detected
    return {"component": label, "declared_privileged": declared, "pairs": len(pairs),
            "max_abs_potential_diff": float(max(diffs)) if diffs else None, "reads": sorted(set(reads))[:5],
            "read_detected": read_detected, "verdict": verdict, "passed": passed,
            "device_parity": "n/a: NumPy on the CPU (device-independent)"}


# ===================================================================================== (i) textures
def check_texture(translator, views, *, label, declared_label_conditioned=False):
    """A label-free stage must give identical frames when the label offered to it is permuted (curvi_render's
    cond_label hook) and must never be asked for it; a wants_label stage is reported as label-conditioned."""
    from .. import view_match as VM
    from .viewmatch_env import _ct_label_output
    rng = np.random.default_rng(13)
    calls = {"n": 0}
    md = 0.0
    for v in views:
        perm = rng.permutation(13)

        def cond(lv, T, Wfv, lab_lin, _perm=perm):
            calls["n"] += 1
            return _perm[np.asarray(lab_lin, np.int64)]

        with _ct_label_output():
            a, _ = VM.curvi_render(v["lv"], v["T"], v["Wfv"], translator, v["cfg"], img_size(v))
            b, _ = VM.curvi_render(v["lv"], v["T"], v["Wfv"], translator, v["cfg"], img_size(v), cond_label=cond)
        md = max(md, _maxdiff(a, b))
    wants = bool(getattr(translator, "wants_label", False))
    read_detected = wants or calls["n"] > 0 or md > 0.0
    if declared_label_conditioned:                 # a wants_label stage must be caught (power of the test)
        verdict = ("label-conditioned (the label-following probe applies)" if read_detected else
                   "FAIL: declared label-conditioned but no label read detected (the test has no power)")
        passed = read_detected
    else:
        verdict = ("non-privileged (label-free: identical frames under a permuted label)" if not read_detected else
                   "FAIL: the stage reads the label slice")
        passed = not read_detected
    return {"component": label, "wants_label": wants, "declared_label_conditioned": bool(declared_label_conditioned),
            "frames": len(views), "label_calls": calls["n"], "max_abs_frame_diff": md,
            "read_detected": read_detected, "verdict": verdict, "passed": passed}


def frame_parity(make_fn, inputs, *, label, device_rng=False):
    """make_fn(device) -> callable(input) -> frame; identical inputs on CPU and MPS. Pixel parity (TOL
    frame_cpu_mps_*). An adapter declared to draw its noise from the DEVICE random generator (device_rng; the
    in-repo LOTUS reimplementation seeds torch and draws speckle on the render device, so CPU and MPS speckle
    fields differ by construction) is held to statistical parity instead: per-frame mean and SD of the
    intensities within TOL frame_stat_abs."""
    out = {"component": label}
    if not mps_ok():
        return dict(out, skipped="MPS not available", passed=True)
    res = {}
    for dev in ("cpu", "mps"):
        f = make_fn(dev)
        fr, ts = [], []
        for x in inputs:
            t0 = time.perf_counter()
            fr.append(np.asarray(f(x), np.float32))
            sync(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        res[dev] = (fr, ts)
    d = np.stack([np.abs(a - b) for a, b in zip(res["cpu"][0], res["mps"][0])])
    mx, mn = float(d.max()), float(d.mean())
    dm = max(abs(float(a.mean()) - float(b.mean())) for a, b in zip(res["cpu"][0], res["mps"][0]))
    ds = max(abs(float(a.std()) - float(b.std())) for a, b in zip(res["cpu"][0], res["mps"][0]))
    if device_rng:
        mode = "statistical (noise drawn from the device random generator)"
        ok = dm <= TOL["frame_stat_abs"] and ds <= TOL["frame_stat_abs"]
    else:
        mode = "pixel"
        ok = mx <= TOL["frame_cpu_mps_max_abs"] and mn <= TOL["frame_cpu_mps_mean_abs"]
    return dict(out, parity_mode=mode, max_abs=mx, mean_abs=mn, frame_mean_abs_diff=dm, frame_sd_abs_diff=ds,
                tolerance={"max_abs": TOL["frame_cpu_mps_max_abs"], "mean_abs": TOL["frame_cpu_mps_mean_abs"],
                           "frame_stat_abs": TOL["frame_stat_abs"]},
                ms_per_frame={k: round(float(np.median(v[1])), 3) for k, v in res.items()}, passed=bool(ok))


def action_parity(make_ctrl, observations, *, label, discrete=False):
    """make_ctrl(device) -> controller; identical observations on CPU and MPS."""
    out = {"component": label}
    if not mps_ok():
        return dict(out, skipped="MPS not available", passed=True)
    acts, ts = {}, {}
    for dev in ("cpu", "mps"):
        c = make_ctrl(dev)
        c.reset(0, None)
        a, t = [], []
        for o in observations:
            t0 = time.perf_counter()
            a.append(np.asarray(c.act(copy.deepcopy(o), {}), np.float32).reshape(-1))
            sync(dev)
            t.append((time.perf_counter() - t0) * 1e3)
        acts[dev], ts[dev] = np.stack(a), t
    mx = _maxdiff(acts["cpu"], acts["mps"])
    ok = mx <= (0.0 if discrete else TOL["action_cpu_mps_max_abs"])
    return dict(out, max_abs_action_diff=mx, discrete=bool(discrete),
                tolerance=0.0 if discrete else TOL["action_cpu_mps_max_abs"],
                ms_per_act={k: round(float(np.median(v)), 3) for k, v in ts.items()}, passed=bool(ok))


# ===================================================================================== (iii) episodes
def run_episode(make_env, make_ctrl, device, episode=0):
    """One TRAIN episode (crossed design, eval_episode k) on ``device``: final pose error, S trajectory,
    the wall time, the peak RSS and the latency per step. CUDA moves are counted."""
    with cuda_watch() as cw:
        t0 = time.perf_counter()
        env = make_env(device)
        ctrl = make_ctrl(device)
        t_build = time.perf_counter() - t0
        obs, info = env.reset(options={"eval_episode": int(episode)})
        ctrl.reset(int(episode), env)
        S, steps = [float(info["match"])], []
        for _ in range(env.unwrapped.n_steps):
            ts = time.perf_counter()
            a = np.asarray(ctrl.act(obs, info), np.float32).reshape(6)
            obs, _r, _te, _tr, info = env.step(a)
            sync(device)
            steps.append((time.perf_counter() - ts) * 1e3)
            S.append(float(info["match"]))
        P = info.get("privileged") or {}
        wall = time.perf_counter() - t0
        env.close()
    return {"device": device, "episode": int(episode), "final_pos_mm": float(P.get("pose_error_mm", np.nan)),
            "final_ang_deg": float(P.get("rotation_error_deg", np.nan)), "S": [round(s, 5) for s in S],
            "wall_s": round(wall, 2), "build_s": round(t_build, 2),
            "step_ms": {"median": round(float(np.median(steps)), 2), "max": round(float(np.max(steps)), 2)},
            "peak_rss_mb_cumulative": peak_rss_mb(), "cuda_moves": int(cw["n"])}


def episode_parity(make_env, make_ctrl, *, label, episode=0, devices=("cpu", "mps")):
    devs = [d for d in devices if d != "mps" or mps_ok()]
    eps = {d: run_episode(make_env, make_ctrl, d, episode) for d in devs}
    out = {"component": label, "episodes": eps, "cuda_moves": int(sum(e["cuda_moves"] for e in eps.values()))}
    if len(devs) < 2:
        return dict(out, cpu_only=True, passed=out["cuda_moves"] == 0)
    a, b = eps["cpu"], eps["mps"]
    dp = abs(a["final_pos_mm"] - b["final_pos_mm"])
    ds = _maxdiff(a["S"], b["S"])
    ok = dp <= TOL["episode_final_pos_mm"] and ds <= TOL["episode_match_max_abs"] and out["cuda_moves"] == 0
    return dict(out, final_pos_diff_mm=dp, match_max_abs_diff=ds,
                tolerance={"final_pos_mm": TOL["episode_final_pos_mm"], "match": TOL["episode_match_max_abs"]},
                passed=bool(ok))


# ===================================================================================== report
def summarize(adapter, slot, components, parity, extra=None):
    """The gate's result JSON body: admitted iff every component check and every parity check passed and CUDA
    is absent."""
    cs = cuda_status()
    comp_ok = all(c.get("passed") for c in components)
    par_ok = all(p.get("passed") for p in parity)
    cuda_ok = not cs["torch_cuda_available"] and all(int(p.get("cuda_moves", 0)) == 0 for p in parity)
    labels = {"deployable": sorted({c["component"] for c in components
                                    if c.get("passed") and not c.get("declared_privileged")
                                    and not c.get("declared_label_conditioned")}),
              "privileged": sorted({c["component"] for c in components if c.get("declared_privileged")}),
              "label_conditioned": sorted({c["component"] for c in components if c.get("declared_label_conditioned")})}
    corpus_key = {"ctrl:sb3": "sb3", "ctrl:goalnet": "goalnet", "ctrl:reference": "reference", "mask:tv": "tv",
                  "mask:nnunet": "nnunet", "mask:sam": "sam", "mask:medsam": "medsam",
                  "reward:terms": "reward"}.get(adapter) or ("texture" if slot == "texture" else
                                                             "renderer" if slot == "renderer" else None)
    return {"gate": GATE_VERSION, "adapter": adapter, "slot": slot, "analysis_status": ANALYSIS_STATUS,
            "admitted": bool(comp_ok and par_ok and cuda_ok),
            "checks": {"privileged_channel": {"passed": comp_ok, "components": components},
                       "pretraining_corpus": dict(PRETRAINING.get(corpus_key, {}), key=corpus_key),
                       "device_parity": {"passed": par_ok, "results": parity, "tolerances": TOL},
                       "cuda_free": dict(cs, passed=cuda_ok)},
            "labels": labels, "gate_case": GATE_CASE, **(extra or {})}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_result(out, body):
    """Write the gate result. Admitted: ``out`` (the runner's DONE marker). Not admitted:
    ``<out minus .json>.failed.json`` only, so a later runner restart never SKIPs a failed gate."""
    from .. import provenance as PROV
    body = dict(body)
    prov = PROV.collect()
    prov["analysis_status"] = ANALYSIS_STATUS
    prov["gate"] = GATE_VERSION
    body["provenance"] = prov
    path = out if body.get("admitted") else (out[:-5] if out.endswith(".json") else out) + ".failed.json"
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(body, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)
    return path


__all__ = ["ANALYSIS_STATUS", "GATE_VERSION", "TOL", "PRETRAINING", "PrivilegedAccess", "GuardedEnv",
           "strip_privileged", "poison_privileged", "check_controller", "LeakyInfoProbe", "LeakyEnvProbe",
           "check_training_env", "check_mask_source", "mask_parity", "check_reward", "check_texture", "frame_parity",
           "action_parity", "run_episode", "episode_parity", "cuda_watch", "cuda_status", "train_views", "summarize",
           "write_result"]
