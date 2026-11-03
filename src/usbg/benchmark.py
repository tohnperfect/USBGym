"""benchmark.py: the USB Gym benchmark protocol.

This module fixes the evaluation protocol so results line up across methods: a frozen
patient-disjoint split, a fixed seed set, and standardized task evaluators. An entry implements
one pluggable interface (a policy, a reward, or a translator) and is scored by these functions on the
HELD-OUT patients only; the numbers sit on the same episodes as the bundled reference baselines.

Two reference tasks (following the paper):
  * GOAL (goal-conditioned reaching): reach a shot close to a target B-mode image of a held-out
    patient. Metric: reach success rate and final embedding similarity, averaged over held-out patients
    and seeds. The pose/angle error is a diagnostic only (the objective is pose-free). Implemented
    here as goal_task() and scored by scripts/run_benchmark.py.
  * VIEW-MATCH (curvilinear view-matching): servo the probe to reproduce a target curvilinear view
    under the segmentation-anchored reward, on this same frozen split. Scored by
    scripts/train_view_match_rl.py (it is not re-implemented in this module).

The target-organ visibility task is a supporting demonstration (usbg.rl.evaluate_success /
usbg.eval.compare_policies), not one of the two reference tasks.

The split is deliberately frozen here; do not shuffle it per run.
"""
from __future__ import annotations

import os

import numpy as np

# Frozen patient-disjoint split (19 liver+gallbladder TotalSegmentator cases with a non-empty
# gallbladder after the canonical merge). Do NOT change these lists without versioning the benchmark.
TRAIN_PATIENTS = ["s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0522", "s0703",
                  "s0720", "s0727", "s0777", "s1086", "s1373"]                       # 13 train
TEST_PATIENTS = ["s0358", "s0513", "s0519", "s0591", "s0649", "s0762"]               # 6 held-out
SEEDS = (0, 1, 2)
BENCHMARK_VERSION = "usbgym-bench-v1"

# Reference render pipeline: physics renderer + domain randomization, NO translator (D5: the v1
# default was a CUT trained on a private clinical pool, now retired). A public translator is an
# explicit plug-in (make_grender(translator_path=...)); it is never picked up implicitly.
DEFAULT_TRANSLATOR = None


def load_case(cid):
    from . import volume as V
    c = os.path.join("data/cache", f"{cid}.npz")
    if os.path.exists(c):
        return V.load_cache(c)
    return V.load_labeled_ct(os.path.join("data/raw/totalseg_small", cid, "ct.nii.gz"),
                             os.path.join("data/raw/totalseg_small", cid, "segmentations"))


def _mean_std(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), 0.0)


def goal_task(act_fns, similarity, grender, history, image_size, device,
              patients=None, n_per_patient=8, seed=1234, n_steps=16):
    """Evaluate one or more goal-reaching act_fns on the HELD-OUT patients (the frozen GOAL task).

    act_fns: {name: fn(cur_stack, target_shot) -> twist}. Returns per-patient results and the
    aggregate over patients for each act_fn: reach success, final similarity, and (diagnostic) angle
    error. Every act_fn sees the identical target setups (same seed per patient)."""
    from . import goal_rl as G
    patients = patients or TEST_PATIENTS
    per = {name: {"success": [], "final_sim": [], "ang": []} for name in act_fns}
    per_patient = {}
    for cid in patients:
        lv = load_case(cid)
        ev = G.evaluate(lv, act_fns, similarity, history, image_size, device, grender=grender,
                        n_episodes=n_per_patient, n_steps=n_steps, seed=seed)
        per_patient[cid] = {k: {"success": v["success_rate"], "final_sim": v["final_sim"],
                                "ang": v["ang_err_deg"]} for k, v in ev.items()}
        for name, v in ev.items():
            per[name]["success"].append(v["success_rate"])
            per[name]["final_sim"].append(v["final_sim"])
            per[name]["ang"].append(v["ang_err_deg"])
    aggregate = {name: {"success": _mean_std(d["success"]), "final_sim": _mean_std(d["final_sim"]),
                        "ang_err_deg": _mean_std(d["ang"])} for name, d in per.items()}
    return {"benchmark": BENCHMARK_VERSION, "task": "GOAL", "patients": list(patients),
            "n_per_patient": n_per_patient, "seed": seed,
            "per_patient": per_patient, "aggregate": aggregate}


def make_grender(translator_path=DEFAULT_TRANSLATOR, randomize=True, device=None):
    """The benchmark render pipeline: physics + domain randomization, plus an optional translator.

    An explicit translator_path that does not exist raises FileNotFoundError: a missing checkpoint
    must never fall back silently to raw physics behind a translator-labelled number."""
    from . import goal_rl as G
    from .render_lotus import get_device
    dev = device or get_device()
    translator = None
    if translator_path:
        if not os.path.exists(translator_path):
            raise FileNotFoundError(f"translator checkpoint not found: {translator_path}")
        from .texture_gan import load_translator
        translator = load_translator(translator_path, device=dev)   # dual-input aware (SonoSPADE)
    return G.GoalRender(kind="physics", translator=translator, randomize=randomize, device=dev)


def baseline_act_fns(rng_seed=123):
    """The bundled non-learned baselines every GOAL entry is compared against."""
    from . import goal_rl as G
    rng = np.random.default_rng(rng_seed)
    return {"static": lambda h, t: np.zeros(6),
            "random": lambda h, t: rng.uniform(-0.5, 0.5, 6) * G.MAX_TWIST}
