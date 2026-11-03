"""Tests that gate the USB Gym benchmark protocol: a frozen, disjoint split, stable baselines, and a
render pipeline. These guard reproducibility (the split and seeds must not drift between runs)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from usbg import benchmark as B


def test_split_is_frozen_and_disjoint():
    tr, te = set(B.TRAIN_PATIENTS), set(B.TEST_PATIENTS)
    assert len(B.TRAIN_PATIENTS) == 13 and len(B.TEST_PATIENTS) == 6      # frozen sizes
    assert len(tr) == 13 and len(te) == 6                                # no duplicates within a split
    assert tr.isdisjoint(te)                                             # patient-disjoint
    assert all(c.startswith("s") for c in B.TRAIN_PATIENTS + B.TEST_PATIENTS)
    assert B.BENCHMARK_VERSION and isinstance(B.SEEDS, tuple) and len(B.SEEDS) >= 3


def test_baseline_act_fns_are_valid():
    from usbg import goal_rl as G
    fns = B.baseline_act_fns()
    assert set(fns) == {"static", "random"}
    h = np.zeros((4, 64, 64), np.float32); t = np.zeros((64, 64), np.float32)
    for name, fn in fns.items():
        a = fn(h, t)
        assert np.asarray(a).shape == (6,)
        assert np.all(np.abs(a) <= G.MAX_TWIST + 1e-6)                   # within twist bounds
    # static is exactly zero (a valid do-nothing baseline)
    assert np.allclose(fns["static"](h, t), 0.0)


def test_make_grender_builds_pipeline():
    gr = B.make_grender(translator_path=None, randomize=True, device="cpu")
    from usbg.goal_rl import GoalRender
    assert isinstance(gr, GoalRender) and gr.kind == "physics" and gr.randomize
    assert gr.translator is None                                        # physics-only reference
    cfg = gr.episode_cfg(np.random.default_rng(0))
    assert cfg is not None                                              # a physics config draw


def test_reference_pipeline_has_no_translator():
    """D5: the reference pipeline has no translator (the v1 default was a private-pool CUT)."""
    assert B.DEFAULT_TRANSLATOR is None
    gr = B.make_grender(randomize=False, device="cpu")
    assert gr.kind == "physics" and gr.translator is None


def test_make_grender_missing_translator_raises():
    """An explicit translator path that does not exist fails loudly (no silent physics fallback)."""
    with pytest.raises(FileNotFoundError):
        B.make_grender(translator_path="/nonexistent.pt", device="cpu")


@pytest.mark.slow
def test_goal_task_runs_on_one_patient():
    """A minimal GOAL-task smoke on one held-out patient (skipped if its cache is absent)."""
    cid = B.TEST_PATIENTS[0]
    if not os.path.exists(os.path.join("data/cache", f"{cid}.npz")):
        pytest.skip(f"cache for {cid} not present")
    from usbg.similarity import SSIMSimilarity
    from usbg.render_lotus import get_device
    dev = get_device()
    gr = B.make_grender(translator_path=None, randomize=False, device=dev)
    res = B.goal_task(B.baseline_act_fns(), SSIMSimilarity(), gr, history=2, image_size=64,
                      device=dev, patients=[cid], n_per_patient=2, seed=1234, n_steps=3)
    assert res["task"] == "GOAL" and cid in res["per_patient"]
    for name in ("static", "random"):
        a = res["aggregate"][name]
        assert 0.0 <= a["success"][0] <= 1.0 and np.isfinite(a["final_sim"][0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
