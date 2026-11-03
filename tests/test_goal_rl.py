"""Tests for goal-conditioned similarity RL (policy, setup, pose diagnostic, rollout)."""
from __future__ import annotations

import numpy as np
import pytest

from usbg import goal_rl as G
from usbg import volume as V


def test_goal_policy_and_gaussian_shapes():
    import torch
    from usbg.policy import GoalPolicy
    policy = G.build_goal_gaussian_policy(GoalPolicy.build(history=3))
    cur = torch.randn(1, 3, 128, 128)
    tgt = torch.randn(1, 1, 128, 128)
    a, logp = policy.act(cur, tgt, deterministic=False)
    assert a.shape == (1, 6) and logp.shape == (1,)
    assert bool((a.abs() <= torch.as_tensor(G.MAX_TWIST) + 1e-6).all())   # bounded twist
    m, none = policy.act(cur, tgt, deterministic=True)
    assert m.shape == (1, 6) and none is None


def test_sample_setup_and_pose_error():
    lv = V.synthetic_phantom()
    Wfv, T_tgt, T_start = G.sample_setup(lv, V.GALLBLADDER, np.random.default_rng(0))
    assert Wfv.shape == (4, 4) and T_tgt.shape == (4, 4) and T_start.shape == (4, 4)
    assert not np.allclose(T_start, T_tgt)                          # start is perturbed
    pe, ae = G._pose_error(T_tgt, T_tgt)
    assert pe < 1e-6 and ae < 1e-6                                  # identical pose -> zero error


def test_same_pipeline_guard():
    from usbg.render_lotus import Renderer, RenderConfig
    a = Renderer("physics", RenderConfig(seed=1), translator=None)
    b = Renderer("physics", RenderConfig(seed=1), translator=(lambda x: x))
    G.assert_same_pipeline(a, a)                                    # identical pipeline -> ok
    with pytest.raises(AssertionError):
        G.assert_same_pipeline(a, b)                               # raw vs translated -> fail loudly


@pytest.mark.slow
def test_rollout_produces_similarity_trace():
    from usbg.policy import GoalPolicy
    from usbg.similarity import SSIMSimilarity
    from usbg.render_lotus import get_device
    lv = V.synthetic_phantom()
    Wfv, T_tgt, T_start = G.sample_setup(lv, V.GALLBLADDER, np.random.default_rng(0))
    env = G._make_env(lv, Wfv, T_start, G.GoalRender(kind="numpy"), np.random.default_rng(0))
    policy = G.build_goal_gaussian_policy(GoalPolicy.build(history=2)).to(get_device())
    roll = G.rollout(env, policy, SSIMSimilarity(), T_tgt, history=2, image_size=64,
                     device=get_device(), n_steps=4)
    env.close()
    assert roll["sims"].shape == (5,) and roll["rewards"].shape == (4,)
    assert len(roll["log_probs"]) == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
