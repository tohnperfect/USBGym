"""Template plug-in for the controller slot.

The factory ``make(seed, device)`` returns a controller with ``reset(episode, env)`` and ``act(obs, info) ->
normalized six-axis twist`` (float32, in [-1, 1]); over the servo base the twist is the residual. If it learns, it
has ``train(make_env, budget, seed)``: ``make_env(privileged_info=False)`` builds a training environment on the 13
training patients (usbg.plug.controllers.make_env), and ``budget`` gives the protocol's 57,600 environment steps
(reinforcement learning) or 40,000 samples (imitation). It never sees a held-out patient before evaluation.

The slot condition (SUBMITTING.md): SonoSPADE frames; CT mask; the plugged controller, residual or full twist;
seeds 0 to 2. The maintainers run this slot.

This toy controller adds a zero residual, so the servo base alone acts.
"""
from __future__ import annotations

import numpy as np


class ZeroResidual:
    privileged = False
    name = "zero_residual"
    observes = "nothing (zero residual); the servo base reads the masks"

    def __init__(self, seed=0, device="cpu"):
        self.seed, self.device = int(seed), device

    def train(self, make_env, budget, seed):
        return None                      # a learner trains here, within budget["rl_env_steps"]

    def reset(self, ep, env):
        pass

    def act(self, obs, info):
        return np.zeros(6, np.float32)


def make(seed=0, device="cpu"):
    return ZeroResidual(seed, device)
