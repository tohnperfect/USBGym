"""Template plug-in for the reward slot (usbg.interfaces.Reward via usbg.plug.rewards.RewardPlug).

The factory ``make()`` returns a potential S(current frame and mask, target frame and mask); the step reward stays
S_t - S_(t-1) - 0.003 |rho|^2, and evaluation and keep-best stay on the match score S. A potential that reads the
probe or target pose is privileged (set privileged = True).

Condition of the slot (SUBMITTING.md): SonoSPADE frames and CT masks; the reward trains the reference REINFORCE
residual from each seed's imitation start, seeds 0 to 2. The maintainers run this slot.

This toy potential is the Dice overlap of the organ masks.
"""
from __future__ import annotations

from usbg.plug.rewards import RewardPlug


class OrganDice(RewardPlug):
    def __init__(self):
        super().__init__("paper")
        self.name, self.privileged, self.app_name = "organ_dice", False, None

    def potential(self, cur_img, cur_lab, tgt_img, tgt_lab, **kw):
        a, b = cur_lab > 0, tgt_lab > 0
        return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))

    def info(self):
        return {"slot": "reward", "reward": self.name, "privileged": False, "citation": "template"}


def make():
    return OrganDice()
