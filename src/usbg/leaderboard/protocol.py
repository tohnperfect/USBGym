"""usbg.leaderboard.protocol: the plug protocol that every leaderboard row follows.

One slot is swapped at a time; the rest of the loop is held at that slot's condition (the slot headers of
Table 5 of the USB Gym article, copied below and checked against the printed table when the leaderboard data
is built). A new method meets the same 60 held-out episodes as the slot's reference component:

* the six held-out patients, two targets (liver, kidney) and five screened windows each, in the crossed design
  (episode k takes the (patient, target) cell k mod 12, patient-major, and window k div 12), evaluation seed 999,
  contact on, no domain randomization, the windows file of the reported runs;
* 12 steps per episode; the final translation error of each episode is the score;
* training only on the 13 training patients, with every setting fixed before any held-out run (no tuning on
  held-out patients);
* reinforcement learners get the 57,600 environment steps of the reference controller (300 iterations of 16
  episodes, 12 steps each) and imitation learners its 40,000 samples;
* learned methods run seeds 0, 1 and 2; a method without training (inference only) may run seed 0 only and is
  then paired with seed 0 of the reference;
* plugged networks run batch normalization in evaluation mode.
"""
from __future__ import annotations

PROTOCOL_VERSION = "usbg-plug-protocol-1"
SUBMISSION_SCHEMA = "usbg-leaderboard-submission-1"

TRAIN_PATIENTS = ("s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0522", "s0703", "s0720", "s0727", "s0777",
                  "s1086", "s1373")
HELD_OUT_PATIENTS = ("s0358", "s0513", "s0519", "s0591", "s0649", "s0762")
TARGETS = ("liver", "kidney")
N_WINDOWS = 5
N_EPISODES = 60
EVAL_SEED = 999
EVAL_DESIGN = "crossed"
N_STEPS = 12
MIN_WINDOWS = 2
CONTACT = "on"
DOMAIN_RANDOMIZATION = "off"
SEEDS = (0, 1, 2)
SEED0 = (0,)
RL_ENV_STEPS = 57_600            # 300 iterations of 16 episodes, 12 steps each
IMITATION_SAMPLES = 40_000
BATCH_NORM = "eval"              # plugged networks
WINDOWS_FILE = "usbg-windows-v1.json"
WINDOWS_MD5 = "af35f0bddeb57c079981eb0ba8aca5b3"
SEAT = "anatomical"
REWARD_WEIGHTS = {"w_seg": 0.6, "w_app": 0.4, "effort": 0.003}
SUCCESS = {"mm": 10.0, "deg": 10.0}


def episode_design():
    """The 60 held-out episodes: [{"ep", "patient", "target"}] (episode k: cell k mod 12 in patient-major order,
    window k div 12 in the cell's rank order)."""
    cells = [(p, t) for p in HELD_OUT_PATIENTS for t in TARGETS]
    return [{"ep": k, "patient": cells[k % len(cells)][0], "target": cells[k % len(cells)][1],
             "window_rank": k // len(cells)} for k in range(N_EPISODES)]


# Slot conditions: the plain text of the slot headers of Table 5 (checked against the printed table by the build).
# who: "submitter" when every artifact the condition needs is public (README, Data preparation; the release
# assets of usbg.leaderboard.assets); "maintainer" when the condition needs an artifact that is not released, so
# the maintainers run the submitted plug-in with the reference artifacts and post the result.
SLOTS = {
    "texture": {
        "title": "Texture stage",
        "interface": "usbg.interfaces.TextureTranslator: a callable from an H x W float frame in [0, 1] to one of "
                     "the same shape (set wants_label = True to also receive the label slice)",
        "condition": "frames from the plugged stage; mask from that stage's own matched U-Net; servo base, "
                     "residual off; seeds 0 to 2",
        "reference": "texture/sonospade",
        "trains": "the stage-matched U-Net of each seed (segtrain, the reference recipe) on frames of the plugged "
                  "stage; the servo base is calibrated on the training patients under that U-Net's mask",
        "who": "maintainer",      # submitter-run once the KidneyUS test assignment (kidneyus_split.npz) is published
        "needs": ["the CT label caches of the 19 patients", "the public real frames of the segmenter recipe "
                  "(Vitale et al. and KidneyUS training split; see README, Data preparation; KidneyUS is CC BY-NC-SA "
                  "and the matched U-Net inherits its terms)", "the fixed KidneyUS test assignment of the reported "
                  "runs (segtrain/data/real/kidneyus_split.npz), not yet published"],
        "seeds": SEEDS,
    },
    "mask": {
        "title": "Mask source",
        "interface": "usbg.interfaces.MaskSource: segment(img01) -> int64 mask in {0 background, 1 liver, "
                     "2 gallbladder, 3 vessel, 4 kidney, 7 spleen}",
        "condition": "SonoSPADE frames; mask from the plugged source; servo base fitted under that mask, residual "
                     "off; seeds 0 to 2, and seed 0 for the SAM-type rows, paired with seed 0",
        "reference": "mask/unet",
        "trains": "nothing beyond the plug-in's own training (on training patients); the servo base is calibrated "
                  "on the training patients under the plugged mask",
        "who": "maintainer",
        "needs": ["the SonoSPADE stage checkpoint of the reported runs (md5 70440a5c6167e69489835ef553bfe37c), "
                  "not released"],
        "seeds": SEEDS,
    },
    "ctrl": {
        "title": "Controller",
        "interface": "a controller: reset(episode) and act(obs) -> normalized six-axis twist (or residual), trained "
                     "by its own train(make_env, budget, seed) on the training patients",
        "condition": "SonoSPADE frames; CT mask†; the plugged controller, residual or full twist as named; seeds 0 "
                     "to 2, and seed 0 for CEM",
        "reference": "ctrl/reinforce",
        "trains": "the plugged controller, within 57,600 environment steps (reinforcement learning) or 40,000 "
                  "samples (imitation), on the training patients",
        "who": "maintainer",
        "needs": ["the SonoSPADE stage checkpoint of the reported runs, not released"],
        "seeds": SEEDS,
    },
    "reward": {
        "title": "Reward",
        "interface": "usbg.interfaces.Reward / usbg.plug.rewards.RewardPlug: a potential S(current frame, target "
                     "frame); the step reward stays S_t - S_(t-1) - 0.003 |rho|^2",
        "condition": "SonoSPADE frames; CT mask†; the REINFORCE residual from the imitation start, trained with the "
                     "plugged potential; seeds 0 to 2",
        "reference": "reward/paper",
        "trains": "the reference REINFORCE residual (300 iterations x 16 episodes) from the imitation checkpoint of "
                  "each seed, with the plugged potential; evaluation and keep-best stay on S",
        "who": "maintainer",
        "needs": ["the SonoSPADE stage checkpoint and the imitation checkpoints of the reported runs, not "
                  "released"],
        "seeds": SEEDS,
    },
    "renderer": {
        "title": "Renderer",
        "interface": "usbg.interfaces.Renderer: render(label_slice, *, device=None, cfg=None, seed=None) -> float32 "
                     "frame in [0, 1] (int64 label slice in, the physics renderer's geometry)",
        "condition": "frames from the plugged renderer, no texture stage; CT mask†; the physics-trained policy, "
                     "residual on; seeds 0 to 2, and seed 0 for SIMUS",
        "reference": "renderer/physics",
        "trains": "nothing: the physics-trained reference policy of each seed (a release asset) is evaluated under "
                  "the plugged renderer",
        "who": "submitter",
        "needs": ["the CT label caches of the six held-out patients", "the release assets "
                  "usbg_physics_policy_s{0,1,2}.pt and .json (python -m usbg.leaderboard fetch)"],
        "seeds": SEEDS,
    },
    "anatomy": {
        "title": "CT anatomy",
        "interface": "a labeled CT volume per patient (the usbg.volume cache), built from the same CTs",
        "condition": "five reference levels of the oracle ladder, each in its oracle-ladder condition with seeds 0 "
                     "to 2. TotalSegmentator v2 labels (fast mode, 3.0 mm) replace the reference labels on the same "
                     "episodes and classes",
        "reference": None,
        "trains": "nothing: the five reference levels are rerun on the plugged label volumes",
        "who": "maintainer",
        "needs": ["the SonoSPADE stage and the learned-mask segmenters of the reported runs, not released"],
        "seeds": SEEDS,
    },
}

PLUG_SLOTS = ("texture", "mask", "ctrl", "reward", "renderer", "anatomy")

__all__ = ["PROTOCOL_VERSION", "SUBMISSION_SCHEMA", "SLOTS", "PLUG_SLOTS", "episode_design", "TRAIN_PATIENTS",
           "HELD_OUT_PATIENTS", "N_STEPS", "N_EPISODES", "EVAL_SEED", "SEEDS", "SEED0", "RL_ENV_STEPS",
           "IMITATION_SAMPLES", "BATCH_NORM", "WINDOWS_MD5"]
