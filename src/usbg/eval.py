"""eval.py: closed-loop rollout of the BC image policy in the MuJoCo env.

Rolls the policy forward from a base pose: at each step it stacks the last `history` rendered
B-mode frames, predicts a 6-DOF twist, and steps the env (which integrates the pose and renders
the next frame). We compare the BC policy against a random-twist and a static (hold) baseline on:

  contact_rate      : fraction of steps in contact (holds the probe on the patient)
  recognizer_agree  : does the recognizer, run on the rollout kinematics, call it the commanded
                      maneuver (i.e. did the policy actually reproduce the maneuver)
  target_visibility : mean fraction of the target organ in view along the rollout

Acceptance: rolled out, the BC policy holds contact and reproduces the maneuver more
than a random or static baseline.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import slicer as S
from . import render_lotus as R
from .contracts_bridge import CLASSES_5, DT_C


def _prep(obs_float01, image_size):
    from PIL import Image
    a = np.clip(np.asarray(obs_float01) * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((image_size, image_size), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def target_visibility(env, target_label) -> float:
    lab = S.slice_at_pose(env.lv, env.world_from_probe, env.world_from_volume)
    return float((lab == target_label).mean())


def load_bc_policy(ckpt_path, device=None):
    import torch
    from .policy import BModePolicy, TwistNorm
    device = device or R.get_device()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = BModePolicy.build(history=ck["history"]).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, TwistNorm.from_dict(ck["twist_norm"]), ck["history"], ck["image_size"], device


def bc_policy_fn(net, tn, device):
    import torch

    def fn(hist_stack, maneuver_id):
        x = torch.from_numpy(hist_stack[None]).to(device)                 # (1,h,H,W)
        m = torch.tensor([maneuver_id], dtype=torch.long, device=device)
        with torch.no_grad():
            z = net(x, m)["twist"][0].cpu().numpy()
        return tn.denormalize(z)                                          # SI twist
    return fn


class ScriptedScan:
    """Open-loop descend-and-fan baseline: a hand-designed (non-learned) controller that presses
    the probe in to seat/hold acoustic contact, then fans it back and forth (a 'tilt' sweep) about
    the contact pose. It keeps the probe on the body but does not servo toward the target, so it
    separates 'holds contact' from 'acquires the view' -- the non-trivial baseline the closed-loop
    rollout otherwise lacks. Axes/magnitudes mirror the scripted demo in demo_env_rollout.py
    (v_z press-in ~0.012, w_x tilt). Stateful: reset() is called at the start of each rollout."""

    def __init__(self, descend_steps=4, press_v=0.012, hold_v=0.004, fan_w=0.20, fan_period=4):
        self.descend_steps = descend_steps
        self.press_v = press_v
        self.hold_v = hold_v
        self.fan_w = fan_w
        self.fan_period = fan_period
        self.reset()

    def reset(self):
        self._k = 0

    def __call__(self, h, m):
        k = self._k
        self._k += 1
        if k < self.descend_steps:                       # (a) press in to seat contact
            return np.array([0.0, 0.0, self.press_v, 0.0, 0.0, 0.0])
        j = k - self.descend_steps                       # (b) fan +/- about the contact pose
        sign = 1.0 if (j // self.fan_period) % 2 == 0 else -1.0
        return np.array([0.0, 0.0, self.hold_v, sign * self.fan_w, 0.0, 0.0])


def rollout(env, maneuver_id, policy_fn, n_steps, history, image_size, target_label):
    obs, info = env.reset()
    if hasattr(policy_fn, "reset"):                       # stateful scripted policy restarts here
        policy_fn.reset()
    frame = _prep(obs, image_size)
    hist = deque([frame] * history, maxlen=history)
    twists, forces, vis, poses = [], [info["contact_force"]], [target_visibility(env, target_label)], []
    for _ in range(n_steps):
        tw = np.asarray(policy_fn(np.stack(hist, 0), maneuver_id), dtype=np.float64).reshape(6)
        obs, info = env.step(tw)
        hist.append(_prep(obs, image_size))
        twists.append(tw); forces.append(info["contact_force"])
        vis.append(target_visibility(env, target_label)); poses.append(env.world_from_probe)
    return {"twists": np.asarray(twists), "forces": np.asarray(forces),
            "visibility": np.asarray(vis), "poses": np.asarray(poses)}


def _rel6_from_poses(poses):
    """start-relative pose [x,y,z,rx,ry,rz] via Repo B's rel6 (tool frame, axis-angle)."""
    from us_rl_sim.demos.action_data import rel6
    prs = [(T[:3, 3], T[:3, :3]) for T in poses]
    return np.asarray(rel6(prs), dtype=np.float64)


def recognizer_agreement(roll, maneuver_id, model, norm, device) -> float:
    """Fraction of valid rollout frames the recognizer labels as the commanded maneuver."""
    from par.eval import predict_record
    from par.data import ClipRecord, _valid_eff
    tw = np.concatenate([np.full((1, 6), np.nan), roll["twists"]], axis=0)   # frame-0 NaN
    rel = _rel6_from_poses(np.concatenate([roll["poses"][:1], roll["poses"]], axis=0)) \
        if len(roll["poses"]) else np.zeros((len(tw), 6))
    rel = rel[:len(tw)]
    valid = _valid_eff(tw, np.ones(len(tw), bool))
    rec = ClipRecord(twist=tw, rel6=rel, valid_eff=valid, label=maneuver_id)
    res = predict_record(model, rec, norm, device)
    pv = res["pred"][np.asarray(res["valid"], bool)]
    return float((pv == maneuver_id).mean()) if pv.size else 0.0


def compare_policies(env_factory, bc_fn, history, image_size, target_label,
                     model, norm, device, n_steps=16, force_thresh=0.5, seed=0,
                     maneuver_ids=None, n_placements=3, extra_policies=None):
    """Run BC vs random vs static across the given maneuvers; return a per-policy summary.

    Each maneuver is rolled out from n_placements randomized anatomy placements (env_factory
    takes an rng) and the metrics are averaged, so the coarse per-rollout recognizer agreement
    is smoothed. maneuver_ids : maneuver indices (into CLASSES_5); None = all five.
    extra_policies : optional {name: policy_fn} added to the comparison (e.g. a scripted baseline);
    a policy_fn with a reset() method is restarted at the start of each rollout.
    """
    rng = np.random.default_rng(seed)
    typ = 0.05                                    # typical twist scale for the random baseline
    policies = {
        "bc": bc_fn,
        "random": lambda h, m: rng.normal(0, typ, size=6),
        "static": lambda h, m: np.zeros(6),
    }
    if extra_policies:
        policies.update(extra_policies)
    if maneuver_ids is None:
        maneuver_ids = list(range(len(CLASSES_5)))
    out = {k: {"contact_rate": [], "recognizer_agree": [], "visibility": []} for k in policies}
    for man_id in maneuver_ids:
        for _ in range(n_placements):
            seed_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
            for name, fn in policies.items():
                env = env_factory(seed_rng)
                roll = rollout(env, man_id, fn, n_steps, history, image_size, target_label)
                env.close()
                out[name]["contact_rate"].append(float((roll["forces"] > force_thresh).mean()))
                out[name]["recognizer_agree"].append(
                    recognizer_agreement(roll, man_id, model, norm, device))
                out[name]["visibility"].append(float(roll["visibility"].mean()))
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in out.items()}
