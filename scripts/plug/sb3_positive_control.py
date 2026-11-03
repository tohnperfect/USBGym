#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): positive control of the RL slot by a standard learner.

Before any SB3 row (PPO, SAC, TD3, A2C, DQN) of the controller slot is read, this control shows on TRAIN
patients that a standard Stable-Baselines3 learner, run through the registered Gymnasium environment
(gym.make("USBGym/ViewMatch-v0")) with the slot's own wrappers and the declared comparison budget, reaches
near-oracle error on an EASY variant of the task. The protocol and the pass rule are fixed in
the pre-specified protocol, before any run.

Easy variant (both privileged, labeled so everywhere; never a controller row of the paper):
  observation  se3_log(T_probe^-1 T_target), the target pose relative to the probe (probe frame); translation
               / 0.02 m and rotation / 0.3 rad (the start-distribution half-widths), float32 (6,)
  reward       the privileged gain in potential on pose distance, from the reward slot (usbg.plug.rewards "pose":
               S = -(mm + 2 deg) / 100; r = S_t - S_{t-1} - effort |rho|^2), through PluggedRewardWrapper
  action       the full normalized twist (no base), bound MAX_TWIST, 12 steps, FiniteHorizonTerminal
  learner      SB3 PPO, MlpPolicy, the controller slot's pre-declared PPO hyperparameters
               (scripts/plug_train_sb3.py HP["ppo"], gamma 0.9, VecNormalize on the reward), 57,600 env steps
               (the paper's RL budget), 1 environment, seeds 0, 1, 2
  patients     fit on 10 TRAIN patients; the evaluation uses 3 OTHER TRAIN patients (s0777, s1086, s1373) in the
               crossed design (reset(options={"eval_episode": k}), eval seed 999, 30 episodes, 5 windows per
               cell). It reads no held-out or cohort patient.
  references   the pose oracle (the analytic base, residual zero) and static (zero action), on those 30
               episodes.

A seed s passes when E_learner(s) <= E_oracle + 3.0 mm, E being the final translation error, mean over patients.
  PASS     every seed passes: the SB3 learners, wrappers, budget and evaluator of the RL slot
           learn a correction that the reward shows;
  PARTIAL  not PASS, but the learner closes on average at least half of the gap between static and oracle over
           the seeds: mean of 1 - (E_learner - E_oracle) / (E_static - E_oracle) >= 0.5;
  FAIL     otherwise.
The verdict job always exits 0 (the verdict is a recorded result, not a crash); the controller slot's table
jobs depend on it, and with PARTIAL or FAIL every SB3 row is printed with the label "RL slot: positive
control not passed".

The log holds learning curves and env-step budgets: the training episode returns and final pose errors (binned)
and a TRAIN-internal held-out curve (the 30 evaluation episodes at 6 env-step milestones).

Secondary arm (--obs image, descriptive, never gates): the same with the slot's image observation
(StackedGoalObs, CnnPolicy) and the privileged pose reward, seed 0.

  USBG_SEAT=anatomical USBG_LEARNED_SEG=0 PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python \\
      scripts/plug/sb3_positive_control.py train --seed 0 --out outputs/v2_1/checks/sb3pos/ppo_poseobs_s0.json
  ... sb3_positive_control.py verdict --inputs <3 seed JSONs> --out outputs/v2_1/checks/sb3pos/verdict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ANALYSIS_STATUS = "exploratory, v2.1 pluggability"
FIT_CASES = ["s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0522", "s0703", "s0720", "s0727"]
EVAL_CASES = ["s0777", "s1086", "s1373"]
EPISODES_PER_CELL = 5
PASS_MARGIN_MM = 3.0
PARTIAL_FRACTION = 0.5
BUDGET = 57600
T_SCALE, R_SCALE = 0.02, 0.3
MILESTONES = 6


def pose_obs(T_probe, T_tgt):
    from usbg.contracts_bridge import invert_T, se3_log
    xi = np.asarray(se3_log(invert_T(np.asarray(T_probe, np.float64)) @ np.asarray(T_tgt, np.float64)), np.float64)
    return np.concatenate([xi[:3] / T_SCALE, xi[3:] / R_SCALE]).astype(np.float32)


def make_env(cases, *, device=None, base=None, privileged_info=True, frame_dtype="float32", eval_seed=999,
             render_seed=0):
    """The registered environment (gym.make), the paper's defaults (physics, CT mask, contact on, 12 steps)."""
    import gymnasium as gym
    from usbg.plug import viewmatch_env as VME
    return gym.make(VME.ENV_ID, patients=list(cases), targets=("liver", "kidney"), texture="physics", mask="ct",
                    base=base, render_seed=render_seed, contact=True, device=device, frame_dtype=frame_dtype,
                    privileged_info=privileged_info, min_windows=2, eval_seed=eval_seed, n_steps=12)


def _wrappers():
    import gymnasium as gym
    from gymnasium import spaces

    class PoseObs(gym.ObservationWrapper):
        """PRIVILEGED observation (positive control only): the relative target pose, scaled."""

        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

        def observation(self, obs):
            u = self.env.unwrapped
            return pose_obs(u._probe.world_from_probe, u._T_tgt)

    return PoseObs


def train_env_fn(a, rank=0):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        from usbg.plug import controllers as C
        from usbg.plug import rewards as RW
        PoseObs = _wrappers()
        e = make_env(a.fit_cases, device=a.env_device, privileged_info=True,
                     frame_dtype="uint8" if a.obs == "image" else "float32")
        e = RW.PluggedRewardWrapper(e, reward="pose")
        e = C.StackedGoalObs(e) if a.obs == "image" else PoseObs(e)
        e = C.FiniteHorizonTerminal(e)
        return Monitor(e, info_keywords=("match",))
    return _f


class PoseObsController:
    """Evaluator adapter of the pose-observation learner (PRIVILEGED by construction)."""
    privileged = True
    observes = "PRIVILEGED: the relative target pose (positive control only)"

    def __init__(self, model, name):
        self.model, self.name = model, name

    def reset(self, ep, env):
        pass

    def act(self, obs, info):
        P = info["privileged"]
        a, _ = self.model.predict(pose_obs(P["probe_pose"], P["target_pose"]), deterministic=True)
        return np.clip(np.asarray(a, np.float32).reshape(6), -1.0, 1.0)


def n_eval_episodes(env):
    return EPISODES_PER_CELL * len(env.unwrapped.cells)


def evaluate(env, ctrl, n):
    import torch
    from usbg.plug import controllers as C
    with torch.no_grad():
        return C.evaluate(env.unwrapped, ctrl, n, log_episodes=True)


def _brief(res):
    pl = res["patient_level"]
    return {"final_pos_mm": pl["final_pos"], "final_ang_deg": pl["final_ang"], "keepbest_pos_mm": pl["keepbest_pos"],
            "success_5mm_5deg": pl.get("success_5mm_5deg"), "success_10mm_10deg": pl.get("success_10mm_10deg"),
            "episode_mean_final_pos_mm": res["pos"], "n_episodes": res["n_episodes"]}


def cmd_train(a):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from usbg import provenance as PROV
    from usbg.benchmark import TRAIN_PATIENTS
    from usbg.plug import controllers as C
    import plug_train_sb3 as P3
    if not (set(a.fit_cases) | set(a.eval_cases)) <= set(TRAIN_PATIENTS):
        raise SystemExit("the positive control uses TRAIN patients only")
    if set(a.fit_cases) & set(a.eval_cases):
        raise SystemExit("fit and evaluation patients overlap")
    out = os.path.abspath(a.out)
    stem = out[:-5] if out.endswith(".json") else out
    for p in (out, stem + ".zip", stem + ".vecnormalize.pkl"):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if a.torch_threads:
        torch.set_num_threads(int(a.torch_threads))
    set_random_seed(a.seed)
    hp = dict(P3.HP["ppo"])
    venv = DummyVecEnv([train_env_fn(a)])
    venv.seed(a.seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=P3.COMMON_HP["gamma"])
    policy = "CnnPolicy" if a.obs == "image" else "MlpPolicy"
    model = PPO(policy, venv, n_steps=hp["n_steps_total"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
                learning_rate=hp["learning_rate"], gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"],
                ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"],
                policy_kwargs={"log_std_init": hp["log_std_init"]}, gamma=P3.COMMON_HP["gamma"], seed=a.seed,
                device=a.learner_device, verbose=0)
    ev_env = make_env(a.eval_cases, device=a.env_device, privileged_info=True)
    n_ev = n_eval_episodes(ev_env) if a.eval_episodes is None else int(a.eval_episodes)

    def learner_ctrl():
        if a.obs == "image":
            return C.SB3Controller(model, f"ppo_imageobs_s{a.seed}")
        return PoseObsController(model, f"ppo_poseobs_s{a.seed}")

    # held-out (TRAIN-internal) curve: MILESTONES - 1 points during training, the final evaluation is the last point
    marks = sorted({int(round(a.total_steps * (i + 1) / MILESTONES)) for i in range(MILESTONES - 1)})

    class Log(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows, self.curve, self.t0, self.next = [], [], time.time(), 0

        def _on_step(self):
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:
                    P = info.get("privileged") or {}
                    self.rows.append([int(self.num_timesteps), round(float(ep["r"]), 5), int(ep["l"]),
                                      round(float(P.get("pose_error_mm", float("nan"))), 3),
                                      round(time.time() - self.t0, 1)])
            if self.next < len(marks) and self.num_timesteps >= marks[self.next]:
                te = time.time()
                r = evaluate(ev_env, learner_ctrl(), n_ev)
                self.curve.append({"env_steps": int(self.num_timesteps), **_brief(r), "sec": round(time.time() - te, 1)})
                print(f"[sb3pos s{a.seed}] {self.num_timesteps} steps: held-out (TRAIN-internal) final "
                      f"{r['patient_level']['final_pos']:.2f} mm", flush=True)
                self.next += 1
            return True

    cb = Log()
    t0 = time.time()
    model.learn(total_timesteps=int(a.total_steps), callback=cb, progress_bar=False)
    train_sec = time.time() - t0
    model.save(stem + ".zip")
    venv.save(stem + ".vecnormalize.pkl")
    venv.close()
    te = time.time()
    res_l = evaluate(ev_env, learner_ctrl(), n_ev)
    or_env = make_env(a.eval_cases, device=a.env_device, base="pose_oracle", privileged_info=True)
    res_o = evaluate(or_env, C.ZeroController("pose_oracle"), n_ev)
    res_s = evaluate(ev_env, C.ZeroController("static"), n_ev)
    eval_sec = time.time() - te
    E_l, E_o, E_s = (r["patient_level"]["final_pos"] for r in (res_l, res_o, res_s))
    cb.curve.append({"env_steps": int(model.num_timesteps), **_brief(res_l), "final": True})
    frac = (E_s - E_l) / (E_s - E_o) if E_s > E_o else float("nan")
    rows = cb.rows
    bins = []
    if rows:
        edges = np.linspace(0, len(rows), 11).astype(int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi > lo:
                seg = rows[lo:hi]
                bins.append({"env_steps": seg[-1][0], "mean_return": float(np.mean([r[1] for r in seg])),
                             "mean_final_pos_mm": float(np.nanmean([r[3] for r in seg])), "n_episodes": len(seg)})
    prov = C.provenance_block(ev_env.unwrapped, method="ppo positive control (privileged easy variant)",
                              extra={"learner_device": a.learner_device, "env_device": a.env_device,
                                     "obs": a.obs, "model_md5": PROV.md5(stem + ".zip")})
    prov.update(train_cases=list(a.fit_cases), eval_cases=list(a.eval_cases), privileged=True,
                analysis_status=ANALYSIS_STATUS)
    body = {"check": "rl_slot_positive_control", "analysis_status": ANALYSIS_STATUS, "privileged": True,
            "rung_class": "privileged positive control (never a controller row)",
            "learner": "SB3 PPO " + ("CnnPolicy (image observation)" if a.obs == "image" else "MlpPolicy (pose observation)"),
            "variant": {"observation": "relative target pose (privileged)" if a.obs == "pose" else
                        "image history + target (StackedGoalObs)", "reward": "pose-distance potential gain (privileged)",
                        "action": "full twist, MAX_TWIST bound, no base", "horizon": 12, "via": "gym.make(USBGym/ViewMatch-v0)"},
            "hyperparameters": {**hp, "gamma": P3.COMMON_HP["gamma"], "norm_reward": True, "policy": policy,
                                "n_envs": 1},
            "budget": {"env_steps": int(model.num_timesteps), "requested": int(a.total_steps),
                       "declared_comparison_budget": BUDGET},
            "seed": a.seed, "fit_cases": list(a.fit_cases), "eval_cases": list(a.eval_cases),
            "eval": {"design": "crossed, reset(options={'eval_episode': k}), eval seed 999", "n_episodes": n_ev,
                     "learner": _brief(res_l), "pose_oracle": _brief(res_o), "static": _brief(res_s)},
            "pass_rule": {"margin_mm": PASS_MARGIN_MM, "learner_minus_oracle_mm": E_l - E_o,
                          "seed_passes": bool(E_l <= E_o + PASS_MARGIN_MM), "recovered_fraction": frac},
            "curves": {"train_binned": bins, "heldout_train_internal": cb.curve,
                       "episode_log_columns": ["env_steps", "return", "length", "final_pos_mm", "sec"],
                       "episode_log": rows},
            "timing": {"train_sec": round(train_sec, 1), "eval_sec": round(eval_sec, 1),
                       "env_steps_per_sec": round(model.num_timesteps / max(train_sec, 1e-9), 2)},
            "artifacts": {"model": stem + ".zip", "vecnormalize": stem + ".vecnormalize.pkl"},
            "episodes": {"learner": res_l.get("episodes"), "pose_oracle": res_o.get("episodes"),
                         "static": res_s.get("episodes")},
            "provenance": prov}
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(body, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, out)
    ev_env.close()
    or_env.close()
    print(f"[sb3pos s{a.seed}] learner {E_l:.2f} mm, oracle {E_o:.2f} mm, static {E_s:.2f} mm; seed passes "
          f"{body['pass_rule']['seed_passes']} (recovered {frac:.2f}) -> {out}")
    return 0


def cmd_verdict(a):
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    seeds, missing = [], []
    for p in a.inputs:
        if not os.path.exists(p):
            missing.append(p)
            continue
        d = json.load(open(p))
        if d.get("check") != "rl_slot_positive_control" or d.get("analysis_status") != ANALYSIS_STATUS:
            raise SystemExit(f"{p}: not a positive-control result")
        pr = d["pass_rule"]
        seeds.append({"path": os.path.abspath(p), "seed": d["seed"], "learner_mm": d["eval"]["learner"]["final_pos_mm"],
                      "oracle_mm": d["eval"]["pose_oracle"]["final_pos_mm"], "static_mm": d["eval"]["static"]["final_pos_mm"],
                      "seed_passes": pr["seed_passes"], "recovered_fraction": pr["recovered_fraction"],
                      "env_steps": d["budget"]["env_steps"]})
    if missing or len(seeds) < 3:
        verdict = "INCOMPLETE"
    elif all(s["seed_passes"] for s in seeds):
        verdict = "PASS"
    elif float(np.nanmean([s["recovered_fraction"] for s in seeds])) >= PARTIAL_FRACTION:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    sentence = {
        "PASS": "On TRAIN patients and in all three seeds, SB3 PPO, trained in the registered environment with the "
                "wrappers of the slot and a budget of 57,600 steps, ends within 3 mm of the pose oracle on a "
                "privileged easy variant; the RL slot can learn a correction that the reward shows (exploratory).",
        "PARTIAL": "On TRAIN patients, SB3 PPO recovers at least half of the static-to-oracle gap on a privileged easy "
                   "variant but does not reach within 3 mm of the oracle in every seed; the RL slot is not certified, "
                   "and every SB3 row is printed with that label (exploratory).",
        "FAIL": "On TRAIN patients, SB3 PPO does not recover half of the static-to-oracle gap on a privileged easy "
                "variant; the RL slot is not certified, and no SB3 row is interpreted as evidence about learnability "
                "(exploratory).",
        "INCOMPLETE": "The RL-slot positive control is incomplete; no SB3 row is interpreted (exploratory)."}[verdict]
    body = {"check": "rl_slot_positive_control_verdict", "analysis_status": ANALYSIS_STATUS, "verdict": verdict,
            "sentence": sentence, "rule": {"seed_pass": f"E_learner <= E_oracle + {PASS_MARGIN_MM} mm (patient-level "
                                                         "mean final translation error, 30 TRAIN-internal episodes)",
                                           "pass": "all 3 seeds", "partial": f"mean recovered fraction >= {PARTIAL_FRACTION}"},
            "seeds": seeds, "missing": missing, "rl_slot_certified": verdict == "PASS",
            "provenance": {"analysis_status": ANALYSIS_STATUS, "argv": list(sys.argv),
                           "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "inputs": [s["path"] for s in seeds]},
            "sb3_row_label": None if verdict == "PASS" else "RL slot: positive control not passed"}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump(body, fh, indent=1)
    os.replace(out + ".tmp", out)
    print(f"[sb3pos] verdict {verdict} -> {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--obs", default="pose", choices=["pose", "image"])
    t.add_argument("--total-steps", type=int, default=BUDGET)
    t.add_argument("--fit-cases", nargs="+", default=FIT_CASES)
    t.add_argument("--eval-cases", nargs="+", default=EVAL_CASES)
    t.add_argument("--eval-episodes", type=int, default=None, help="default 5 per valid (patient, organ) cell")
    t.add_argument("--env-device", default=None, help="render device (default MPS if present, never CUDA)")
    t.add_argument("--learner-device", default="cpu")
    t.add_argument("--torch-threads", type=int, default=4)
    t.add_argument("--out", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--inputs", nargs="+", required=True)
    v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ.setdefault("USBG_SEAT", "anatomical")
    return cmd_train(a) if a.cmd == "train" else cmd_verdict(a)


if __name__ == "__main__":
    raise SystemExit(main())
