#!/usr/bin/env python3
"""v2.1 pluggability, slot "controller / policy learning" (EXPLORATORY, opt-in; new file, touches no queued job).

Trains a published policy-learning method on USBGym/ViewMatch-v0 (usbg.plug.viewmatch_env, a copy of the
usbg.gym_env environment), or builds a reference controller, and evaluates it on the paper's crossed held-out
design through ``reset(options={"eval_episode": k})`` (6 held-out patients, eval seed 999, 60 episodes), so
every row meets the same episodes as the oracle-ladder rows. The environment evaluator reproduces the trainer's evaluate
exactly (TRAIN-patient smoke: no difference for any of the pose oracle, static, the servo base and a v2
REINFORCE policy, under the CT mask and under the matched learned mask with SonoSPADE).

Methods (--algo)
  ppo, sac, td3, a2c   Stable-Baselines3 2.6 learners, CnnPolicy (NatureCNN on the stacked current history and
                       target). --base servo (default): residual over the paper's image-servo base, the matrix
                       taken from the paper's v2 policy checkpoint of the same texture, mask and seed
                       (--servo-base), residual bound 0.5 MAX_TWIST, so the base, action bound, reward and RL
                       budget (57,600 env steps) equal the paper's REINFORCE row. NOT equal: the SB3 learner
                       starts from a random NatureCNN policy, while the paper's row starts RL from its own
                       imitation warm start (4,000 iterations x batch 10 on the GoalPolicy net) and keeps a BC
                       anchor; state this when the rows are compared. --base none: from scratch (full twist).
  dqn                  SB3 DQN on Discrete(13) (no-op, +-1 normalized step on each twist axis), from scratch.
  bc                   behavior cloning in plain PyTorch: the GoalPolicy net of the paper, fitted to rollouts of
                       the privileged pose oracle.
  dagger               DAgger (round 0 = teacher rollouts, then rounds of learner rollouts relabeled by the
                       teacher, aggregated, warm-started refits); same net, same env-step and gradient budgets
                       as bc.
  v2_policy            the paper's own REINFORCE (or imitation) policy reloaded from --policy-pt (a parity row).
  random, static, servo_base, pose_oracle, cem
                       reference controllers (uniform random; zero action; the servo base alone; the PRIVILEGED
                       analytic base; the PRIVILEGED CEM look-ahead bound of usbg.controllers_v21).

Rules: training uses TRAIN patients only (held-out patients are refused); --out must not exist (nothing is
overwritten); the result JSON has a provenance block with analysis_status "exploratory, v2.1 pluggability".

  USBG_SEAT=anatomical USBG_LEARNED_SEG=0 PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python \\
      scripts/plug_train_sb3.py --algo ppo --texture physics --mask ct --seed 0 \\
      --servo-base outputs/v2/table4/ctmask/physics_deploy_s0.pt --out outputs/v2_1/plug/ctrl/ppo_res/physics_ct_s0.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SB3_ALGOS = ("ppo", "sac", "td3", "a2c", "dqn")
IMIT_ALGOS = ("bc", "dagger")
REF_ALGOS = ("random", "static", "servo_base", "pose_oracle", "cem", "v2_policy")
PAPER_BUDGET = {"rl_online_env_steps": 57600, "imitation_samples": 40000,
                "note": "the paper's v2 runs: 300 REINFORCE iterations x 16 episodes x 12 steps after 4000 imitation "
                        "iterations x batch 10 (outputs/v2/table4/*/<stage>_deploy_s<seed>.json budget)"}

# Pre-declared hyperparameters (fixed before any run; not tuned on held-out patients). gamma = 0.9 is the
# paper's REINFORCE discount; the rest are Stable-Baselines3 / RL Zoo defaults scaled to the budget, except
# that SAC and TD3 take one gradient step per 2 env steps at batch 128 (update-to-data ratio 0.5): one SB3
# SAC gradient step on the 128 px image costs about 170 ms at batch 256 and 60 ms at batch 128 on MPS
# (TRAIN smoke), so UTD 1 at batch 256 would take about 2.7 h of updates for the 57,600-step budget. The
# replay buffers hold 20,000 transitions (uint8 obs, optimize_memory_usage: about 1.6 GB per run).
HP = {
    "ppo": {"n_steps_total": 768, "batch_size": 64, "n_epochs": 10, "learning_rate": 3e-4, "gae_lambda": 0.95,
            "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5, "log_std_init": math.log(0.5)},
    "a2c": {"n_steps_total": 12, "learning_rate": 7e-4, "gae_lambda": 1.0, "ent_coef": 0.0, "vf_coef": 0.5,
            "max_grad_norm": 0.5, "log_std_init": math.log(0.5)},
    "sac": {"buffer_size": 20000, "learning_starts": 1000, "batch_size": 128, "learning_rate": 3e-4, "tau": 0.005,
            "train_freq": 2, "gradient_steps": 1, "ent_coef": "auto"},
    "td3": {"buffer_size": 20000, "learning_starts": 1000, "batch_size": 128, "learning_rate": 1e-3, "tau": 0.005,
            "train_freq": 2, "gradient_steps": 1, "policy_delay": 2, "target_policy_noise": 0.2,
            "target_noise_clip": 0.5, "action_noise_sigma": 0.1},
    "dqn": {"buffer_size": 20000, "learning_starts": 1000, "batch_size": 32, "learning_rate": 1e-4, "train_freq": 4,
            "gradient_steps": 1, "target_update_interval": 1000, "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "step_frac": 1.0},
    "bc": {"imitation_env_steps": 40000, "fit_steps": 8000, "batch_size": 64, "learning_rate": 3e-4},
    "dagger": {"imitation_env_steps": 40000, "rounds": 4, "fit_steps": 8000, "batch_size": 64, "learning_rate": 3e-4,
               "beta": "1 in round 0 (teacher rollouts), 0 afterwards (learner rollouts, teacher relabels)"},
}
COMMON_HP = {"gamma": 0.9, "policy": "CnnPolicy (NatureCNN, features_dim 512) on the stacked uint8 image",
             "norm_reward": True, "horizon_terminal": True}
DEFAULT_BASE = {"ppo": "servo", "sac": "servo", "td3": "servo", "a2c": "servo", "dqn": "none", "bc": "servo",
                "dagger": "servo", "random": "none", "static": "none", "servo_base": "servo",
                "pose_oracle": "pose_oracle", "cem": "none", "v2_policy": "servo"}
RUNG_CLASS = {"pose_oracle": "privileged reference (analytic pose base)",
              "cem": "privileged reference (CEM look-ahead; renders hypothetical poses)",
              "static": "reference (no commanded motion)", "random": "reference (uniform random actions)",
              "servo_base": "non-privileged base (pose-free at run time; calibrated on privileged analytic twists)",
              "v2_policy": "non-privileged policy (the paper's v2 policy, reloaded through the environment)",
              "bc": "non-privileged policy (pose-free at run time; trained from a privileged teacher)",
              "dagger": "non-privileged policy (pose-free at run time; trained from a privileged teacher)"}
for _a in SB3_ALGOS:
    RUNG_CLASS[_a] = "non-privileged policy (pose-free at run time; reward needs no pose)"


def parse_args(argv=None):
    from usbg.benchmark import TRAIN_PATIENTS, TEST_PATIENTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", required=True, choices=[*SB3_ALGOS, *IMIT_ALGOS, *REF_ALGOS])
    ap.add_argument("--texture", default="physics", choices=["physics", "cut", "cyclegan", "sonospade"])
    ap.add_argument("--texture-ckpt", default=None, help="checkpoint of the texture stage ('none' for physics)")
    ap.add_argument("--mask", default="ct", choices=["ct", "learned"])
    ap.add_argument("--seg-ckpt", default=None, help="texture-matched segmenter (mask learned)")
    ap.add_argument("--base", default=None, choices=["none", "servo", "pose_oracle"],
                    help="default per algo: residual over the servo base for ppo/sac/td3/a2c/bc/dagger")
    ap.add_argument("--servo-base", default=None, help="v2 policy .pt whose 'ibvs' matrix is the servo base")
    ap.add_argument("--residual-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-seed", type=int, default=None, help="render cfg seed (default = --seed, as the paper)")
    ap.add_argument("--total-steps", type=int, default=57600, help="RL env steps (the paper's 57,600)")
    ap.add_argument("--imitation-steps", type=int, default=None, help="bc/dagger env steps (default 40,000)")
    ap.add_argument("--fit-steps", type=int, default=None, help="bc/dagger gradient steps (default 8,000)")
    ap.add_argument("--dagger-rounds", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--vec", default="dummy", choices=["dummy", "subproc"])
    ap.add_argument("--env-device", default="auto", choices=["auto", "cpu", "mps"],
                    help="render device of the training envs (auto = MPS if present, never CUDA)")
    ap.add_argument("--learner-device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--norm-reward", default="on", choices=["on", "off"])
    ap.add_argument("--hp", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a pre-declared hyperparameter (recorded; JSON values)")
    ap.add_argument("--train-cases", nargs="+", default=list(TRAIN_PATIENTS))
    ap.add_argument("--eval-cases", nargs="+", default=list(TEST_PATIENTS))
    ap.add_argument("--targets", nargs="+", default=["liver", "kidney"], choices=["liver", "kidney", "spleen"])
    ap.add_argument("--min-windows", type=int, default=2)
    ap.add_argument("--eval-episodes", type=int, default=60)
    ap.add_argument("--eval-seed", type=int, default=999)
    ap.add_argument("--eval-contact", nargs="+", default=["on"], choices=["on", "off"])
    ap.add_argument("--eval-device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--bn-modes", nargs="+", default=["train", "eval"], choices=["train", "eval"],
                    help="GoalPolicy nets (bc, dagger, v2_policy): BatchNorm mode(s) at evaluation; the first is "
                         "primary (train = the paper's convention)")
    ap.add_argument("--policy-pt", default=None, help="v2_policy: the v2 policy checkpoint (.pt or _imit.pt)")
    ap.add_argument("--model", default=None, help="evaluate a saved model (.zip / .pt) instead of training")
    ap.add_argument("--cem-k", type=int, default=64); ap.add_argument("--cem-rounds", type=int, default=2)
    ap.add_argument("--cem-elite", type=int, default=8); ap.add_argument("--cem-seed-offset", type=int, default=7907)
    ap.add_argument("--no-episodes", action="store_true", help="omit per-episode records from the JSON")
    ap.add_argument("--smoke", action="store_true", help="smoke run: evaluation cases must be TRAIN patients")
    ap.add_argument("--label", default=None, help="row label in the summary table (default <algo>_<base>)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.base = a.base or DEFAULT_BASE[a.algo]
    if a.render_seed is None:
        a.render_seed = a.seed
    leak = sorted(set(a.train_cases) & set(TEST_PATIENTS))
    if leak:
        ap.error(f"training cases include held-out patients {leak}")
    if a.smoke and not set(a.eval_cases) <= set(TRAIN_PATIENTS):
        ap.error("--smoke evaluates TRAIN patients only")
    if a.base == "servo" and not a.servo_base:
        ap.error("--base servo needs --servo-base (the v2 policy checkpoint of the same texture, mask and seed)")
    if a.mask == "learned" and not a.seg_ckpt:
        ap.error("--mask learned needs --seg-ckpt (the texture-matched segmenter)")
    if a.algo == "dqn" and a.base != "none":
        ap.error("dqn runs from scratch (--base none), as the discrete-action probe-navigation papers did")
    if a.algo == "cem" and (a.base != "none" or a.mask != "ct"):
        ap.error("cem needs --base none and --mask ct")
    if a.algo == "pose_oracle" and a.base != "pose_oracle":
        ap.error("pose_oracle needs --base pose_oracle")
    if a.algo == "servo_base" and a.base != "servo":
        ap.error("servo_base needs --base servo")
    if a.algo == "v2_policy" and not a.policy_pt:
        ap.error("v2_policy needs --policy-pt")
    for k, v in (kv.split("=", 1) for kv in a.hp):
        HP.setdefault(a.algo, {})[k] = json.loads(v)
    return a


def resolve_device(choice):
    from usbg import render_lotus as R
    return R.get_device() if choice == "auto" else choice


def artifact_stem(out):
    return out[:-5] if out.endswith(".json") else out


def refuse_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p} (outputs are never overwritten)")


def env_kwargs(a, patients, *, device, privileged, frame_dtype, contact=True):
    return dict(patients=list(patients), texture=a.texture, texture_ckpt=a.texture_ckpt, mask=a.mask,
                seg_ckpt=a.seg_ckpt, base=None if a.base == "none" else a.base,
                servo_base=a.servo_base if a.base == "servo" else None, residual_frac=a.residual_frac,
                render_seed=a.render_seed, contact=contact, device=device, frame_dtype=frame_dtype,
                privileged_info=privileged, targets=tuple(a.targets), min_windows=a.min_windows,
                eval_seed=a.eval_seed)


# ===================================================================================== SB3 training
def _make_train_env_fn(a, rank, env_device, threads):
    def _f():
        import torch
        if threads:
            torch.set_num_threads(int(threads))
        from stable_baselines3.common.monitor import Monitor
        from usbg.plug import controllers as C
        env = C.make_env(**env_kwargs(a, a.train_cases, device=env_device, privileged=False, frame_dtype="uint8"))
        if a.algo == "dqn":
            env = C.DiscreteTwist(env, step_frac=HP["dqn"]["step_frac"])
        env = C.StackedGoalObs(env)
        env = C.FiniteHorizonTerminal(env)
        return Monitor(env, info_keywords=("match", "seg_iou"))
    return _f


def train_sb3(a, stem, learner_device, env_device):
    import torch
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    set_random_seed(a.seed)
    fns = [_make_train_env_fn(a, i, env_device, a.torch_threads if a.vec == "subproc" else None)
           for i in range(a.n_envs)]
    venv = SubprocVecEnv(fns) if a.vec == "subproc" else DummyVecEnv(fns)
    venv.seed(a.seed)
    gamma = COMMON_HP["gamma"]
    if a.norm_reward == "on":
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=gamma)
    hp = dict(HP[a.algo])
    kw = dict(policy="CnnPolicy", env=venv, gamma=gamma, seed=a.seed, device=learner_device, verbose=0)
    if a.algo == "ppo":
        model = PPO(n_steps=max(1, hp["n_steps_total"] // a.n_envs), batch_size=hp["batch_size"],
                    n_epochs=hp["n_epochs"], learning_rate=hp["learning_rate"], gae_lambda=hp["gae_lambda"],
                    clip_range=hp["clip_range"], ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"],
                    max_grad_norm=hp["max_grad_norm"], policy_kwargs={"log_std_init": hp["log_std_init"]}, **kw)
    elif a.algo == "a2c":
        model = A2C(n_steps=max(1, hp["n_steps_total"] // a.n_envs), learning_rate=hp["learning_rate"],
                    gae_lambda=hp["gae_lambda"], ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"],
                    max_grad_norm=hp["max_grad_norm"], policy_kwargs={"log_std_init": hp["log_std_init"]}, **kw)
    else:
        rb = dict(buffer_size=hp["buffer_size"], learning_starts=hp["learning_starts"], batch_size=hp["batch_size"],
                  learning_rate=hp["learning_rate"], train_freq=hp["train_freq"], gradient_steps=hp["gradient_steps"],
                  optimize_memory_usage=True, replay_buffer_kwargs={"handle_timeout_termination": False})
        if a.algo == "sac":
            model = SAC(tau=hp["tau"], ent_coef=hp["ent_coef"], **rb, **kw)
        elif a.algo == "td3":
            noise = NormalActionNoise(np.zeros(6), hp["action_noise_sigma"] * np.ones(6))
            model = TD3(tau=hp["tau"], policy_delay=hp["policy_delay"], target_policy_noise=hp["target_policy_noise"],
                        target_noise_clip=hp["target_noise_clip"], action_noise=noise, **rb, **kw)
        else:
            model = DQN(target_update_interval=hp["target_update_interval"],
                        exploration_fraction=hp["exploration_fraction"],
                        exploration_initial_eps=hp["exploration_initial_eps"],
                        exploration_final_eps=hp["exploration_final_eps"], **rb, **kw)

    class EpisodeLog(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows, self.t0 = [], time.time()

        def _on_step(self):
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:
                    self.rows.append([int(self.num_timesteps), round(float(ep["r"]), 5), int(ep["l"]),
                                      round(float(info.get("match", float("nan"))), 4),
                                      round(time.time() - self.t0, 1)])
            return True

    cb = EpisodeLog()
    t0 = time.time()
    model.learn(total_timesteps=int(a.total_steps), callback=cb, progress_bar=False)
    sec = time.time() - t0
    model.save(stem + ".zip")
    if a.norm_reward == "on":
        venv.save(stem + ".vecnormalize.pkl")
    venv.close()
    rows = cb.rows
    curve = []
    if rows:
        edges = np.linspace(0, len(rows), 11).astype(int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi > lo:
                seg = rows[lo:hi]
                curve.append({"env_steps": seg[-1][0], "mean_return": float(np.mean([r[1] for r in seg])),
                              "mean_final_match": float(np.nanmean([r[3] for r in seg])), "n_episodes": len(seg)})
    train = {"sec": round(sec, 1), "env_steps": int(model.num_timesteps),
             "env_steps_per_sec": round(model.num_timesteps / max(sec, 1e-9), 2),
             "n_updates": int(getattr(model, "_n_updates", 0)), "curve": curve, "episodes_logged": len(rows),
             "episode_log_columns": ["env_steps", "return", "length", "final_match", "sec"],
             "episode_log": rows}
    return model, train


def load_sb3(a, path, learner_device):
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C, "dqn": DQN}[a.algo]
    return cls.load(path, device=learner_device)


# ===================================================================================== BC / DAgger
def _bn_snapshot(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items() if "running" in k or "num_batches" in k}


def train_imitation(a, stem, learner_device, env_device):
    import torch
    from usbg.plug import controllers as C
    from usbg.policy import GoalPolicy
    hp = dict(HP[a.algo])
    n_env = int(a.imitation_steps or hp["imitation_env_steps"])
    n_fit = int(a.fit_steps or hp["fit_steps"])
    torch.manual_seed(a.seed)
    rng = np.random.default_rng([a.seed, 4242])
    env = C.make_env(**env_kwargs(a, a.train_cases, device=env_device, privileged=True, frame_dtype="uint8"))
    net = GoalPolicy.build(history=env.history).to(learner_device)
    store = C.EpisodeStore(env.history)
    log = {"rounds": []}
    t0 = time.time()
    if a.algo == "bc":
        tc = time.time()
        st = C.collect_episodes(env, store, n_env, beta=1.0, rng=rng, first_seed=a.seed)
        tcol = time.time() - tc
        tf = time.time()
        _, losses = C.fit_goal_net(net, store, n_fit, hp["batch_size"], hp["learning_rate"], learner_device, rng)
        log["rounds"].append({"round": 0, "beta": 1.0, **st, "collect_sec": round(tcol, 1),
                              "fit_steps": n_fit, "fit_sec": round(time.time() - tf, 1), "loss": losses,
                              "dataset_samples": len(store)})
    else:
        rounds = int(a.dagger_rounds if a.dagger_rounds is not None else hp["rounds"])
        per_env, per_fit = n_env // (rounds + 1), n_fit // (rounds + 1)
        opt = None
        for r in range(rounds + 1):
            tc = time.time()
            if r == 0:
                st = C.collect_episodes(env, store, per_env, beta=1.0, rng=rng, first_seed=a.seed)
            else:
                snap = _bn_snapshot(net)          # learner rollouts must not move the BatchNorm running stats
                st = C.collect_episodes(env, store, per_env, beta=0.0, learner=C.goal_net_learner(net, learner_device),
                                        rng=rng)
                net.load_state_dict({**net.state_dict(), **snap})
            tcol = time.time() - tc
            tf = time.time()
            opt, losses = C.fit_goal_net(net, store, per_fit, hp["batch_size"], hp["learning_rate"], learner_device,
                                         rng, opt=opt)
            log["rounds"].append({"round": r, "beta": 1.0 if r == 0 else 0.0, **st, "collect_sec": round(tcol, 1),
                                  "fit_steps": per_fit, "fit_sec": round(time.time() - tf, 1), "loss": losses,
                                  "dataset_samples": len(store)})
            print(f"[{a.algo}] round {r}: {st['env_steps']} env steps (teacher share {st['teacher_share']:.2f}, "
                  f"rolled final {st['mean_final_pos_mm']:.2f} mm), dataset {len(store)}, loss {losses[-1][1]:.4f}",
                  flush=True)
    sec = time.time() - t0
    env_steps = sum(r["env_steps"] for r in log["rounds"])
    meta = {"algo": a.algo, "history": env.history, "base": a.base, "residual_frac": a.residual_frac,
            "a_bound": [float(x) for x in env.a_bound], "env_steps": env_steps, "fit_steps": n_fit,
            "dataset_samples": len(store), "store_bytes": store.nbytes()}
    torch.save({"state_dict": net.state_dict(), "meta": meta}, stem + ".pt")
    env.close()
    train = {"sec": round(sec, 1), "env_steps": env_steps, "env_steps_per_sec": round(env_steps / max(sec, 1e-9), 2),
             **log, "dataset_samples": len(store), "store_bytes": store.nbytes()}
    return net, train


# ===================================================================================== evaluation
def evaluate_all(a, controllers, eval_device):
    """controllers: [(weights name, factory() -> controller, extra dict)]. One block per (controller,
    contact); the env is rebuilt per contact setting with the same everything else."""
    import torch
    from usbg.plug import controllers as C
    blocks, env_prov = [], None
    for contact in a.eval_contact:
        env = C.make_env(**env_kwargs(a, a.eval_cases, device=eval_device, privileged=True, frame_dtype="float32",
                                      contact=(contact == "on")))
        if env_prov is None:
            env_prov = env
        for name, factory, extra in controllers:
            ctrl = factory()
            te = time.time()
            with torch.no_grad():
                res = C.evaluate(env, ctrl, a.eval_episodes, log_episodes=not a.no_episodes)
            sec = time.time() - te
            print(f"[eval {a.algo}/{name} contact={contact}] start {res['start_pos']:.2f} -> final {res['pos']:.2f} mm "
                  f"/ {res['ang']:.2f} deg | keep-best {res['keepbest_pos']:.2f} | succ10 {res['success_10mm_10deg']:.2f} "
                  f"succ5 {res['success_5mm_5deg']:.2f} | S {res['static_match']:.3f}->{res['match']:.3f} | peak "
                  f"force {res['peak_force']:.2f} N | {sec:.0f}s", flush=True)
            blocks.append({"weights": name, "contact": contact, **extra, "sec": round(sec, 1), **res})
        if env is not env_prov:
            env.close()
    return blocks, env_prov


def main(argv=None):
    a = parse_args(argv)
    import torch
    if a.torch_threads:
        torch.set_num_threads(int(a.torch_threads))
    from usbg import provenance as PROV
    from usbg import view_match as VM
    from usbg.plug import controllers as C
    t_wall = time.time()
    a.out = os.path.abspath(a.out)
    stem = artifact_stem(a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    trains = a.algo in SB3_ALGOS + IMIT_ALGOS and a.model is None
    refuse_existing([a.out, a.out + ".tmp"] + ([stem + ".zip", stem + ".vecnormalize.pkl", stem + ".pt"]
                                              if trains else []))
    if a.algo == "cem" and VM._US is not None:
        raise SystemExit("cem scores by CT-mask S: run it with USBG_LEARNED_SEG=0")
    learner_device = resolve_device(a.learner_device)
    env_device, eval_device = resolve_device(a.env_device), resolve_device(a.eval_device)
    train, model_path, ctrls = None, None, []
    if a.algo in SB3_ALGOS:
        if a.model:
            model, model_path = load_sb3(a, a.model, learner_device), os.path.abspath(a.model)
        else:
            model, train = train_sb3(a, stem, learner_device, env_device)
            model_path = stem + ".zip"
        table = C.discrete_table(HP["dqn"]["step_frac"]) if a.algo == "dqn" else None
        ctrls.append(("final", lambda: C.SB3Controller(model, f"{a.algo}_s{a.seed}", table), {}))
    elif a.algo in IMIT_ALGOS + ("v2_policy",):
        from usbg.policy import GoalPolicy
        if a.algo == "v2_policy":
            model_path = os.path.abspath(a.policy_pt)
            ck = torch.load(model_path, map_location="cpu", weights_only=False)
            if ck.get("policy", "cnn") != "cnn":
                raise SystemExit(f"{model_path}: only the Gaussian GoalPolicy head is supported")
            if a.base == "servo":
                ib = ck.get("ibvs") or {}
                sb = torch.load(a.servo_base, map_location="cpu", weights_only=False).get("ibvs") or {}
                if not np.array_equal(np.asarray(ib.get("model")), np.asarray(sb.get("model"))):
                    raise SystemExit("v2_policy: --servo-base must be the policy's own calibrated base")
            sd, hist = ck["state_dict"], int(ck["history"])
        elif a.model:
            model_path = os.path.abspath(a.model)
            ck = torch.load(model_path, map_location="cpu", weights_only=False)
            sd, hist = ck["state_dict"], int(ck["meta"]["history"])
        else:
            net, train = train_imitation(a, stem, learner_device, env_device)
            model_path = stem + ".pt"
            sd, hist = {k: v.detach().cpu() for k, v in net.state_dict().items()}, net.enc[0][0].in_channels - 1

        def goal_factory(mode, _sd=sd, _h=hist):
            def f():
                n = GoalPolicy.build(history=_h)
                n.load_state_dict(copy.deepcopy(_sd))
                return C.GoalNetController(n.to(learner_device), learner_device, f"{a.algo}_bn{mode}", bn_mode=mode)
            return f
        for mode in a.bn_modes:
            ctrls.append((f"bn_{mode}", goal_factory(mode), {"bn_mode": mode}))
    elif a.algo == "random":
        ctrls.append(("random", lambda: C.RandomController(a.seed), {}))
    elif a.algo in ("static", "servo_base", "pose_oracle"):
        ctrls.append((a.algo, lambda: C.ZeroController(a.algo), {}))
    elif a.algo == "cem":
        ctrls.append(("cem", lambda: C.CEMController(a.seed, k=a.cem_k, rounds=a.cem_rounds, n_elite=a.cem_elite,
                                                     seed_offset=a.cem_seed_offset),
                      {"cem": {"k": a.cem_k, "rounds": a.cem_rounds, "n_elite": a.cem_elite,
                               "seed_offset": a.cem_seed_offset}}))
    blocks, env = evaluate_all(a, ctrls, eval_device)
    method_key = a.algo if a.algo in C.CITATIONS else {"v2_policy": "reinforce", "servo_base": "servo"}.get(a.algo)
    import gymnasium
    versions = {"torch": torch.__version__, "gymnasium": gymnasium.__version__}
    try:
        import stable_baselines3
        versions["stable_baselines3"] = stable_baselines3.__version__
    except Exception:
        pass
    servo_md5 = PROV.md5(a.servo_base) if a.base == "servo" else None
    prov = C.provenance_block(env, method=a.algo, extra={
        "versions": versions, "learner_device": learner_device, "env_device": env_device, "eval_device": eval_device,
        "servo_base": os.path.abspath(a.servo_base) if a.base == "servo" else None, "servo_base_md5": servo_md5,
        "model": model_path, "model_md5": PROV.md5(model_path), "smoke": bool(a.smoke)})
    prov.update(train_cases=list(a.train_cases), eval_cases=list(a.eval_cases), targets=list(a.targets),
                min_windows=a.min_windows, eval_seed=a.eval_seed, privileged=a.algo in ("pose_oracle", "cem"))
    out = {"slot": C.SLOT, "method": a.algo, "label": a.label or f"{a.algo}_{a.base}",
           "citation": C.CITATIONS.get(method_key),
           "sb3_citation": C.CITATIONS["sb3"] if a.algo in SB3_ALGOS else None,
           "rung_class": RUNG_CLASS.get(a.algo), "privileged": a.algo in ("pose_oracle", "cem"),
           "condition": {"texture": a.texture, "texture_ckpt": a.texture_ckpt, "mask": a.mask, "seg_ckpt": a.seg_ckpt,
                         "base": a.base, "residual_frac": a.residual_frac if a.base != "none" else None,
                         "servo_base": a.servo_base, "servo_base_md5": servo_md5, "seed": a.seed,
                         "render_seed": a.render_seed, "train_contact": "on", "randomize": "off"},
           "hyperparameters": {**COMMON_HP, "norm_reward": a.norm_reward == "on", **HP.get(a.algo, {}),
                               "n_envs": a.n_envs, "vec": a.vec, "overrides": list(a.hp)}
           if a.algo in SB3_ALGOS + IMIT_ALGOS else None,
           "budget": ({"env_steps": int(train["env_steps"]) if train else None, "requested": a.total_steps
                       if a.algo in SB3_ALGOS else (a.imitation_steps or HP[a.algo]["imitation_env_steps"]),
                       "paper_reference": PAPER_BUDGET} if a.algo in SB3_ALGOS + IMIT_ALGOS else None),
           "train": train, "results": blocks, "artifacts": {"model": model_path, "model_md5": PROV.md5(model_path)},
           "analysis_status": C.ANALYSIS_STATUS, "smoke": bool(a.smoke), "provenance": prov,
           "wall_sec": round(time.time() - t_wall, 1)}
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, a.out)
    print(f"wrote {a.out} ({out['wall_sec']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
