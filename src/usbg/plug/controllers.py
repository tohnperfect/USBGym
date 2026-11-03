"""usbg.plug.controllers: published policy-learning methods and reference controllers plugged into
USBGym/ViewMatch-v0 (v2.1 pluggability study; EXPLORATORY and OPT-IN).

Nothing in the v2 or v2.1 job queues imports this module. It is reached only through
``scripts/plug_train_sb3.py``. Every result it helps write carries ``analysis_status =
"exploratory, v2.1 pluggability"`` (ANALYSIS_STATUS).

What plugs in where
    The environment is ``usbg.plug.viewmatch_env.ViewMatchEnv`` (USBGym/ViewMatch-v0): the paper's
    observation (the last 4 frames and the target frame, 128 px), the paper's action box (a normalized
    body twist, scaled by MAX_TWIST without a base or by residual_frac * MAX_TWIST over the image-servo
    base), the paper's reward (the gain in S, which weights S_seg by 0.6 and S_app by 0.4, minus the effort term) and the paper's
    12-step episodes. Learning algorithms see only that interface; the wrappers below adapt it to a
    library without changing what the agent observes or does:

    StackedGoalObs         Dict(current, target) -> one uint8 image (history + 1, S, S): the current history
                           and the target as input channels, exactly the channel stack the paper's GoalPolicy
                           encoder reads. SB3's image policies (NatureCNN, Mnih et al. 2015) consume it.
    FiniteHorizonTerminal  the 12-step truncation becomes a termination, so every algorithm treats the
                           horizon as the end of the task, as the paper's REINFORCE does (returns over the
                           episode, no bootstrap past step 12). The observation carries no time index in
                           either case.
    DiscreteTwist          Discrete(13) -> the Box action: a no-op and +-step_frac on each of the six twist
                           axes, one axis per step (the discrete probe-motion sets of Hase et al. 2020 and
                           Li et al. 2021, adapted to this 6-DoF twist).

Controllers for the evaluator (``evaluate``): ``reset(ep, env)`` then ``act(obs, info) -> normalized action``.
    ZeroController        zero action: "static" without a base; the base alone with a base ("servo":
                          the paper's least-squares moment servo; "pose_oracle": privileged analytic base).
    RandomController      uniform actions in the box, a seeded stream per episode.
    SB3Controller         a trained Stable-Baselines3 model, deterministic (mean action).
    GoalNetController     a GoalPolicy net (the paper's encoder): this module's BC and DAgger policies and
                          the paper's own v2 REINFORCE policies (normalized action = tanh(net(cur, tgt))).
    CEMController         usbg.controllers_v21.LookaheadController (the privileged CEM look-ahead bound)
                          driven through the environment's simulator state. PRIVILEGED reference only.

Evaluation (``evaluate``) walks ``reset(options={"eval_episode": k})`` for k = 0..n-1: the paper's crossed
held-out design (setup seed eval_seed + k, cell k % n_cells patient-major, window k // n_cells), and writes
the per-episode keys and aggregates of ``train_view_match_rl.evaluate`` plus success at 5 mm / 5 deg and
10 mm / 10 deg (final and keep-best) and patient-level means. Pose errors come from ``info["privileged"]``
(evaluation only; no controller receives them except the privileged teacher and reference rungs).

Imitation (plain PyTorch, no extra package): ``teacher_action`` is the privileged teacher of the pose oracle:
the twist that corrects the true pose analytically, minus the base the environment adds, in normalized units.
``collect_episodes`` rolls a mixture of teacher and learner and stores every visited state with its teacher
label (beta = 1: behavior cloning on teacher trajectories; beta = 0: DAgger's on-policy relabeling, Ross,
Gordon and Bagnell 2011), and ``fit_goal_net`` regresses the GoalPolicy output on the labels.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except ImportError:                                   # keep the module importable without gymnasium
    gym, spaces, _HAVE_GYM = None, None, False

ANALYSIS_STATUS = "exploratory, v2.1 pluggability"
SLOT = "controller"
SUCCESS_TOLS = ((5.0, 5.0), (10.0, 10.0))

# Real, verifiable references for every method this module plugs in (used in the result JSON).
CITATIONS = {
    "ppo": "Schulman, Wolski, Dhariwal, Radford, Klimov, Proximal policy optimization algorithms, arXiv preprint arXiv:1707.06347, 2017",
    "sac": "Haarnoja, Zhou, Abbeel, Levine, Soft Actor-Critic, ICML 2018, PMLR 80:1861-1870, arXiv:1801.01290",
    "td3": "Fujimoto, van Hoof, Meger, Addressing Function Approximation Error in Actor-Critic Methods, ICML 2018, "
           "PMLR 80:1587-1596, arXiv:1802.09477",
    "a2c": "Mnih et al., Asynchronous Methods for Deep Reinforcement Learning, ICML 2016, PMLR 48:1928-1937, "
           "arXiv:1602.01783",
    "dqn": "Mnih et al., Human-level control through deep reinforcement learning, Nature 518 (2015) 529-533, "
           "doi:10.1038/nature14236; discrete probe actions as in Hase et al., IROS 2020, and Li et al., ICRA 2021",
    "bc": "Pomerleau, Efficient training of artificial neural networks for autonomous navigation, Neural Comput. 3 "
          "(1991) 88-97, doi:10.1162/neco.1991.3.1.88",
    "dagger": "Ross, Gordon, Bagnell, A Reduction of Imitation Learning and Structured Prediction to No-Regret Online "
              "Learning, AISTATS 2011, PMLR 15:627-635, arXiv:1011.0686",
    "cem": "Rubinstein, The cross-entropy method for combinatorial and continuous optimization, Methodol. Comput. "
           "Appl. Probab. 1 (1999) 127-190, doi:10.1023/A:1010091220143; de Boer, Kroese, Mannor, Rubinstein, "
           "Ann. Oper. Res. 134 (2005) 19-67, doi:10.1007/s10479-005-5724-z",
    "sb3": "Raffin, Hill, Gleave, Kanervisto, Ernestus, Dormann, Stable-Baselines3: Reliable Reinforcement Learning "
           "Implementations, JMLR 22 (2021) 268:1-8",
    "nature_cnn": "Mnih et al., Nature 518 (2015) 529-533 (the image encoder SB3 uses for image observations)",
    "reinforce": "Williams, Simple statistical gradient-following algorithms for connectionist reinforcement "
                 "learning, Mach. Learn. 8 (1992) 229-256, doi:10.1007/BF00992696 (the paper's own learner)",
    "servo": "own: least-squares linear map from organ-mask moment features to the corrective twist (the paper's "
             "image-servo base); moment features in the spirit of Mebarki, Krupa, Chaumette, IEEE TRO 26 (2010) "
             "296-306, doi:10.1109/TRO.2010.2042533",
}


# ===================================================================================== environment
def make_env(*, patients, texture="physics", texture_ckpt=None, mask="ct", seg_ckpt=None, base=None,
             servo_base=None, residual_frac=0.5, render_seed=0, contact=True, device=None,
             frame_dtype="float32", privileged_info=True, targets=("liver", "kidney"), min_windows=2,
             eval_seed=999, n_steps=12, history=4, image_size=128, w_seg=0.6, w_app=0.4, effort=0.003):
    """USBGym/ViewMatch-v0 with the paper's defaults (anatomical seat, liver and kidney, min 2 windows,
    12 steps, history 4, 128 px, S weighting S_seg by 0.6 and S_app by 0.4, effort 0.003, start offsets up to
    0.02 m lateral, 0.01 m axial and 18 deg in fan, DR off)."""
    from .viewmatch_env import ViewMatchEnv
    return ViewMatchEnv(patients=patients, targets=tuple(targets), texture=texture, texture_ckpt=texture_ckpt,
                        mask=mask, seg_ckpt=seg_ckpt, base=base, servo_base=servo_base,
                        residual_frac=residual_frac, contact=contact, randomize="off", render_seed=render_seed,
                        n_steps=n_steps, history=history, image_size=image_size, w_seg=w_seg, w_app=w_app,
                        effort=effort, min_windows=min_windows, eval_seed=eval_seed, include_masks=False,
                        frame_dtype=frame_dtype, privileged_info=privileged_info, device=device)


def stack_uint8(obs):
    """Dict(current (H,S,S), target (1,S,S)) in float32 [0,1] or uint8 -> uint8 (H+1, S, S). Lossless:
    the frames are 8-bit quantized by the environment (usbg.eval._prep)."""
    c, t = np.asarray(obs["current"]), np.asarray(obs["target"])
    x = np.concatenate([c, t], 0)
    if x.dtype != np.uint8:
        x = np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    return x


if _HAVE_GYM:
    class StackedGoalObs(gym.ObservationWrapper):
        """Dict(current, target) -> Box uint8 (history + 1, S, S) (see the module docstring)."""

        def __init__(self, env):
            super().__init__(env)
            cur = env.observation_space["current"]
            h, s = cur.shape[0], cur.shape[1]
            self.observation_space = spaces.Box(0, 255, (h + 1, s, s), np.uint8)

        def observation(self, obs):
            return stack_uint8(obs)

    class FiniteHorizonTerminal(gym.Wrapper):
        """terminated = True at the fixed horizon (the environment truncates there and never terminates)."""

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            if trunc and not term:
                term, trunc = True, False
                info = dict(info, horizon_end=True)
            return obs, r, term, trunc, info

    class DiscreteTwist(gym.ActionWrapper):
        """Discrete(13): 0 = no-op, 1 + 2i = +step_frac on axis i, 2 + 2i = -step_frac on axis i, for the axes
        (vx, vy, vz, wx, wy, wz) of the normalized twist."""

        def __init__(self, env, step_frac=1.0):
            super().__init__(env)
            self.table = discrete_table(step_frac)
            self.step_frac = float(step_frac)
            self.action_space = spaces.Discrete(len(self.table))

        def action(self, a):
            return self.table[int(a)]
else:                                                    # pragma: no cover
    StackedGoalObs = FiniteHorizonTerminal = DiscreteTwist = None


def discrete_table(step_frac=1.0):
    """(13, 6) float32: row 0 the no-op, then +-step_frac on each axis."""
    rows = [np.zeros(6, np.float32)]
    for i in range(6):
        for sgn in (1.0, -1.0):
            r = np.zeros(6, np.float32)
            r[i] = sgn * float(step_frac)
            rows.append(r)
    return np.stack(rows, 0)


# ===================================================================================== privileged teacher
def teacher_action(env):
    """The privileged pose-oracle teacher in the environment's normalized action units: the analytic
    corrective twist from the TRUE probe pose to the target pose (goal_rl.analytic_twist, clipped to
    MAX_TWIST), minus the base twist the environment adds at this state, divided by the action bound and
    clipped to [-1, 1]. Without a base it is the analytic twist itself; over the servo base it is the gap
    the base leaves (the paper's imitation target, train_view_match_rl.residual_target). Simulator-only."""
    from .. import goal_rl as G
    u = env.unwrapped
    full = G.analytic_twist(u._probe.world_from_probe, u._T_tgt)
    base = u._base_twist()
    return np.clip((full - base) / u.a_bound.astype(np.float64), -1.0, 1.0).astype(np.float32)


# ===================================================================================== controllers
class ZeroController:
    """Zero normalized action. Without a base: no commanded motion (static); with a base: the base alone."""
    privileged = False

    def __init__(self, name="zero", observes="nothing (zero action); the base, if any, reads the masks"):
        self.name, self.observes = name, observes

    def reset(self, ep, env):
        pass

    def act(self, obs, info):
        return np.zeros(6, np.float32)


class RandomController:
    """Uniform normalized actions in [-1, 1]^6; stream default_rng([seed, 7717, ep]) per episode."""
    privileged = False
    observes = "nothing (uniform random actions)"

    def __init__(self, seed):
        self.seed = int(seed)
        self.name = f"random_s{self.seed}"

    def reset(self, ep, env):
        self.rng = np.random.default_rng([self.seed, 7717, int(ep)])

    def act(self, obs, info):
        return self.rng.uniform(-1.0, 1.0, 6).astype(np.float32)


class SB3Controller:
    """A trained Stable-Baselines3 model on the StackedGoalObs image, deterministic (mean / greedy action).
    ``table`` maps a discrete action index to the normalized twist (DQN)."""
    privileged = False
    observes = "the current 4-frame history and the target frame (one stacked uint8 image)"

    def __init__(self, model, name, table=None):
        self.model, self.name, self.table = model, name, table

    def reset(self, ep, env):
        pass

    def act(self, obs, info):
        a, _ = self.model.predict(stack_uint8(obs), deterministic=True)
        if self.table is not None:
            return self.table[int(np.asarray(a).reshape(-1)[0])]
        return np.clip(np.asarray(a, np.float32).reshape(6), -1.0, 1.0)


class GoalNetController:
    """A GoalPolicy net (usbg.policy.GoalPolicy: the paper's encoder over the current history stacked with
    the target): normalized action = tanh(net(cur, tgt)). For a v2 REINFORCE policy this equals its
    deterministic residual divided by its residual bound. bn_mode "train" keeps BatchNorm on per-call
    statistics at B = 1 as the paper's evaluation does (disclosed there); "eval" uses the running statistics."""
    privileged = False
    observes = "the current 4-frame history and the target frame"

    def __init__(self, net, device, name, bn_mode="train"):
        self.net, self.device, self.name, self.bn_mode = net, device, name, bn_mode
        if bn_mode == "train":
            self.net.train()
        elif bn_mode == "eval":
            self.net.eval()
        else:
            raise ValueError(f"bn_mode {bn_mode!r}")

    def reset(self, ep, env):
        pass

    def act(self, obs, info):
        import torch
        cur = torch.as_tensor(np.asarray(obs["current"], np.float32)[None], device=self.device)
        tgt = torch.as_tensor(np.asarray(obs["target"], np.float32)[None], device=self.device)
        if cur.max() > 1.0:                               # uint8 frames
            cur, tgt = cur / 255.0, tgt / 255.0
        with torch.no_grad():
            return torch.tanh(self.net(cur, tgt))[0].float().cpu().numpy().astype(np.float32)


class CEMController:
    """The PRIVILEGED CEM look-ahead bound of usbg.controllers_v21 (LookaheadController: K candidate twists
    per round, rounds, elites; a cloned MuJoCo state runs, renders and scores each candidate with the
    CT-mask S, against a target image drawn with an independent speckle seed), driven through this
    environment's simulator state. The target pose is never read. Requires base=None (the action is the
    full twist) and mask="ct". Its twist is returned in normalized units (twist / MAX_TWIST, float32)."""
    privileged = True
    observes = ("PRIVILEGED: clones the MuJoCo state and renders hypothetical poses from the CT; CT-mask S "
                "against an independently seeded target view")

    def __init__(self, seed, k=64, rounds=2, n_elite=8, seed_offset=7907):
        from .. import controllers_v21 as C21
        self.inner = C21.LookaheadController(seed, k=k, rounds=rounds, n_elite=n_elite, seed_offset=seed_offset)
        self.name = "cem_" + self.inner.name
        self.cfg = {"k": k, "rounds": rounds, "n_elite": n_elite, "seed_offset": seed_offset, "seed": seed}

    def reset(self, ep, env):
        u = env.unwrapped
        if u.base is not None or u.pipeline.mask_name != "ct":
            raise ValueError("CEMController needs base=None and mask='ct' (it scores by CT-mask S)")
        self.env = env
        self.inner.reset(ep)

    def act(self, obs, info):
        from .. import controllers_v21 as C21
        from .. import view_match as VM
        u = self.env.unwrapped
        tr, lv, T_tgt, Wfv, S = u.pipeline.translator, u._lv, u._T_tgt, u._Wfv, u.image_size
        sim = C21.SimAccess(env=u._probe, lv=lv, Wfv=Wfv, translator=tr, cfg=u._cfg_ep, image_size=S,
                            w_seg=u.w_seg, w_app=u.w_app,
                            render_target=lambda cf: VM.curvi_render(lv, T_tgt, Wfv, tr, cf, S))
        o = C21.Obs(t=u._t, n_steps=u.n_steps, cur_stack=None, cur_img=u._img, cur_mask=u._mask,
                    tgt_img=u._tgt_img, tgt_mask=u._tgt_mask, tgt_prep=u._tgt_prep, tgt_t=None, s=float(u._s),
                    s_seg=float(info["seg_iou"]), s_app=float(info["appearance"]),
                    force=float(info["contact_force"]), odo_cmd=None)
        tw, self.last_info = self.inner.act(o, sim)
        return (np.asarray(tw, np.float64) / u.a_bound.astype(np.float64)).astype(np.float32)

    def episode_summary(self):
        return self.inner.episode_summary()


# ===================================================================================== evaluation
def _window_info(lv, organ, w):
    try:
        from .. import windows as W
        from .viewmatch_env import ORGANS
        m = W.windows_meta(lv, ORGANS[organ])[int(w)]
        return {"window_organ": m.get("organ"), "window_id": m.get("window_id"), "window_phi": m.get("phi"),
                "window_tilt_deg": m.get("tilt_deg"), "window_th": m.get("th")}
    except Exception:
        return {}


def evaluate(env, controller, n_episodes=60, log_episodes=True, step_hook=None):
    """The paper's crossed held-out evaluation through ``reset(options={"eval_episode": k})``.

    Per episode the keys of train_view_match_rl.evaluate (start / final / keep-best translation and rotation
    error, seg and match, peak force, contact fraction, trajectories) plus the rotation trajectory, the
    return, and success at 5 mm / 5 deg and 10 mm / 10 deg (final and keep-best). Keep-best is the step of
    maximal non-privileged S (pose-free). Aggregates: the trainer's episode means, the success rates and
    patient-level means (mean over patients of per-patient means)."""
    u = env.unwrapped
    if not u.privileged_info:
        raise ValueError("evaluate needs privileged_info=True (pose errors are read from info['privileged'])")
    eps = []
    t_start = time.time()
    for k in range(int(n_episodes)):
        obs, info = env.reset(options={"eval_episode": k})
        controller.reset(k, env)
        P = info["privileged"]
        p0, a0 = float(P["pose_error_mm"]), float(P["rotation_error_deg"])
        rec = dict(ep=k, patient=info["patient"], target=info["target_organ"], window=info["window"],
                   n_windows=info["n_windows"], start_clamped=bool(info["start_clamped"]),
                   clamp_mm=float(info["clamp_mm"]), start_pos=p0, start_ang=a0,
                   static_seg=float(info["seg_iou"]), static_match=float(info["match"]),
                   **_window_info(u._lv, info["target_organ"], info["window"]))
        pos_tr, ang_tr, match_tr, seg_tr = [p0], [a0], [float(info["match"])], [float(info["seg_iou"])]
        forces, ret, acts = [], 0.0, []
        for t in range(u.n_steps):
            a = np.asarray(controller.act(obs, info), np.float32).reshape(6)
            acts.append(a)
            obs, r, term, trunc, info = env.step(a)
            ret += float(r)
            P = info["privileged"]
            pos_tr.append(float(P["pose_error_mm"])); ang_tr.append(float(P["rotation_error_deg"]))
            seg_tr.append(float(info["seg_iou"])); match_tr.append(float(info["match"]))
            forces.append(float(info["contact_force"]))
            if step_hook is not None:
                step_hook(k, t, a, info)
        kb = int(np.argmax(match_tr))
        rec.update(final_pos=pos_tr[-1], final_ang=ang_tr[-1], final_seg=seg_tr[-1], final_match=match_tr[-1],
                   keepbest_step=kb, keepbest_pos=pos_tr[kb], keepbest_ang=ang_tr[kb], keepbest_seg=seg_tr[kb],
                   peak_force=float(max(forces) if forces else 0.0),
                   contact_frac=float(np.mean([f > 0 for f in forces])) if forces else 0.0,
                   pos_traj=[round(x, 3) for x in pos_tr], ang_traj=[round(x, 3) for x in ang_tr],
                   match_traj=[round(x, 4) for x in match_tr], ret=ret,
                   mean_abs_action=[round(float(v), 4) for v in np.mean(np.abs(np.asarray(acts)), 0)])
        for mm, deg in SUCCESS_TOLS:
            tag = f"{int(mm)}mm_{int(deg)}deg"
            rec[f"success_{tag}"] = bool(pos_tr[-1] <= mm and ang_tr[-1] <= deg)
            rec[f"kb_success_{tag}"] = bool(pos_tr[kb] <= mm and ang_tr[kb] <= deg)
        if hasattr(controller, "episode_summary"):
            rec["controller"] = controller.episode_summary()
        eps.append(rec)
    return summarize(eps, u, controller, time.time() - t_start, log_episodes)


METRIC_KEYS = {"pos": "final_pos", "ang": "final_ang", "seg": "final_seg", "match": "final_match",
               "start_pos": "start_pos", "start_ang": "start_ang", "static_seg": "static_seg",
               "static_match": "static_match", "keepbest_pos": "keepbest_pos", "keepbest_ang": "keepbest_ang",
               "keepbest_seg": "keepbest_seg", "peak_force": "peak_force"}


def summarize(eps, u, controller, sec, log_episodes=True):
    res = {k: float(np.mean([e[v] for e in eps])) for k, v in METRIC_KEYS.items()}
    res["start_clamped_frac"] = float(np.mean([e["start_clamped"] for e in eps]))
    res["clamp_mm_mean"] = float(np.mean([e["clamp_mm"] for e in eps]))
    res["contact_frac"] = float(np.mean([e["contact_frac"] for e in eps]))
    res["return"] = float(np.mean([e["ret"] for e in eps]))
    succ_keys = [k for k in eps[0] if k.startswith("success_") or k.startswith("kb_success_")]
    for k in succ_keys:
        res[k] = float(np.mean([e[k] for e in eps]))
    pats = sorted({e["patient"] for e in eps})
    pl = {}
    for k in ["final_pos", "final_ang", "keepbest_pos", "keepbest_ang", "final_match", "final_seg",
              "peak_force", *succ_keys]:
        pl[k] = float(np.mean([np.mean([e[k] for e in eps if e["patient"] == p]) for p in pats]))
    res["patient_level"] = pl

    def _group(key, other):
        g = {}
        for e in eps:
            g.setdefault(e[key], []).append(e)
        return {k: {"n": len(L), "pos": float(np.mean([e["final_pos"] for e in L])),
                    "ang": float(np.mean([e["final_ang"] for e in L])),
                    "start_pos": float(np.mean([e["start_pos"] for e in L])),
                    other + "s": sorted({e[other] for e in L})} for k, L in g.items()}
    res["per_patient"] = _group("patient", "target")
    res["per_target"] = _group("target", "patient")
    res.update(n_episodes=len(eps), design="crossed (ViewMatch-v0 eval_episode k)", eval_seed=u.eval_seed,
               contact="on" if u.contact else "off", seat="anatomical", n_cells=len(u.cells),
               patients=list(u.patients), controller={"name": controller.name,
                                                      "privileged": bool(controller.privileged),
                                                      "observes": getattr(controller, "observes", None)},
               eval_sec=round(sec, 1), eval_sec_per_episode=round(sec / max(len(eps), 1), 3))
    if log_episodes:
        res["episodes"] = eps
    return res


# ===================================================================================== BC and DAgger
class EpisodeStore:
    """Visited states with their teacher labels, stored per episode as uint8 frames (the start frame and
    one frame per step) plus the target frame, so a sample's 4-frame history is rebuilt exactly as the
    environment's deque builds it (the start frame repeated at the front)."""

    def __init__(self, history=4):
        self.history = int(history)
        self.frames, self.targets, self.labels = [], [], []
        self.index = []                                    # (episode, t)

    def add(self, frames, target, labels):
        e = len(self.frames)
        self.frames.append(np.asarray(frames, np.uint8))
        self.targets.append(np.asarray(target, np.uint8))
        self.labels.append(np.asarray(labels, np.float32))
        self.index += [(e, t) for t in range(len(labels))]

    def __len__(self):
        return len(self.index)

    def stack(self, e, t):
        f = self.frames[e]
        idx = [max(0, t - self.history + 1 + j) for j in range(self.history)]
        return f[idx]

    def batch(self, rng, n):
        sel = rng.integers(0, len(self.index), int(n))
        cur = np.stack([self.stack(*self.index[i]) for i in sel], 0)
        tgt = np.stack([self.targets[self.index[i][0]][None] for i in sel], 0)
        y = np.stack([self.labels[self.index[i][0]][self.index[i][1]] for i in sel], 0)
        return cur, tgt, y

    def nbytes(self):
        return int(sum(f.nbytes for f in self.frames) + sum(t.nbytes for t in self.targets))


def _last_frame_u8(obs):
    x = np.asarray(obs["current"][-1])
    return x if x.dtype == np.uint8 else np.rint(np.clip(x, 0, 1) * 255.0).astype(np.uint8)


def _target_u8(obs):
    x = np.asarray(obs["target"][0])
    return x if x.dtype == np.uint8 else np.rint(np.clip(x, 0, 1) * 255.0).astype(np.uint8)


def collect_episodes(env, store, n_env_steps, beta, learner=None, rng=None, first_seed=None):
    """Roll episodes until n_env_steps steps were taken; at every visited state store the teacher label.
    The executed action is the teacher's with probability beta (per step), else the learner's
    (``learner(obs) -> normalized action``). Returns a stats dict (episodes, steps, teacher share, mean
    final translation error of the rolled episodes)."""
    rng = rng or np.random.default_rng(0)
    steps, n_ep, n_teacher, finals = 0, 0, 0, []
    while steps < n_env_steps:
        obs, info = env.reset(seed=first_seed) if (first_seed is not None and n_ep == 0) else env.reset()
        frames, labels = [_last_frame_u8(obs)], []
        tgt = _target_u8(obs)
        u = env.unwrapped
        for t in range(u.n_steps):
            y = teacher_action(env)
            use_teacher = (learner is None) or (rng.random() < beta)
            a = y if use_teacher else np.asarray(learner(obs), np.float32)
            n_teacher += int(use_teacher)
            labels.append(y)
            obs, r, term, trunc, info = env.step(a)
            frames.append(_last_frame_u8(obs))
            steps += 1
            if term or trunc:
                break
        store.add(frames, tgt, labels)
        n_ep += 1
        if "privileged" in info:
            finals.append(float(info["privileged"]["pose_error_mm"]))
    return {"episodes": n_ep, "env_steps": steps, "teacher_share": n_teacher / max(steps, 1),
            "mean_final_pos_mm": float(np.mean(finals)) if finals else None}


def fit_goal_net(net, store, steps, batch, lr, device, rng, opt=None, log_every=200):
    """MSE regression of tanh(net(cur, tgt)) on the normalized teacher labels (Adam). Returns (opt, losses)."""
    import torch
    net.train()
    opt = opt or torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    for i in range(int(steps)):
        cur, tgt, y = store.batch(rng, batch)
        cur_t = torch.as_tensor(cur, device=device).float() / 255.0
        tgt_t = torch.as_tensor(tgt, device=device).float() / 255.0
        y_t = torch.as_tensor(y, device=device)
        loss = ((torch.tanh(net(cur_t, tgt_t)) - y_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % log_every == 0 or i == int(steps) - 1:
            losses.append([i, float(loss)])
    return opt, losses


def goal_net_learner(net, device):
    """obs -> normalized action from a GoalPolicy net in train-mode BatchNorm (the mode it is trained in)."""
    import torch

    def f(obs):
        cur = torch.as_tensor(np.asarray(obs["current"], np.float32)[None], device=device)
        tgt = torch.as_tensor(np.asarray(obs["target"], np.float32)[None], device=device)
        if cur.max() > 1.0:
            cur, tgt = cur / 255.0, tgt / 255.0
        with torch.no_grad():
            return torch.tanh(net(cur, tgt))[0].float().cpu().numpy()
    return f


# ===================================================================================== provenance
def provenance_block(env, *, method, extra=None):
    """usbg.provenance.collect plus the environment's own fingerprint, with the mask and texture entries
    the v2.1 runner reads (seat.resolved, texture.md5, mask.seg_md5, windows.md5). The learned mask is
    applied by the environment's pipeline (USBG_LEARNED_SEG stays 0 for these jobs), so the mask entry is
    taken from the pipeline, not from the process-wide view_match source."""
    from .. import provenance as PROV
    u = env.unwrapped
    pi = u.pipeline.info()
    prov = PROV.collect(stage=pi["texture"], texture_ckpt=pi["texture_ckpt"])
    prov["mask"] = {"learned": pi["mask"] == "learned", "mask_source": pi["mask"], "seg_ckpt": pi["seg_ckpt"],
                    "seg_md5": pi["seg_md5"], "USBG_LEARNED_SEG": os.environ.get("USBG_LEARNED_SEG"),
                    "applied_by": "usbg.plug.viewmatch_env.ViewPipeline"}
    prov["analysis_status"] = ANALYSIS_STATUS
    prov["slot"] = SLOT
    prov["method"] = method
    prov["env"] = env_fingerprint(u)
    prov["extra"] = extra or {}
    return prov


def env_fingerprint(u):
    p = u.provenance()
    p.pop("git", None)
    return json.loads(json.dumps(p, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


__all__ = ["ANALYSIS_STATUS", "CITATIONS", "make_env", "stack_uint8", "StackedGoalObs", "FiniteHorizonTerminal",
           "DiscreteTwist", "discrete_table", "teacher_action", "ZeroController", "RandomController",
           "SB3Controller", "GoalNetController", "CEMController", "evaluate", "summarize", "EpisodeStore",
           "collect_episodes", "fit_goal_net", "goal_net_learner", "provenance_block"]
