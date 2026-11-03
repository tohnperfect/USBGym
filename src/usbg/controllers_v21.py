"""controllers_v21.py: v2.1 (EXPLORATORY) extended-logging evaluator and reference rungs.

Everything here is OPT-IN. scripts/eval_view_ladder.py routes an evaluation through this module only
when a v2.1 flag is given (--ext-log, a v2.1 rung, --bn-mode eval, --n-steps-override). Without those
flags the v2 path (train_view_match_rl.evaluate) runs unchanged, so every archived v2 number stays
reproducible. Archived v2 numbers stay primary; every output written through this module carries
provenance.analysis_status = "exploratory, v2.1" (ANALYSIS_STATUS).

1. evaluate_ext(T, ...) is a step-for-step replica of train_view_match_rl.evaluate: the same crossed
   (patient x target x window) episodes, the same RNG streams (default_rng(seed + ep) for the setup and
   the env), the same renders, the same pose-free keep-best, the same per-episode keys and the same
   top-level aggregates. The extra logging only READS state (pose, labels, contact), so a legacy rung
   gives the same numbers through either evaluator (checked by the v2.1 smoke test). It adds, per
   episode and step (index 0 = the start, then one entry per executed step):
     pos / ang          translation norm of the SE(3) log (mm) and the geodesic rotation angle (deg),
                        exactly view_match.pose_error;
     axes               signed per-axis error in the TARGET probe frame, from the same SE(3) log:
                        lateral (probe x, along the array, in plane), elevational (probe y, out of
                        plane), axial (probe z, along the beam) in mm; roll (about the beam axis z),
                        in-plane tilt (about y, rocking) and out-of-plane tilt (about x, fanning) in
                        deg. sqrt(lat^2 + elev^2 + ax^2) = pos and the rotation-vector norm = ang;
     euclid             Euclidean probe-face-centre distance (mm) and its target-frame components;
     S, S_seg, S_app    the non-privileged score and its two terms (S = w_seg S_seg + w_app S_app);
     S_seg fallback     True when neither mask has any MATCH_ORGANS organ with >= 20 px, so
                        seg_organ_iou returns 1.0 by convention;
     seg_ct             organ IoU of the CT label slices (current vs target): a privileged diagnostic
                        that differs from S_seg only under learned masks;
     force              total probe-skin normal contact force (N);
     gap                height of the probe face above the skin plane z = 0 (mm): face centre and the
                        lowest of the centre and four face corners (the clamp_start_outside_skin
                        geometry). > 0 = hovering (the renderer still images through 'gel'),
                        < 0 = pressed in; v2.1.2: under the opt-in skin-tangent surface it is measured
                        along the normal of the env's contact plane;
     cmd_lag            distance between the commanded (mocap) and actual probe pose (mm);
     twist              the executed 6-DoF twist; plus each controller's own per-step diagnostics.
   The arrays go to an npz sidecar next to the result JSON (save_traj_npz); the JSON keeps the v2
   per-episode keys plus ang_traj and final-step scalars.

2. Controllers and what each may OBSERVE (enforced by construction: a non-privileged controller's act()
   receives only an Obs).
   Obs (non-privileged, pose-free): the current B-mode and its 'history' stack, the current mask from the
   run's mask source (CT label slice under USBG_LEARNED_SEG=0, the learned segmenter otherwise), the
   target image and target mask, the non-privileged score S of current vs target and its two terms, the
   contact force, and ODOMETRY = the robot's commanded probe pose relative to the episode start
   (inv(C_0) @ C_t, kinematics only; never the pose relative to the anatomy or to the target).
     PolicyBaseController  legacy rungs (policy, ibvs_base, static, scripted, pose_oracle, cnn_base):
                           a + base_twist(...) exactly as train_view_match_rl.evaluate. Observes what
                           its base observes: 'image' mode the masks, 'cnn' mode the image history,
                           'oracle' mode the privileged true and target poses (pose_oracle only).
     SGuardController      non-privileged. The linear servo step u = clip(g * base(mask, target mask)); if
                           the next observed S drops below S_ref - eps, the move is undone through
                           odometry on the following step and g is halved (never raised). eps is a
                           TRAIN-fixed hyperparameter (pilot on TRAIN patients, select_param).
     HillClimbController   non-privileged (1+1)-ES on S over twists: offspring = parent pose composed with
                           exp(d * DT_C), d = MAX_TWIST * clip(sigma * z, -1, 1), z ~ N(0, I6); the move
                           from the current commanded pose is executed through odometry (clipped to
                           the action bounds) and S is read at the pose actually reached. Success
                           (S > S_parent) makes the reached pose the parent, sigma *= 1.5; failure
                           sigma *= 1.5 ** -0.25 (the 1/5th rule; sigma in [0.02, 1.0]). The LAST step
                           returns to the best-S parent via inverse odometry (keep-best return), or
                           holds if already there. sigma0 is TRAIN-fixed (pilot, select_param).
   SimAccess (PRIVILEGED; never available on a real probe): a clone of the MuJoCo state and the CT
   volume, so a controller can render HYPOTHETICAL poses it has not visited.
     LookaheadController   PRIVILEGED REFERENCE RUNG, never a controller. At each step a cross-entropy
                           search runs 2 rounds of K = 64 candidate twists: round 1 holds the zero
                           twist, the twist executed last and 62 uniform draws inside the action box;
                           round 2: 64 Gaussian draws around the 8 best of round 1, std floor
                           0.05 * MAX_TWIST, clipped to the box). A CLONED MuJoCo state runs every
                           candidate (press limit, weld and contact settle as in env.step); the clone
                           renders the pose that the candidate reaches, and S with CT masks compares
                           that render with a target image drawn with an INDEPENDENT speckle seed
                           (cfg.seed + look_seed_offset), so identical speckle cannot make S = 1 near the
                           target. The best of the 128 is then executed on the real state (deterministic,
                           so it reaches the predicted pose). The target pose is never read. CT masks
                           only (USBG_LEARNED_SEG=0).

3. base_fit_report: per-DoF R^2, OLS gain (shrinkage) and correlation of the image servo base's
   twist against the analytic twist, on fresh TRAIN pairs and on held-out pairs.
4. select_param: the TRAIN-pilot selection rule for eps and sigma0 (lowest mean final translation
   error, contact on, DR off, averaged over the given pilot files; ties to the smaller value). Pilot
   files must evaluate TRAIN patients only.
"""
from __future__ import annotations

import copy
import json
import os
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

ANALYSIS_STATUS = "exploratory, v2.1"
BN_MODE_EVAL = ("eval (BatchNorm uses the running statistics saved with the policy; v2.1 exploratory "
                "arm beside the v2 train-mode default)")
AXES = ("lat_mm", "elev_mm", "ax_mm", "tilt_out_deg", "tilt_in_deg", "roll_deg")
AXES_DOC = ("signed SE(3)-log error in the target probe frame: lat = probe x (along the array, in "
            "plane), elev = probe y (out of plane), ax = probe z (beam); tilt_out = rotation about x "
            "(out-of-plane fanning), tilt_in = about y (in-plane rocking), roll = about z (beam axis)")
SUCCESS_TOLS = ((5.0, 5.0), (10.0, 10.0))


# ----------------------------------------------------------------------------------- geometry helpers
def pose_axes(T_cur, T_tgt):
    """-> (axes 6-vector in AXES order, euclid_xyz_mm 3-vector) for the current pose in the target
    probe frame. axes[:3] are the SE(3)-log translation (so their norm is pose_error's mm) and axes[3:]
    the rotation vector in deg (norm = pose_error's geodesic angle)."""
    from .contracts_bridge import invert_T, se3_log
    rel_T = invert_T(np.asarray(T_tgt, np.float64)) @ np.asarray(T_cur, np.float64)
    xi = se3_log(rel_T)
    ax = np.concatenate([np.asarray(xi[:3]) * 1000.0, np.degrees(np.asarray(xi[3:]))])
    return ax, np.asarray(rel_T[:3, 3]) * 1000.0


def face_gap_mm(T, plane=None):
    """(face-centre height, lowest of centre + 4 face corners) above the skin plane z = 0, in mm. v2.1.2: with
    plane = (p, n) (the env's opt-in contact_plane) the heights are measured along n above that plane."""
    from .placement import PROBE_HALF
    if plane is not None:
        from .placement import plane_gap_m
        gc, gm = plane_gap_m(T, plane[0], plane[1])
        return float(gc * 1000.0), float(gm * 1000.0)
    bx, by, _ = PROBE_HALF
    C = np.array([[0, 0, 0, 1.0]] + [[sx * bx, sy * by, 0.0, 1.0] for sx in (-1, 1) for sy in (-1, 1)])
    z = (np.asarray(T, np.float64) @ C.T)[2]
    return float(z[0] * 1000.0), float(z.min() * 1000.0)


def twist_between(C_from, C_to):
    """Body twist that moves the commanded pose C_from to C_to in one step, clipped to the action bounds
    (both poses may be odometry-frame poses: only their relative transform is used)."""
    from .contracts_bridge import invert_T, se3_log, DT_C
    from .goal_rl import MAX_TWIST
    xi = se3_log(invert_T(np.asarray(C_from, np.float64)) @ np.asarray(C_to, np.float64))
    return np.clip(np.asarray(xi) / DT_C, -MAX_TWIST, MAX_TWIST)


def ct_label_at(lv, T, Wfv, size=128):
    """The CT label slice in the curvilinear frame at pose T: exactly the lab_sec that
    view_match.curvi_render conditions and (under CT masks) returns, without rendering an image."""
    from . import slicer as S
    from .segmenter import align_label_to_image
    from .view_match import warp_to_sector
    lab_lin = S.slice_at_pose(lv, np.asarray(T, np.float64), Wfv)
    lab_sq = align_label_to_image(lab_lin, (size, size))
    _, lab_sec = warp_to_sector(np.zeros((size, size), np.float32), lab_sq)
    return np.asarray(lab_sec, np.int64)


def seg_fallback(lab_a, lab_b, min_pixels=20):
    """True when seg_organ_iou falls back to 1.0 (no MATCH_ORGANS organ with >= min_pixels in either)."""
    from .view_match import MATCH_ORGANS
    a = np.asarray(lab_a); b = np.asarray(lab_b)
    for o in MATCH_ORGANS:
        if (a == o).sum() >= min_pixels or (b == o).sum() >= min_pixels:
            return False
    return True


# ------------------------------------------------------------------------------- observation contract
@dataclass
class Obs:
    """What a NON-PRIVILEGED controller may read at step t (all pose-free)."""
    t: int
    n_steps: int
    cur_stack: np.ndarray        # (history, H, W) prepped image history, current frame last
    cur_img: np.ndarray          # current curvilinear B-mode in [0, 1]
    cur_mask: np.ndarray         # current mask from the run's mask source (CT label or learned)
    tgt_img: np.ndarray
    tgt_mask: np.ndarray
    tgt_prep: np.ndarray
    tgt_t: object                # torch tensor of tgt_prep (1, 1, H, W) on the device
    s: float                     # non-privileged S(current, target)
    s_seg: float
    s_app: float
    force: float                 # contact force (N): a force-sensor reading
    odo_cmd: np.ndarray          # 4x4 commanded probe pose relative to the episode start (odometry)


@dataclass
class SimAccess:
    """PRIVILEGED access: the live env (to clone its MuJoCo state), the CT volume and the renderer, so a
    controller can render hypothetical poses. It holds no target pose."""
    env: object
    lv: object
    Wfv: np.ndarray
    translator: object
    cfg: object
    image_size: int
    w_seg: float
    w_app: float
    render_target: object        # callable(cfg) -> (img, label) of the target view under another cfg


@dataclass
class LegacyAccess:
    """What the legacy residual bases read: the true current pose and the target pose. Only the
    'oracle' residual mode (pose_oracle rung) actually uses them; 'image' and 'cnn' modes ignore both."""
    world_from_probe: np.ndarray
    T_tgt: np.ndarray


class PolicyBaseController:
    """Legacy rungs, numerically identical to train_view_match_rl.evaluate:
    a_exec = clip(policy(cur, tgt) + base_twist(pose, T_tgt, cur_lab, tgt_lab, stack, tgt_prep, residual))."""
    privileged = False
    legacy = True

    def __init__(self, T, policy, residual, device, name="policy", privileged=False):
        self.T, self.policy, self.residual, self.device, self.name = T, policy, residual, device, name
        self.privileged = bool(privileged)
        mode = T._RESIDUAL_MODE if residual else "none"
        self.observes = {"oracle": "PRIVILEGED: true current pose and target pose (analytic base) + image history",
                         "image": "masks (current, target) for the base + image history (policy net)",
                         "cnn": "image history and target image (CNN appearance base) + image history",
                         "none": "image history and target image (policy net only)"}[mode]

    def reset(self, ep):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, obs, legacy):
        import torch
        from . import goal_rl as G
        cur = torch.from_numpy(obs.cur_stack[None]).to(self.device)
        a, _ = self.policy.act(cur, obs.tgt_t, deterministic=True)
        a_exec = np.clip(a[0].detach().cpu().numpy()
                         + self.T.base_twist(legacy.world_from_probe, legacy.T_tgt, obs.cur_mask, obs.tgt_mask,
                                             obs.cur_stack, obs.tgt_prep, self.residual), -G.MAX_TWIST, G.MAX_TWIST)
        return a_exec, {}

    def episode_summary(self):
        return {}


class SGuardController:
    """S-guarded linear servo. Non-privileged: masks, S and odometry only."""
    privileged = False
    legacy = False
    observes = ("current and target masks (linear servo base), non-privileged S, odometry (commanded pose "
                "relative to the start) for the undo")

    def __init__(self, T, eps, gain0=1.0, shrink=0.5):
        self.T, self.eps, self.gain0, self.shrink = T, float(eps), float(gain0), float(shrink)
        self.name = f"sguard_eps{self.eps:g}"

    def reset(self, ep):
        self.gain, self.s_ref, self.c_ref, self.last = self.gain0, None, None, None
        self.n_reject, self.n_move = 0, 0

    def act(self, obs):
        from . import goal_rl as G
        info = {"gain": self.gain, "reject": 0.0, "undo": 0.0}
        if self.s_ref is None or self.last == "undo":
            self.s_ref, self.c_ref = obs.s, obs.odo_cmd.copy()      # (re-)observed reference pose
        elif self.last == "move":
            if obs.s < self.s_ref - self.eps:                       # S dropped: undo via odometry
                self.gain *= self.shrink
                self.n_reject += 1
                self.last = "undo"
                info.update(reject=1.0, undo=1.0, gain=self.gain)
                return twist_between(obs.odo_cmd, self.c_ref), info
            self.s_ref, self.c_ref = obs.s, obs.odo_cmd.copy()      # accepted
        u = np.clip(self.gain * self.T.image_base_twist(obs.cur_mask, obs.tgt_mask), -G.MAX_TWIST, G.MAX_TWIST)
        self.last = "move"
        self.n_move += 1
        return u, info

    def episode_summary(self):
        return {"n_reject": self.n_reject, "n_move": self.n_move, "final_gain": self.gain}


class HillClimbController:
    """Hill climbing on S by a (1+1)-ES, with the 1/5th rule and a keep-best return. Non-privileged."""
    privileged = False
    legacy = False
    observes = "non-privileged S (current vs target images and masks) and odometry; no servo base, no pose"
    SUCC, FAIL_POW, SIG_MIN, SIG_MAX = 1.5, -0.25, 0.02, 1.0

    def __init__(self, sigma0, seed):
        self.sigma0, self.seed = float(sigma0), int(seed)
        self.name = f"hillclimb_sigma{self.sigma0:g}"

    def reset(self, ep):
        self.rng = np.random.default_rng([self.seed, 7411, int(ep)])
        self.sigma, self.parent, self.s_par, self.pending = self.sigma0, None, None, False
        self.n_success, self.n_trial, self.returned = 0, 0, False

    def act(self, obs):
        from .contracts_bridge import se3_exp, DT_C
        from .goal_rl import MAX_TWIST
        info = {"sigma": self.sigma, "success": 0.0, "return": 0.0}
        if self.parent is None:
            self.parent, self.s_par = obs.odo_cmd.copy(), obs.s
        elif self.pending:                                        # judge the offspring just reached
            self.pending = False
            if obs.s > self.s_par:
                self.parent, self.s_par = obs.odo_cmd.copy(), obs.s
                self.sigma = min(self.sigma * self.SUCC, self.SIG_MAX)
                self.n_success += 1
                info["success"] = 1.0
            else:
                self.sigma = max(self.sigma * self.SUCC ** self.FAIL_POW, self.SIG_MIN)
        at_parent = np.allclose(obs.odo_cmd, self.parent, atol=1e-9)
        if obs.t == obs.n_steps - 1:                              # last step: keep-best return
            info["return"] = 1.0
            self.returned = not at_parent
            return (np.zeros(6) if at_parent else twist_between(obs.odo_cmd, self.parent)), info
        d = np.clip(self.sigma * self.rng.standard_normal(6), -1.0, 1.0) * MAX_TWIST
        X = self.parent @ se3_exp(d * DT_C)
        self.pending = True
        self.n_trial += 1
        return twist_between(obs.odo_cmd, X), info

    def episode_summary(self):
        return {"n_success": self.n_success, "n_trial": self.n_trial, "final_sigma": self.sigma,
                "final_return_move": bool(self.returned), "s_parent": self.s_par}


class LookaheadController:
    """PRIVILEGED look-ahead bound. Renders hypothetical poses through a cloned MuJoCo
    state; scored by CT-mask S against an independently seeded target. Never a controller."""
    privileged = True
    legacy = False
    observes = ("PRIVILEGED: clones the MuJoCo state and renders K x rounds hypothetical poses per step "
                "from the CT; scores them by CT-mask S against the target view rendered with an "
                "independent speckle seed; never reads the target pose")

    def __init__(self, seed, k=64, rounds=2, n_elite=8, seed_offset=7907, std_floor=0.05):
        self.seed, self.k, self.rounds, self.n_elite = int(seed), int(k), int(rounds), int(n_elite)
        self.seed_offset, self.std_floor = int(seed_offset), float(std_floor)
        self.name = f"lookahead_k{self.k}_r{self.rounds}"

    def reset(self, ep):
        self.rng = np.random.default_rng([self.seed, 1931, int(ep)])
        self.prev, self.tgt_img_i, self.tgt_lab, self.pred = None, None, None, None
        self.n_cand, self.s_at_target, self.max_pred_err = 0, None, 0.0

    def _begin(self, obs, sim):
        from dataclasses import replace
        from . import view_match as VM
        cfg_i = replace(sim.cfg, seed=int(sim.cfg.seed) + self.seed_offset)
        self.tgt_img_i, _ = sim.render_target(cfg_i)               # independent speckle, same pose
        self.tgt_lab = obs.tgt_mask                                 # CT label (CT-mask runs only)
        # S of a PERFECT pose against the independently seeded target: the ceiling of this score
        self.s_at_target = VM.combined_score(obs.tgt_img, obs.tgt_mask, self.tgt_img_i, self.tgt_lab,
                                             None, sim.w_seg, sim.w_app)

    def _predict_pose(self, sim, u):
        import mujoco
        from .contracts_bridge import se3_exp, DT_C
        from .mujoco_env import _mat_from_T, _quat_from_mat
        env = sim.env
        d = copy.copy(env.data)
        tgt = env._press_limited(env._target @ se3_exp(np.asarray(u, np.float64) * DT_C))
        d.mocap_pos[0] = tgt[:3, 3]
        d.mocap_quat[0] = _quat_from_mat(_mat_from_T(tgt))
        for _ in range(env.settle_steps):
            mujoco.mj_step(env.model, d)
        T = np.eye(4)
        T[:3, 3] = d.xpos[env._probe_bid]
        T[:3, :3] = d.xmat[env._probe_bid].reshape(3, 3)
        return T

    def _score(self, sim, u):
        from . import view_match as VM
        T = self._predict_pose(sim, u)
        img, lab = VM.curvi_render(sim.lv, T, sim.Wfv, sim.translator, sim.cfg, sim.image_size)
        self.n_cand += 1
        return VM.combined_score(img, lab, self.tgt_img_i, self.tgt_lab, None, sim.w_seg, sim.w_app), T

    def act(self, obs, sim):
        from .goal_rl import MAX_TWIST
        if self.tgt_img_i is None:
            self._begin(obs, sim)
        pred_err = float("nan")
        if self.pred is not None:                                   # clone prediction vs the real step
            pred_err = float(np.linalg.norm(self.pred[:3, 3] - sim.env.world_from_probe[:3, 3]) * 1000.0)
            self.max_pred_err = max(self.max_pred_err, pred_err)
        U1 = self.rng.uniform(-1.0, 1.0, (self.k, 6)) * MAX_TWIST
        U1[0] = 0.0                                                 # stay
        if self.prev is not None and self.k > 1:
            U1[1] = self.prev                                       # warm start
        scored = []
        for u in U1:
            s, Tp = self._score(sim, u)
            scored.append((s, u, Tp))
        s_stay = scored[0][0]
        for _ in range(self.rounds - 1):
            el = np.asarray([u for s, u, _ in sorted(scored, key=lambda x: -x[0])[: self.n_elite]])
            mu, sd = el.mean(0), np.maximum(el.std(0), self.std_floor * MAX_TWIST)
            U = np.clip(mu + sd * self.rng.standard_normal((self.k, 6)), -MAX_TWIST, MAX_TWIST)
            for u in U:
                s, Tp = self._score(sim, u)
                scored.append((s, u, Tp))
        i = int(np.argmax([s for s, _, _ in scored]))
        s_best, u_best, self.pred = scored[i]
        self.prev = np.asarray(u_best, np.float64).copy()
        return self.prev.copy(), {"look_s_best": float(s_best), "look_s_stay": float(s_stay),
                                  "look_gain": float(s_best - s_stay), "look_n_cand": float(len(scored)),
                                  "look_pred_err_mm": pred_err}

    def episode_summary(self):
        return {"n_candidates_rendered": self.n_cand, "look_s_at_target": self.s_at_target,
                "look_target_seed_offset": self.seed_offset,
                "max_clone_prediction_err_mm": self.max_pred_err}


# ------------------------------------------------------------------------------------- the evaluator
_STEP_KEYS = ("pos", "ang", "euclid", "S", "S_seg", "S_app", "S_seg_fallback", "seg_ct", "force",
              "gap_center", "gap_min", "cmd_lag")


class _EpLog:
    def __init__(self):
        self.d = {k: [] for k in _STEP_KEYS}
        self.axes, self.exyz, self.twist, self.ctrl = [], [], [], {}

    def add(self, env, T_tgt, pos, ang, S, s_seg, s_app, fb, seg_ct, force):
        T_cur = env.world_from_probe
        ax, exyz = pose_axes(T_cur, T_tgt)
        gc, gm = face_gap_mm(T_cur, getattr(env, "contact_plane", None))
        lag = float(np.linalg.norm(np.asarray(env._target)[:3, 3] - T_cur[:3, 3]) * 1000.0)
        for k, v in zip(_STEP_KEYS, (pos, ang, float(np.linalg.norm(exyz)), S, s_seg, s_app, fb, seg_ct,
                                     force, gc, gm, lag)):
            self.d[k].append(v)
        self.axes.append(ax); self.exyz.append(exyz)

    def add_ctrl(self, t, info):
        for k, v in info.items():
            self.ctrl.setdefault(k, {})[t] = float(v)


def evaluate_ext(T, lv_list, cases, translator, cfg, controller, sim, history, image_size, device, w_seg, w_app,
                 setup_kw, n_steps, n_episodes, seed=999, design="crossed", eval_randomize="off",
                 log_episodes=True, target_seed_offset=None, *, setup_only=False):
    """train_view_match_rl.evaluate with a pluggable controller and extended logging. Returns
    (res, traj): res has every key of T.evaluate (same values for a legacy controller) plus 'ext';
    traj is the per-step array dict for save_traj_npz.

    target_seed_offset (opt-in, v2.1 independent target speckle; default None = the archived shared
    seed): the TARGET view is rendered with speckle seed cfg.seed + target_seed_offset, so target and
    observations no longer share one speckle field. Observations, setups and every other draw are
    unchanged. With None nothing changes and res gains no key.

    v2.1.2 (opt-in): the episode's start pose and contact plane come from
    T.contact_setup (the switch T._CONTACT_SURFACE, the same one T.evaluate reads); the probe-skin gap is measured
    against the env's contact plane; setup_only=True returns ({"setups": [...]}, None) without an env or a render.
    With the switch at its default nothing changes."""
    import torch
    from . import view_match as VM, render_lotus as R
    from .contracts_bridge import invert_T
    from .placement import NoAcceptedWindow
    learned = VM._US is not None
    cells, skipped_cells = T.eval_cells(lv_list, cases, design)
    if cells is not None and not cells:
        raise SystemExit(f"no eval (patient, target) cell has a screened window: {skipped_cells}")
    eps, skipped_eps, logs = [], Counter(), []
    setups = []
    contact_setup = getattr(T, "contact_setup", None)
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed + ep)
        if cells is None:                                                   # legacy assignment
            vi, organ, window = ep % len(lv_list), T.SEAT_ORGANS[ep % len(T.SEAT_ORGANS)], None
        else:
            vi, organ = cells[ep % len(cells)]
            window = ep // len(cells)
        lv = lv_list[vi]
        try:
            Wfv, T_tgt, T_start, meta = T.make_setup(lv, rng, organ, setup_kw, window=window, count=False)
        except NoAcceptedWindow:
            skipped_eps[f"{cases[vi]}:{organ}"] += 1
            continue
        if contact_setup is not None:                                       # v2.1.2 opt-in contact surface
            T_run, plane, plane_rec = contact_setup(lv, Wfv, T_tgt, T_start, meta)
        else:
            T_run, plane, plane_rec = T_start, None, None
        if setup_only:
            setups.append(T.setup_record(ep, cases[vi], meta, T_tgt, T_run, plane, plane_rec, Wfv))
            continue
        if plane is None:
            env = T.make_env(lv, Wfv, T_run, device, np.random.default_rng(seed + ep))
        else:
            env = T.make_env(lv, Wfv, T_run, device, np.random.default_rng(seed + ep), contact_plane=plane)
        c = R.domain_randomize(R.RenderConfig(), np.random.default_rng([seed, ep, T._DR_SALT])) \
            if eval_randomize == "episode" else cfg
        controller.reset(ep)
        c_tgt = c if target_seed_offset is None else c.with_seed(int(c.seed) + int(target_seed_offset))
        tgt_img, tgt_lab = VM.curvi_render(lv, T_tgt, Wfv, translator, c_tgt, image_size)
        tgt_prep = T._prep(tgt_img, image_size)
        tgt_t = torch.from_numpy(tgt_prep[None, None]).to(device)
        env.reset()
        p0, a0 = VM.pose_error(env.world_from_probe, T_tgt)
        s0_img, s0_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, c, image_size)
        m0 = VM.combined_score(s0_img, s0_lab, tgt_img, tgt_lab, sim, w_seg, w_app)
        rec = dict(ep=ep, patient=cases[vi], target=meta["target_used"], target_requested=meta["target_requested"],
                   window=meta.get("window"), n_windows=meta.get("n_windows"),
                   start_clamped=bool(meta.get("start_clamped")), clamp_mm=float(meta.get("clamp_mm") or 0.0),
                   start_pos=p0, start_ang=a0, static_seg=VM.seg_organ_iou(s0_lab, tgt_lab), static_match=m0,
                   **T._window_info(lv, meta["target_used"], meta.get("window")))
        hist = deque([T._prep(s0_img, image_size)] * history, maxlen=history)
        cur_lab, cur_img = s0_lab, s0_img
        pos_tr, ang_tr, match_tr, seg_tr, forces = [p0], [a0], [m0], [rec["static_seg"]], []
        # ---- v2.1 extended logging (read-only) ----
        tgt_ct = ct_label_at(lv, T_tgt, Wfv, image_size) if learned else tgt_lab
        L = _EpLog()
        app0 = VM.sector_appearance(s0_img, tgt_img)
        f0 = env.contact_force()
        L.add(env, T_tgt, p0, a0, m0, rec["static_seg"], app0, seg_fallback(s0_lab, tgt_lab),
              VM.seg_organ_iou(ct_label_at(lv, env.world_from_probe, Wfv, image_size), tgt_ct) if learned
              else rec["static_seg"], f0)
        C0inv = invert_T(np.asarray(env._target, np.float64))
        access = None
        if controller.privileged and not controller.legacy:
            access = SimAccess(env=env, lv=lv, Wfv=Wfv, translator=translator, cfg=c, image_size=image_size,
                               w_seg=w_seg, w_app=w_app,
                               render_target=lambda cf, _lv=lv, _T=T_tgt, _W=Wfv:
                               VM.curvi_render(_lv, _T, _W, translator, cf, image_size))
        s_seg_cur, s_app_cur, f_cur = rec["static_seg"], app0, f0
        for t in range(n_steps):
            cur_np = np.stack(hist, 0)
            obs = Obs(t=t, n_steps=n_steps, cur_stack=cur_np, cur_img=cur_img, cur_mask=cur_lab, tgt_img=tgt_img,
                      tgt_mask=tgt_lab, tgt_prep=tgt_prep, tgt_t=tgt_t, s=match_tr[-1], s_seg=s_seg_cur,
                      s_app=s_app_cur, force=f_cur, odo_cmd=C0inv @ np.asarray(env._target, np.float64))
            if controller.legacy:
                a_exec, info = controller.act(obs, LegacyAccess(env.world_from_probe, T_tgt))
            elif controller.privileged:
                a_exec, info = controller.act(obs, access)
            else:
                a_exec, info = controller.act(obs)
            a_exec = np.asarray(a_exec, np.float64)
            env.step(a_exec)
            forces.append(env.contact_force())
            oi, cur_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, c, image_size)
            hist.append(T._prep(oi, image_size))
            pe, ae = VM.pose_error(env.world_from_probe, T_tgt)
            pos_tr.append(pe); ang_tr.append(ae)
            seg_tr.append(VM.seg_organ_iou(cur_lab, tgt_lab))
            match_tr.append(VM.combined_score(oi, cur_lab, tgt_img, tgt_lab, sim, w_seg, w_app))
            cur_img = oi
            s_seg_cur, s_app_cur, f_cur = seg_tr[-1], VM.sector_appearance(oi, tgt_img), forces[-1]
            L.add(env, T_tgt, pe, ae, match_tr[-1], s_seg_cur, s_app_cur, seg_fallback(cur_lab, tgt_lab),
                  VM.seg_organ_iou(ct_label_at(lv, env.world_from_probe, Wfv, image_size), tgt_ct) if learned
                  else s_seg_cur, f_cur)
            L.twist.append(a_exec.copy())
            L.add_ctrl(t, info)
        kb = int(np.argmax(match_tr))                                      # pose-free keep-best
        rec.update(final_pos=pos_tr[-1], final_ang=ang_tr[-1], final_seg=seg_tr[-1], final_match=match_tr[-1],
                   keepbest_step=kb, keepbest_pos=pos_tr[kb], keepbest_ang=ang_tr[kb], keepbest_seg=seg_tr[kb],
                   peak_force=float(max(forces) if forces else 0.0),
                   contact_frac=float(np.mean([f > 0 for f in forces])) if forces else 0.0,
                   pos_traj=[round(float(x), 3) for x in pos_tr], match_traj=[round(float(x), 4) for x in match_tr])
        fa = L.axes[-1]
        gm = np.asarray(L.d["gap_min"])
        rec.update(ang_traj=[round(float(x), 3) for x in ang_tr],
                   final_axes={k: round(float(v), 4) for k, v in zip(AXES, fa)},
                   final_disp_xyz_mm=[round(float(v), 4) for v in L.exyz[-1]],
                   final_euclid_mm=float(L.d["euclid"][-1]), final_app=float(L.d["S_app"][-1]),
                   final_seg_ct=float(L.d["seg_ct"][-1]), static_seg_ct=float(L.d["seg_ct"][0]),
                   fallback_steps=int(np.sum(L.d["S_seg_fallback"])),
                   fallback_final=bool(L.d["S_seg_fallback"][-1]),
                   final_gap_min_mm=float(gm[-1]), mean_gap_min_mm=float(gm[1:].mean()) if len(gm) > 1 else float(gm[0]),
                   hover_frac=float(np.mean(gm[1:] > 0)) if len(gm) > 1 else float(gm[0] > 0),
                   **{f"success_{int(a)}mm_{int(b)}deg": bool(pos_tr[-1] <= a and ang_tr[-1] <= b) for a, b in SUCCESS_TOLS},
                   controller=controller.episode_summary())
        if plane_rec is not None:                                           # v2.1.2 opt-in (None = no change)
            T.plane_episode_fields(rec, plane_rec, meta, env.world_from_probe)
        eps.append(rec)
        logs.append(L)
        env.close()
    if setup_only:
        return {"setups": setups, "contact_surface": getattr(T, "_CONTACT_SURFACE", "target_plane"),
                "n_episodes": len(setups), "skipped_episodes": dict(skipped_eps), "design": design,
                "eval_seed": seed}, None
    if not eps:
        raise SystemExit(f"every eval episode was skipped: {dict(skipped_eps)}")
    keys = {"pos": "final_pos", "ang": "final_ang", "seg": "final_seg", "match": "final_match",
            "start_pos": "start_pos", "start_ang": "start_ang", "static_seg": "static_seg",
            "static_match": "static_match", "keepbest_pos": "keepbest_pos", "keepbest_ang": "keepbest_ang",
            "keepbest_seg": "keepbest_seg", "peak_force": "peak_force"}
    res = {k: float(np.mean([e[v] for e in eps])) for k, v in keys.items()}
    res["start_clamped_frac"] = float(np.mean([e["start_clamped"] for e in eps]))
    res["clamp_mm_mean"] = float(np.mean([e["clamp_mm"] for e in eps]))

    def _group(key, other):
        g = {}
        for e in eps:
            g.setdefault(e[key], []).append(e)
        return {k: {"n": len(L_), "pos": float(np.mean([e["final_pos"] for e in L_])),
                    "ang": float(np.mean([e["final_ang"] for e in L_])),
                    "start_pos": float(np.mean([e["start_pos"] for e in L_])),
                    other + "s": sorted({e[other] for e in L_})} for k, L_ in g.items()}
    res["per_patient"] = _group("patient", "target")
    res["per_target"] = _group("target", "patient")
    res.update(n_episodes=len(eps), design=design, eval_randomize=eval_randomize, eval_seed=seed,
               contact="on" if T._CONTACT else "off", seat=T._seat(),
               skipped_cells=skipped_cells, skipped_episodes=dict(skipped_eps))
    if target_seed_offset is not None:                                   # opt-in only: no key by default
        res["target_seed_offset"] = int(target_seed_offset)
        res["target_speckle"] = ("independent: target rendered with cfg.seed + target_seed_offset "
                                 "(the per-episode DR seed + offset under eval_randomize episode)")
    if getattr(T, "_PLANE_LOG", False):                                  # v2.1.2 opt-in only: no key by default
        res["contact_surface"] = T._CONTACT_SURFACE
    traj = _stack_traj(logs, eps, n_steps)
    res["ext"] = _ext_summary(eps, traj, controller)
    if log_episodes:
        res["episodes"] = eps
    return res, traj


def _stack_traj(logs, eps, n_steps):
    out = {k: np.asarray([L.d[k] for L in logs], np.float32 if k != "S_seg_fallback" else bool) for k in _STEP_KEYS}
    out["axes"] = np.asarray([L.axes for L in logs], np.float32)            # (E, N+1, 6) in AXES order
    out["euclid_xyz"] = np.asarray([L.exyz for L in logs], np.float32)      # (E, N+1, 3)
    out["twist"] = np.asarray([L.twist for L in logs], np.float32).reshape(len(logs), n_steps, 6)
    ck = sorted({k for L in logs for k in L.ctrl})
    for k in ck:
        a = np.full((len(logs), n_steps), np.nan, np.float32)
        for i, L in enumerate(logs):
            for t, v in L.ctrl.get(k, {}).items():
                a[i, t] = v
        out[f"ctrl_{k}"] = a
    out["ep"] = np.asarray([e["ep"] for e in eps], np.int32)
    out["patient"] = np.asarray([e["patient"] for e in eps])
    out["target"] = np.asarray([e["target"] for e in eps])
    out["window"] = np.asarray([-1 if e.get("window") is None else int(e["window"]) for e in eps], np.int32)
    return out


def _ext_summary(eps, traj, controller):
    fa = np.asarray([[e["final_axes"][k] for k in AXES] for e in eps], np.float64)
    pp = {}
    for i, e in enumerate(eps):
        pp.setdefault(e["patient"], []).append(i)
    per_pat_abs = {p: {k: float(np.mean(np.abs(fa[ix, j]))) for j, k in enumerate(AXES)} for p, ix in pp.items()}
    s = {"axes_doc": AXES_DOC,
         "final_abs_axes_mean": {k: float(np.mean(np.abs(fa[:, j]))) for j, k in enumerate(AXES)},
         "final_abs_axes_patient_mean": {k: float(np.mean([per_pat_abs[p][k] for p in per_pat_abs]))
                                         for k in AXES},
         "final_signed_axes_mean": {k: float(np.mean(fa[:, j])) for j, k in enumerate(AXES)},
         "final_euclid_mm": float(np.mean([e["final_euclid_mm"] for e in eps])),
         "final_app": float(np.mean([e["final_app"] for e in eps])),
         "final_seg_ct": float(np.mean([e["final_seg_ct"] for e in eps])),
         "fallback_steps_total": int(traj["S_seg_fallback"].sum()),
         "fallback_steps_frac": float(traj["S_seg_fallback"].mean()),
         "fallback_endpoints": int(sum(e["fallback_final"] for e in eps)) + int(traj["S_seg_fallback"][:, 0].sum()),
         "mean_gap_min_mm": float(np.mean(traj["gap_min"][:, 1:])) if traj["gap_min"].shape[1] > 1 else None,
         "hover_frac": float(np.mean(traj["gap_min"][:, 1:] > 0)) if traj["gap_min"].shape[1] > 1 else None,
         "pos_by_step": [float(x) for x in traj["pos"].mean(0)],
         "ang_by_step": [float(x) for x in traj["ang"].mean(0)],
         "S_by_step": [float(x) for x in traj["S"].mean(0)],
         "controller": {"name": getattr(controller, "name", None), "privileged": bool(controller.privileged),
                        "observes": getattr(controller, "observes", None)}}
    for a, b in SUCCESS_TOLS:
        k = f"success_{int(a)}mm_{int(b)}deg"
        s[k] = float(np.mean([e[k] for e in eps]))
    return s


def save_traj_npz(path, blocks):
    """blocks: [(meta dict, traj dict)]. Keys 'b<i>__<name>'; 'index' holds the JSON list of block metas.
    Written atomically. Returns the md5 of the file."""
    from .provenance import md5_file
    arrs = {"index": np.asarray(json.dumps([m for m, _ in blocks], default=str))}
    for i, (_m, tr) in enumerate(blocks):
        for k, v in tr.items():
            arrs[f"b{i}__{k}"] = v
    base = path[:-4] if path.endswith(".npz") else path
    tmp = base + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, base + ".npz")
    return md5_file(base + ".npz")


def traj_path(out_json):
    return (out_json[:-5] if out_json.endswith(".json") else out_json) + ".traj.npz"


# ------------------------------------------------------------------------- TRAIN-pilot selection rule
def select_param(files, name, contact="on"):
    """-> (value, record). Lowest mean final translation error over the pilot files (contact on, DR off),
    ties to the smaller value. Every value must appear in every file; files must be TRAIN-only."""
    from .benchmark import TRAIN_PATIENTS
    from .provenance import md5_file
    per_file, recs = [], []
    for f in files:
        d = json.load(open(f))
        ev = set((d.get("provenance") or {}).get("eval_cases") or [])
        if not ev or not ev <= set(TRAIN_PATIENTS):
            raise SystemExit(f"select_param: {f} evaluates {sorted(ev)}, not TRAIN patients only")
        tab = {}
        for r in d["results"]:
            p = r.get("param") or {}
            if p.get("name") == name and r.get("contact") == contact and r.get("eval_randomize", "off") == "off":
                tab[float(p["value"])] = float(r["pos"])
        if not tab:
            raise SystemExit(f"select_param: {f} has no '{name}' grid at contact={contact}")
        per_file.append(tab)
        recs.append({"path": os.path.abspath(f), "md5": md5_file(f), "mean_final_pos_mm": tab,
                     "translator": d.get("translator"), "mask_source": d.get("mask_source")})
    vals = sorted(set.intersection(*[set(t) for t in per_file]))
    if not vals:
        raise SystemExit(f"select_param: no '{name}' value is common to all pilot files")
    means = {v: float(np.mean([t[v] for t in per_file])) for v in vals}
    best = min(vals, key=lambda v: (means[v], v))
    return best, {"param": name, "rule": "lowest mean final translation error (mm) over the TRAIN pilot files, "
                  "contact on, DR off; ties to the smaller value", "values": vals,
                  "mean_over_files": means, "selected": best, "files": recs}


# --------------------------------------------------------------------- base fit diagnostics (2.3 (v))
def base_fit_report(T, lv_train, train_cases, lv_eval, eval_cases, translator, cfg, image_size, setup_kw,
                    n_pairs=300, seed=0):
    """Per-DoF R^2, OLS gain (slope of base twist on analytic twist; < 1 = shrinkage) and correlation of
    the calibrated image base (T._IBVS) against the analytic twist, on n_pairs fresh TRAIN pairs and
    n_pairs held-out pairs (the make_setup start distribution, one render cfg). Uses its own RNG, so it
    never shifts the calibration or the evaluation streams."""
    from . import goal_rl as G, view_match as VM
    out = {"n_pairs": int(n_pairs), "kind": (T._IBVS or {}).get("kind"), "rich": (T._IBVS or {}).get("rich"),
           "dof": ["vx", "vy", "vz", "wx", "wy", "wz"],
           "doc": "base = clipped executed base twist; analytic = clipped analytic twist (the calibration "
                  "target); R2 = 1 - SSres/SStot of base vs analytic; gain = cov(analytic, base)/var(analytic)"}
    for split, lvs, cs, salt in (("train_fresh", lv_train, train_cases, 4049), ("heldout", lv_eval, eval_cases, 4051)):
        cells, _sk = T.target_cells(lvs, cs)
        rng = np.random.default_rng([int(seed), salt])
        A, B = [], []
        for _ in range(int(n_pairs)):
            vi, organ = cells[int(rng.integers(len(cells)))]
            Wfv, T_tgt, T_cur, _m = T.make_setup(lvs[vi], rng, organ, setup_kw, count=False)
            _, cur_lab = VM.curvi_render(lvs[vi], T_cur, Wfv, translator, cfg, image_size)
            _, tgt_lab = VM.curvi_render(lvs[vi], T_tgt, Wfv, translator, cfg, image_size)
            A.append(G.analytic_twist(T_cur, T_tgt)); B.append(T.image_base_twist(cur_lab, tgt_lab))
        A, B = np.asarray(A), np.asarray(B)
        r2, gain, corr = [], [], []
        for j in range(6):
            a, b = A[:, j], B[:, j]
            sst = float(((a - a.mean()) ** 2).sum())
            r2.append(1.0 - float(((a - b) ** 2).sum()) / sst if sst > 0 else float("nan"))
            va = float(a.var())
            gain.append(float(np.mean((a - a.mean()) * (b - b.mean())) / va) if va > 0 else float("nan"))
            corr.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"))
        out[split] = {"r2": r2, "gain": gain, "corr": corr, "n_cells": len(cells),
                      "analytic_std": [float(x) for x in A.std(0)], "base_std": [float(x) for x in B.std(0)]}
    return out
