#!/usr/bin/env python3
"""Curvilinear ultrasound VIEW-REPRODUCTION RL (the redefined closed-loop task).

An agent moves a virtual probe over a user-provided CT + organ segmentation (a LabeledVolume) and
applies a sequence of probe-manipulation twists to render a synthetic B-mode -- in the clinical
CURVILINEAR geometry, textured per-tissue by SonoSPADE -- that reproduces a GIVEN target ultrasound
view. The per-step reward combines organ-layout match (segmentation IoU of the current CT label slice
vs the target's segmentation) and appearance similarity (render vs target). Because the match needs
the right texture in the right organ, per-tissue synthesis helps and a global recolor does not.

Regimes:
  * self-supervised (default): the target is this volume rendered through the SAME translator at a
    hidden pose; the agent reproduces it. Ground-truth pose is known -> clean eval + imitation warm
    start. We compare SonoSPADE vs CUT vs CycleGAN (curvilinear checkpoints) on ground-truth pose error.
  * real registration (--real-eval, legacy, canonical traceability only): the in-run registration of
    the v1 runs. Use scripts/eval_real_registration.py for reported numbers.

Sim input: --ct/--seg to build the volume from a user's CT + TotalSegmentator-style masks; otherwise
uses the cached abdominal cases in data/cache/ (data/cache_anat/ under the anatomical seat).

v2 (oracle-ladder rerun). The seat comes from $USBG_SEAT (usbg.seatmode, default anatomical):
  * anatomical: every target is a screened acoustic window of (patient, organ) (usbg.windows); targets
    default to liver + kidney (spleen dropped). A (patient, target) cell with fewer than --min-windows
    screened windows is SKIPPED (logged, counted), never substituted by liver. Starts are clamped
    outside the skin (logged per episode). Defaults: TRAIN = usbg.benchmark.TRAIN_PATIENTS (13),
    eval = TEST_PATIENTS (6), lateral perturbation 0.02 m, crossed eval design, an explicit texture
    checkpoint (--texture-ckpt or --tex-manifest; the v1 CKPTS are canonical-only).
  * canonical: the v1 seat; with --eval-design legacy (and --allow-legacy-seg for the withdrawn
    shared segmenter) every number of the shipped runs reproduces exactly.
  Every run writes a provenance block (usbg.provenance.collect), refuses the withdrawn segmenter
  (assert_not_withdrawn), saves <out>_imit.pt right after imitation and <out>.pt at the end (both with
  the residual bound, residual mode and the calibrated IBVS matrix, so eval-only reloads need no RNG
  replay), a <ckpt>.provenance.json sidecar per checkpoint, and per-episode eval records.
  BatchNorm stays in TRAIN mode at eval (per-call batch statistics at B=1), exactly as in the v1
  runs; this is disclosed in provenance.bn_mode, not changed.

v2.1 (exploratory, opt-in; the v2.1 study, the RL positive control). --init-from <run>_imit.pt starts the
REINFORCE phase from a saved policy and its calibrated image base (calibration and imitation skipped);
--base-bias / --base-gain plant a deficit on the EXECUTED residual base that the imitation / BC-anchor
teacher never sees, so only the reward can reveal it; --curve-every N evaluates on the held-out patients
every N RL iterations. With any of them the output gains "curve", "rlpos" and analysis_status
"exploratory, v2.1"; without them the trainer and its outputs are the v2 ones.

v2.1.2 (the pre-specified protocol, exploratory, opt-in): the module switch _CONTACT_SURFACE ("target_plane" default,
"skin_tangent") and _PLANE_LOG select the contact surface of evaluate's episodes (contact_setup); evaluate also
takes setup_only=True. eval_view_ladder.py --contact-surface sets them; the trainer never does, and
calibrate_ibvs never reads them.

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  USBG_SEAT=anatomical USBG_LEARNED_SEG=0 PYTHONPATH=src .venv/bin/python scripts/train_view_match_rl.py \
      --translator sonospade --tex-manifest outputs/v2/textures/manifest.json --fix-residual \
      --residual-image --ibvs-features rich --seed 0 --out outputs/v2/table4/ctmask/sonospade_deploy_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, deque

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# v1 10-case TRAIN list (canonical default; the anatomical default is usbg.benchmark.TRAIN_PATIENTS)
TRAIN_CASES = ["s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0703", "s0720", "s1086", "s1373"]
# The benchmark's frozen held-out split (usbg.benchmark.TEST_PATIENTS). Evaluating here instead of on
# TRAIN_CASES is what makes a view-match pose error a generalization number rather than an
# in-distribution diagnostic.
EVAL_CASES = ["s0358", "s0513", "s0519", "s0591", "s0649", "s0762"]
# v1 textures: canonical-only defaults (they were trained on pairs that include held-out patients, D3)
CKPTS = {"physics": None,
         "cut": "data/dataset/exp_real_cut_curvi.pt",
         "cyclegan": "data/dataset/exp_real_cyclegan_curvi.pt",
         "sonospade": "data/dataset/exp_real_sonospade_curvi.pt"}
LEGACY_TARGETS = ["liver", "kidney", "spleen"]
SEAT_ORGANS = list(LEGACY_TARGETS)   # target organs (set from --targets in main; module global so helpers follow)
ANG_W = 2.0     # weight (deg -> mm-equivalent) in the pose-error potential used for reward shaping (#1)

# ---- v2 run-time switches (set once in main / by the eval drivers; module globals so every helper follows)
_RANDOMIZE = "off"      # "off": one fixed cfg (RenderConfig(seed)) for every render, as in the v1 runs
                        # "episode": ONE domain_randomize draw per episode/sample, shared by target+obs
_CONTACT = True         # patient-surface contact in the MuJoCo env (False = the skin box does not collide)
_DR_SALT = 7919         # separate RNG stream for DR draws so DR on/off arms share every setup draw
_MIN_WINDOWS = 1        # anatomical: a (case, target) cell needs >= this many screened windows
_CELLS = None           # anatomical: {volume index: [valid targets]} for the TRAIN samplers; None = legacy draws
_SKIPS = Counter()      # training draws rejected because their (case, target) cell was skipped
_TRAIN_SETUPS = {"n": 0, "clamped": 0, "clamp_mm_sum": 0.0, "substituted": 0}
BN_MODE = "train (BatchNorm uses per-call batch statistics, eval at B=1; as in the v1 runs, disclosed)"
# v2.1 RL positive control (the v2.1 study; exploratory, opt-in via --base-bias / --base-gain). A planted
# deficit on the EXECUTED residual base: a_exec = clip(a_policy + gain * base + bias). The imitation and
# BC-anchor teacher (residual_target) never sees it, so the reward is the only signal about it.
# None = off (every v2 run).
_DEFICIT = None
ANALYSIS_STATUS_V21 = "exploratory, v2.1"
# v2.1.2 (opt-in): the contact surface of the EVALUATION episodes. "target_plane" (default) is
# the box top at z = 0 through the target window face, exactly the v2.1.1 path; "skin_tangent" poses the box on the
# plane tangent to the patient's skin where the pre-lift start's beam axis enters the body
# (usbg.body_surface.skin_plane) and lifts the pre-lift start along that plane's normal. Read by evaluate (through
# contact_setup and make_env) and by usbg.controllers_v21.evaluate_ext; calibrate_ibvs, the imitation and DAgger
# samplers and the training rollouts never read it. _PLANE_LOG adds the per-episode contact-plane record (set by
# eval_view_ladder.py --contact-surface); with both at their defaults nothing changes.
CONTACT_SURFACES = ("target_plane", "skin_tangent")
_CONTACT_SURFACE = "target_plane"
_PLANE_LOG = False


def _organ_id(name):
    from usbg import volume as V
    return {"liver": V.LIVER, "kidney": V.KIDNEY, "spleen": V.SPLEEN}[name]


def _seat():
    from usbg.seatmode import seat_mode
    return seat_mode()


def _case(lv):
    return (getattr(lv, "meta", None) or {}).get("case") or "user_ct"


def n_target_windows(lv, target):
    """Screened windows for (case, target) under the anatomical seat. Canonical: 1 if the organ is
    present (the v1 seat is the organ centroid), else 0."""
    from usbg import placement as P
    oid = _organ_id(target)
    if _seat() == "canonical":
        return 1 if lv.has(oid) else 0
    try:
        return int(P.n_windows(lv, oid))
    except P.NoAcceptedWindow:
        return 0


def target_cells(lv_list, cases, targets=None, min_windows=None):
    """([(volume index, target)] with enough screened windows, {"case:target": n_windows} skipped), in
    patient-major order (the crossed eval design walks it). Canonical: a cell is valid iff the organ is
    present."""
    targets = list(targets or SEAT_ORGANS)
    mw = 1 if _seat() == "canonical" else (_MIN_WINDOWS if min_windows is None else int(min_windows))
    ok, skipped = [], {}
    for i, lv in enumerate(lv_list):
        for t in targets:
            n = n_target_windows(lv, t)
            if n >= mw:
                ok.append((i, t))
            else:
                skipped[f"{cases[i]}:{t}"] = n
    return ok, skipped


def init_train_cells(lv_list, cases, log=print):
    """Fix the (case, target) cells the TRAINING samplers draw from. Anatomical: cells without enough
    screened windows are skipped (logged with their window count), never substituted by liver.
    Canonical: None (the v1 draws, with the v1 liver substitution for an absent organ)."""
    global _CELLS
    if _seat() == "canonical":
        _CELLS = None
        return {"valid": None, "skipped": {}, "min_windows": None}
    ok, skipped = target_cells(lv_list, cases)
    if not ok:
        raise SystemExit("no TRAIN (case, target) cell has a screened window")
    _CELLS = {}
    for i, t in ok:
        _CELLS.setdefault(i, []).append(t)
    for k, n in sorted(skipped.items()):
        log(f"  [cells] SKIP train {k}: {n} screened window(s) < --min-windows {_MIN_WINDOWS} (not substituted)")
    log(f"  [cells] train: {len(ok)} valid (case, target) cells, {len(skipped)} skipped")
    return {"valid": [f"{cases[i]}:{t}" for i, t in ok], "skipped": skipped, "min_windows": _MIN_WINDOWS}


def draw_cell(lv_list, rng):
    """Training draw of (volume, target): a uniform volume, then a uniform target (the v1 draw order).
    Anatomical: redrawn while the cell is skipped (uniform over valid cells), each rejection counted."""
    while True:
        i = int(rng.integers(len(lv_list)))
        t = SEAT_ORGANS[int(rng.integers(len(SEAT_ORGANS)))]
        if _CELLS is None or t in _CELLS.get(i, ()):
            return lv_list[i], t
        _SKIPS[f"{_case(lv_list[i])}:{t}"] += 1


def cycle_cell(lv_list, k):
    """RL-loop deterministic cycle: volume k % n, target k % n_targets (the v1 order); anatomical: the
    next valid cell at or after k, each skipped cell counted."""
    for j in range(k, k + len(lv_list) * len(SEAT_ORGANS) + 1):
        i, t = j % len(lv_list), SEAT_ORGANS[j % len(SEAT_ORGANS)]
        if _CELLS is None or t in _CELLS.get(i, ()):
            return lv_list[i], t
        _SKIPS[f"{_case(lv_list[i])}:{t}"] += 1
    raise RuntimeError("no valid (case, target) cell in the RL cycle")


def episode_cfg(cfg, dr_rng):
    """The render cfg for ONE episode (or one imitation sample). DR off -> the run's fixed cfg; DR on ->
    one draw from the dedicated DR stream, used for the target AND every observation of the episode."""
    if _RANDOMIZE == "episode":
        from usbg import render_lotus as R
        return R.domain_randomize(R.RenderConfig(), dr_rng)
    return cfg


def make_env(lv, Wfv, T_start, device, rng, contact_plane=None):
    """goal_rl._make_env (physics, no translator, no env-side DR) plus the contact switch; the env's own
    observation render is skipped (observe=False) because every caller renders its own curvilinear view
    through VM.curvi_render. Dynamics and every reported number are unchanged. v2.1.2: contact_plane (p, n)
    is passed to the env only when given (the skin-tangent evaluation surface, see contact_setup)."""
    from usbg.mujoco_env import USProbeEnv
    kw = {} if contact_plane is None else {"contact_plane": contact_plane}
    return USProbeEnv(lv, Wfv, T_start, renderer="physics", device=device, translator=None,
                      randomize=False, rng=rng, contact=_CONTACT, observe=False, **kw)


def _pose_list(T):
    return [[float(x) for x in row] for row in np.asarray(T, np.float64)]


def contact_setup(lv, Wfv, T_tgt, T_start, meta):
    """v2.1.2 (opt-in): the start pose and contact plane of one evaluation episode under
    _CONTACT_SURFACE. -> (T_start to run, contact_plane (p, n) or None, plane record or None).

    target_plane: the sampled start as before and no plane (the env keeps the box top at z = 0); the record (only
      with _PLANE_LOG) describes that plane (p = the pre-lift face centre projected onto z = 0, n = +z).
    skin_tangent: (p, n) = usbg.body_surface.skin_plane of the PRE-LIFT start (meta['start_prelift']), and the
      pre-lift start lifted along n until the face clears the plane (placement.clamp_start_outside_plane).
    The record holds p (mm, world), n, the acute angle that n makes with the beam axis of the target, the signed
    distance of the target window face from the plane (mm, + = outside), the lift (mm), how p was found, and the target and
    pre-lift start poses (for the setup identity checks)."""
    from usbg import placement as P
    surface = _CONTACT_SURFACE
    if surface not in CONTACT_SURFACES:
        raise ValueError(f"_CONTACT_SURFACE {surface!r}; expected one of {CONTACT_SURFACES}")
    if surface == "target_plane" and not _PLANE_LOG:
        return T_start, None, None
    T_pre = np.asarray(meta.get("start_prelift", T_start), np.float64)
    if surface == "target_plane":
        c = T_pre[:3, 3]
        p, n = np.array([c[0], c[1], 0.0]), np.array([0.0, 0.0, 1.0])
        T_run, plane, lift_mm = T_start, None, float(meta.get("clamp_mm") or 0.0)
        info = {"point_source": "target_plane", "entry_t_mm": None, "n_faces": None, "orientation": None}
    else:
        if meta.get("seat") != "anatomical" or "start_prelift" not in meta:
            raise SystemExit("--contact-surface skin_tangent needs USBG_SEAT=anatomical")
        from usbg import body_surface as B
        p, n, info = B.skin_plane(lv, Wfv, T_pre)
        T_run, lift = P.clamp_start_outside_plane(T_pre, p, n)
        plane, lift_mm = (p, n), float(lift * 1000.0)
    b_tgt = np.asarray(T_tgt, np.float64)[:3, 2]
    rec = {"surface": surface, "p_mm": [float(x) * 1000.0 for x in p], "n": [float(x) for x in n],
           "angle_to_target_beam_deg": float(np.degrees(np.arccos(np.clip(abs(float(n @ b_tgt)), 0.0, 1.0)))),
           "target_face_signed_mm": float(n @ (np.asarray(T_tgt, np.float64)[:3, 3] - p)) * 1000.0,
           "lift_mm": lift_mm, "point_source": info.get("point_source"), "entry_t_mm": info.get("entry_t_mm"),
           "n_faces": info.get("n_faces"), "orientation": info.get("orientation"),
           "target_pose": _pose_list(T_tgt), "start_prelift": _pose_list(T_pre)}
    return T_run, plane, rec


def plane_episode_fields(rec, plane_rec, meta, T_final):
    """v2.1.2: add the contact-plane record to an episode record (only when contact_setup returned one). Under
    skin_tangent start_clamped / clamp_mm hold the applied skin lift and the sampled target-plane lift moves to
    target_plane_clamped / target_plane_clamp_mm. The final probe-skin gap (mm) is measured
    against the episode's own plane."""
    if plane_rec is None:
        return rec
    from usbg import placement as P
    pr = dict(plane_rec)
    p = np.asarray(pr["p_mm"], np.float64) / 1000.0
    n = np.asarray(pr["n"], np.float64)
    gc, gm = P.plane_gap_m(T_final, p, n)
    pr.update(final_gap_center_mm=gc * 1000.0, final_gap_min_mm=gm * 1000.0)
    if pr["surface"] == "skin_tangent":
        rec["target_plane_clamped"] = bool(meta.get("start_clamped"))
        rec["target_plane_clamp_mm"] = float(meta.get("clamp_mm") or 0.0)
        rec["start_clamped"] = bool(pr["lift_mm"] > 0.0)
        rec["clamp_mm"] = float(pr["lift_mm"])
    rec["contact_plane"] = pr
    return rec


def setup_record(ep, case, meta, T_tgt, T_run, plane, plane_rec, Wfv=None):
    """v2.1.2 setup-only record of one episode (no render, no env): what both contact paths must share."""
    return {"ep": int(ep), "patient": case, "target": meta["target_used"], "target_requested": meta["target_requested"],
            "window": meta.get("window"), "n_windows": meta.get("n_windows"),
            "world_from_volume": None if Wfv is None else np.asarray(Wfv, np.float64).copy(),
            "target_pose": np.asarray(T_tgt, np.float64).copy(),
            "start_prelift": np.asarray(meta.get("start_prelift", T_run), np.float64).copy(),
            "start_pose": np.asarray(T_run, np.float64).copy(),
            "plane": None if plane is None else (np.asarray(plane[0]).copy(), np.asarray(plane[1]).copy()),
            "lift_mm": None if plane_rec is None else plane_rec["lift_mm"],
            "point_source": None if plane_rec is None else plane_rec["point_source"]}


def make_setup(lv, rng, organ, setup_kw, window=None, count=True):
    """Target pose at a (patient, organ[, window]) + a perturbed start (goal_rl.sample_setup geometry).

    Returns (Wfv, T_tgt, T_start, meta); meta = sample_setup's {seat, window, n_windows, start_clamped,
    clamp_mm} + target_requested / target_used. Canonical: an absent organ is replaced by liver exactly as
    in the v1 code (recorded). Anatomical: no substitution, NoAcceptedWindow propagates."""
    from usbg import goal_rl as G
    used = organ
    if _seat() == "canonical" and not lv.has(_organ_id(organ)):
        used = "liver"
    Wfv, T_tgt, T_start, meta = G.sample_setup(lv, _organ_id(used), rng, window=window, return_meta=True,
                                               **setup_kw)
    meta = dict(meta, target_requested=organ, target_used=used)
    if count:
        _TRAIN_SETUPS["n"] += 1
        _TRAIN_SETUPS["clamped"] += int(bool(meta.get("start_clamped")))
        _TRAIN_SETUPS["clamp_mm_sum"] += float(meta.get("clamp_mm") or 0.0)
        _TRAIN_SETUPS["substituted"] += int(used != organ)
    return Wfv, T_tgt, T_start, meta


def setup_stats():
    n = max(_TRAIN_SETUPS["n"], 1)
    return {"n": _TRAIN_SETUPS["n"], "clamped": _TRAIN_SETUPS["clamped"],
            "clamp_frac": _TRAIN_SETUPS["clamped"] / n, "clamp_mm_mean": _TRAIN_SETUPS["clamp_mm_sum"] / n,
            "substituted_liver": _TRAIN_SETUPS["substituted"], "skip_draws": dict(_SKIPS)}


def _window_info(lv, organ, w):
    """Side / id / view angles of window w for (case, organ) (anatomical seat), for the episode record."""
    if w is None or _seat() == "canonical":
        return {}
    try:
        from usbg import windows as W
        m = W.windows_meta(lv, _organ_id(organ))[int(w)]
        return {"window_organ": m.get("organ"), "window_id": m.get("window_id"), "window_phi": m.get("phi"),
                "window_tilt_deg": m.get("tilt_deg"), "window_th": m.get("th")}
    except Exception:
        return {}


def pose_potential(pose, T_tgt):
    """Potential Phi(s) = -(translation_mm + ANG_W*rotation_deg) to the target pose, for the
    potential-based reward shaping in fix #1. Uses the ground-truth pose available in the sim (a
    training-time signal only, like the imitation teacher)."""
    from usbg import view_match as VM
    pos_mm, ang_deg = VM.pose_error(pose, T_tgt)
    return -(pos_mm + ANG_W * ang_deg)


_IBVS = None               # calibrated image visual-servoing base: {"kind","model","rich"} (set in main)
_RESIDUAL_MODE = "oracle"  # "oracle" (analytic pose base, sim only) | "image" (segmentation-feature base, non-privileged)


def _organ_moments(lab, organ, min_px=20):
    """Per-organ shape descriptors in normalized image coords: centroid, area, presence, and (for the
    rich feature set) principal-axis orientation and elongation from the second moments."""
    m = (lab == organ); a = int(m.sum()); H, W = lab.shape
    if a < min_px:
        return dict(cx=0.5, cy=0.5, area=0.0, present=0.0, theta=0.0, elong=0.0)
    ys, xs = np.nonzero(m); x = xs / W; y = ys / H
    cx, cy = float(x.mean()), float(y.mean())
    dx, dy = x - cx, y - cy
    cxx, cyy, cxy = float((dx * dx).mean()), float((dy * dy).mean()), float((dx * dy).mean())
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy + 1e-12)
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    elong = (l1 - l2) / (l1 + l2 + 1e-9)
    return dict(cx=cx, cy=cy, area=a / (H * W), present=1.0, theta=float(theta), elong=float(elong))


def ibvs_features(cur_lab, tgt_lab, rich=False):
    """Image-only features for the visual-servoing base (NO pose): per-organ centroid offset, area
    change, and co-presence over the matched organs; the rich set adds elongation change and the
    orientation change (as sin/cos deltas) so the base can also infer in-plane rotation."""
    from usbg.view_match import MATCH_ORGANS
    f = []
    for o in MATCH_ORGANS:
        c = _organ_moments(cur_lab, o); t = _organ_moments(tgt_lab, o)
        both = c["present"] * t["present"]
        f += [(t["cx"] - c["cx"]) * both, (t["cy"] - c["cy"]) * both, (t["area"] - c["area"]) * both, both]
        if rich:
            f += [(t["elong"] - c["elong"]) * both,
                  (np.sin(t["theta"]) - np.sin(c["theta"])) * both,
                  (np.cos(t["theta"]) - np.cos(c["theta"])) * both]
    f.append(1.0)
    return np.asarray(f, np.float64)


def _fit_mlp_base(X, Y, epochs=400):
    """Small MLP fit (image features -> twist) for a nonlinear visual-servoing base."""
    import torch, torch.nn as nn
    Xt = torch.as_tensor(X, dtype=torch.float32); Yt = torch.as_tensor(Y, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 6))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad(); (((net(Xt) - Yt) ** 2).mean()).backward(); opt.step()
    net.eval()
    return net


def calibrate_ibvs(lv_list, rng, translator, cfg, image_size, setup_kw, kind="linear", rich=False, n=500,
                   dr_rng=None):
    """Fit the image visual-servoing base (features -> corrective twist) with the analytic twist as
    target on sampled (current, target) pairs. Calibration uses the privileged pose (a sim
    training-time step, like the imitation teacher); at RUN time the base needs only the images."""
    from usbg import goal_rl as G, view_match as VM
    X, Y = [], []
    for _ in range(n):
        lv, organ = draw_cell(lv_list, rng)
        Wfv, T_tgt, T_cur, _m = make_setup(lv, rng, organ, setup_kw)
        c = episode_cfg(cfg, dr_rng)                       # one DR draw per (current, target) pair
        _, cur_lab = VM.curvi_render(lv, T_cur, Wfv, translator, c, image_size)
        _, tgt_lab = VM.curvi_render(lv, T_tgt, Wfv, translator, c, image_size)
        X.append(ibvs_features(cur_lab, tgt_lab, rich)); Y.append(G.analytic_twist(T_cur, T_tgt))
    X, Y = np.asarray(X), np.asarray(Y)
    model = np.linalg.lstsq(X, Y, rcond=None)[0] if kind == "linear" else _fit_mlp_base(X, Y)
    return {"kind": kind, "model": model, "rich": rich}


def image_base_twist(cur_lab, tgt_lab):
    from usbg import goal_rl as G
    if _IBVS is None:
        return np.zeros(6, dtype=np.float64)
    feat = ibvs_features(cur_lab, tgt_lab, _IBVS["rich"])
    if _IBVS["kind"] == "linear":
        tw = feat @ _IBVS["model"]
    else:
        import torch
        with torch.no_grad():
            tw = _IBVS["model"](torch.as_tensor(feat, dtype=torch.float32)).numpy()
    return np.clip(tw, -G.MAX_TWIST, G.MAX_TWIST)


def calibrate_cnn_base(lv_list, rng, translator, cfg, image_size, history, setup_kw, device,
                       n_traj=120, steps=8, epochs=400, dagger_rounds=0, dr_rng=None):
    """Learned appearance-encoder base over a FRAME HISTORY (the 'combined' direction), pose-free at
    run time. Round 0 collects (appearance-history, target, analytic twist) by rolling the analytic
    controller. With dagger_rounds>0 it then alternates: roll the CURRENT CNN base, relabel every
    visited state with the analytic expert, aggregate, and refit -- DAgger (Ross 2011) that trains the
    base on ITS OWN deployment states to beat the covariate shift that collapses a naive fit. The
    multi-frame appearance makes the out-of-plane DoF observable that a single mask cannot see (speckle
    decorrelation; Gee 2006 / Prevost 2018 / Guo 2020)."""
    import torch
    from usbg import goal_rl as G, view_match as VM
    from usbg.policy import GoalPolicy
    net = GoalPolicy.build(history=history).to(device)
    mt = torch.as_tensor(G.MAX_TWIST, dtype=torch.float32, device=device)
    Xc, Xt, Y = [], [], []

    def net_twist(stack, tgt_p):
        with torch.no_grad():
            c = torch.as_tensor(stack[None], dtype=torch.float32, device=device)
            t = torch.as_tensor(tgt_p[None, None], dtype=torch.float32, device=device)
            return (mt * torch.tanh(net(c, t)))[0].cpu().numpy()

    def collect(use_net):
        for _ in range(n_traj):
            lv, organ = draw_cell(lv_list, rng)
            Wfv, T_tgt, T_start, _m = make_setup(lv, rng, organ, setup_kw)
            env = make_env(lv, Wfv, T_start, device, rng)
            c = episode_cfg(cfg, dr_rng)                       # one DR draw per trajectory
            tgt_img, _ = VM.curvi_render(lv, T_tgt, Wfv, translator, c, image_size)
            tgt_p = _prep(tgt_img, image_size)
            env.reset()
            obs_img, _ = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, c, image_size)
            hist = deque([_prep(obs_img, image_size)] * history, maxlen=history)
            for _ in range(steps):
                stack = np.stack(hist, 0).astype(np.float32)
                expert = G.analytic_twist(env.world_from_probe, T_tgt)      # expert relabel of the visited state
                Xc.append(stack); Xt.append(tgt_p[None].astype(np.float32)); Y.append(expert)
                drive = net_twist(stack, tgt_p) if use_net else expert       # roll the current base (DAgger) or analytic
                a = np.clip(drive + rng.normal(0, 0.3, 6) * G.MAX_TWIST, -G.MAX_TWIST, G.MAX_TWIST)
                env.step(a)
                obs_img, _ = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, c, image_size)
                hist.append(_prep(obs_img, image_size))
            env.close()

    def fit():
        Xc_t = torch.as_tensor(np.asarray(Xc), device=device)
        Xt_t = torch.as_tensor(np.asarray(Xt), device=device)
        Y_t = torch.as_tensor(np.asarray(Y, np.float32), device=device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        last = torch.zeros((), device=device)
        for _ in range(epochs):
            idx = torch.randint(0, Y_t.shape[0], (min(64, Y_t.shape[0]),), device=device)
            last = ((mt * torch.tanh(net(Xc_t[idx], Xt_t[idx])) - Y_t[idx]) ** 2).mean()
            opt.zero_grad(); last.backward(); opt.step()
        return float(last)

    collect(use_net=False)
    print(f"  [cnn-base calib] round 0 (analytic): {len(Y)} samples, fit twistMSE {fit():.4f}")
    for r in range(dagger_rounds):
        collect(use_net=True)
        print(f"  [cnn-base calib] DAgger round {r + 1}: {len(Y)} samples, fit twistMSE {fit():.4f}")
    net.eval()
    return {"kind": "cnn", "net": net, "mt": mt, "device": device}


def cnn_base_twist(cur_stack, tgt_prep):
    import torch
    from usbg import goal_rl as G
    if _IBVS is None:
        return np.zeros(6, dtype=np.float64)
    net, mt, device = _IBVS["net"], _IBVS["mt"], _IBVS["device"]
    with torch.no_grad():
        c = torch.as_tensor(cur_stack[None], dtype=torch.float32, device=device)
        t = torch.as_tensor(tgt_prep[None, None], dtype=torch.float32, device=device)
        tw = (mt * torch.tanh(net(c, t)))[0].cpu().numpy()
    return np.clip(tw, -G.MAX_TWIST, G.MAX_TWIST)


def base_twist(pose, T_tgt, cur_lab, tgt_lab, cur_stack, tgt_prep, residual):
    """Residual-policy base action (#3) as EXECUTED. Zero when residual is off; otherwise the analytic pose
    base (_RESIDUAL_MODE 'oracle', sim upper bound), the segmentation visual-servoing base (_RESIDUAL_MODE
    'image', pose-free), or the learned appearance-encoder-over-history base (_RESIDUAL_MODE 'cnn').
    v2.1 (opt-in): with a planted deficit (_DEFICIT) the executed base is gain * base + bias."""
    b = nominal_base_twist(pose, T_tgt, cur_lab, tgt_lab, cur_stack, tgt_prep, residual)
    if residual and _DEFICIT is not None:
        b = _DEFICIT["gain"] * b + _DEFICIT["bias"]
    return b


def nominal_base_twist(pose, T_tgt, cur_lab, tgt_lab, cur_stack, tgt_prep, residual):
    """The residual base without any planted deficit (identical to base_twist when _DEFICIT is None).
    The imitation / BC-anchor teacher uses this, so a planted deficit stays invisible to supervision."""
    if not residual:
        return np.zeros(6, dtype=np.float64)
    if _RESIDUAL_MODE == "cnn":
        return cnn_base_twist(cur_stack, tgt_prep)
    if _RESIDUAL_MODE == "image":
        return image_base_twist(cur_lab, tgt_lab)
    from usbg import goal_rl as G
    return G.analytic_twist(pose, T_tgt)


def residual_target(pose, T_tgt, cur_lab, tgt_lab, cur_stack, tgt_prep, residual):
    """Imitation target for the policy: the FULL analytic corrective if not residual, else the part
    the base does not supply (analytic minus base), so residual training regresses only the gap."""
    from usbg import goal_rl as G
    full = G.analytic_twist(pose, T_tgt)
    if not residual:
        return full
    return full - nominal_base_twist(pose, T_tgt, cur_lab, tgt_lab, cur_stack, tgt_prep, True)


def load_volumes(cases):
    """data/cache/<case>.npz via volume.load_cache (which reads data/cache_anat/ under the anatomical
    seat); lv.meta['case'] is the id the windows are looked up by."""
    from usbg import volume as V
    lvs = [V.load_cache(os.path.join(_REPO, "data/cache", f"{c}.npz")) for c in cases]
    for lv, c in zip(lvs, cases):
        lv.meta.setdefault("case", c)
    return lvs


def resolve_volume(args, which="train"):
    """Load the volumes for a split. `which` is "train" or "eval"; they are separate lists so a
    reported pose error is not measured on the volumes the policy trained on."""
    from usbg import volume as V
    if args.ct and args.seg:
        return [V.load_labeled_ct(args.ct, args.seg)], ["user_ct"]
    cases = args.train_cases if which == "train" else (args.eval_cases or args.train_cases)
    return load_volumes(cases), cases


def _prep(img01, size):
    from usbg.eval import _prep as p
    return p(img01, size)


def rollout(env, lv, Wfv, T_tgt, translator, cfg, policy, sim, history, image_size, device,
            w_seg, w_app, n_steps, deterministic=False, effort=0.003,
            fix_reward=False, shape_coef=0.02, gamma=0.9, residual=False):
    """One view-matching episode. Reward = per-step GAIN in the combined seg+appearance match to the
    target view (telescopes to net improvement), minus an action-cost term. With fix_reward (#1) the
    action-cost term is dropped and a potential-based distance-to-goal shaping is added; with residual
    (#3) the executed action is the analytic base twist plus the policy's bounded residual."""
    import torch
    from usbg import view_match as VM, goal_rl as G
    tgt_img, tgt_lab = VM.curvi_render(lv, T_tgt, Wfv, translator, cfg, image_size)
    tgt_prep = _prep(tgt_img, image_size)
    tgt_t = torch.from_numpy(tgt_prep[None, None]).to(device)
    env.reset()
    obs_img, obs_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, cfg, image_size)
    hist = deque([_prep(obs_img, image_size)] * history, maxlen=history)
    s_prev = VM.combined_score(obs_img, obs_lab, tgt_img, tgt_lab, sim, w_seg, w_app)
    phi_prev = pose_potential(env.world_from_probe, T_tgt)
    eff = 0.0 if fix_reward else effort
    logps, rewards = [], []
    for _ in range(n_steps):
        cur_np = np.stack(hist, 0)
        cur = torch.from_numpy(cur_np[None]).to(device)
        a, logp = policy.act(cur, tgt_t, deterministic=deterministic)
        a_pol = a[0].detach().cpu().numpy()
        a_exec = np.clip(a_pol + base_twist(env.world_from_probe, T_tgt, obs_lab, tgt_lab,
                                            cur_np, tgt_prep, residual), -G.MAX_TWIST, G.MAX_TWIST)
        env.step(a_exec)
        obs_img, obs_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, cfg, image_size)
        hist.append(_prep(obs_img, image_size))
        s = VM.combined_score(obs_img, obs_lab, tgt_img, tgt_lab, sim, w_seg, w_app)
        r = (s - s_prev) - eff * float(a_pol @ a_pol)
        if fix_reward:
            phi = pose_potential(env.world_from_probe, T_tgt)
            r += shape_coef * (gamma * phi - phi_prev)
            phi_prev = phi
        rewards.append(r); s_prev = s
        if logp is not None:
            logps.append(logp[0])
    return {"log_probs": logps, "rewards": np.asarray(rewards), "s_final": s_prev,
            "pose": env.world_from_probe, "tgt_img": tgt_img, "tgt_lab": tgt_lab}


def imitation_batch(lv_list, rng, translator, cfg, history, image_size, batch, setup_kw, device,
                    residual=False, dr_rng=None):
    """Privileged warm-start: from a perturbed start, the corrective twist toward the (hidden) target
    pose, with the current curvilinear render as the policy input and the target render as the goal.
    In residual mode (#3) the target is zero, since the analytic base already supplies the corrective."""
    from usbg import view_match as VM
    curs, tgts, tws = [], [], []
    for _ in range(batch):
        lv, organ = draw_cell(lv_list, rng)
        Wfv, T_tgt, T_cur, _m = make_setup(lv, rng, organ, setup_kw)
        c = episode_cfg(cfg, dr_rng)                       # one DR draw per (current, target) sample
        cur_img, cur_lab = VM.curvi_render(lv, T_cur, Wfv, translator, c, image_size)
        tgt_img, tgt_lab = VM.curvi_render(lv, T_tgt, Wfv, translator, c, image_size)
        cstack = np.stack([_prep(cur_img, image_size)] * history, 0); tprep = _prep(tgt_img, image_size)
        curs.append(cstack)
        tgts.append(tprep[None])
        tws.append(residual_target(T_cur, T_tgt, cur_lab, tgt_lab, cstack, tprep, residual))
    return (np.asarray(curs, np.float32), np.asarray(tgts, np.float32), np.asarray(tws, np.float32))


def dagger_batch(lv_list, rng, translator, cfg, policy, history, image_size, batch, setup_kw, device,
                 n_steps, residual=False, dr_rng=None):
    """On-policy relabeling (fix #4, DAgger): roll the current deterministic policy from perturbed
    starts and, at each visited (current, target) state, relabel with the analytic corrective twist
    from that DRIFTED pose, so imitation trains on the policy's own state distribution (killing the
    covariate shift that makes a plain-BC policy under-react on states it never saw a label for)."""
    import torch
    from usbg import goal_rl as G, view_match as VM
    curs, tgts, tws = [], [], []
    while len(curs) < batch:
        lv, organ = draw_cell(lv_list, rng)
        Wfv, T_tgt, T_start, _m = make_setup(lv, rng, organ, setup_kw)
        env = make_env(lv, Wfv, T_start, device, rng)
        cfg_ep = episode_cfg(cfg, dr_rng)                  # one DR draw per trajectory (target + obs)
        tgt_img, tgt_lab = VM.curvi_render(lv, T_tgt, Wfv, translator, cfg_ep, image_size)
        tgt_t = torch.from_numpy(_prep(tgt_img, image_size)[None, None]).to(device)
        env.reset()
        obs_img, obs_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, cfg_ep, image_size)
        hist = deque([_prep(obs_img, image_size)] * history, maxlen=history)
        for _ in range(n_steps):
            cur_stack = np.stack(hist, 0).astype(np.float32)
            tgt_p = _prep(tgt_img, image_size).astype(np.float32)
            label = residual_target(env.world_from_probe, T_tgt, obs_lab, tgt_lab, cur_stack, tgt_p, residual)
            curs.append(cur_stack)
            tgts.append(tgt_p[None])
            tws.append(np.asarray(label, np.float32))
            a, _ = policy.act(torch.from_numpy(cur_stack[None]).to(device), tgt_t, deterministic=True)
            a_exec = np.clip(a[0].detach().cpu().numpy()
                             + base_twist(env.world_from_probe, T_tgt, obs_lab, tgt_lab, cur_stack, tgt_p, residual), -G.MAX_TWIST, G.MAX_TWIST)
            env.step(a_exec)
            obs_img, obs_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, cfg_ep, image_size)
            hist.append(_prep(obs_img, image_size))
            if len(curs) >= batch:
                break
        env.close()
    return (np.asarray(curs, np.float32), np.asarray(tgts, np.float32), np.asarray(tws, np.float32))


def eval_cells(lv_list, cases, design="crossed"):
    """Episode -> (volume index, target) assignment.

    legacy : ep -> (ep % n_vol, SEAT_ORGANS[ep % n_org]), window drawn by sample_setup. With 6 held-out
             volumes and 3 organs this CONFOUNDS patient and target (gcd 3): s0358/s0591 only ever got
             liver, s0513/s0649 only kidney, s0519/s0762 only spleen. Kept to reproduce the v1 numbers.
    crossed: every valid (patient, target) cell in patient-major order; episode ep takes cell
             ep % n_cells and screened window ep // n_cells (modulo the cell's window count), so each
             patient is evaluated on every target it has and the windows are walked in rank order.
    Returns (cells or None, skipped {"case:target": n_windows})."""
    if design == "legacy":
        return None, {}
    return target_cells(lv_list, cases)


def evaluate(lv_list, cases, translator, cfg, policy, sim, history, image_size, device, w_seg, w_app,
             setup_kw, n_steps, n_episodes, seed=999, residual=False, design="crossed",
             eval_randomize="off", log_episodes=True, episode_hook=None, *, setup_only=False):
    """Self-supervised eval on held-out target views: ground-truth pose error (translator-independent),
    plus final seg-IoU and combined match, vs a static (no-move) baseline on the SAME setups.

    v2: per-episode records (episodes are nested in target and patient, so the mixed model needs them),
    the crossed (patient x target x window) design, a pose-free keep-best (argmax of the non-privileged
    combined score over the trajectory), contact-force stats, start-clamp logging, and an optional
    per-episode DR draw identical across every arm and stage (seeded by (seed, ep), independent of the
    policy's training seed). Same top-level keys as before, so the earlier aggregators still work.

    v2.1.2 (opt-in): the contact surface follows _CONTACT_SURFACE (contact_setup); setup_only=True returns
    {"setups": [setup_record per episode], ...} without building an env or rendering anything."""
    import torch
    from usbg import goal_rl as G, view_match as VM, render_lotus as R
    from usbg.placement import NoAcceptedWindow
    cells, skipped_cells = eval_cells(lv_list, cases, design)
    if cells is not None and not cells:
        raise SystemExit(f"no eval (patient, target) cell has a screened window: {skipped_cells}")
    eps, skipped_eps = [], Counter()
    setups = []
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed + ep)
        if cells is None:                                                   # legacy assignment
            vi, organ, window = ep % len(lv_list), SEAT_ORGANS[ep % len(SEAT_ORGANS)], None
        else:
            vi, organ = cells[ep % len(cells)]
            window = ep // len(cells)
        lv = lv_list[vi]
        try:
            Wfv, T_tgt, T_start, meta = make_setup(lv, rng, organ, setup_kw, window=window, count=False)
        except NoAcceptedWindow:                       # legacy design under the anatomical seat only
            skipped_eps[f"{cases[vi]}:{organ}"] += 1
            continue
        T_run, plane, plane_rec = contact_setup(lv, Wfv, T_tgt, T_start, meta)
        if setup_only:
            setups.append(setup_record(ep, cases[vi], meta, T_tgt, T_run, plane, plane_rec, Wfv))
            continue
        env = make_env(lv, Wfv, T_run, device, np.random.default_rng(seed + ep), contact_plane=plane)
        c = R.domain_randomize(R.RenderConfig(), np.random.default_rng([seed, ep, _DR_SALT])) \
            if eval_randomize == "episode" else cfg
        if hasattr(policy, "reset"):
            policy.reset()
        tgt_img, tgt_lab = VM.curvi_render(lv, T_tgt, Wfv, translator, c, image_size)
        tgt_prep = _prep(tgt_img, image_size)
        tgt_t = torch.from_numpy(tgt_prep[None, None]).to(device)
        env.reset()
        p0, a0 = VM.pose_error(env.world_from_probe, T_tgt)
        s0_img, s0_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, c, image_size)
        m0 = VM.combined_score(s0_img, s0_lab, tgt_img, tgt_lab, sim, w_seg, w_app)
        rec = dict(ep=ep, patient=cases[vi], target=meta["target_used"], target_requested=meta["target_requested"],
                   window=meta.get("window"), n_windows=meta.get("n_windows"),
                   start_clamped=bool(meta.get("start_clamped")), clamp_mm=float(meta.get("clamp_mm") or 0.0),
                   start_pos=p0, start_ang=a0, static_seg=VM.seg_organ_iou(s0_lab, tgt_lab), static_match=m0,
                   **_window_info(lv, meta["target_used"], meta.get("window")))
        hist = deque([_prep(s0_img, image_size)] * history, maxlen=history)
        cur_lab = s0_lab
        pos_tr, ang_tr, match_tr, seg_tr, forces = [p0], [a0], [m0], [rec["static_seg"]], []
        for _ in range(n_steps):
            cur_np = np.stack(hist, 0)
            cur = torch.from_numpy(cur_np[None]).to(device)
            a, _ = policy.act(cur, tgt_t, deterministic=True)
            a_exec = np.clip(a[0].detach().cpu().numpy()
                             + base_twist(env.world_from_probe, T_tgt, cur_lab, tgt_lab, cur_np, tgt_prep, residual), -G.MAX_TWIST, G.MAX_TWIST)
            env.step(a_exec)
            forces.append(env.contact_force())
            oi, cur_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, c, image_size)
            hist.append(_prep(oi, image_size))
            pe, ae = VM.pose_error(env.world_from_probe, T_tgt)
            pos_tr.append(pe); ang_tr.append(ae)
            seg_tr.append(VM.seg_organ_iou(cur_lab, tgt_lab))
            match_tr.append(VM.combined_score(oi, cur_lab, tgt_img, tgt_lab, sim, w_seg, w_app))
        kb = int(np.argmax(match_tr))                                      # pose-free keep-best
        rec.update(final_pos=pos_tr[-1], final_ang=ang_tr[-1], final_seg=seg_tr[-1], final_match=match_tr[-1],
                   keepbest_step=kb, keepbest_pos=pos_tr[kb], keepbest_ang=ang_tr[kb], keepbest_seg=seg_tr[kb],
                   peak_force=float(max(forces) if forces else 0.0),
                   contact_frac=float(np.mean([f > 0 for f in forces])) if forces else 0.0,
                   pos_traj=[round(float(x), 3) for x in pos_tr], match_traj=[round(float(x), 4) for x in match_tr])
        if episode_hook is not None:                   # v2.1 opt-in extra fields (None = the v2 record)
            rec.update(episode_hook(env.world_from_probe, T_tgt))
        plane_episode_fields(rec, plane_rec, meta, env.world_from_probe)     # v2.1.2 opt-in (None = no change)
        eps.append(rec)
        env.close()
    if setup_only:
        return {"setups": setups, "contact_surface": _CONTACT_SURFACE, "n_episodes": len(setups),
                "skipped_episodes": dict(skipped_eps), "design": design, "eval_seed": seed}
    if not eps:
        raise SystemExit(f"every eval episode was skipped: {dict(skipped_eps)}")
    keys = {"pos": "final_pos", "ang": "final_ang", "seg": "final_seg", "match": "final_match",
            "start_pos": "start_pos", "start_ang": "start_ang", "static_seg": "static_seg",
            "static_match": "static_match", "keepbest_pos": "keepbest_pos", "keepbest_ang": "keepbest_ang",
            "keepbest_seg": "keepbest_seg", "peak_force": "peak_force"}
    res = {k: float(np.mean([e[v] for e in eps])) for k, v in keys.items()}   # v1 top-level keys + v2 ones
    res["start_clamped_frac"] = float(np.mean([e["start_clamped"] for e in eps]))
    res["clamp_mm_mean"] = float(np.mean([e["clamp_mm"] for e in eps]))

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
    res.update(n_episodes=len(eps), design=design, eval_randomize=eval_randomize, eval_seed=seed,
               contact="on" if _CONTACT else "off", seat=_seat(),
               skipped_cells=skipped_cells, skipped_episodes=dict(skipped_eps))
    if _PLANE_LOG:                                     # v2.1.2 opt-in only: no key by default
        res["contact_surface"] = _CONTACT_SURFACE
    if log_episodes:
        res["episodes"] = eps
    return res


# ------------------------------------------------- v2.1 RL positive control (the v2.1 study; exploratory)
# Everything below is reached only through the opt-in flags --init-from, --base-bias, --base-gain and
# --curve-every; a run without them is the v2 trainer, byte for byte in its outputs.
def deficit_twist(bias_step):
    """--base-bias, given per control step in the probe body frame ([mm, mm, mm, deg, deg, deg] for
    [x lateral, y elevational, z axial, wx, wy, wz]) -> the SI body twist (m/s, rad/s) env.step integrates
    over one DT_C."""
    from usbg.contracts_bridge import DT_C
    b = np.asarray(bias_step, np.float64).reshape(6)
    return np.concatenate([b[:3] / 1000.0, np.radians(b[3:])]) / DT_C


def twist_to_step_units(tw):
    """SI body twist -> per-step [mm, mm, mm, deg, deg, deg] (the inverse of deficit_twist)."""
    from usbg.contracts_bridge import DT_C
    t = np.asarray(tw, np.float64).reshape(6) * DT_C
    return np.concatenate([t[:3] * 1000.0, np.degrees(t[3:])])


def final_axes(T_cur, T_tgt):
    """Signed final offset in the TARGET probe frame (probe x lateral, y elevational, z axial into the
    tissue): the probe-origin displacement of inv(T_tgt) @ T_cur in mm, and its rotation vector in deg.
    This translation is the plain displacement; pose_error's norm uses the se(3) log translation, which
    differs slightly when the rotation error is large."""
    from usbg.contracts_bridge import invert_T, se3_log
    rel = invert_T(np.asarray(T_tgt)) @ np.asarray(T_cur)
    t = rel[:3, 3] * 1000.0
    w = np.degrees(se3_log(rel)[3:])
    return {"final_dlat_mm": float(t[0]), "final_delev_mm": float(t[1]), "final_daxial_mm": float(t[2]),
            "final_rot_x_deg": float(w[0]), "final_rot_y_deg": float(w[1]), "final_rot_z_deg": float(w[2])}


class ActRecorder:
    """Evaluation-only wrapper. act() is forwarded unchanged; the policy's deterministic output (the
    residual, before the base and the planted deficit are added) is summed per episode. evaluate() calls
    reset() at each episode start and hook() at each episode end, so every record gets the episode's mean
    residual (per-step units) and the signed per-axis final offset."""

    def __init__(self, policy):
        self.policy, self._acc, self._n = policy, np.zeros(6), 0

    def reset(self):
        self._acc, self._n = np.zeros(6), 0
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, cur, tgt, deterministic=False):
        a, logp = self.policy.act(cur, tgt, deterministic=deterministic)
        self._acc += a[0].detach().cpu().numpy().astype(np.float64)
        self._n += 1
        return a, logp

    def hook(self, T_cur, T_tgt):
        rec = final_axes(T_cur, T_tgt)
        rec["resid_mean_step"] = [round(float(x), 5) for x in twist_to_step_units(self._acc / max(self._n, 1))]
        return rec


def buffers_snapshot(module):
    """Copies of every buffer (BatchNorm running statistics included). In train-mode BatchNorm an eval
    forward pass still updates the running statistics, so a held-out curve eval restores them afterwards
    and leaves training (and the saved weights) exactly as without the curve."""
    return {k: v.detach().clone() for k, v in module.named_buffers()}


def buffers_restore(module, snap):
    import torch
    with torch.no_grad():
        for k, v in module.named_buffers():
            v.copy_(snap[k])


_CURVE_EP_KEYS = ("ep", "patient", "target", "window", "start_pos", "final_pos", "final_ang", "final_match",
                  "final_seg", "keepbest_pos", "final_dlat_mm", "final_delev_mm", "final_daxial_mm",
                  "final_rot_x_deg", "final_rot_y_deg", "final_rot_z_deg", "resid_mean_step")


def curve_point(it, ev, steps_per_iter, log_std=None):
    """One held-out error-curve point: the summary of an evaluate() result plus compact per-episode
    records (the patient-cluster bootstrap needs them)."""
    eps = ev.get("episodes") or []
    pt = {"iter": int(it), "rl_online_steps": int(it * steps_per_iter)}
    pt.update({k: ev[k] for k in ("pos", "ang", "match", "seg", "keepbest_pos", "start_pos", "static_match",
                                  "n_episodes")})
    pt["per_patient"] = {p: {"n": v["n"], "pos": v["pos"], "ang": v["ang"]} for p, v in ev["per_patient"].items()}
    if eps and "final_dlat_mm" in eps[0]:
        pt["dlat_mm_mean"] = float(np.mean([e["final_dlat_mm"] for e in eps]))
        pt["resid_mean_step"] = [float(x) for x in np.mean([e["resid_mean_step"] for e in eps], 0)]
    if log_std is not None:
        pt["log_std"] = log_std
    pt["episodes"] = [{k: e[k] for k in _CURVE_EP_KEYS if k in e} for e in eps]
    return pt


V21_TRAINER_FLAGS = ("init_from", "base_bias", "base_gain", "curve_every")


def config_record(args, v21):
    """The resolved config written into the output JSON and the checkpoints. Without a v2.1 flag the
    v2.1 argparse keys are left out, so a default run's config has exactly the v2 keys (v2 schema)."""
    c = dict(vars(args))
    if not v21:
        for k in V21_TRAINER_FLAGS:
            c.pop(k, None)
    return c


def load_init_policy(path, args, prov):
    """--init-from: read a saved trainer checkpoint (<run>_imit.pt or <run>.pt) and check that it fits this
    run. Hard errors: policy head, history, image size, residual mode and bound, image-base kind and
    features, translator and texture md5, target set, mask source and segmenter md5, and seat. Other
    differences (reward weights,
    start perturbation, train cases, seed, and eval cases the checkpoint was trained on, which makes the
    curve an in-distribution diagnostic) are printed and recorded as warnings.
    Returns (checkpoint dict, image-base dict for _IBVS or None, init record)."""
    import torch
    from usbg import provenance as PROV
    if not os.path.exists(path):
        raise SystemExit(f"--init-from {path} does not exist")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c0 = ck.get("config") or {}
    p0 = ck.get("provenance") or {}
    bad, warn = [], []

    def need(name, got, want):
        if got != want:
            bad.append(f"{name}: checkpoint {got!r} vs this run {want!r}")

    def soft(name, got, want):
        if got != want:
            warn.append(f"{name}: checkpoint {got!r} vs this run {want!r}")

    mode_now = _RESIDUAL_MODE if args.fix_residual else "none"
    need("policy", ck.get("policy", "cnn"), args.policy)
    need("history", ck.get("history"), args.history)
    need("image_size", ck.get("image_size"), args.image_size)
    need("residual_mode", ck.get("residual_mode"), mode_now)
    if args.fix_residual:
        need("residual_frac", float(ck.get("residual_frac", -1)), float(args.residual_frac))
    need("translator", c0.get("translator"), args.translator)
    need("texture_md5", (p0.get("texture") or {}).get("md5"), (prov.get("texture") or {}).get("md5"))
    need("targets", list(ck.get("targets") or []), list(SEAT_ORGANS))
    need("mask_learned", bool((p0.get("mask") or {}).get("learned")), bool((prov.get("mask") or {}).get("learned")))
    need("seg_md5", (p0.get("mask") or {}).get("seg_md5"), (prov.get("mask") or {}).get("seg_md5"))
    need("seat", p0.get("seat_mode"), prov.get("seat_mode"))
    overlap = sorted(set(c0.get("train_cases") or []) & set(args.eval_cases or []))
    if overlap:                                          # in-distribution diagnostic, never a held-out number
        warn.append(f"the checkpoint was trained on this run's eval cases {overlap}: the curve is NOT held-out")
    ib, ibvs = ck.get("ibvs"), None
    if mode_now == "image":
        if ib is None:
            bad.append("residual mode 'image' but the checkpoint has no calibrated image base")
        else:
            need("ibvs_base", ib.get("kind"), args.ibvs_base)
            need("ibvs_rich", bool(ib.get("rich")), args.ibvs_features == "rich")
            if ib.get("kind") == "linear":
                model = np.asarray(ib["model"], np.float64)
            else:
                import torch.nn as nn
                sd = ib["model"]
                model = nn.Sequential(nn.Linear(sd["0.weight"].shape[1], 64), nn.ReLU(), nn.Linear(64, 64),
                                      nn.ReLU(), nn.Linear(64, 6))
                model.load_state_dict(sd)
                model.eval()
            ibvs = {"kind": ib.get("kind"), "model": model, "rich": bool(ib.get("rich"))}
    elif mode_now == "cnn":
        bad.append("the CNN appearance base is not saved in checkpoints; --init-from supports the image and "
                   "oracle bases only")
    for k in ("w_seg", "w_app", "lateral_m", "axial_m", "fan_deg", "min_windows", "contact", "randomize",
              "eval_design", "eval_seed", "n_steps", "seed"):
        soft(k, c0.get(k), getattr(args, k))
    soft("train_cases", sorted(c0.get("train_cases") or []), sorted(args.train_cases or []))
    if bad:
        raise SystemExit("--init-from " + path + " does not fit this run:\n  " + "\n  ".join(bad))
    sc = PROV.sidecar_path(path)
    init = {"path": os.path.abspath(path), "md5": PROV.md5_file(path), "phase": ck.get("phase"),
            "provenance_json": sc if os.path.exists(sc) else None,
            "provenance_json_md5": PROV.md5(sc) if os.path.exists(sc) else None,
            "seed": c0.get("seed"), "imitation": c0.get("imitation"), "imitation_batch": c0.get("imitation_batch"),
            "train_cases": c0.get("train_cases"), "ablation": c0.get("ablation"),
            "eval_overlaps_init_train": overlap,
            "residual_mode": ck.get("residual_mode"), "ibvs_kind": None if ib is None else ib.get("kind"),
            "log_std_source": "checkpoint" if "log_std" in ck else "rebuilt at the shipped initial value "
                              "(log(0.5 * max_twist); the v2 trainer never updates it without --fix-explore)",
            "warnings": warn,
            "rng_note": "calibration and imitation are skipped, so the RL-phase setup draws start at "
                        "default_rng(seed) and differ from the archived run's RL phase"}
    return ck, ibvs, init


def resolve_texture(translator, texture_ckpt=None, tex_manifest=None):
    """-> texture checkpoint path or None (physics). Order: --texture-ckpt (or 'none'), then the texture
    manifest {physics: null, cut: path, cyclegan: path, sonospade: path}, then (canonical seat only) the
    v1 CKPTS. A relative path resolves against the gym repo."""
    if texture_ckpt is not None:
        p = None if str(texture_ckpt).lower() == "none" else texture_ckpt
    elif tex_manifest is not None:
        m = json.load(open(tex_manifest))
        if translator not in m:
            raise SystemExit(f"{tex_manifest} has no entry for {translator!r} (keys {sorted(m)})")
        p = m[translator]
    elif _seat() == "canonical":
        p = CKPTS[translator]
    elif translator == "physics":
        p = None
    else:
        raise SystemExit(f"--translator {translator} under USBG_SEAT=anatomical needs --texture-ckpt or "
                         "--tex-manifest (the v1 CKPTS were trained on pairs that include held-out patients)")
    if p is not None and not os.path.isabs(p):
        p = os.path.join(_REPO, p)
    if (p is None) != (translator == "physics"):
        raise SystemExit(f"--translator {translator} with texture checkpoint {p!r}")
    if p is not None and not os.path.exists(p):
        raise SystemExit(f"texture checkpoint {p} does not exist")
    return p


def main(argv=None):
    t0 = time.time()   # wall-clock, written to the output JSON as wall_sec so the dashboard can show it
    ap = argparse.ArgumentParser()
    ap.add_argument("--translator", default="sonospade", choices=list(CKPTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ct", default=None); ap.add_argument("--seg", default=None)
    ap.add_argument("--train-cases", nargs="+", default=None,
                    help="default: canonical = the v1 10-case list; anatomical = usbg.benchmark.TRAIN_PATIENTS")
    ap.add_argument("--eval-cases", nargs="+", default=None,
                    help="patients to EVALUATE on. Canonical default: --train-cases (an in-distribution "
                         "diagnostic, as in the v1 code). Anatomical default: the benchmark held-out "
                         f"split {' '.join(EVAL_CASES)}")
    ap.add_argument("--similarity", default="ssim",
                    help="inert for view-match: the appearance term is fixed sector-interior NCC by "
                         "design (whole-frame similarity is dominated by the shared black border). "
                         "Kept only for CLI symmetry with the goal task; combined_score ignores it.")
    ap.add_argument("--w-seg", type=float, default=0.5); ap.add_argument("--w-app", type=float, default=0.5)
    ap.add_argument("--imitation", type=int, default=200); ap.add_argument("--imitation-batch", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20); ap.add_argument("--episodes-per-iter", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=12); ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--rl-lr", type=float, default=1e-4)
    ap.add_argument("--bc-anchor", type=float, default=0.3); ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lateral-m", type=float, default=None,
                    help="in-plane start perturbation; default 0.03 (canonical, v1) / 0.02 (anatomical: the "
                         "radius over which the window's skin is screened flat, usbg.windows.FLAT_RADIUS_M)")
    ap.add_argument("--axial-m", type=float, default=0.01)
    ap.add_argument("--fan-deg", type=float, default=18.0)
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--real-eval", default=None, help="LEGACY npz of real US targets (needs --allow-legacy-seg)")
    ap.add_argument("--out", required=True); ap.add_argument("--device", default=None)
    # under-movement fixes (compose as an ablation ladder)
    ap.add_argument("--fix-reward", action="store_true", help="#1 drop the action-cost penalty + add potential-based distance-to-goal shaping")
    ap.add_argument("--fix-explore", action="store_true", help="#2 unfreeze the std (put log_std in the RL optimizer) + entropy bonus")
    ap.add_argument("--fix-residual", action="store_true", help="#3 execute analytic base twist + a learned bounded residual")
    ap.add_argument("--fix-dagger", action="store_true", help="#4 relabel the policy's drifted states with the analytic expert (DAgger)")
    ap.add_argument("--shape-coef", type=float, default=0.02); ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--residual-frac", type=float, default=0.5); ap.add_argument("--dagger-frac", type=float, default=0.5)
    ap.add_argument("--residual-image", action="store_true", help="#3-image: calibrated image visual-servoing base (pose-free, non-privileged) instead of the analytic pose oracle")
    ap.add_argument("--residual-cnn", action="store_true", help="#3-cnn: learned appearance-encoder-over-history base (pose-free), calibrated on short analytic rollouts")
    ap.add_argument("--cnn-traj", type=int, default=120); ap.add_argument("--cnn-epochs", type=int, default=400)
    ap.add_argument("--cnn-dagger", type=int, default=0, help="DAgger rounds for the CNN base (roll base, relabel with analytic, refit)")
    ap.add_argument("--policy", default="cnn", choices=["cnn", "mdn"], help="#7: mdn = K-component Gaussian-mixture head (multimodal), fights BC regression-to-the-mean")
    ap.add_argument("--n-mix", type=int, default=5)
    ap.add_argument("--ibvs-base", default="linear", choices=["linear", "mlp"], help="image visual-servoing base model")
    ap.add_argument("--ibvs-features", default="simple", choices=["simple", "rich"], help="image base features (rich adds orientation/elongation for rotation)")
    ap.add_argument("--ablation", default="", help="label recorded in the output JSON")
    # ---- v2 flags ----
    ap.add_argument("--texture-ckpt", default=None,
                    help="texture checkpoint for --translator ('none' for physics); overrides --tex-manifest")
    ap.add_argument("--tex-manifest", default=None,
                    help="texture manifest {physics: null, cut: path, cyclegan: path, sonospade: path}")
    ap.add_argument("--targets", nargs="+", default=None, choices=["liver", "kidney", "spleen"],
                    help="target organs. Default: canonical = liver kidney spleen (v1); anatomical = liver kidney")
    ap.add_argument("--min-windows", type=int, default=1,
                    help="anatomical: a (case, target) cell needs >= this many screened windows, else it is "
                         "skipped (logged). 2 skips s0058/s0522/s0727 kidney on usbg-windows-v1")
    ap.add_argument("--randomize", default="off", choices=["off", "episode"],
                    help="DR during training/calibration: one draw per episode/sample, shared by target+obs")
    ap.add_argument("--eval-randomize", default="off", choices=["off", "episode"],
                    help="DR during the in-run evals (default render = off)")
    ap.add_argument("--contact", default=None, choices=["on", "off"],
                    help="patient-surface contact (default: $USBG_CONTACT, else on)")
    ap.add_argument("--eval-design", default="crossed", choices=["crossed", "legacy"],
                    help="episode->(patient,target,window) assignment; legacy reproduces the v1 confound")
    ap.add_argument("--eval-seed", type=int, default=999)
    ap.add_argument("--allow-legacy-seg", action="store_true",
                    help="permit the withdrawn shared segmenter / an implicit US_SEG_CKPT / --real-eval "
                         "(canonical-seat traceability only)")
    # ---- v2.1 (exploratory) RL positive control, the v2.1 study. All opt-in; defaults = the v2 trainer ----
    ap.add_argument("--init-from", default=None,
                    help="v2.1: start from a saved trainer checkpoint (e.g. <run>_imit.pt): its weights, residual "
                         "bound and calibrated image base. Skips calibration and imitation (pass --imitation 0)")
    ap.add_argument("--base-bias", type=float, nargs=6, default=None, metavar="B",
                    help="v2.1: planted deficit, a constant bias added to the EXECUTED residual base each step, "
                         "in the probe body frame: x lateral, y elevational, z axial (mm per step), then wx wy wz "
                         "(deg per step). e.g. 0.5 0 0 0 0 0. The imitation/BC teacher never sees it")
    ap.add_argument("--base-gain", type=float, default=1.0,
                    help="v2.1: planted deficit, the executed residual base is multiplied by this gain "
                         "(e.g. 0.5); the imitation/BC teacher never sees it")
    ap.add_argument("--curve-every", type=int, default=0,
                    help="v2.1: held-out evaluation every N RL iterations (0 = off); each evaluation restores the "
                         "BatchNorm buffers afterwards, which leaves training unchanged")
    args = ap.parse_args(argv)
    if args.real_eval and not args.allow_legacy_seg:
        ap.error("--real-eval is the legacy in-run registration (training CTs, frames the old segmenter saw, "
                 "GT target mask); use scripts/eval_real_registration.py, or pass --allow-legacy-seg")
    v21 = bool(args.init_from or args.base_bias is not None or args.base_gain != 1.0 or args.curve_every)
    if (args.base_bias is not None or args.base_gain != 1.0) and not args.fix_residual:
        ap.error("--base-bias / --base-gain plant a deficit on the residual base; they need --fix-residual")
    if args.base_gain <= 0:
        ap.error("--base-gain must be > 0")
    if args.curve_every < 0:
        ap.error("--curve-every must be >= 0")
    if args.init_from and args.imitation != 0:
        ap.error("--init-from skips calibration and imitation; pass --imitation 0 so the config records what ran")
    if args.init_from and args.fix_dagger:
        ap.error("--fix-dagger acts during imitation, which --init-from skips")
    if v21 and args.real_eval:
        ap.error("the v2.1 flags are not supported with the legacy --real-eval")

    import torch
    from usbg import goal_rl as G, render_lotus as R, view_match as VM
    from usbg import provenance as PROV
    from usbg.benchmark import TRAIN_PATIENTS, TEST_PATIENTS
    from usbg.mujoco_env import contact_default
    from usbg.policy import GoalPolicy, GoalMDNPolicy
    from usbg.similarity import load_similarity
    global SEAT_ORGANS, _RANDOMIZE, _CONTACT, _MIN_WINDOWS, _DEFICIT
    seat = _seat()
    anat = seat == "anatomical"
    # ---- resolve the seat-dependent defaults (written back into args so config records what ran) ----
    if args.targets is None:
        args.targets = ["liver", "kidney"] if anat else list(LEGACY_TARGETS)
    if anat and "spleen" in args.targets and os.environ.get("USBG_ALLOW_SPLEEN", "0") != "1":
        ap.error("spleen is dropped as a target under the anatomical seat (set USBG_ALLOW_SPLEEN=1 to override)")
    SEAT_ORGANS = list(args.targets)
    _MIN_WINDOWS = int(args.min_windows)
    if not (args.ct and args.seg):
        if args.train_cases is None:
            args.train_cases = list(TRAIN_PATIENTS) if anat else list(TRAIN_CASES)
        if args.eval_cases is None and anat:
            args.eval_cases = list(TEST_PATIENTS)
        if anat and set(args.train_cases) & set(TEST_PATIENTS):
            ap.error(f"--train-cases includes held-out benchmark patients {sorted(set(args.train_cases) & set(TEST_PATIENTS))}")
    if args.lateral_m is None:
        args.lateral_m = 0.02 if anat else 0.03
    args.contact = args.contact or ("on" if contact_default() else "off")
    _RANDOMIZE, _CONTACT = args.randomize, (args.contact == "on")
    tex_ckpt = resolve_texture(args.translator, args.texture_ckpt, args.tex_manifest)
    args.texture_ckpt = tex_ckpt or "none"

    device = args.device or R.get_device()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dr_rng = np.random.default_rng([args.seed, _DR_SALT])     # DR draws never shift the setup stream

    prov = PROV.collect(stage=args.translator, texture_ckpt=tex_ckpt, randomize=args.randomize,
                        eval_randomize=args.eval_randomize, contact=args.contact,
                        train_cases=args.train_cases, eval_cases=args.eval_cases, targets=SEAT_ORGANS,
                        min_windows=_MIN_WINDOWS, eval_design=args.eval_design, eval_seed=args.eval_seed,
                        lateral_m=args.lateral_m, bn_mode=BN_MODE)
    if v21:                                              # v2.1 opt-in: deficit, init and status in provenance
        if args.base_bias is not None or args.base_gain != 1.0:
            bias_step = list(args.base_bias) if args.base_bias is not None else [0.0] * 6
            _DEFICIT = {"gain": float(args.base_gain), "bias": deficit_twist(bias_step)}
        prov["analysis_status"] = ANALYSIS_STATUS_V21
        prov["rlpos"] = {
            "plan_step": "RL positive control: a planted deficit that the reward can see",
            "deficit": None if _DEFICIT is None else {
                "bias_step_mm_deg": bias_step, "bias_twist_si": [float(x) for x in _DEFICIT["bias"]],
                "gain": _DEFICIT["gain"], "frame": "probe body (x lateral, y elevational, z axial)",
                "applies_to": "executed residual base in RL rollouts and every evaluation; NOT the "
                              "imitation / BC-anchor teacher (residual_target uses the nominal base)"},
            "init_from": None, "curve_every": int(args.curve_every)}
    PROV.assert_not_withdrawn(prov, allow=args.allow_legacy_seg)
    for role, p in (("texture", tex_ckpt), ("segmenter", (prov["mask"] or {}).get("seg_ckpt"))):
        why = PROV.is_private_descendant(p) if p else None
        if why:
            raise SystemExit(f"refusing to run: {role} {p} is private or a private-pool descendant ({why})")
    translator = None if tex_ckpt is None else \
        __import__("usbg.texture_gan", fromlist=["load_translator"]).load_translator(tex_ckpt, device)
    cfg = R.RenderConfig(seed=args.seed)
    sim = load_similarity(args.similarity, device=device)
    lv_list, case_names = resolve_volume(args, "train")
    ev_list, ev_names = resolve_volume(args, "eval")
    setup_kw = dict(lateral_m=args.lateral_m, axial_m=args.axial_m, fan_deg=args.fan_deg)
    print(f"[view-match {args.translator} s{args.seed}] seat={seat} volumes={case_names} eval={ev_names} "
          f"targets={SEAT_ORGANS} device={device} w_seg={args.w_seg} w_app={args.w_app} "
          f"mask={'learned ' + str(prov['mask'].get('seg_ckpt')) if prov['mask'].get('learned') else 'ct'} "
          f"texture={tex_ckpt} contact={args.contact} dr={args.randomize}")
    prov["cells"] = {"train": init_train_cells(lv_list, case_names)}
    _ev_cells, _ev_skip = eval_cells(ev_list, ev_names, args.eval_design)
    prov["cells"]["eval"] = {"design": args.eval_design, "skipped": _ev_skip,
                             "valid": None if _ev_cells is None else [f"{ev_names[i]}:{t}" for i, t in _ev_cells]}
    for k, n in sorted(_ev_skip.items()):
        print(f"  [cells] SKIP eval {k}: {n} screened window(s) (not substituted)")
    timing = {}

    global _RESIDUAL_MODE, _IBVS
    _RESIDUAL_MODE = "cnn" if args.residual_cnn else ("image" if args.residual_image else "oracle")
    tc = time.time()
    init_ck = None
    if args.init_from:                                   # v2.1: the checkpoint's own calibrated base, no RNG use
        init_ck, _IBVS, prov["rlpos"]["init_from"] = load_init_policy(args.init_from, args, prov)
        for w in prov["rlpos"]["init_from"]["warnings"]:
            print(f"  [init-from] WARNING {w}")
        print(f"  [init-from] {args.init_from} (phase {init_ck.get('phase')}, base {_IBVS and _IBVS['kind']}); "
              "calibration and imitation skipped")
    elif args.fix_residual and args.residual_cnn:
        _IBVS = calibrate_cnn_base(lv_list, rng, translator, cfg, args.image_size, args.history, setup_kw, device,
                                   n_traj=args.cnn_traj, epochs=args.cnn_epochs, dagger_rounds=args.cnn_dagger,
                                   dr_rng=dr_rng)
        print("  calibrated learned appearance-encoder-over-history base (CNN, pose-free)")
    elif args.fix_residual and args.residual_image:
        _IBVS = calibrate_ibvs(lv_list, rng, translator, cfg, args.image_size, setup_kw,
                               kind=args.ibvs_base, rich=(args.ibvs_features == "rich"), dr_rng=dr_rng)
        print(f"  calibrated image visual-servoing base (kind={args.ibvs_base}, rich={args.ibvs_features == 'rich'}, pose-free)")
    timing["calib_sec"] = round(time.time() - tc, 2)

    mt = (args.residual_frac * G.MAX_TWIST) if args.fix_residual else G.MAX_TWIST
    if args.policy == "mdn":
        policy = GoalMDNPolicy.build(history=args.history, n_mix=args.n_mix, max_twist=mt).to(device)
    else:
        policy = G.build_goal_gaussian_policy(GoalPolicy.build(history=args.history), max_twist=mt).to(device)
    if init_ck is not None:                              # v2.1: load the saved weights (bound checked first)
        if not np.allclose(np.asarray(init_ck.get("max_twist"), np.float64), np.asarray(mt, np.float64)):
            raise SystemExit(f"--init-from residual bound {init_ck.get('max_twist')} != this run's {mt}")
        (policy.net if hasattr(policy, "net") else policy).load_state_dict(init_ck["state_dict"])
        if "log_std" in init_ck and hasattr(policy, "log_std"):
            with torch.no_grad():
                policy.log_std.copy_(torch.as_tensor(init_ck["log_std"], dtype=torch.float32, device=device))
    if _DEFICIT is not None:
        cancellable = bool(np.all(np.abs(_DEFICIT["bias"]) <= np.asarray(mt, np.float64) + 1e-12))
        prov["rlpos"]["deficit"]["bias_within_residual_bound"] = cancellable
        print(f"  [deficit] executed base = {_DEFICIT['gain']} * base + bias "
              f"{[round(float(x), 4) for x in _DEFICIT['bias']]} (SI twist; within the residual bound: {cancellable})")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    fx = "+".join(n for n, on in [("reward", args.fix_reward), ("explore", args.fix_explore),
                  ("residual", args.fix_residual), ("dagger", args.fix_dagger)] if on) or "baseline"
    print(f"  ablation={args.ablation or fx}  fixes={fx}")

    # imitation warm-start (privileged corrective toward the hidden target pose; DAgger relabels the
    # policy's own drifted states in the second half when --fix-dagger)
    ti = time.time()
    for it in range(args.imitation):
        if args.fix_dagger and it >= args.imitation // 2 and rng.random() < args.dagger_frac:
            cur, tgt, tw = dagger_batch(lv_list, rng, translator, cfg, policy, args.history,
                                        args.image_size, args.imitation_batch, setup_kw, device,
                                        args.n_steps, residual=args.fix_residual, dr_rng=dr_rng)
        else:
            cur, tgt, tw = imitation_batch(lv_list, rng, translator, cfg, args.history, args.image_size,
                                           args.imitation_batch, setup_kw, device, residual=args.fix_residual,
                                           dr_rng=dr_rng)
        loss = policy.imitation_loss(torch.from_numpy(cur).to(device),
                                     torch.from_numpy(tgt).to(device), torch.from_numpy(tw).to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % max(1, args.imitation // 20) == 0:
            print(f"  imitation {it+1}/{args.imitation} twistMSE {loss.item():.4f}")
    timing["imitation_sec"] = round(time.time() - ti, 2)
    timing["imitation_sec_per_iter"] = round(timing["imitation_sec"] / max(args.imitation, 1), 4)

    def save_ckpt(path, phase):
        """Everything an eval-only reload needs: weights, the residual bound (NOT recoverable from the
        weights), the residual mode, the calibrated IBVS base (so no RNG replay is needed), the resolved
        config and the run provenance; then the <ckpt>.provenance.json sidecar."""
        sd_ = (policy.net if hasattr(policy, "net") else policy).state_dict()
        ib = None
        if _IBVS is not None and _IBVS.get("kind") in ("linear", "mlp"):
            m = _IBVS["model"]
            ib = {"kind": _IBVS["kind"], "rich": _IBVS["rich"],
                  "model": (np.asarray(m) if _IBVS["kind"] == "linear" else m.state_dict())}
        torch.save({"state_dict": sd_, "history": args.history, "image_size": args.image_size,
                    "policy": args.policy, "max_twist": np.asarray(mt), "residual_frac": args.residual_frac,
                    "residual_mode": (_RESIDUAL_MODE if args.fix_residual else "none"), "ibvs": ib,
                    "phase": phase, "config": config_record(args, v21), "targets": list(SEAT_ORGANS),
                    "provenance": prov,
                    **({"log_std": policy.log_std.detach().cpu().numpy()}
                       if v21 and hasattr(policy, "log_std") else {})}, path)
        if args.ct and args.seg:
            return
        parents = [PROV.parent_entry(tex_ckpt, "translator")] if tex_ckpt else []
        seg_ck = (prov["mask"] or {}).get("seg_ckpt")
        if seg_ck and os.path.exists(seg_ck):
            parents.append(PROV.parent_entry(seg_ck, "segmenter"))
        if args.init_from:
            parents.append(PROV.parent_entry(os.path.abspath(args.init_from), "init_policy"))
        v21_extra = {"analysis_status": ANALYSIS_STATUS_V21, "rlpos": prov.get("rlpos")} if v21 else {}
        PROV.write_provenance(
            path, "policy", train_cases=case_names, val_cases=[], iters=args.imitation + (args.iters if phase == "final" else 0),
            seed=args.seed, dr_mode="full" if args.randomize == "episode" else "off",
            dr_notes="one domain_randomize draw per episode/sample (target+obs share it)" if args.randomize == "episode" else "",
            parents=parents, seat_mode=seat,
            extra={"phase": phase, "translator": args.translator, "targets": list(SEAT_ORGANS),
                   "mask": prov["mask"], "windows": prov.get("windows"), "contact": args.contact,
                   "eval_cases": args.eval_cases, "bn_mode": BN_MODE, "ablation": args.ablation or fx, **v21_extra})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_ckpt(args.out.replace(".json", "_imit.pt"), "after_imitation")
    # v2.1 opt-in: evaluations run through an ActRecorder (mean residual and signed per-axis final offset
    # per episode); without the v2.1 flags the policy and the records are exactly the v2 ones.
    rec_pol = ActRecorder(policy) if v21 else None
    eval_pol, eval_hook = (rec_pol, rec_pol.hook) if v21 else (policy, None)
    te = time.time()
    ev_im = evaluate(ev_list, ev_names, translator, cfg, eval_pol, sim, args.history, args.image_size,
                     device, args.w_seg, args.w_app, setup_kw, args.n_steps, args.eval_episodes,
                     seed=args.eval_seed, residual=args.fix_residual, design=args.eval_design,
                     eval_randomize=args.eval_randomize, episode_hook=eval_hook)
    timing["eval_sec_per_episode"] = round((time.time() - te) / max(ev_im["n_episodes"], 1), 3)
    print(f"  [after imitation] start_pos {ev_im['start_pos']:.1f}mm -> pos {ev_im['pos']:.1f}mm "
          f"ang {ev_im['ang']:.1f}deg | seg {ev_im['static_seg']:.3f}->{ev_im['seg']:.3f} match {ev_im['match']:.3f} "
          f"| clamped starts {ev_im['start_clamped_frac']:.2f} peak force {ev_im['peak_force']:.2f} N")

    # RL fine-tune on the combined view-match reward (REINFORCE + BC anchor). With --fix-explore the
    # log_std joins the optimizer and an entropy bonus keeps exploration from collapsing.
    rl_base = policy.net if hasattr(policy, "net") else policy
    rl_params = list(rl_base.parameters()) + ([policy.log_std] if (args.fix_explore and hasattr(policy, "log_std")) else [])
    rl_opt = torch.optim.Adam(rl_params, lr=args.rl_lr)
    steps_per_iter = args.episodes_per_iter * args.n_steps

    def _log_std():
        return [round(float(x), 5) for x in policy.log_std.detach().cpu()] if hasattr(policy, "log_std") else None

    curve, iter_log, curve_sec = [], [], 0.0             # v2.1 only (stay empty / 0 otherwise)
    if v21:
        curve.append(curve_point(0, ev_im, steps_per_iter, _log_std()))
    tr0 = time.time()
    for it in range(args.iters):
        logps, advs, rews = [], [], []
        for ei in range(args.episodes_per_iter):
            lv, organ = cycle_cell(lv_list, it + ei)
            Wfv, T_tgt, T_start, _m = make_setup(lv, rng, organ, setup_kw)
            env = make_env(lv, Wfv, T_start, device, np.random.default_rng(args.seed + it * 91 + ei))
            roll = rollout(env, lv, Wfv, T_tgt, translator, episode_cfg(cfg, dr_rng), policy, sim, args.history,
                           args.image_size, device, args.w_seg, args.w_app, args.n_steps,
                           fix_reward=args.fix_reward, shape_coef=args.shape_coef, gamma=args.gamma,
                           residual=args.fix_residual)
            env.close()
            if not roll["log_probs"]:
                continue
            ret = G.discounted_returns(roll["rewards"], args.gamma)
            logps.extend(roll["log_probs"]); advs.extend(ret.tolist()); rews.extend(roll["rewards"])
        if not logps:
            continue
        raw_ret = sum(rews) / len(rews) if rews else float("nan")   # mean per-step view-match reward
        adv = torch.tensor(advs, dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        bc = torch.zeros((), device=device)
        if args.bc_anchor > 0:
            cur_b, tgt_b, tw_b = imitation_batch(lv_list, rng, translator, cfg, args.history,
                                                 args.image_size, args.imitation_batch, setup_kw, device,
                                                 residual=args.fix_residual, dr_rng=dr_rng)
            bc = policy.imitation_loss(torch.from_numpy(cur_b).to(device),
                                       torch.from_numpy(tgt_b).to(device), torch.from_numpy(tw_b).to(device))
        loss = -(torch.stack(logps) * adv).mean() + args.bc_anchor * bc
        if args.fix_explore and hasattr(policy, "log_std"):
            loss = loss - args.ent_coef * policy.log_std.sum()   # entropy bonus (up to a constant)
        rl_opt.zero_grad(); loss.backward(); rl_opt.step()
        if (it + 1) % max(1, args.iters // 20) == 0:
            ls = float(policy.log_std.mean().detach()) if hasattr(policy, "log_std") else float("nan")
            print(f"  iter {it+1}/{args.iters} reward {raw_ret:+.4f} bc {float(bc.detach()):.4f} logstd_mean {ls:.3f}")
        if v21:                                          # learning curve of the training return
            iter_log.append({"iter": it + 1, "reward_mean_step": round(float(raw_ret), 6),
                             "bc": round(float(bc.detach()), 6)})
        if v21 and args.curve_every and (it + 1) % args.curve_every == 0 and (it + 1) < args.iters:
            tcv = time.time()                            # held-out curve point; BN buffers restored after
            snap = buffers_snapshot(policy)
            ev_k = evaluate(ev_list, ev_names, translator, cfg, eval_pol, sim, args.history, args.image_size,
                            device, args.w_seg, args.w_app, setup_kw, args.n_steps, args.eval_episodes,
                            seed=args.eval_seed, residual=args.fix_residual, design=args.eval_design,
                            eval_randomize=args.eval_randomize, episode_hook=eval_hook)
            buffers_restore(policy, snap)
            curve.append(curve_point(it + 1, ev_k, steps_per_iter, _log_std()))
            curve_sec += time.time() - tcv
            print(f"  [curve] iter {it+1}: held-out pos {ev_k['pos']:.2f}mm ang {ev_k['ang']:.2f}deg "
                  f"match {ev_k['match']:.3f} dlat {curve[-1].get('dlat_mm_mean', float('nan')):+.2f}mm")
    timing["rl_sec"] = round(time.time() - tr0 - curve_sec, 2)
    timing["rl_sec_per_iter"] = round(timing["rl_sec"] / max(args.iters, 1), 3)

    ev_final = evaluate(ev_list, ev_names, translator, cfg, eval_pol, sim, args.history, args.image_size,
                        device, args.w_seg, args.w_app, setup_kw, args.n_steps, args.eval_episodes,
                        seed=args.eval_seed, residual=args.fix_residual, design=args.eval_design,
                        eval_randomize=args.eval_randomize, episode_hook=eval_hook)
    print(f"  [FINAL self-sup] pos {ev_final['pos']:.1f}mm ang {ev_final['ang']:.1f}deg "
          f"(start {ev_final['start_pos']:.1f}) | seg {ev_final['seg']:.3f} match {ev_final['match']:.3f} "
          f"| keep-best {ev_final['keepbest_pos']:.1f}mm | peak force {ev_final['peak_force']:.2f} N")
    st = setup_stats()
    print(f"  [setups] train starts clamped {st['clamped']}/{st['n']} (mean lift {st['clamp_mm_mean']:.2f} mm), "
          f"liver substitutions {st['substituted_liver']}, skipped-cell draws {sum(_SKIPS.values())} {dict(_SKIPS)}")

    out = {"translator": args.translator, "seed": args.seed, "ablation": args.ablation or fx,
           "policy": args.policy, "residual_mode": (_RESIDUAL_MODE if args.fix_residual else "none"),
           "fixes": {"reward": args.fix_reward, "explore": args.fix_explore,
                     "residual": args.fix_residual, "dagger": args.fix_dagger},
           "after_imitation": ev_im, "final": ev_final, "config": config_record(args, v21),
           "budget": {"n_steps": args.n_steps, "imitation_samples": args.imitation * args.imitation_batch,
                      "rl_online_steps": args.iters * args.episodes_per_iter * args.n_steps},
           "train_setups": st, "timing": timing, "provenance": prov}
    if v21:
        curve.append(curve_point(args.iters, ev_final, steps_per_iter, _log_std()))
        timing["curve_sec"] = round(curve_sec, 2)
        out["analysis_status"] = ANALYSIS_STATUS_V21
        out["rlpos"] = {"deficit": prov["rlpos"]["deficit"], "init_from": prov["rlpos"]["init_from"],
                        "curve_every": args.curve_every, "final_log_std": _log_std(),
                        "teacher_sees_deficit": False, "iter_log": iter_log}
        out["curve"] = curve

    if args.real_eval:
        out["real"] = real_registration_eval(args.real_eval, lv_list, translator, cfg, policy, sim,
                                              args.history, args.image_size, device, args.w_seg,
                                              args.w_app, setup_kw, args.n_steps, residual=args.fix_residual)
        print(f"  [FINAL real-reg] seg {out['real']['seg']:.3f} (static {out['real']['static_seg']:.3f}) "
              f"match {out['real']['match']:.3f}")

    out["wall_sec"] = round(time.time() - t0, 1)
    out["provenance"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    save_ckpt(args.out.replace(".json", ".pt"), "final")
    tmp = args.out + ".tmp"                                  # atomic: a crash never leaves a half JSON
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(tmp, args.out)
    print(f"  wrote {args.out}")
    return 0


def real_registration_eval(npz, lv_list, translator, cfg, policy, sim, history, image_size, device,
                           w_seg, w_app, setup_kw, n_steps, n_targets=12, residual=False):
    """LEGACY (v1 traceability; reported numbers come from scripts/eval_real_registration.py).
    Register to REAL US targets: the agent servos a CT volume to match a real US image + its organ
    segmentation. Target is not exactly achievable (different patient), so we report the achieved
    seg-IoU / match to target vs a static start baseline."""
    import torch
    from usbg import goal_rl as G, view_match as VM, volume as V
    d = np.load(npz, allow_pickle=True)
    imgs = d["images"].astype(np.float32) / 255.0
    labs = d["labels"].astype(np.int64)
    rng = np.random.default_rng(12345)
    idx = rng.permutation(len(imgs))[:n_targets]
    acc = {k: [] for k in ["seg", "match", "static_seg"]}
    use_base = residual and _RESIDUAL_MODE in ("image", "cnn")   # only pose-free bases (the real target has no CT pose)
    for j, i in enumerate(idx):
        tgt_img, tgt_lab = imgs[i], labs[i]                # real US image + its organ segmentation
        tgt_prep = _prep(tgt_img, image_size)
        tgt_t = torch.from_numpy(tgt_prep[None, None]).to(device)
        lv = lv_list[j % len(lv_list)]
        Wfv, _T, T_start = G.sample_setup(lv, V.LIVER, rng, **setup_kw)
        env = make_env(lv, Wfv, T_start, device, np.random.default_rng(7 + j))
        env.reset()
        s0i, s0l = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, cfg, image_size)
        acc["static_seg"].append(VM.seg_organ_iou(s0l, tgt_lab))
        hist = deque([_prep(s0i, image_size)] * history, maxlen=history)
        cur_lab = s0l
        for _ in range(n_steps):
            cur_np = np.stack(hist, 0)
            cur = torch.from_numpy(cur_np[None]).to(device)
            a, _ = policy.act(cur, tgt_t, deterministic=True)
            a_exec = np.clip(a[0].detach().cpu().numpy()
                             + base_twist(None, None, cur_lab, tgt_lab, cur_np, tgt_prep, use_base), -G.MAX_TWIST, G.MAX_TWIST)
            env.step(a_exec)
            oi, cur_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, cfg, image_size)
            hist.append(_prep(oi, image_size))
        fi, fl = VM.curvi_render(lv, env.world_from_probe, Wfv, translator, cfg, image_size)
        acc["seg"].append(VM.seg_organ_iou(fl, tgt_lab))
        acc["match"].append(VM.combined_score(fi, fl, tgt_img, tgt_lab, sim, w_seg, w_app))
        env.close()
    return {k: float(np.mean(v)) for k, v in acc.items()}


if __name__ == "__main__":
    raise SystemExit(main())
