#!/usr/bin/env python3
"""Oracle ladder, contact on/off and DR evaluation (eval only, no training).

Every rung runs train_view_match_rl.evaluate on the SAME held-out setups (eval seed 999; the crossed
patient x target x window design under the anatomical seat) and writes per-episode records.

Rungs
  pose_oracle : analytic pose base G.analytic_twist(true pose, target pose) + ZERO residual. A
                PRIVILEGED ceiling with no learned weights. Its pose error is texture- and mask-independent
                by construction (only seg/match depend on the stage).
  ibvs_base   : the calibrated segmentation-moment IBVS base alone + ZERO residual, under whatever mask
                source the environment selects (USBG_LEARNED_SEG / US_SEG_CKPT). Calibrated on the TRAIN
                cohort with --calib-seed, replaying default_rng(seed) exactly as the trainer does (so it is
                the base a trainer run with that seed, cohort, texture and mask would use).
  static      : no commanded motion (contact may still move the probe; the start is clamped outside
                the skin under the anatomical seat).
  scripted    : open-loop descend-and-fan (usbg.eval.ScriptedScan), plus the pose-free keep-best.
  policy      : a saved trained policy (--policy-json; --weights imit final). The net is rebuilt with the
                saved residual bound and kept in TRAIN mode for BatchNorm exactly as in the trainer (it
                never calls .eval(); .eval() changes the numbers). The IBVS matrix comes from the v2 .pt;
                for a pre-v2 run it is recalibrated by replaying the run's RNG.

Several --contact / --eval-randomize / --weights values run in ONE process with one calibration.

  # canonical regression (v1 shipped run; the withdrawn segmenter is allowed for traceability only)
  USBG_SEAT=canonical USBG_LEARNED_SEG=1 US_SEG_CKPT=data/dataset/us_seg_sim_real.pt \\
  PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/eval_view_ladder.py --rung policy \\
      --policy-json outputs/view_match/policy_heldout/sonospade_deploy_s0.json --weights final \\
      --eval-design legacy --contact on off --allow-legacy-seg --out outputs/v2/validation/legacy_sonospade_deploy_s0.json
  # anatomical rung
  USBG_SEAT=anatomical USBG_LEARNED_SEG=0 PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 \\
  .venv/bin/python scripts/eval_view_ladder.py --rung ibvs_base --translator sonospade \\
      --tex-manifest outputs/v2/textures/manifest.json --calib-seed 0 --contact on off \\
      --out outputs/v2/table4/ladder/ibvs_base_sonospade_ct_s0.json

v2.1 (EXPLORATORY, opt-in; without these flags every v2 rung runs exactly as above). Any of them routes
the evaluation through usbg.controllers_v21.evaluate_ext (a step-for-step replica of
train_view_match_rl.evaluate with extended per-step logging and an <out>.traj.npz sidecar) and stamps
provenance.analysis_status = "exploratory, v2.1":
  --ext-log              extended logging for any rung (per-axis error, geodesic rotation, S and its
                         terms, the S_seg = 1 fallback, CT-label IoU, force, probe-skin gap, twists)
  --bn-mode eval         policy rung: BatchNorm with the saved running statistics (v2 = train mode)
  --n-steps-override N   horizon that also wins over the policy run's own n_steps (the 36-step arm)
  --base-fit-pairs N     ibvs_base / sguard: per-DoF R^2 and gain of the base on fresh TRAIN and held-out pairs
  --target-seed-offset N the target view is rendered with speckle seed cfg.seed + N (independent target
                         speckle); observations keep cfg.seed. Without it target and observation share a seed
  new rungs (what each observes is documented in usbg/controllers_v21.py):
    sguard     non-privileged S-guarded linear servo (--guard-eps grid on TRAIN, or --guard-select <pilots>)
    hillclimb  non-privileged (1+1)-ES on S with keep-best return (--es-sigma0 grid, or --es-select <pilots>)
    cnn_base   pose-free CNN appearance-history base, calibrated with DAgger on analytic-twist labels
               (--cnn-dagger; or --cnn-base-ckpt to reload one), zero residual
    lookahead  PRIVILEGED look-ahead bound: CEM over K x rounds twists rendered through a cloned MuJoCo
               state, scored by CT-mask S against an independently seeded target (CT masks only)
  The MLP base is the existing --rung ibvs_base --ibvs-base mlp.

v2.1.2 (opt-in):
  --contact-surface {target_plane, skin_tangent}  the contact surface of the evaluation episodes
                         (train_view_match_rl.contact_setup); the run-time calibration never reads it, so a base
                         rung keeps the target-plane calibration. Given, each episode gains a contact_plane record
                         (p, n, angle to the target beam, offset of the target face, lift, final probe-skin gap,
                         target and pre-lift start poses); absent, the output is the v2.1.1 one
  --save-ibvs            write the linear IBVS matrix in force to <out stem>.ibvs.npz (provenance only)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# v2.1 rungs (exploratory; usbg.controllers_v21). The v2 rungs above are unchanged.
V21_RUNGS = ("sguard", "hillclimb", "cnn_base", "lookahead")
V21_RUNG_CLASS = {"pose_oracle": "privileged reference (analytic pose base)",
                  "lookahead": "privileged reference (look-ahead bound; renders hypothetical poses)",
                  "static": "reference (no commanded motion)", "scripted": "reference (open-loop sweep)",
                  "ibvs_base": "non-privileged base (pose-free at run time; calibrated on privileged analytic twists)",
                  "sguard": "non-privileged (pose-free; masks, S, odometry)",
                  "hillclimb": "non-privileged (pose-free; S and odometry only)",
                  "cnn_base": "non-privileged base (pose-free at run time; calibrated with DAgger on privileged analytic twists)",
                  "policy": "non-privileged policy (pose-free at run time; image base + learned residual)"}


class ZeroPolicy:
    """Residual of exactly zero: executes the base twist only (or nothing, without a base)."""

    def act(self, cur, tgt, deterministic=True):
        import torch
        return torch.zeros(cur.shape[0], 6, device=cur.device), None


class ScriptedPolicy:
    """usbg.eval.ScriptedScan (press 4 steps, then fan +/-) as a policy-shaped object."""

    def __init__(self):
        from usbg.eval import ScriptedScan
        self.s = ScriptedScan()

    def reset(self):
        self.s.reset()

    def act(self, cur, tgt, deterministic=True):
        import torch
        return torch.as_tensor(self.s(None, None)[None], dtype=torch.float32, device=cur.device), None


def load_policy(pt_path, residual_frac, fix_residual, device):
    """Rebuild a saved Gaussian goal policy (train-mode BatchNorm, as in the trainer)."""
    import torch
    from usbg import goal_rl as G
    from usbg.policy import GoalPolicy
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    if ck.get("policy", "cnn") != "cnn":
        raise SystemExit(f"{pt_path}: only the Gaussian ('cnn') policy head is supported here")
    mt = ck.get("max_twist")
    mt = np.asarray(mt, np.float64) if mt is not None else \
        ((residual_frac * G.MAX_TWIST) if fix_residual else G.MAX_TWIST)
    pol = G.build_goal_gaussian_policy(GoalPolicy.build(history=ck["history"]), max_twist=mt).to(device)
    pol.net.load_state_dict(ck["state_dict"])
    pol.train()
    return pol, ck


def ibvs_path(out_json):
    """<out stem>.ibvs.npz, the --save-ibvs file beside the result JSON."""
    return (out_json[:-5] if out_json.endswith(".json") else out_json) + ".ibvs.npz"


def save_ibvs(out_json, ib, *, source):
    """v2.1.2 --save-ibvs (provenance only): write the linear IBVS base in force (kind, rich, the exact model array)
    to <out stem>.ibvs.npz, atomically, and return its provenance entry {path, md5, kind, rich, shape, source}."""
    from usbg.provenance import md5_file
    p = ibvs_path(out_json)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = p[:-4] + ".tmp.npz"
    model = np.asarray(ib["model"])
    np.savez(tmp, model=model, kind=np.asarray(str(ib["kind"])), rich=np.asarray(bool(ib["rich"])))
    os.replace(tmp, p)
    return {"path": os.path.abspath(p), "md5": md5_file(p), "kind": str(ib["kind"]), "rich": bool(ib["rich"]),
            "shape": [int(x) for x in model.shape], "dtype": str(model.dtype), "source": source}


def load_ibvs(path):
    """(kind, rich, model) of a --save-ibvs file (the exact array as saved)."""
    with np.load(path, allow_pickle=False) as z:
        return str(z["kind"]), bool(z["rich"]), np.array(z["model"])


def provenance_mismatch(run_prov, now_prov):
    """[(key, run, now)] where the current environment differs from the run's own provenance."""
    def g(p, *path):
        for k in path:
            p = (p or {}).get(k)
        return p
    out = []
    for key, path in (("seat", ("seat", "resolved")), ("seg_md5", ("mask", "seg_md5")),
                      ("learned_mask", ("mask", "learned")), ("texture_md5", ("texture", "md5")),
                      ("windows_md5", ("windows", "md5"))):
        a, b = g(run_prov, *path), g(now_prov, *path)
        if a != b:
            out.append((key, a, b))
    return out


def _v21_grid(values, select_files, pname, flag, sel_flag, prov, C21, ap):
    """v2.1: the hyperparameter values to evaluate: an explicit grid (a TRAIN pilot, or a pre-declared
    value), or the one value select_param picks from TRAIN pilot outputs (recorded in provenance)."""
    if (values is None) == (select_files is None):
        ap.error(f"give exactly one of {flag} (grid or pre-declared value) or {sel_flag} (TRAIN pilot files)")
    if select_files is not None:
        best, rec = C21.select_param(select_files, pname)
        prov["v21"].setdefault("selection", {})[pname] = rec
        print(f"[v2.1] {pname} = {best:g} selected on TRAIN pilots {rec['mean_over_files']}")
        return [best]
    prov["v21"].setdefault("grid", {})[pname] = [float(v) for v in values]
    return [float(v) for v in values]


def _cnn_base(args, T, lv_train, train_cases, rng, tr, cfg, setup_kw, dev, seed, prov):
    """v2.1 cnn_base rung: calibrate the pose-free CNN appearance-history base on TRAIN with DAgger
    rounds (train_view_match_rl.calibrate_cnn_base, analytic-twist labels), or reload a saved one. The
    calibrated weights are saved to <out>.cnn_base.pt so a later cohort run reuses them unchanged."""
    import torch
    from usbg import goal_rl as G
    from usbg import provenance as PROV
    from usbg.policy import GoalPolicy
    info = {"n_traj": args.cnn_traj, "steps": args.cnn_steps, "epochs": args.cnn_epochs,
            "dagger_rounds": args.cnn_dagger, "history": args.history, "labels": "analytic twist (privileged)"}
    mt = torch.as_tensor(G.MAX_TWIST, dtype=torch.float32, device=dev)
    if args.cnn_base_ckpt:
        ck = torch.load(args.cnn_base_ckpt, map_location=dev, weights_only=False)
        if int(ck["history"]) != int(args.history):
            raise SystemExit(f"{args.cnn_base_ckpt}: history {ck['history']} != --history {args.history}")
        net = GoalPolicy.build(history=int(ck["history"])).to(dev)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        info.update(ck.get("calib") or {}, reloaded_from=os.path.abspath(args.cnn_base_ckpt),
                    reloaded_md5=PROV.md5(args.cnn_base_ckpt))
        return {"kind": "cnn", "net": net, "mt": mt, "device": dev}, info
    T.init_train_cells(lv_train, train_cases)
    T._RANDOMIZE = args.calib_randomize
    dr_rng = np.random.default_rng([seed, T._DR_SALT])
    ib = T.calibrate_cnn_base(lv_train, rng, tr, cfg, args.image_size, args.history, setup_kw, dev,
                              n_traj=args.cnn_traj, steps=args.cnn_steps, epochs=args.cnn_epochs,
                              dagger_rounds=args.cnn_dagger, dr_rng=dr_rng)
    T._RANDOMIZE = "off"
    p = (args.out[:-5] if args.out.endswith(".json") else args.out) + ".cnn_base.pt"
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    torch.save({"state_dict": ib["net"].state_dict(), "history": args.history, "image_size": args.image_size,
                "calib": dict(info, seed=seed, train_cases=list(train_cases)), "provenance": prov}, p)
    info.update(saved=os.path.abspath(p), saved_md5=PROV.md5(p), seed=seed)
    return ib, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=["pose_oracle", "ibvs_base", "static", "scripted", "policy",
                                                       *V21_RUNGS])
    ap.add_argument("--translator", default=None, choices=["physics", "cut", "cyclegan", "sonospade"])
    ap.add_argument("--texture-ckpt", default=None, help="'none' for physics; overrides --tex-manifest")
    ap.add_argument("--tex-manifest", default=None)
    ap.add_argument("--policy-json", default=None)
    ap.add_argument("--weights", nargs="+", default=["final"], choices=["final", "imit"])
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--calib-randomize", default="off", choices=["off", "episode"],
                    help="ibvs_base: calibrate with one DR draw per pair (as a --randomize episode run)")
    ap.add_argument("--render-seed", type=int, default=None, help="cfg seed for renders (default = calib/policy seed)")
    ap.add_argument("--ibvs-features", default="rich", choices=["simple", "rich"])
    ap.add_argument("--ibvs-base", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--train-cases", nargs="+", default=None)
    ap.add_argument("--eval-cases", nargs="+", default=None)
    ap.add_argument("--targets", nargs="+", default=None, choices=["liver", "kidney", "spleen"])
    ap.add_argument("--min-windows", type=int, default=None)
    ap.add_argument("--contact", nargs="+", default=["on"], choices=["on", "off"])
    ap.add_argument("--eval-randomize", nargs="+", default=["off"], choices=["off", "episode"])
    ap.add_argument("--eval-design", default=None, choices=["crossed", "legacy"],
                    help="default: the policy run's own design, else crossed")
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=12)
    ap.add_argument("--w-seg", type=float, default=0.6); ap.add_argument("--w-app", type=float, default=0.4)
    ap.add_argument("--lateral-m", type=float, default=None); ap.add_argument("--axial-m", type=float, default=0.01)
    ap.add_argument("--fan-deg", type=float, default=18.0)
    ap.add_argument("--history", type=int, default=4); ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--no-episodes", action="store_true", help="omit per-episode records from the JSON")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="policy rung: allow env (seat/mask/texture/windows) to differ from the run's provenance")
    ap.add_argument("--allow-legacy-seg", action="store_true")
    ap.add_argument("--out", required=True)
    # ---- v2.1 (exploratory, opt-in); defaults leave every v2 rung on the v2 code path ----
    ap.add_argument("--ext-log", action="store_true",
                    help="v2.1: extended per-step logging (usbg.controllers_v21.evaluate_ext) + <out>.traj.npz")
    ap.add_argument("--bn-mode", default="train", choices=["train", "eval"],
                    help="policy rung: BatchNorm mode at evaluation (train = v2 default; eval is a v2.1 arm)")
    ap.add_argument("--n-steps-override", type=int, default=None,
                    help="v2.1: horizon that wins over --n-steps and over the policy run's own n_steps")
    ap.add_argument("--guard-eps", type=float, nargs="+", default=None, help="sguard: eps grid (TRAIN pilot)")
    ap.add_argument("--guard-select", nargs="+", default=None,
                    help="sguard: TRAIN pilot result JSONs; eps = select_param(..., 'guard_eps')")
    ap.add_argument("--guard-gain0", type=float, default=1.0, help="sguard: initial servo gain")
    ap.add_argument("--es-sigma0", type=float, nargs="+", default=None, help="hillclimb: sigma0 grid (TRAIN pilot)")
    ap.add_argument("--es-select", nargs="+", default=None,
                    help="hillclimb: TRAIN pilot result JSONs; sigma0 = select_param(..., 'es_sigma0')")
    ap.add_argument("--cnn-traj", type=int, default=120); ap.add_argument("--cnn-steps", type=int, default=8)
    ap.add_argument("--cnn-epochs", type=int, default=400); ap.add_argument("--cnn-dagger", type=int, default=3)
    ap.add_argument("--cnn-base-ckpt", default=None, help="cnn_base: reload a saved <out>.cnn_base.pt")
    ap.add_argument("--look-k", type=int, default=64); ap.add_argument("--look-rounds", type=int, default=2)
    ap.add_argument("--look-elite", type=int, default=8)
    ap.add_argument("--look-seed-offset", type=int, default=7907,
                    help="lookahead: the scoring target is rendered with cfg.seed + this (independent speckle)")
    ap.add_argument("--base-fit-pairs", type=int, default=0,
                    help="ibvs_base/sguard: per-DoF R^2 and gain of the calibrated base on this many fresh "
                         "TRAIN and held-out pairs (computed after the evaluation)")
    ap.add_argument("--analysis-status", default=None, help="v2.1: provenance analysis_status (default "
                    "'exploratory, v2.1')")
    ap.add_argument("--eval-cases-override", nargs="+", default=None,
                    help="v2.1: evaluation patients that also win over the policy run's own eval cohort "
                         "(TRAIN-patient smoke tests, a replication cohort)")
    ap.add_argument("--target-seed-offset", type=int, default=None,
                    help="v2.1 (opt-in): render the TARGET view with speckle seed cfg.seed + this offset, so target "
                         "and observations do not share one speckle field (independent target speckle; the "
                         "decision rows use 7907). Default: the archived shared seed")
    # ---- v2.1.2 (opt-in); absent, every rung runs exactly as at v2.1.1 ----
    ap.add_argument("--contact-surface", default=None, choices=["target_plane", "skin_tangent"],
                    help="v2.1.2: contact surface of the evaluation episodes (absent = target_plane without the "
                         "per-episode plane record). skin_tangent: the box top is the plane tangent to the skin "
                         "where the pre-lift start's beam axis enters the body; the calibration never reads it")
    ap.add_argument("--save-ibvs", action="store_true",
                    help="v2.1.2 (provenance only): write the linear IBVS matrix in force (run-time calibration or "
                         "checkpoint) to <out stem>.ibvs.npz; the evaluation is unchanged")
    args = ap.parse_args(argv)
    t0 = time.time()
    v21 = (args.ext_log or args.rung in V21_RUNGS or args.bn_mode != "train" or args.n_steps_override is not None
           or args.base_fit_pairs > 0 or args.analysis_status is not None or args.eval_cases_override is not None
           or args.target_seed_offset is not None or args.contact_surface is not None or args.save_ibvs)
    if args.save_ibvs and (args.rung not in ("ibvs_base", "policy", "sguard") or args.ibvs_base != "linear"):
        ap.error("--save-ibvs applies to the rungs with a linear image base (ibvs_base, policy, sguard)")
    if v21:
        from usbg import controllers_v21 as C21
    if args.bn_mode != "train" and args.rung != "policy":
        ap.error("--bn-mode applies to --rung policy only")

    import torch
    import train_view_match_rl as T
    from usbg import goal_rl as G, render_lotus as R
    from usbg import provenance as PROV
    from usbg.benchmark import TRAIN_PATIENTS, TEST_PATIENTS
    from usbg.similarity import load_similarity
    seat = T._seat()
    anat = seat == "anatomical"

    J = None
    if args.rung == "policy":
        if not args.policy_json:
            ap.error("--rung policy needs --policy-json")
        J = json.load(open(args.policy_json)); c = J["config"]
        # the run's own cohort/setup/flags win over CLI defaults
        args.translator = c["translator"]; args.calib_seed = c["seed"]
        for k in ("train_cases", "eval_cases", "w_seg", "w_app", "lateral_m", "axial_m", "fan_deg",
                  "history", "image_size", "ibvs_features", "ibvs_base", "n_steps"):
            if c.get(k) is not None:
                setattr(args, k, c[k])
        args.targets = args.targets or c.get("targets")
        args.min_windows = args.min_windows if args.min_windows is not None else c.get("min_windows")
        args.eval_design = args.eval_design or c.get("eval_design")
        args.eval_seed = args.eval_seed if args.eval_seed is not None else c.get("eval_seed")
        args.eval_episodes = args.eval_episodes or c.get("eval_episodes")
        if args.texture_ckpt is None and args.tex_manifest is None and c.get("texture_ckpt"):
            args.texture_ckpt = c["texture_ckpt"]
    if args.n_steps_override is not None:                # v2.1: wins over the run's own n_steps
        args.n_steps = int(args.n_steps_override)
    if args.eval_cases_override is not None:             # v2.1: wins over the run's own eval cohort
        args.eval_cases = list(args.eval_cases_override)
    if args.translator is None:
        ap.error("--translator is required")
    args.eval_design = args.eval_design or "crossed"
    args.eval_seed = 999 if args.eval_seed is None else args.eval_seed
    args.eval_episodes = args.eval_episodes or 60
    train_cases = args.train_cases or (list(TRAIN_PATIENTS) if anat else list(T.TRAIN_CASES))
    eval_cases = args.eval_cases or (list(TEST_PATIENTS) if anat else list(T.EVAL_CASES))
    T.SEAT_ORGANS = list(args.targets or (["liver", "kidney"] if anat else T.LEGACY_TARGETS))
    T._MIN_WINDOWS = int(args.min_windows or 1)
    lateral = args.lateral_m if args.lateral_m is not None else (0.02 if anat else 0.03)
    dev = R.get_device()
    seed = args.calib_seed
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)                      # calibrate_ibvs is its first consumer (as in the trainer)
    tex = T.resolve_texture(args.translator, args.texture_ckpt, args.tex_manifest)
    tr = None if tex is None else __import__("usbg.texture_gan", fromlist=["load_translator"]).load_translator(tex, dev)
    cfg = R.RenderConfig(seed=seed if args.render_seed is None else args.render_seed)
    sim = load_similarity("ssim", device=dev)
    prov = PROV.collect(stage=args.translator, texture_ckpt=tex, rung=args.rung, train_cases=train_cases,
                        eval_cases=eval_cases, targets=T.SEAT_ORGANS, min_windows=T._MIN_WINDOWS,
                        policy_json=os.path.abspath(args.policy_json) if args.policy_json else None,
                        policy_json_md5=PROV.md5(args.policy_json) if args.policy_json else None,
                        calib_seed=seed, eval_design=args.eval_design, eval_seed=args.eval_seed,
                        lateral_m=lateral, bn_mode=T.BN_MODE if args.bn_mode == "train" else C21.BN_MODE_EVAL,
                        privileged=args.rung in ("pose_oracle", "lookahead"))
    if v21:                                              # only v2.1 outputs carry these keys
        prov["analysis_status"] = args.analysis_status or C21.ANALYSIS_STATUS
        prov["v21"] = {"rung_class": V21_RUNG_CLASS.get(args.rung), "ext_log": True, "bn_mode": args.bn_mode,
                       "n_steps": args.n_steps, "n_steps_override": args.n_steps_override,
                       "eval_cases_override": args.eval_cases_override, "archived_v2_primary": True}
        if args.target_seed_offset is not None:          # opt-in only: no key by default
            prov["v21"]["target_seed_offset"] = int(args.target_seed_offset)
            prov["v21"]["target_seed"] = int(cfg.seed) + int(args.target_seed_offset)
            prov["v21"]["obs_seed"] = int(cfg.seed)
        if args.contact_surface is not None:             # v2.1.2 opt-in only: no key by default
            if args.contact_surface == "skin_tangent" and not anat:
                raise SystemExit("--contact-surface skin_tangent needs USBG_SEAT=anatomical")
            prov["v21"]["contact_surface"] = {
                "surface": args.contact_surface, "read_by_calibration": False,
                "plane": ("box top through the target window face, normal to the target beam (z = 0)"
                          if args.contact_surface == "target_plane" else
                          "plane at the point where the beam axis of the start (before its lift) first enters the CT body; "
                          "its normal is the face-area-weighted mean of the outward skin-mesh normals less "
                          "than 10 mm from that point (usbg.body_surface.skin_plane); starts lifted along it; "
                          "press cap 5 mm along it")}
    PROV.assert_not_withdrawn(prov, allow=args.allow_legacy_seg)
    if args.rung == "lookahead" and prov["mask"].get("learned"):
        raise SystemExit("--rung lookahead scores by CT-mask S: run it with USBG_LEARNED_SEG=0")
    if J is not None and "provenance" in J:
        mm = provenance_mismatch(J["provenance"], prov)
        prov["mismatch_vs_run"] = [list(x) for x in mm]
        if mm and not args.allow_mismatch:
            raise SystemExit(f"provenance mismatch vs the run: {mm} (use --allow-mismatch to evaluate a policy "
                             "under a deliberately different condition)")

    lv_train = T.load_volumes(train_cases)
    lv_eval = T.load_volumes(eval_cases)
    setup_kw = dict(lateral_m=lateral, axial_m=args.axial_m, fan_deg=args.fan_deg)

    # ---- the base + the residual policies per rung ----
    residual = False
    policies = {}
    params = {}                                          # v2.1 grids: policy name -> {"name", "value"}
    calib_sec = 0.0
    ibvs_source = None                                   # v2.1.2 --save-ibvs: where the matrix in force came from
    if args.rung == "pose_oracle":
        T._RESIDUAL_MODE, residual = "oracle", True
        policies["zero_residual"] = ZeroPolicy()
    elif args.rung in ("ibvs_base", "policy", "sguard"):
        ck0 = None
        if args.rung == "policy":
            residual = bool(J["config"]["fix_residual"])
            T._RESIDUAL_MODE = J.get("residual_mode", "none") if residual else "oracle"
            if T._RESIDUAL_MODE == "cnn":
                raise SystemExit("the CNN base is not reloadable here (its weights are not saved)")
            p_final = args.policy_json.replace(".json", ".pt")
            ck0 = torch.load(p_final, map_location=dev, weights_only=False)
        else:
            T._RESIDUAL_MODE, residual = "image", True
        if residual and T._RESIDUAL_MODE == "image":
            ib = (ck0 or {}).get("ibvs")
            if ib is not None and ib["kind"] == "linear":     # v2 checkpoints carry the base: no RNG replay
                T._IBVS = {"kind": "linear", "model": np.asarray(ib["model"]), "rich": ib["rich"]}
                ibvs_source = "checkpoint"
            else:
                ibvs_source = "run-time calibration (target-plane lift)"
                tc = time.time()
                T.init_train_cells(lv_train, train_cases)
                T._RANDOMIZE = args.calib_randomize if args.rung != "policy" else \
                    J["config"].get("randomize", "off")
                dr_rng = np.random.default_rng([seed, T._DR_SALT])
                T._IBVS = T.calibrate_ibvs(lv_train, rng, tr, cfg, args.image_size, setup_kw,
                                           kind=args.ibvs_base, rich=(args.ibvs_features == "rich"), dr_rng=dr_rng)
                T._RANDOMIZE = "off"
                calib_sec = time.time() - tc
                print(f"[calib] IBVS base recalibrated on {train_cases} seed {seed} in {calib_sec:.0f}s")
        if args.rung == "ibvs_base":
            policies["zero_residual"] = ZeroPolicy()
        elif args.rung == "sguard":
            grid = _v21_grid(args.guard_eps, args.guard_select, "guard_eps", "--guard-eps", "--guard-select",
                             prov, C21, ap)
            for e in grid:
                c_ = C21.SGuardController(T, e, gain0=args.guard_gain0)
                policies[c_.name] = c_
                params[c_.name] = {"name": "guard_eps", "value": float(e)}
        else:
            for w in args.weights:
                p = args.policy_json.replace(".json", ".pt" if w == "final" else "_imit.pt")
                if not os.path.exists(p):
                    print(f"!! LOUD SKIP weights={w}: {p} missing (pre-v2 runs saved only the final .pt)")
                    continue
                pol, _ck = load_policy(p, J["config"].get("residual_frac", 0.5), residual, dev)
                if args.bn_mode == "eval":                  # v2.1 arm; v2 keeps train mode
                    pol.eval()
                policies[w] = pol
            if not policies:
                raise SystemExit("no policy weights to evaluate")
    elif args.rung == "static":
        policies["static"] = ZeroPolicy()
    elif args.rung == "scripted":
        policies["scripted"] = ScriptedPolicy()
    elif args.rung == "hillclimb":
        grid = _v21_grid(args.es_sigma0, args.es_select, "es_sigma0", "--es-sigma0", "--es-select", prov, C21, ap)
        for s0 in grid:
            c_ = C21.HillClimbController(s0, seed)
            policies[c_.name] = c_
            params[c_.name] = {"name": "es_sigma0", "value": float(s0)}
    elif args.rung == "cnn_base":
        T._RESIDUAL_MODE, residual = "cnn", True
        tc = time.time()
        T._IBVS, cnn_info = _cnn_base(args, T, lv_train, train_cases, rng, tr, cfg, setup_kw, dev, seed, prov)
        calib_sec = time.time() - tc
        prov["v21"]["cnn_base"] = cnn_info
        policies["zero_residual"] = ZeroPolicy()
    elif args.rung == "lookahead":
        c_ = C21.LookaheadController(seed, k=args.look_k, rounds=args.look_rounds, n_elite=args.look_elite,
                                     seed_offset=args.look_seed_offset)
        policies[c_.name] = c_
        prov["v21"]["lookahead"] = {"k": args.look_k, "rounds": args.look_rounds, "n_elite": args.look_elite,
                                    "target_seed": int(cfg.seed) + args.look_seed_offset, "obs_seed": int(cfg.seed),
                                    "score": "CT-mask S (w_seg, w_app) vs the independently seeded target view"}

    if args.save_ibvs:                                   # v2.1.2 provenance only: the matrix in force, written once
        ib = T._IBVS or {}
        if ib.get("kind") != "linear":
            raise SystemExit("--save-ibvs: this run has no linear image base to save")
        prov["v21"]["save_ibvs"] = save_ibvs(args.out, ib, source=ibvs_source)
    if args.contact_surface is not None:                 # v2.1.2: set after the calibration, which never reads it
        T._CONTACT_SURFACE = args.contact_surface
        T._PLANE_LOG = True

    results, npz_blocks = [], []
    for (name, pol), contact, er in itertools.product(policies.items(), args.contact, args.eval_randomize):
        T._CONTACT = (contact == "on")
        te = time.time()
        with torch.no_grad():
            if v21:                                      # v2.1: extended evaluator (same episodes and numbers)
                ctrl = pol if isinstance(pol, (C21.SGuardController, C21.HillClimbController,
                                               C21.LookaheadController)) else \
                    C21.PolicyBaseController(T, pol, residual, dev, name=name, privileged=args.rung == "pose_oracle")
                ev, traj = C21.evaluate_ext(T, lv_eval, eval_cases, tr, cfg, ctrl, sim, args.history, args.image_size,
                                            dev, args.w_seg, args.w_app, setup_kw, args.n_steps, args.eval_episodes,
                                            seed=args.eval_seed, design=args.eval_design, eval_randomize=er,
                                            log_episodes=not args.no_episodes,
                                            target_seed_offset=args.target_seed_offset)
            else:
                ev = T.evaluate(lv_eval, eval_cases, tr, cfg, pol, sim, args.history, args.image_size, dev,
                                args.w_seg, args.w_app, setup_kw, args.n_steps, args.eval_episodes,
                                seed=args.eval_seed, residual=residual, design=args.eval_design,
                                eval_randomize=er, log_episodes=not args.no_episodes)
        sec = time.time() - te
        print(f"[{args.rung}/{name} {args.translator} contact={contact} dr={er}] start {ev['start_pos']:.2f} -> "
              f"final {ev['pos']:.2f} mm / {ev['ang']:.2f} deg | keep-best {ev['keepbest_pos']:.2f} | "
              f"seg {ev['static_seg']:.3f}->{ev['seg']:.3f} | peak force {ev['peak_force']:.2f} N | "
              f"clamped {ev['start_clamped_frac']:.2f} | {sec:.0f}s ({sec / max(ev['n_episodes'], 1):.2f} s/ep)")
        blk = {"weights": name, "contact": contact, "eval_randomize": er, "sec": round(sec, 1), **ev}
        if name in params:
            blk["param"] = params[name]
        results.append(blk)
        if v21:
            npz_blocks.append(({"block": len(results) - 1, "weights": name, "contact": contact,
                                "eval_randomize": er, "param": params.get(name)}, traj))

    out = {"rung": args.rung, "translator": args.translator, "calib_seed": seed,
           "seed": J["seed"] if J is not None else seed,
           "arm": (J.get("ablation") if J is not None else None),
           "mask_source": "learned" if prov["mask"].get("learned") else "ct",
           "privileged": args.rung in ("pose_oracle", "lookahead"),
           "residual_mode": T._RESIDUAL_MODE if residual else "none", "n_steps": args.n_steps,
           "results": results, "calib_sec": round(calib_sec, 1), "wall_sec": round(time.time() - t0, 1),
           "provenance": prov}
    if args.contact_surface is not None:                 # v2.1.2 opt-in only: no key by default
        out["contact_surface"] = args.contact_surface
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if v21:
        if args.base_fit_pairs > 0 and (T._IBVS or {}).get("kind") in ("linear", "mlp"):
            tb = time.time()
            out["base_fit"] = C21.base_fit_report(T, lv_train, train_cases, lv_eval, eval_cases, tr, cfg,
                                                  args.image_size, setup_kw, n_pairs=args.base_fit_pairs, seed=seed)
            out["base_fit"]["sec"] = round(time.time() - tb, 1)
        tp = C21.traj_path(args.out)
        out["analysis_status"] = prov["analysis_status"]
        out["rung_class"] = V21_RUNG_CLASS.get(args.rung)
        out["traj_npz"] = {"path": os.path.abspath(tp), "md5": C21.save_traj_npz(tp, npz_blocks),
                           "layout": "b<block>__<key>; 'index' = JSON list of block metas; per-step arrays have "
                                     "n_steps + 1 columns (index 0 = start); axes order " + ",".join(C21.AXES)}
        out["wall_sec"] = round(time.time() - t0, 1)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
