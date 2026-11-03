#!/usr/bin/env python3
"""Real-target registration for oracle-ladder policies (D12): a standalone, held-out replacement for
the in-run train_view_match_rl.real_registration_eval, which servoed the TRAINING CTs, drew its 12
targets from Vitale frames the shared segmenter had trained on, fed the GROUND-TRUTH real mask to the
reward and the IBVS base, and scored every organ (penalising organs a partial label does not annotate).

Per saved matched-arm policy (--policy-json, several allowed):
  targets  : (frame, organ) pairs, organ in --organs (default: the run's targets, liver kidney), with
             >= --min-px ground-truth pixels of that organ:
               * Vitale GROUP holdout (vitale_holdout_v2.npz from the segmenter repo's make_vitale_split.py,
                 letter groups g,b; no v2 segmenter trained on them; fully labelled) = PRIMARY;
               * a seeded AUL liver subset (aul_liver_test.npz; liver-only labels);
               * a seeded KidneyUS holdout subset (kidneyus_holdout_test.npz; kidney-only labels).
             AUL and KidneyUS frames are resized screenshots, NOT in the convex-sector geometry of the
             sim renders, so their appearance term and IoU are geometry-mismatched: report per source.
  CT       : the held-out benchmark patients; each target is paired with --cts-per-target of them
             (rotating, so every CT gets the same share) and seated at a screened window of the target
             organ (window index rotates with the target), start perturbed and clamped as in training.
  target mask fed to the controller (IBVS base and non-privileged reward):
             'seg' = the segmenter of this run on the real image (non-privileged, PRIMARY) or 'gt' (reference).
  scoring  : organ-restricted IoU against the real GT mask, of the sim view's CT label (iou_ct, primary:
             the anatomy actually reached) and of the learned mask of the sim render (iou_seg), at the
             start (static), the end (final) and the keep-best step chosen by the NON-PRIVILEGED score
             (no GT); per step the non-privileged reward r_dep and the GT-mask reward r_gt are logged for a
             pooled and a within-episode Spearman.

  USBG_SEAT=anatomical USBG_LEARNED_SEG=1 US_SEG_CKPT=<seg ckpt of THIS run> PYTHONPATH=src \\
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/eval_real_registration.py \\
      --policy-json outputs/v2/table4/matched/sonospade_deploy_s0.json --out outputs/v2/table4/registration/sonospade_s0.json

v2.1 keep-best null controllers (OPT-IN, exploratory).
Without --controller the script, its episodes and its output format are exactly the v2 ones above.
--controller policy random_walk scripted base runs every listed controller on the SAME setups (target,
CT, window, perturbed start, env seed, target mask) for the SAME n_steps budget and applies the SAME
keep-best rule: the argmax of the non-privileged score r_dep over the n_steps + 1 states, start included
(first index on ties). So "keep-best minus static" of the trained policy can be set against the gain
that selection alone yields under undirected or non-servoing motion:
  policy      : the v2 path above, run again (the residual policy as trained, on top of its base, which
                uses no pose); it repeats the archived v2 episodes, and those stay primary;
  random_walk : the step lengths of the policy with random directions. Its twist at step t is the absolute
                twist |a_t| of the policy at step t of the paired episode (same target, CT, start and target
                mask), channel by channel, times 6 random signs drawn from
                default_rng([--rw-seed, target, CT, start, mask]); every channel keeps the motion budget of
                the policy at every step but loses its direction. 'policy' must be listed in the same run (it
                runs first);
  scripted    : usbg.eval.ScriptedScan (press 4 steps, then fan +/-), the oracle-ladder scripted rung, open
                loop, no base;
  base        : the run's pose-free IBVS base alone with a zero residual (the oracle-ladder ibvs_base rung),
                driven by the same target mask as the policy.
Each episode then also logs: controller, subject (cluster id of the real frame: Vitale session,
KidneyUS patient prefix, AUL frame stem), keepbest_iou_valid, the per-step iou_ct and iou_ct_valid
trajectories and the commanded motion budget (twist_trans_sum, twist_rot_sum: the sum over steps of the
executed twist norms in SI body-twist units, m/s and rad/s like MAX_TWIST (not normalized by it); multiply
by DT_C for the per-step motion). Summary keys
gain a '<controller>:' prefix and the provenance carries "analysis_status": "exploratory, v2.1".
--target-index runs a subset of the full target list while keeping each target's full-list index (so
its CTs, window and start are those of a full run). A separate aggregation turns the outputs
into the registration contrasts (crossed subject x CT x seed bootstrap, Holm over cells).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SIX_TO_REPO = np.array([0, 1, 2, 3, 4, 7], dtype=np.int64)
CONTROLLERS = ("policy", "random_walk", "scripted", "base")      # run order (policy first: random_walk pairs with it)
ANALYSIS_STATUS = "exploratory, v2.1"
_MASK_SALT = {"seg": 0, "gt": 1, "oof": 2}
SEG_REPO = os.environ.get("USBG_SEG_REPO", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "segtrain"))
VITALE_HOLDOUT = os.path.join(SEG_REPO, "data/real/vitale_holdout_v2.npz")
AUL_NPZ = os.path.join(SEG_REPO, "data/real/aul_liver_test.npz")
KIDNEYUS_NPZ = os.path.join(SEG_REPO, "data/real/kidneyus_holdout_test.npz")


def _subset(n_total, n, seed):
    if not n or n >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).permutation(n_total)[:n])


def _subject_id(src, d, i):
    """Cluster id of a real frame for a clustered bootstrap (v2.1 controller mode only): the Vitale
    acquisition session (patient identity unknown; letter groups g/b are coarser), the KidneyUS patient
    prefix of the stem ('117_IM-0149-0027_anon' -> '117'), the AUL frame stem (AUL has no subject ids,
    so each frame is its own cluster). None when the file carries no ids."""
    if "sessions" in d.files:
        return f"{src}:{d['sessions'][i]}"
    if "stems" in d.files:
        s = str(d["stems"][i])
        return f"{src}:{s.split('_')[0] if src == 'kidneyus' else s}"
    return None


def load_targets(args, organs_repo):
    """-> list of dict(src, idx, img (128,128) float [0,1], gt (repo ids), valid (repo organ ids), organ)."""
    out = []

    def add(src, idx, img, gt, valid, **extra):
        for o in organs_repo:
            if o in valid and int((gt == o).sum()) >= args.min_px:
                out.append(dict(src=src, idx=int(idx), img=np.asarray(img, np.float32), gt=gt,
                                valid=valid, organ=int(o), **extra))

    def partial(src, path, n, seed):
        d = np.load(path, allow_pickle=True)
        valid = {int(_SIX_TO_REPO[c]) for c in np.asarray(d["valid_classes"]).ravel() if int(c) != 0}
        idx = _subset(len(d["images"]), n, seed)
        for i in idx[:args.max_frames_per_source or None]:
            add(src, i, d["images"][i], _SIX_TO_REPO[d["label6"][i].astype(np.int64)], valid,
                subject=_subject_id(src, d, i))
        return {"path": os.path.abspath(path), "n_frames_file": int(len(d["images"])), "n_frames_used": int(len(idx))}

    srcs = {}
    if args.vitale_npz and args.vitale_npz.lower() != "none":
        if not os.path.exists(args.vitale_npz):
            raise SystemExit(f"{args.vitale_npz} missing: build it with the segmenter repo's "
                             "scripts/make_vitale_split.py --mode group --holdout-groups g,b")
        srcs["vitale_holdout"] = partial("vitale_holdout", args.vitale_npz, 0, 0)
    if args.aul_npz and args.aul_npz.lower() != "none":
        srcs["aul"] = partial("aul", args.aul_npz, args.aul_n, args.aul_seed)
    if args.kidneyus_npz and args.kidneyus_npz.lower() != "none":
        srcs["kidneyus"] = partial("kidneyus", args.kidneyus_npz, args.kidneyus_n, args.kidneyus_seed)
    if args.vitale_all:                       # cross-check only: the 60 Vitale frames and their out-of-fold masks
        if not args.oof_npz:
            raise SystemExit("--vitale-all needs --oof-npz (out-of-fold real predictions for THIS stage/seed)")
        d = np.load(args.vitale_all, allow_pickle=True)
        oof = np.load(args.oof_npz)[args.oof_key]
        for i in range(len(d["images"])):
            add("vitale_all", i, d["images"][i].astype(np.float32) / 255.0, d["labels"][i].astype(np.int64),
                {1, 2, 3, 4, 7}, oof=_SIX_TO_REPO[np.asarray(oof[i]).astype(np.int64)])
        srcs["vitale_all"] = {"path": os.path.abspath(args.vitale_all), "oof": os.path.abspath(args.oof_npz)}
    return out, srcs


def ct_label_at(lv, T, Wfv, size):
    """The CT-derived curvilinear label at pose T (what curvi_render returns with USBG_LEARNED_SEG=0),
    without the physics render or the translator: the sim-side ground truth."""
    from usbg import slicer as S, view_match as VM
    from usbg.segmenter import align_label_to_image
    lab_lin = S.slice_at_pose(lv, np.asarray(T, np.float64), Wfv)
    lab_sq = align_label_to_image(lab_lin, (size, size))
    _, lab_sec = VM.warp_to_sector(np.zeros((size, size), np.float32), lab_sq)
    return lab_sec


def _spearman_within(eps):
    from scipy.stats import spearmanr
    rs = []
    for e in eps:
        a, b = np.asarray(e["r_dep"]), np.asarray(e["r_gt"])
        if a.std() > 1e-9 and b.std() > 1e-9:
            rs.append(float(spearmanr(a, b)[0]))
    return {"mean_rho": float(np.mean(rs)) if rs else float("nan"), "n_episodes": len(rs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-json", nargs="+", required=True)
    ap.add_argument("--vitale-npz", default=VITALE_HOLDOUT, help="Vitale GROUP holdout npz ('none' to skip)")
    ap.add_argument("--aul-npz", default=AUL_NPZ); ap.add_argument("--aul-n", type=int, default=60)
    ap.add_argument("--aul-seed", type=int, default=0)
    ap.add_argument("--kidneyus-npz", default=KIDNEYUS_NPZ); ap.add_argument("--kidneyus-n", type=int, default=60)
    ap.add_argument("--kidneyus-seed", type=int, default=0)
    ap.add_argument("--vitale-all", default=None, help="cross-check: data/eval/kaggle_vitale_test.npz (needs --oof-npz)")
    ap.add_argument("--oof-npz", default=None); ap.add_argument("--oof-key", default="pred6")
    ap.add_argument("--max-frames-per-source", type=int, default=0, help="smoke tests only (0 = all)")
    ap.add_argument("--organs", nargs="+", default=None, choices=["liver", "kidney"],
                    help="default: the run's targets restricted to liver/kidney")
    ap.add_argument("--ct-cases", nargs="+", default=None, help="default: usbg.benchmark.TEST_PATIENTS")
    ap.add_argument("--cts-per-target", type=int, default=3, help="CTs per target, rotating (0 = all)")
    ap.add_argument("--starts-per-ct", type=int, default=1)
    ap.add_argument("--target-mask", nargs="+", default=["seg", "gt"], choices=["seg", "gt", "oof"])
    ap.add_argument("--seg-ckpt", default=None, help="override the run's segmenter (e.g. for a CT-mask run)")
    ap.add_argument("--contact", default="on", choices=["on", "off"])
    ap.add_argument("--min-px", type=int, default=20)
    ap.add_argument("--start-seed", type=int, default=4242)
    ap.add_argument("--no-episodes", action="store_true")
    ap.add_argument("--allow-mismatch", action="store_true")
    ap.add_argument("--allow-legacy-seg", action="store_true")
    ap.add_argument("--controller", nargs="+", default=None, choices=list(CONTROLLERS),
                    help="v2.1 opt-in (the pre-specified protocol): run these controllers on identical setups with the "
                         "identical non-privileged keep-best; omitted = the v2 single-policy path and format")
    ap.add_argument("--rw-seed", type=int, default=20261012,
                    help="random_walk sign seed (fixed in advance; mixed with target, CT, start and mask)")
    ap.add_argument("--target-index", type=int, nargs="+", default=None,
                    help="run only these indices of the full target list (each keeps its full-list index, so "
                         "its CTs, window and start equal a full run's); for smoke tests and sharding")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    ctrls = ["policy"] if args.controller is None else [c for c in CONTROLLERS if c in args.controller]
    if "random_walk" in ctrls and "policy" not in ctrls:
        ap.error("random_walk takes its step sizes from the twists the policy executed: list 'policy' too")

    import torch
    from scipy.stats import spearmanr
    import train_view_match_rl as T
    from eval_view_ladder import load_policy, provenance_mismatch
    from usbg import goal_rl as G, render_lotus as R, volume as V, view_match as VM
    from usbg import provenance as PROV, us_segmenter as USS
    from usbg.benchmark import TEST_PATIENTS
    from usbg.similarity import load_similarity  # noqa: F401  (combined_score ignores sim)
    organ_id = {"liver": V.LIVER, "kidney": V.KIDNEY}
    organ_name = {v: k for k, v in organ_id.items()}
    T._CONTACT = (args.contact == "on")
    dev = R.get_device()
    ct_cases = args.ct_cases or list(TEST_PATIENTS)
    lvs = dict(zip(ct_cases, T.load_volumes(ct_cases)))
    results, targets_n_all, srcs = [], {}, {}
    for pj in args.policy_json:
        J = json.load(open(pj)); c = J["config"]; jp = J.get("provenance") or {}
        organs = args.organs or [o for o in (c.get("targets") or ["liver"]) if o in organ_id]
        targets, srcs = load_targets(args, [organ_id[o] for o in organs])
        n_per = {}
        for tg in targets:
            k = f"{tg['src']}:{organ_name[tg['organ']]}"
            n_per[k] = n_per.get(k, 0) + 1
        targets_n_all[pj] = n_per
        print(f"[{os.path.basename(pj)}] targets: {n_per}")
        seg_ckpt = args.seg_ckpt or (jp.get("mask") or {}).get("seg_ckpt") or os.environ.get("US_SEG_CKPT")
        if not seg_ckpt:
            raise SystemExit(f"{pj}: no segmenter recorded (a CT-mask run?); pass --seg-ckpt explicitly")
        env_ck = os.environ.get("US_SEG_CKPT")
        if env_ck and os.path.abspath(env_ck) != os.path.abspath(seg_ckpt) and not args.allow_mismatch:
            raise SystemExit(f"US_SEG_CKPT={env_ck} differs from the run's segmenter {seg_ckpt}")
        seg = USS.load_segmenter(seg_ckpt)
        VM.set_mask_source(seg)                          # sim-side current mask = this run's segmenter
        T.SEAT_ORGANS = list(c.get("targets") or organs)
        T._MIN_WINDOWS = int(c.get("min_windows") or 1)
        tex = T.resolve_texture(c["translator"], c.get("texture_ckpt"), None)
        v21 = {}                                         # v2.1 opt-in fields only (the v2 block is unchanged)
        if args.controller is not None:
            from usbg.eval import ScriptedScan
            v21 = dict(analysis_status=ANALYSIS_STATUS, plan_step="keep-best nulls on real-frame registration",
                       controllers=ctrls, rw_seed=args.rw_seed,
                       random_walk="signflip of the paired policy episode's executed per-step twist "
                                   "magnitudes, 6 i.i.d. signs per step",
                       scripted={k: v for k, v in vars(ScriptedScan()).items() if not k.startswith("_")},
                       keepbest_rule="argmax r_dep over n_steps+1 states, "
                       "start included, first index on ties (identical for every controller)",
                       max_twist=[float(x) for x in G.MAX_TWIST])
        if args.target_index is not None:
            v21["target_index"] = list(args.target_index)
        prov = PROV.collect(stage=c["translator"], texture_ckpt=tex, policy_json=os.path.abspath(pj),
                            policy_json_md5=PROV.md5(pj), seg_ckpt=os.path.abspath(seg_ckpt),
                            seg_md5=PROV.md5(seg_ckpt), targets_n=n_per, sources=srcs, ct_cases=ct_cases,
                            cts_per_target=args.cts_per_target, starts_per_ct=args.starts_per_ct,
                            contact=args.contact, organs=organs, bn_mode=T.BN_MODE, **v21)
        if prov["seg_md5"] in PROV.WITHDRAWN_SEG_MD5 and not args.allow_legacy_seg:
            raise SystemExit(f"{seg_ckpt} is the withdrawn shared segmenter (D2): refusing")
        if jp:
            mm = provenance_mismatch(jp, prov)
            prov["mismatch_vs_run"] = [list(x) for x in mm]
            if mm and not args.allow_mismatch:
                raise SystemExit(f"{pj}: provenance mismatch vs the run: {mm}")
        seed = c["seed"]
        torch.manual_seed(seed); np.random.seed(seed)
        tr = None if tex is None else __import__("usbg.texture_gan", fromlist=["load_translator"]).load_translator(tex, dev)
        cfg = R.RenderConfig(seed=seed)
        setup_kw = dict(lateral_m=c["lateral_m"], axial_m=c["axial_m"], fan_deg=c["fan_deg"])
        size, hist_n, n_steps = c["image_size"], c["history"], c["n_steps"]
        w_seg, w_app = c["w_seg"], c["w_app"]
        residual = bool(c["fix_residual"])
        T._RESIDUAL_MODE = J.get("residual_mode", "none") if residual else "oracle"
        pol, ck = load_policy(pj.replace(".json", ".pt"), c.get("residual_frac", 0.5), residual, dev)
        if residual and T._RESIDUAL_MODE == "image":
            ib = ck.get("ibvs")
            if ib is not None and ib["kind"] == "linear":
                T._IBVS = {"kind": "linear", "model": np.asarray(ib["model"]), "rich": ib["rich"]}
            else:                                         # pre-v2 run: replay the trainer's RNG
                rng = np.random.default_rng(seed)
                lv_train = T.load_volumes(c["train_cases"])
                T.init_train_cells(lv_train, c["train_cases"])
                T._IBVS = T.calibrate_ibvs(lv_train, rng, tr, cfg, size, setup_kw, kind=c["ibvs_base"],
                                           rich=(c["ibvs_features"] == "rich"))
        use_base = residual and T._RESIDUAL_MODE in ("image", "cnn")   # pose-free bases only (no CT pose of a real target)
        if "base" in ctrls and not use_base:
            raise SystemExit(f"{pj}: --controller base needs a pose-free residual base (residual_mode image/cnn); "
                             f"this run has residual={residual}, mode={T._RESIDUAL_MODE}")
        scan = None
        if "scripted" in ctrls:
            from usbg.eval import ScriptedScan
            scan = ScriptedScan()
        eps, pairs = [], []
        te = time.time()
        only_ti = None if args.target_index is None else set(args.target_index)
        for ti, tg in enumerate(targets):
            if only_ti is not None and ti not in only_ti:
                continue
            tgt_img, tgt_gt, o = tg["img"], tg["gt"], tg["organ"]
            oname = organ_name[o]
            tgt_seg = seg.segment(tgt_img)
            tgt_prep = T._prep(tgt_img, size)
            tgt_t = torch.from_numpy(tgt_prep[None, None]).to(dev)
            valid = tuple(sorted(tg["valid"]))
            ok_cts = [ci for ci, case in enumerate(ct_cases) if T.n_target_windows(lvs[case], oname) >= T._MIN_WINDOWS]
            if not ok_cts:
                continue
            k_ct = len(ok_cts) if args.cts_per_target <= 0 else min(args.cts_per_target, len(ok_cts))
            chosen = [ok_cts[(ti + j) % len(ok_cts)] for j in range(k_ct)]
            for ci in chosen:
                case = ct_cases[ci]
                lv = lvs[case]
                for k in range(args.starts_per_ct):
                    srng = np.random.default_rng([args.start_seed, ti, ci, k])
                    Wfv, _Tt, T_start, meta = T.make_setup(lv, srng, oname, setup_kw,
                                                           window=ti * args.starts_per_ct + k, count=False)
                    for mode in args.target_mask:
                        tmask = {"seg": tgt_seg, "gt": tgt_gt, "oof": tg.get("oof")}[mode]
                        if tmask is None:
                            continue
                        pol_abs = None                   # |twist| the policy executed per step (random_walk budget)
                        for ctrl in ctrls:               # ['policy'] without --controller: the v2 path
                            env = T.make_env(lv, Wfv, T_start, dev, np.random.default_rng([ti, ci, k]))
                            env.reset()
                            if ctrl == "scripted":
                                scan.reset()
                            elif ctrl == "random_walk":
                                rw_sign = 2.0 * np.random.default_rng(
                                    [args.rw_seed, ti, ci, k, _MASK_SALT[mode]]).integers(0, 2, (n_steps, 6)) - 1.0
                            img, lab = VM.curvi_render(lv, env.world_from_probe, Wfv, tr, cfg, size)
                            hist = deque([T._prep(img, size)] * hist_n, maxlen=hist_n)
                            traj, forces, twists = [], [], []
                            for step in range(n_steps + 1):
                                ct_lab = ct_label_at(lv, env.world_from_probe, Wfv, size)
                                traj.append(dict(
                                    iou_ct=VM.seg_organ_iou(ct_lab, tgt_gt, organs=(o,)),
                                    iou_seg=VM.seg_organ_iou(lab, tgt_gt, organs=(o,)),
                                    iou_ct_valid=VM.seg_organ_iou(ct_lab, tgt_gt, organs=valid),
                                    r_dep=VM.combined_score(img, lab, tgt_img, tmask, None, w_seg, w_app),
                                    r_gt=VM.combined_score(img, ct_lab, tgt_img, tgt_gt, None, w_seg, w_app)))
                                if step == n_steps:
                                    break
                                cur_np = np.stack(hist, 0)
                                if ctrl == "policy":
                                    with torch.no_grad():
                                        a, _ = pol.act(torch.from_numpy(cur_np[None]).to(dev), tgt_t, deterministic=True)
                                    a_exec = np.clip(a[0].cpu().numpy() + T.base_twist(None, None, lab, tmask, cur_np,
                                                     tgt_prep, use_base), -G.MAX_TWIST, G.MAX_TWIST)
                                elif ctrl == "base":     # zero residual + the same pose-free base
                                    a_exec = np.clip(np.zeros(6) + T.base_twist(None, None, lab, tmask, cur_np,
                                                     tgt_prep, use_base), -G.MAX_TWIST, G.MAX_TWIST)
                                elif ctrl == "scripted":
                                    a_exec = np.clip(scan(None, None), -G.MAX_TWIST, G.MAX_TWIST)
                                else:                    # random_walk: same |twist| per step and channel, random signs
                                    a_exec = rw_sign[step] * pol_abs[step]
                                twists.append(np.asarray(a_exec, np.float64))
                                env.step(a_exec)
                                forces.append(env.contact_force())
                                img, lab = VM.curvi_render(lv, env.world_from_probe, Wfv, tr, cfg, size)
                                hist.append(T._prep(img, size))
                            env.close()
                            if ctrl == "policy":
                                pol_abs = np.abs(np.asarray(twists, np.float64).reshape(n_steps, 6))
                            kb = int(np.argmax([s["r_dep"] for s in traj]))      # non-privileged keep-best, no GT
                            rec = dict(src=tg["src"], frame=tg["idx"], organ=oname, ct=case, start=k, mask=mode,
                                       window=meta.get("window"), start_clamped=bool(meta.get("start_clamped")),
                                       static_iou=traj[0]["iou_ct"], final_iou=traj[-1]["iou_ct"],
                                       keepbest_iou=traj[kb]["iou_ct"], keepbest_step=kb,
                                       static_iou_seg=traj[0]["iou_seg"], final_iou_seg=traj[-1]["iou_seg"],
                                       keepbest_iou_seg=traj[kb]["iou_seg"],
                                       static_iou_valid=traj[0]["iou_ct_valid"], final_iou_valid=traj[-1]["iou_ct_valid"],
                                       static_r=traj[0]["r_dep"], final_r=traj[-1]["r_dep"],
                                       peak_force=float(max(forces) if forces else 0.0),
                                       r_dep=[round(s["r_dep"], 4) for s in traj],
                                       r_gt=[round(s["r_gt"], 4) for s in traj])
                            if args.controller is not None:   # v2.1 fields (absent from the v2 format)
                                tw = np.asarray(twists, np.float64).reshape(-1, 6)
                                rec = dict(controller=ctrl, subject=tg.get("subject"), **rec,
                                           keepbest_iou_valid=traj[kb]["iou_ct_valid"],
                                           iou_ct_traj=[round(s["iou_ct"], 4) for s in traj],
                                           iou_valid_traj=[round(s["iou_ct_valid"], 4) for s in traj],
                                           twist_trans_sum=float(np.linalg.norm(tw[:, :3], axis=1).sum()),
                                           twist_rot_sum=float(np.linalg.norm(tw[:, 3:], axis=1).sum()))
                            eps.append(rec)
                            pairs += [(s["r_dep"], s["r_gt"], mode, ctrl) for s in traj]
        sec = time.time() - te
        summ = {}
        keys = ("static_iou", "final_iou", "keepbest_iou", "static_iou_seg", "final_iou_seg", "keepbest_iou_seg",
                "static_iou_valid", "final_iou_valid", "peak_force")
        if args.controller is not None:
            keys = keys + ("keepbest_iou_valid", "twist_trans_sum", "twist_rot_sum")
        for ctrl in ctrls:                               # one pass, unprefixed, without --controller (v2 keys)
            pre = "" if args.controller is None else f"{ctrl}:"
            eps_c = [e for e in eps if e.get("controller", "policy") == ctrl]
            for mode in args.target_mask:
                for src in sorted({e["src"] for e in eps_c}):
                    for org in sorted({e["organ"] for e in eps_c}):
                        E = [e for e in eps_c if e["mask"] == mode and e["src"] == src and e["organ"] == org]
                        if not E:
                            continue
                        s = {k: float(np.mean([e[k] for e in E])) for k in keys}
                        s.update(n_episodes=len(E), n_frames=len({e["frame"] for e in E}),
                                 n_cts=len({e["ct"] for e in E}))
                        summ[f"{pre}{mode}:{src}:{org}"] = s
                P = [(a, b) for a, b, m, cc in pairs if m == mode and cc == ctrl]
                if len(P) > 3:
                    rho, p = spearmanr([a for a, _ in P], [b for _, b in P])
                    summ[f"{pre}{mode}:spearman_r_dep_vs_r_gt"] = {
                        "pooled_rho": float(rho), "pooled_p": float(p), "n_pairs": len(P),
                        "within_episode": _spearman_within([e for e in eps_c if e["mask"] == mode])}
        print(f"[{os.path.basename(pj)}] {len(eps)} episodes in {sec:.0f}s ({sec / max(len(eps), 1):.2f} s/ep)")
        for k, v in summ.items():
            print(f"   {k}: {json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})}")
        res = {"policy_json": pj, "translator": c["translator"], "seed": seed, "arm": J.get("ablation"),
               "summary": summ, "n_episodes": len(eps), "sec": round(sec, 1),
               "sec_per_episode": round(sec / max(len(eps), 1), 3), "provenance": prov}
        if args.controller is not None:
            res.update(analysis_status=ANALYSIS_STATUS, controllers=ctrls,
                       n_episodes_per_controller={cc: sum(e["controller"] == cc for e in eps) for cc in ctrls})
        if not args.no_episodes:
            res["episodes"] = eps
        results.append(res)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    top = {"results": results, "targets_n": targets_n_all, "sources": srcs,
           "provenance": results[0]["provenance"] if results else None,
           "wall_sec": round(time.time() - t0, 1)}
    if args.controller is not None:
        top = {"analysis_status": ANALYSIS_STATUS, "controllers": ctrls, **top}
    with open(tmp, "w") as fh:
        json.dump(top, fh, indent=1, default=float)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
