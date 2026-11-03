#!/usr/bin/env python3
"""fig_matched_layout.py: the matched-layout Fig. 1. For each annotated real Vitale frame, find
the held-out CT view whose organ layout best matches its annotation, then show real | annotation |
matched CT label | physics | each texture stage at that view.

Stage 1 (search on label slices only): per held-out CT, --n-cand probe poses around its screened acoustic
  windows (usbg.windows; organs alternate, windows round-robin): slide +/-3 cm in the skin's tangent
  plane, tilt +/-25 deg, roll about the beam +/-90 deg, clamp outside the skin, then press 0 to 5 mm.
  Each CT label slice is sector-warped (view_match.SECTOR), reduced to MATCH_ORGANS and stored at 64x64
  with its left-right mirror. Every (real frame, candidate) pair is scored at once by the mean per-organ
  IoU with view_match.seg_organ_iou semantics (organs present in either map with >= min px; an organ in
  only one map scores 0).
Stage 2 (refine): cross-entropy search at 128x128 around the --topk candidates of each real frame. A
  mirrored match is the pose T @ Rz(180 deg), which is exact for the symmetric convex grid: rendered
  images are never flipped.
Stage 3: physics (DR off, fixed speckle seed) and every texture stage rendered at the best pose. The JSON
  holds, per real frame: best IoU, per-organ IoU, coarse vs refined IoU, case, window, mirror flag and the
  4x4 world pose (plus the probe->volume pose).
Figure rows by a rule fixed before the search (no picking by eye): one row per organ stratum of the real
  annotation with >= 3 frames, in the order liver; liver+gallbladder; vessel; liver+vessel; kidney;
  liver+kidney (kidney strata only when kidney is a searched organ), taking the best-IoU frame of the
  stratum; then filled to --rows by global IoU rank; at most --max-per-patient rows per CT patient and
  no repeated real frame. The caption numbers are the median and IQR of best IoU over all frames.
--rerender-from JSON --arrays-out NPZ (option; the default run above is unchanged): skip stages 1 and 2, re-run
  stage 3 at the saved best poses of that JSON's figure rows with its settings, check each row's pose and IoU
  against it, and save the rendered arrays (used by make_figures.py layout of the paper build).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib

import numpy as np

ORGANS = (1, 2, 3, 4, 7)                         # view_match.MATCH_ORGANS (liver, GB, vessel, kidney, spleen)
ORGAN_NAME = {1: "liver", 2: "gallbladder", 3: "vessel", 4: "kidney", 7: "spleen"}
STRATA = [("liver",), ("liver", "gallbladder"), ("vessel",), ("liver", "vessel"), ("kidney",), ("liver", "kidney")]
JIT = dict(slide=0.03, tilt=25.0, roll=90.0, press=0.005)
CE_SD = np.array([0.006, 0.006, 0.002, 5.0, 5.0, 10.0])    # tx ty press rx ry rz


def to64(lab128):
    return lab128[::2, ::2]


def iou_matrix(C, Rl, min_px):
    """C: (Nc, P) uint8 candidate maps, Rl: (Nr, P) real maps. (Nr, Nc) mean per-organ IoU."""
    Nc, Nr = len(C), len(Rl)
    s = np.zeros((Nr, Nc), np.float32); n = np.zeros((Nr, Nc), np.float32)
    for o in ORGANS:
        c = (C == o).astype(np.float32); r = (Rl == o).astype(np.float32)
        ca, ra = c.sum(1), r.sum(1)
        inter = r @ c.T
        union = ra[:, None] + ca[None, :] - inter
        present = (ra[:, None] >= min_px) | (ca[None, :] >= min_px)
        s += np.where(present, inter / np.maximum(union, 1), 0)
        n += present
    return np.where(n > 0, s / np.maximum(n, 1), 1.0)


def per_organ_iou(a, b, min_px):
    out = {}
    for o in ORGANS:
        pa, pb = a == o, b == o
        if pa.sum() < min_px and pb.sum() < min_px:
            continue
        out[ORGAN_NAME[o]] = float((pa & pb).sum() / max(int((pa | pb).sum()), 1))
    return out


def stratum(organs_present):
    return tuple(o for o in ("liver", "gallbladder", "vessel", "kidney", "spleen") if o in organs_present)


def pick_rows(frames, rows, max_per_patient, kidney):
    """The row rule of the module doc, fixed before the search. Returns [(frame_record, reason)]."""
    chosen, used, per_pat = [], set(), {}

    def ok(f):
        return f["frame"] not in used and per_pat.get(f["case"], 0) < max_per_patient

    def take(f, why):
        chosen.append((f, why)); used.add(f["frame"]); per_pat[f["case"]] = per_pat.get(f["case"], 0) + 1
    for st in STRATA:
        if "kidney" in st and not kidney:
            continue
        members = sorted([f for f in frames if tuple(f["stratum"]) == st], key=lambda f: -f["best_iou"])
        if len(members) < 3:
            continue
        for f in members:
            if ok(f):
                take(f, f"stratum {'+'.join(st)} (n={len(members)})")
                break
        if len(chosen) >= rows:
            break
    for f in sorted(frames, key=lambda f: -f["best_iou"]):
        if len(chosen) >= rows:
            break
        if ok(f):
            take(f, "global IoU rank")
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=None, help="default: benchmark.TEST_PATIENTS")
    ap.add_argument("--organs", nargs="+", default=["liver", "kidney"], choices=["liver", "kidney", "gallbladder"])
    ap.add_argument("--real", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--n-cand", type=int, default=2000, help="candidate poses per case")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--ce-iters", type=int, default=4)
    ap.add_argument("--ce-pop", type=int, default=48)
    ap.add_argument("--min-px", type=int, default=20)
    ap.add_argument("--stages", nargs="+", default=["physics="], metavar="STAGE=CKPT")
    ap.add_argument("--tex-manifest", default=None, help="JSON {stage: ckpt|null} (adds every stage)")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--max-per-patient", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-canonical", action="store_true")
    ap.add_argument("--out-json", default=None, help="required unless --rerender-from")
    ap.add_argument("--out", default=None, help="figure PNG (required unless --rerender-from)")
    ap.add_argument("--contact-sheet", default=None, help="all-frames contact sheet PNG")
    ap.add_argument("--rerender-from", default=None, metavar="JSON",
                    help="no search: re-run only stage 3 (render) at the saved best poses of this JSON's figure "
                         "rows, with its cases, organs, stages, seed, real set and min_px; the pose and the IoU "
                         "of every row are checked against the JSON. Writes --arrays-out (and, with --out, the "
                         "figure in this script's own format, for a pixel comparison); never writes --out-json")
    ap.add_argument("--arrays-out", default=None, metavar="NPZ",
                    help="with --rerender-from: the re-rendered rows (CT label and every stage) plus a .json sidecar")
    args = ap.parse_args(argv)
    saved = None
    if args.rerender_from:
        if not args.arrays_out:
            ap.error("--rerender-from needs --arrays-out")
        if args.out_json:
            ap.error("--rerender-from never writes --out-json")
        with open(args.rerender_from) as fh:
            saved = json.load(fh)
        args.cases, args.organs = list(saved["cases"]), list(saved["organs"])
        args.real, args.min_px, args.seed = saved["real"]["path"], int(saved["min_px"]), int(saved["render_cfg"]["seed"])
        args.stages, args.tex_manifest = [f"{k}={v or ''}" for k, v in saved["stages"].items()], None
    elif not (args.out_json and args.out):
        ap.error("--out-json and --out are required")

    from usbg import volume as V, slicer as S, view_match as VM, render_lotus as R, windows as W
    from usbg.placement import seat_volume_under_probe, clamp_start_outside_skin, seat_mode
    from usbg.segmenter import align_label_to_image
    from usbg.contracts_bridge import pose_from_rpy
    from usbg.texture_gan import load_translator
    from usbg import provenance as P

    if seat_mode() == "canonical" and not args.allow_canonical:
        raise SystemExit("the matched-layout search runs at the anatomical windows; USBG_SEAT=canonical "
                         "needs --allow-canonical")
    if args.cases is None:
        from usbg.benchmark import TEST_PATIENTS
        args.cases = list(TEST_PATIENTS)
    stage_paths = {}
    for kv in args.stages:
        k, p = kv.split("=", 1)
        stage_paths[k] = p or None
    if args.tex_manifest:
        with open(args.tex_manifest) as fh:
            for k, v in json.load(fh).items():
                if not k.startswith("_"):
                    stage_paths.setdefault(k, v)
    stage_paths.setdefault("physics", None)
    stages = {k: (load_translator(p, args.device) if p else None) for k, p in stage_paths.items()}

    t0 = time.time()
    real = np.load(args.real)
    RL = np.where(np.isin(real["labels"], ORGANS), real["labels"], 0).astype(np.uint8)
    RI = real["images"].astype(np.float32) / 255.0
    sessions = [str(s) for s in real["sessions"]]
    RL64 = np.stack([to64(r) for r in RL]).reshape(len(RL), -1)
    Rz180 = pose_from_rpy(rz=180.0)
    lab_of = {"liver": V.LIVER, "kidney": V.KIDNEY, "gallbladder": V.GALLBLADDER}

    def contact(T0, q):
        T, _ = clamp_start_outside_skin(T0 @ pose_from_rpy(tx=q[0], ty=q[1], tz=0.0, rx=q[3], ry=q[4], rz=q[5]))
        T = T.copy(); T[2, 3] -= abs(q[2])
        return T

    def sector_of(l):
        _, wl = VM.warp_to_sector(np.zeros((128, 128), np.float32), l)
        return np.where(np.isin(wl, ORGANS), wl, 0).astype(np.uint8)

    def label_sector(lv, T, Wfv):
        return sector_of(align_label_to_image(S.slice_at_pose(lv, T, Wfv), (128, 128)))

    if saved is not None:
        return _rerender_rows(args, saved, stages, stage_paths, RI, RL, contact, Rz180, lab_of)

    # ---------------------------------------------------------------- stage 1: candidates
    cands, meta, lvs, seats, wcount, skipped = [], [], {}, {}, {}, []
    for case in args.cases:
        lv = V.load_cache(os.path.join("data/cache", f"{case}.npz")); lvs[case] = lv
        per = []
        wcount[case] = {}
        for organ in args.organs:
            try:
                metas = W.windows_meta(lv, lab_of[organ]); why = "no accepted window"
            except W.NoAcceptedWindow as e:
                metas, why = [], str(e)
            wcount[case][organ] = len(metas)
            if metas:
                per.append((organ, metas))
            else:
                skipped.append({"case": case, "organ": organ, "reason": why})
        rng = np.random.default_rng([args.seed, zlib.crc32(case.encode())])
        for j in range(args.n_cand if per else 0):
            organ, metas = per[j % len(per)]
            wi = (j // len(per)) % len(metas)
            key = (case, organ, wi)
            if key not in seats:
                seats[key] = seat_volume_under_probe(lv, lab_of[organ], window=wi)
            Wfv, T0 = seats[key]
            d = rng.uniform
            q = np.array([d(-JIT["slide"], JIT["slide"]), d(-JIT["slide"], JIT["slide"]), d(0, JIT["press"]),
                          d(-JIT["tilt"], JIT["tilt"]), d(-JIT["tilt"], JIT["tilt"]), d(-JIT["roll"], JIT["roll"])])
            l = align_label_to_image(S.slice_at_pose(lv, contact(T0, q), Wfv), (128, 128))
            for mirror in (0, 1):               # mirror the LINEAR label (lateral axis), then warp:
                L = sector_of(l[:, ::-1] if mirror else l)          # = the Rz(180) view, sector in place
                cands.append(to64(L).ravel())
                meta.append((case, organ, wi, f"{metas[wi]['organ']}/w{metas[wi]['window_id']}", mirror, q))
        print(f"  {case}: {args.n_cand if per else 0} candidates ({wcount[case]})", flush=True)
    if not cands:
        raise SystemExit("no candidates (no accepted windows)")
    C = np.stack(cands)
    t_cand = time.time() - t0
    t1 = time.time()
    I = iou_matrix(C, RL64, max(1, args.min_px // 4))
    t_score = time.time() - t1

    # ---------------------------------------------------------------- stage 2: CE refinement
    t2 = time.time()
    results = []
    for r in range(len(RL)):
        top = np.argsort(-I[r])[:args.topk]
        best = (-1.0, None)
        rng = np.random.default_rng([args.seed, 1000 + r])
        for c in top:
            case, organ, wi, wname, mirror, q0 = meta[c]
            lv = lvs[case]; Wfv, T0 = seats[(case, organ, wi)]
            mu, sd = q0.copy(), CE_SD.copy()
            for _ in range(args.ce_iters):
                pop = mu + sd * rng.standard_normal((args.ce_pop, 6))
                pop[0] = mu
                sc = []
                for q in pop:
                    T = contact(T0, q)
                    T = T @ Rz180 if mirror else T
                    sc.append(VM.seg_organ_iou(label_sector(lv, T, Wfv), RL[r], organs=ORGANS, min_pixels=args.min_px))
                sc = np.asarray(sc)
                if sc.max() > best[0]:
                    qb = pop[int(sc.argmax())]
                    best = (float(sc.max()), dict(case=case, organ=organ, window_index=int(wi), window=wname,
                                                  mirror=int(mirror), q=[float(x) for x in qb]))
                elite = pop[np.argsort(-sc)[:max(4, args.ce_pop // 8)]]
                mu, sd = elite.mean(0), np.maximum(elite.std(0), CE_SD * 0.3)
        organs_real = [ORGAN_NAME[o] for o in ORGANS if (RL[r] == o).sum() >= args.min_px]
        results.append(dict(frame=int(r), session=sessions[r], real_organs=organs_real,
                            stratum=list(stratum(organs_real)), coarse_best=float(I[r].max()),
                            best_iou=best[0], **best[1]))
    t_ce = time.time() - t2

    # ---------------------------------------------------------------- stage 3: render at the best pose
    t3 = time.time()
    renders = {}
    for f in results:
        lv = lvs[f["case"]]; Wfv, T0 = seats[(f["case"], f["organ"], f["window_index"])]
        T = contact(T0, np.asarray(f["q"]))
        T = T @ Rz180 if f["mirror"] else T
        lab = S.slice_at_pose(lv, T, Wfv)
        img = R.render(lab, "physics", device=args.device, cfg=R.RenderConfig(seed=args.seed))
        wi_, wl_ = VM.warp_to_sector(VM._resize01(img, 128), align_label_to_image(lab, (128, 128)))
        ims = {}
        for k, tr in stages.items():
            x = wi_ if tr is None else (tr.translate_aligned(wi_, wl_) if getattr(tr, "wants_label", False) else tr(wi_))
            ims[k] = np.asarray(x, np.float32)
        Lm = np.where(np.isin(wl_, ORGANS), wl_, 0).astype(np.uint8)
        f["refined_iou_check"] = VM.seg_organ_iou(Lm, RL[f["frame"]], organs=ORGANS, min_pixels=args.min_px)
        f["per_organ_iou"] = per_organ_iou(Lm, RL[f["frame"]], args.min_px)
        f["pose_world"] = np.round(T, 8).tolist()
        f["pose_vp"] = np.round(np.linalg.inv(Wfv) @ T, 8).tolist()
        renders[f["frame"]] = (Lm, ims)
    t_render = time.time() - t3

    b = np.array([f["best_iou"] for f in results])
    rows = pick_rows(results, args.rows, args.max_per_patient, "kidney" in args.organs)
    win_by_case = {}
    for f in results:
        win_by_case[f["case"]] = win_by_case.get(f["case"], 0) + 1
    out = {"seat_mode": seat_mode(), "cases": list(args.cases), "organs": list(args.organs),
           "window_counts": wcount, "skipped": skipped, "n_cand_per_case": args.n_cand, "mirror": True,
           "topk": args.topk, "ce_iters": args.ce_iters, "ce_pop": args.ce_pop, "min_px": args.min_px,
           "jitter": {**JIT, "rule": "slide+tilt+roll about the contact point, clamp outside the skin, press"},
           "real": {"path": args.real, "md5": P.md5(args.real), "n": int(len(RL))},
           "stages": stage_paths, "render_cfg": {"dr": "off", "seed": args.seed},
           "best_iou_summary": {"median": float(np.median(b)), "mean": float(b.mean()),
                                "p25": float(np.percentile(b, 25)), "p75": float(np.percentile(b, 75)),
                                "coarse_median": float(np.median([f["coarse_best"] for f in results]))},
           "frames_won_per_case": win_by_case,
           "figure_rows": [{"frame": f["frame"], "session": f["session"], "case": f["case"],
                            "best_iou": f["best_iou"], "reason": why} for f, why in rows],
           "row_rule": "pre-specified: strata liver; liver+gallbladder; vessel; liver+vessel; kidney; "
                       "liver+kidney (>= 3 frames, best IoU), then global IoU rank; "
                       f"<= {args.max_per_patient} rows per CT patient, no repeated real frame",
           "timing_s": {"candidates": round(t_cand, 1), "score": round(t_score, 1), "ce": round(t_ce, 1),
                        "render": round(t_render, 1), "total": round(time.time() - t0, 1)},
           "frames": results}
    out["provenance"] = P.collect(stages={k: P.texture_info(k, p) for k, p in stage_paths.items()},
                                  cohort={"eval": list(args.cases)},
                                  stage_blocks={k: P.result_block(texture=p, eval_cases=args.cases,
                                                                  target_set=list(args.organs))
                                                for k, p in stage_paths.items() if p})
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    s = out["best_iou_summary"]
    print(f"best IoU over {len(results)} frames: median {s['median']:.3f} (IQR {s['p25']:.3f} to {s['p75']:.3f}); "
          f"coarse median {s['coarse_median']:.3f}; frames won per case {win_by_case}")
    print(json.dumps(out["timing_s"]))
    _figure([(f, why) for f, why in rows], renders, RI, RL, list(stages), args.out, s)
    if args.contact_sheet:
        _contact_sheet(results, renders, RI, RL, args.contact_sheet)
    print(f"wrote {args.out_json}, {args.out}" + (f", {args.contact_sheet}" if args.contact_sheet else ""))
    return 0


def _rerender_rows(args, saved, stages, stage_paths, RI, RL, contact, Rz180, lab_of):
    """--rerender-from: stage 3 of main() for the saved figure rows only, at the saved best poses."""
    from usbg import volume as V, slicer as S, view_match as VM, render_lotus as R
    from usbg.placement import seat_volume_under_probe, seat_mode
    from usbg.segmenter import align_label_to_image
    from usbg import provenance as P

    if seat_mode() != saved["seat_mode"]:
        raise SystemExit(f"seat {seat_mode()} != saved {saved['seat_mode']}")
    if P.is_private_path(args.real):
        raise SystemExit(f"{args.real}: private pool, refused")
    if P.md5(args.real) != saved["real"]["md5"]:
        raise SystemExit(f"{args.real}: md5 differs from the saved run")
    for k, p in stage_paths.items():
        want = (saved["provenance"]["stages"].get(k) or {}).get("md5")
        if (P.md5(p) if p else None) != want:
            raise SystemExit(f"stage {k}: checkpoint md5 differs from the saved run")
    frames = {f["frame"]: f for f in saved["frames"]}
    rows = [(frames[r["frame"]], r["reason"]) for r in saved["figure_rows"]]
    renders, checks, lvs = {}, [], {}
    for f, _why in rows:                                   # stage 3 of main(), unchanged
        if f["case"] not in lvs:
            lvs[f["case"]] = V.load_cache(os.path.join("data/cache", f"{f['case']}.npz"))
        lv = lvs[f["case"]]
        Wfv, T0 = seat_volume_under_probe(lv, lab_of[f["organ"]], window=f["window_index"])
        T = contact(T0, np.asarray(f["q"]))
        T = T @ Rz180 if f["mirror"] else T
        lab = S.slice_at_pose(lv, T, Wfv)
        img = R.render(lab, "physics", device=args.device, cfg=R.RenderConfig(seed=args.seed))
        wi_, wl_ = VM.warp_to_sector(VM._resize01(img, 128), align_label_to_image(lab, (128, 128)))
        ims = {}
        for k, tr in stages.items():
            x = wi_ if tr is None else (tr.translate_aligned(wi_, wl_) if getattr(tr, "wants_label", False) else tr(wi_))
            ims[k] = np.asarray(x, np.float32)
        Lm = np.where(np.isin(wl_, ORGANS), wl_, 0).astype(np.uint8)
        iou = VM.seg_organ_iou(Lm, RL[f["frame"]], organs=ORGANS, min_pixels=args.min_px)
        po = per_organ_iou(Lm, RL[f["frame"]], args.min_px)
        dpose = float(np.abs(np.round(T, 8) - np.asarray(f["pose_world"])).max())
        if dpose > 1e-8 or iou != f["refined_iou_check"] or po != f["per_organ_iou"]:
            raise SystemExit(f"frame {f['frame']}: re-render does not reproduce the saved row "
                             f"(pose diff {dpose:g}, IoU {iou} vs {f['refined_iou_check']})")
        checks.append({"frame": f["frame"], "pose_world_maxabs_diff": dpose, "iou": iou, "per_organ_iou": po})
        renders[f["frame"]] = (Lm, ims)
    order = [f["frame"] for f, _ in rows]
    os.makedirs(os.path.dirname(os.path.abspath(args.arrays_out)), exist_ok=True)
    np.savez_compressed(args.arrays_out, frame=np.asarray(order), ct_label=np.stack([renders[i][0] for i in order]),
                        **{f"stage_{k}": np.stack([renders[i][1][k] for i in order]) for k in stages})
    side = os.path.splitext(args.arrays_out)[0] + ".json"
    with open(side, "w") as fh:
        json.dump({"what": "fig_matched_layout.py stage 3 re-run at the saved best poses of the figure rows",
                   "source_json": args.rerender_from, "source_json_md5": P.md5(args.rerender_from),
                   "arrays": os.path.basename(args.arrays_out), "frames": order, "stages": list(stages),
                   "real": {"path": args.real, "md5": P.md5(args.real)}, "render_cfg": {"dr": "off", "seed": args.seed},
                   "device": args.device, "checks": checks,
                   "provenance": P.collect(stages={k: P.texture_info(k, p) for k, p in stage_paths.items()},
                                           cohort={"eval": list(args.cases)})}, fh, indent=1)
    if args.out:
        _figure(rows, renders, RI, RL, list(stages), args.out, saved["best_iou_summary"])
    print(f"re-rendered {len(order)} rows (pose and IoU match {args.rerender_from}); wrote {args.arrays_out}, {side}"
          + (f", {args.out}" if args.out else ""))
    return 0


_CMAP = {1: (0.85, 0.37, 0.0), 2: (0.0, 0.62, 0.45), 3: (0.0, 0.45, 0.70), 4: (0.80, 0.47, 0.65), 7: (0.94, 0.89, 0.26)}


def _contours(ax, lab):
    for o, c in _CMAP.items():
        m = (lab == o)
        if m.sum() >= 5:
            ax.contour(m.astype(float), levels=[0.5], colors=[c], linewidths=0.7)


def _label_rgb(lab):
    rgb = np.zeros(lab.shape + (3,), np.float32)
    for o, c in _CMAP.items():
        rgb[lab == o] = c
    return rgb


def _figure(rows, renders, RI, RL, stage_names, path, summ):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = ["real", "annotation", "matched CT label"] + stage_names
    n = max(1, len(rows))
    fig, axes = plt.subplots(n, len(cols), figsize=(1.15 * len(cols), 1.2 * n), squeeze=False)
    for i, (f, _why) in enumerate(rows):
        Lm, ims = renders[f["frame"]]
        panels = [(RI[f["frame"]], RL[f["frame"]], "gray"), (_label_rgb(RL[f["frame"]]), None, None),
                  (_label_rgb(Lm), None, None)] + [(ims[k], Lm, "gray") for k in stage_names]
        for j, (im, cl, cm) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(im, cmap=cm, vmin=0, vmax=1) if cm else ax.imshow(im)
            if cl is not None:
                _contours(ax, cl)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=6)
        axes[i][0].set_ylabel(f"{f['session']} | {f['case']}\nIoU {f['best_iou']:.2f}", fontsize=5)
    fig.text(0.5, 0.002, f"best per-organ IoU over all frames: median {summ['median']:.2f} "
             f"(IQR {summ['p25']:.2f} to {summ['p75']:.2f})", ha="center", fontsize=6)
    fig.tight_layout(pad=0.2, rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=220); plt.close(fig)


def _contact_sheet(results, renders, RI, RL, path, per_col=20):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(results); ncg = int(np.ceil(n / per_col)); nr = min(per_col, n)
    fig, axes = plt.subplots(nr, 4 * ncg, figsize=(4 * ncg * 0.9, nr * 0.95), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, f in enumerate(results):
        r, g = idx % per_col, idx // per_col
        Lm, ims = renders[f["frame"]]
        for j, im in enumerate([RI[f["frame"]], _label_rgb(RL[f["frame"]]), _label_rgb(Lm), ims["physics"]]):
            ax = axes[r][4 * g + j]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
        axes[r][4 * g].set_title(f"{f['session']} {f['case']} {f['best_iou']:.2f}", fontsize=4, loc="left")
    fig.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
