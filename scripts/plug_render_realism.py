#!/usr/bin/env python3
"""plug_render_realism.py: v2 sector realism of the plugged renderers, in the paper's protocol.

v2.1 pluggability study, renderer slot (exploratory). This is scripts/eval_realism.py --sector with a renderer
option: the SAME views (eval_realism.window_views: --views-per-case jittered poses per patient at the screened
liver and kidney windows, anatomical seat, clamp outside the skin, press 0-5 mm), the SAME pipeline (linear
render -> view_match._resize01 -> align_label_to_image -> view_match.warp_to_sector, i.e. the measured 128 px
Vitale sector the curvilinear stages were trained in), the SAME real reference (the 60 annotated public Vitale
TEST frames, data/eval/kaggle_vitale_test.npz), the SAME in-sector eroded mask on both domains, and the SAME
metrics and functions (intensity W1 pooled and per patient, KID primary / FID secondary via eval_fid_kid, the
grouped CNN C2ST of eval_realism.c2st_cnn with a permutation null on the physics row, robust renorm). Only the
renderer changes: one row per --renderers entry, each seeded per view by the view seed exactly as the realism table
seeds its physics (RenderConfig(seed=vseed)) and prior-LOTUS (seed=vseed) rows.

Parity: with the paper's arguments (6 held-out patients, 50 views each, seed 0, erode 3) the "physics" row must
reproduce the realism table's physics row and "lotus_reimpl" its "prior LOTUS" row (same frames by construction); with
--parity-ref the differences are written to the output (parity_with_table2).

No texture stage is applied (the renderer row is the raw forward model, as the realism table's renderer rows are).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE2_NAME = {"physics": "physics", "lotus_reimpl": "prior LOTUS"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renderers", nargs="+", default=["physics", "lotus_reimpl", "lotus_orig"])
    ap.add_argument("--cases", nargs="+", default=None, help="default benchmark.TEST_PATIENTS (the realism table)")
    ap.add_argument("--organs", nargs="+", default=["liver", "kidney"], choices=["liver", "kidney", "gallbladder"])
    ap.add_argument("--views-per-case", type=int, default=50)
    ap.add_argument("--press-max-mm", type=float, default=5.0)
    ap.add_argument("--real", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--erode", type=int, default=3)
    ap.add_argument("--renorm", choices=["none", "robust", "both"], default="both")
    ap.add_argument("--kid-subset", type=int, default=50)
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--c2st-repeats", type=int, default=5)
    ap.add_argument("--c2st-folds", type=int, default=5)
    ap.add_argument("--c2st-epochs", type=int, default=40)
    ap.add_argument("--c2st-size", type=int, default=64)
    ap.add_argument("--c2st-device", default="cpu")
    ap.add_argument("--null-repeats", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="render device of the torch renderers (the realism table: cpu)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parity-ref", default=None, help="realism-table JSON (outputs/v2/realism/realism_sector.json)")
    ap.add_argument("--save-frames", action="store_true", help="also write <out>_frames.npz (uint8)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")

    from scipy.ndimage import binary_erosion
    sys.path.insert(0, HERE)
    import eval_realism as ER
    from usbg import render_lotus as R, view_match as VM
    from usbg import provenance as P
    from usbg.placement import seat_mode
    from usbg.segmenter import align_label_to_image
    from usbg.texture_gan import intensity_w1
    from usbg.plug import renderers as PR
    if seat_mode() != "anatomical":
        raise SystemExit("the sector protocol is anatomical-seat only")
    if args.cases is None:
        from usbg.benchmark import TEST_PATIENTS
        args.cases = list(TEST_PATIENTS)
    if P.is_private_path(args.real):
        raise SystemExit(f"--real {args.real} is a private pool")
    PR.register_all()
    names = [PR.canonical_name(r) for r in args.renderers]

    t0 = time.time()
    _, _, M = VM._sector_coords(128, 128, VM.SECTOR)
    Me = binary_erosion(M, iterations=args.erode) if args.erode > 0 else M
    real_npz = np.load(args.real)
    real = real_npz["images"].astype(np.float32) / 255.0 * Me
    real_groups = [str(s) for s in real_npz["sessions"]]

    def render(name, lab, vseed):
        if name == "physics":
            return R.render(lab, "physics", device=args.device, cfg=R.RenderConfig(seed=vseed))
        return R.render(lab, name, device=args.device, cfg=R.RenderConfig(seed=vseed), seed=vseed)

    ranges = dict(slide=0.02, tilt=15.0, roll=10.0)
    pools = {n: [] for n in names}
    rsec = {n: 0.0 for n in names}
    groups, cases_of, skipped, wcount, lifts, presses = [], [], [], {}, [], []
    for case in args.cases:
        lv = ER.resolve_case(case)
        views, sk, cnt = ER.window_views(lv, case, args.organs, args.views_per_case, args.seed, ranges,
                                         args.press_max_mm / 1000.0)
        skipped += sk; wcount[case] = cnt
        for v in views:
            lab, vseed = v["lab"], v["vseed"]
            groups.append(f"{case}:{v['window']}"); cases_of.append(case)
            lifts.append(v["lift"]); presses.append(v["press"])
            for n in names:
                tr = time.time()
                img_lin = render(n, lab, vseed)
                rsec[n] += time.time() - tr
                img, _ = VM.warp_to_sector(VM._resize01(img_lin, 128), align_label_to_image(lab, (128, 128)))
                pools[n].append(img)
        print(f"  {case}: {len(views)} views ({cnt}); render s so far {({k: round(v, 1) for k, v in rsec.items()})}",
              flush=True)
    if not groups:
        raise SystemExit("no views rendered")
    groups = np.asarray(groups); cases_arr = np.asarray(cases_of)
    li = np.asarray(lifts)
    res = {"protocol": "sector v2 (the realism table) with the renderer swapped: anatomical windows, in-sector eroded mask "
                       "on both domains, the 60 annotated public Vitale TEST frames as the reference",
           "renderers": names, "citations": {n: PR.CITATIONS[n] for n in names},
           "seat_mode": P.current_seat_mode(), "cases": list(args.cases), "organs": list(args.organs),
           "views_per_case": args.views_per_case, "n_views": int(len(groups)), "window_counts": wcount,
           "skipped": skipped, "jitter": {**ranges, "press_max_mm": args.press_max_mm},
           "clamp": {"clamped_frac": float((li > 1e-4).mean()), "lift_mm_mean": float(li.mean() * 1e3),
                     "press_mm_mean": float(np.mean(presses) * 1e3)},
           "n_real": int(len(real)), "real": args.real, "real_md5": P.md5(args.real), "erode_px": args.erode,
           "renorm": args.renorm, "render_device": args.device, "real_mean_in_sector": float(real[:, Me].mean()),
           "c2st": {"repeats": args.c2st_repeats, "folds": args.c2st_folds, "epochs": args.c2st_epochs,
                    "size": args.c2st_size, "groups": "sim: case x window; real: frame"},
           "render_s_per_frame": {n: round(rsec[n] / len(groups), 4) for n in names},
           "variants": {}, "timing_s": {"render": round(sum(rsec.values()), 1)}}
    fr = inc = None
    if not args.no_fid:
        from eval_fid_kid import _inception, _features, _fid, _kid
        fdev = R.get_device()
        inc = _inception(fdev)
        fr = _features(inc, list(real), fdev)
    renorms = {"none": ["none"], "robust": ["robust"], "both": ["none", "robust"]}[args.renorm]
    t_c2st = t_inc = 0.0
    for name in names:
        Praw = np.asarray(pools[name], np.float32)
        Pm = Praw * Me
        v = {"n": int(len(Pm)), "mean_in_sector": float(Pm[:, Me].mean()),
             "w1_in_sector": intensity_w1(Pm[:, Me], real[:, Me]),
             "w1_per_case": {c: intensity_w1(Pm[cases_arr == c][:, Me], real[:, Me])
                             for c in args.cases if (cases_arr == c).any()}}
        if "robust" in renorms:
            Pr = np.stack([ER.robust_norm(x) for x in Praw]) * Me
            v["w1_in_sector_robust_renorm"] = intensity_w1(Pr[:, Me], real[:, Me])
            v["w1_per_case_robust_renorm"] = {c: intensity_w1(Pr[cases_arr == c][:, Me], real[:, Me])
                                              for c in args.cases if (cases_arr == c).any()}
        if fr is not None:
            ti = time.time()
            fg = _features(inc, list(Pm), R.get_device())
            v["fid_secondary"] = _fid(fr, fg)
            km, ks = _kid(fr, fg, subset=min(args.kid_subset, len(fr), len(fg)))
            v["kid"], v["kid_std"] = km, ks
            v["kid_per_case"] = {c: _kid(fr, fg[cases_arr == c],
                                         subset=min(args.kid_subset, len(fr), int((cases_arr == c).sum())))[0]
                                 for c in args.cases if (cases_arr == c).sum() >= 2}
            t_inc += time.time() - ti
        tc = time.time()
        a = np.asarray(ER.c2st_cnn(Pm, groups, real, real_groups, repeats=args.c2st_repeats, folds=args.c2st_folds,
                                   epochs=args.c2st_epochs, size=args.c2st_size, seed=args.seed,
                                   device=args.c2st_device), float)
        t_c2st += time.time() - tc
        v["c2st_auc_mean"] = float(np.nanmean(a)) if a.size else float("nan")
        v["c2st_auc_std"] = float(np.nanstd(a)) if a.size else float("nan")
        v["c2st_n"] = int(np.isfinite(a).sum())
        res["variants"][name] = v
        print(f"{name:14s} W1 {v['w1_in_sector']:.4f} KID {v.get('kid', float('nan')):.4f} "
              f"FID {v.get('fid_secondary', float('nan')):7.1f} C2ST {v['c2st_auc_mean']:.3f}+-{v['c2st_auc_std']:.3f}",
              flush=True)
    if "physics" in pools and args.null_repeats > 0:
        tc = time.time()
        null = np.asarray(ER.c2st_cnn(np.asarray(pools["physics"]) * Me, groups, real, real_groups,
                                      repeats=args.null_repeats, folds=args.c2st_folds, epochs=args.c2st_epochs,
                                      size=args.c2st_size, seed=args.seed + 99, permute=True,
                                      device=args.c2st_device), float)
        t_c2st += time.time() - tc
        res["c2st_permutation_null_physics"] = {"mean": float(np.nanmean(null)) if null.size else float("nan"),
                                                "std": float(np.nanstd(null)) if null.size else float("nan"),
                                                "n": int(np.isfinite(null).sum())}
    ph = res["variants"].get("physics")
    if ph is not None:
        res["physics_vs_each"] = {n: {"w1_physics_lower_all_cases": bool(all(
            ph["w1_per_case"][c] < res["variants"][n]["w1_per_case"][c] for c in ph["w1_per_case"])),
            "w1_per_case": {c: [ph["w1_per_case"][c], res["variants"][n]["w1_per_case"][c]] for c in ph["w1_per_case"]}}
            for n in names if n != "physics"}
    if args.parity_ref and os.path.exists(args.parity_ref):
        ref = json.load(open(args.parity_ref))
        same = (list(ref.get("cases", [])) == list(args.cases) and ref.get("views_per_case") == args.views_per_case
                and ref.get("erode_px") == args.erode)
        par = {"ref": os.path.abspath(args.parity_ref), "ref_md5": P.md5(args.parity_ref), "same_design": bool(same)}
        for n, tn in TABLE2_NAME.items():
            if n in res["variants"] and tn in ref.get("variants", {}):
                a, b = res["variants"][n], ref["variants"][tn]
                par[n] = {k: {"here": a.get(k), "table2": b.get(k),
                              "abs_diff": (abs(a[k] - b[k]) if isinstance(a.get(k), (int, float))
                                           and isinstance(b.get(k), (int, float)) else None)}
                          for k in ("w1_in_sector", "kid", "fid_secondary", "c2st_auc_mean", "mean_in_sector")}
        res["parity_with_table2"] = par
    res["timing_s"].update(inception=round(t_inc, 1), c2st=round(t_c2st, 1), total=round(time.time() - t0, 1))
    stem = args.out[:-5] if args.out.endswith(".json") else args.out
    if args.save_frames:
        np.savez_compressed(stem + "_frames.npz", **{n: (np.asarray(pools[n]) * 255).astype(np.uint8) for n in names},
                            groups=groups)
        res["frames_npz"] = os.path.abspath(stem + "_frames.npz")
    prov = P.collect(real={"path": args.real, "md5": P.md5(args.real)}, cohort={"eval": list(args.cases)})
    prov["analysis_status"] = PR.PLUG_STATUS
    prov["plug"] = {n: PR.plug_info(n) for n in names}
    res["provenance"] = prov
    res["analysis_status"] = PR.PLUG_STATUS
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
