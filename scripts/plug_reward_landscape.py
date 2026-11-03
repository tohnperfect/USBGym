#!/usr/bin/env python3
"""plug_reward_landscape.py: appearance-term learnability of published reward terms (v2.1 pluggability,
exploratory, opt-in). Reuses scripts/view_match_landscape.py UNCHANGED.

The paper measures how learnable the appearance term is: around each held-out target view it renders 40
perturbed poses and takes, per view, the Spearman correlation between the term and closeness (the negative
translation distance to the target). This wrapper runs that exact protocol (view_match_landscape.main with
the same arguments, the same per-view seeds, samples, renders and bins) and, while it runs, scores every
(sample, target) pair it scores with the paper's term with each plugged term as well
(usbg.plug.rewards). It does so by wrapping three view_match functions for the duration of the run:

    sector_appearance  returns the paper's value unchanged and records every plugged term on the same pair
    pose_error         returns the value unchanged and records (translation mm, rotation deg)
    seg_organ_iou      returns the value unchanged and records the CT-oracle IoU (and the visible area)

so the base output (<out>_base.json, the view_match_landscape.py schema) is exactly what the unwrapped
script writes. The recorded calls are then aligned to (view, DR arm, sample, stage) from the base script's
fixed loop order, and the alignment is CHECKED: the reference term's per-view correlations recomputed here
must equal the base output's "app" correlations bit for bit, and the CT-oracle IoU's must equal "seg_ct".
A mismatch aborts before anything is written.

Per plugged term, arm and stage the output gives the per-view correlation (mean, std, list), the means by
pose-distance bin, the pooled correlation, the same for the combined potential w_seg * S_seg + w_app * term
(CT-oracle S_seg) for each --weights pair, and two paired Wilcoxon families (by view id, descriptive):
each term against the paper's ncc_lit within a stage, and the --ref stage against each other stage within
a term. The privileged potentials (pose, pose_area) are reported as ceilings. A secondary closeness, the
rotation-aware pose distance (mm + 2 deg), is reported next to the paper's translation closeness.

Only the CT-oracle segmentation term is supported (as in the paper's held-out landscape); --seg-ckpt and
--seg-manifest are refused. Raw per-sample values go to <out>.npz.

Usage (the paper's held-out landscape plus plugged terms; Python of the plugin venv if lpips is listed):
  USBG_SEAT=anatomical USBG_LEARNED_SEG=0 PYTHONPATH=src python scripts/plug_reward_landscape.py \
      --terms ncc_sector ncc_frame ssim nmi mi --out outputs/v2_1/plug/reward/landscape/heldout.json -- \
      --cases s0358 s0513 s0519 s0591 s0649 s0762 --organs liver kidney --targets-per-case 5 \
      --samples-per-target 40 --tex-manifest outputs/v2/textures/manifest.json --dr off --device cpu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(REPO, "outputs", "v2_1", "plug", "reward")
SECONDARY_ANG_W = 2.0


def load_landscape_module():
    path = os.path.join(REPO, "scripts", "view_match_landscape.py")
    spec = importlib.util.spec_from_file_location("view_match_landscape", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["view_match_landscape"] = mod
    spec.loader.exec_module(mod)
    return mod


def _check_out(path, smoke):
    p = os.path.abspath(path)
    if smoke:
        if "/plug_smoke/" not in p:
            raise SystemExit(f"--smoke outputs must go under a plug_smoke directory, got {p}")
    elif not p.startswith(OUT_ROOT + os.sep):
        raise SystemExit(f"outputs must go under {OUT_ROOT}/, got {p}")
    return p


def _nan_eq(a, b):
    return (a != a and b != b) or a == b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terms", nargs="+", default=["ncc_sector", "ncc_frame", "ssim", "nmi", "mi"],
                    help="plugged appearance terms (usbg.plug.rewards.APPEARANCE); ncc_lit, the paper's "
                         "term, is always the reference")
    ap.add_argument("--potentials", nargs="*", default=["pose", "pose_area"],
                    help="privileged potentials reported as ceilings")
    ap.add_argument("--lpips-alexnet", default=None, help="local torchvision AlexNet weights for lpips")
    ap.add_argument("--lpips-device", default="cpu")
    ap.add_argument("--lpips-random-trunk-smoke", action="store_true",
                    help="SMOKE ONLY: random AlexNet trunk (no weights); requires --smoke")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: TRAIN patients allowed (the texture-lineage refusal is recorded, not "
                         "raised); outputs must go under a plug_smoke directory")
    ap.add_argument("--parity-ref", default=None,
                    help="optional landscape JSON (e.g. outputs/v2/view_match/landscape_heldout.json) whose "
                         "per-view app correlations the reference term is compared with (recorded)")
    ap.add_argument("--out", required=True, help="plug result JSON (<out>_base.json, <out>.npz beside it)")
    args, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if args.lpips_random_trunk_smoke and not args.smoke:
        ap.error("--lpips-random-trunk-smoke is a code-path smoke test only; it needs --smoke")
    for bad in ("--out", "--seg-ckpt", "--seg-manifest", "--seg-seed"):
        if any(r == bad or r.startswith(bad + "=") for r in rest):
            ap.error(f"{bad} is not supported here (the plug output is --out; the landscape is CT-oracle only)")

    from usbg.plug import rewards as RW
    from usbg import provenance as P
    from usbg import view_match as VM

    unknown = [t for t in args.terms if t not in RW.APPEARANCE]
    if unknown:
        ap.error(f"unknown terms {unknown}; expected {RW.APPEARANCE}")
    terms = ["ncc_lit"] + [t for t in args.terms if t != "ncc_lit"]
    pots = [p for p in (args.potentials or []) if p]
    for p in pots:
        if p not in RW.PRIVILEGED:
            ap.error(f"unknown potential {p!r}; expected {RW.PRIVILEGED}")

    out = _check_out(args.out, args.smoke)
    base_out = out[:-5] + "_base.json" if out.endswith(".json") else out + "_base.json"
    npz_out = out[:-5] + ".npz" if out.endswith(".json") else out + ".npz"
    for p in (out, base_out, npz_out, base_out.replace(".json", ".png"), out.replace(".json", ".png")):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    if os.environ.get("USBG_LEARNED_SEG", "1").strip().lower() not in ("0", "false", "no", ""):
        print("note: USBG_LEARNED_SEG is not 0; view_match_landscape's anatomical path never reads the "
              "import-time segmenter (it calls set_mask_source(None)), so the seg term is the CT oracle")

    lp_kw = {"alexnet_path": args.lpips_alexnet, "device": args.lpips_device,
             "random_trunk": args.lpips_random_trunk_smoke}
    fns = {t: RW.appearance_fn(t, lp_kw if t == "lpips" else None) for t in terms}

    # ---------------------------------------------------------------- recorders (values pass through)
    rec = {"app": [], "pose": [], "seg": [], "vis": []}
    t_cost = {t: 0.0 for t in terms}

    def sector_appearance_rec(cur_img, tgt_img, thresh=RW.LIT_THRESH):
        if thresh != RW.LIT_THRESH:
            raise RuntimeError("the landscape called sector_appearance with a non-default threshold")
        row = []
        for t in terms:
            t0 = time.perf_counter()
            row.append(float(fns[t](cur_img, tgt_img)))
            t_cost[t] += time.perf_counter() - t0
        rec["app"].append(row)
        return row[0]                                   # the paper's value, unchanged

    def pose_error_rec(T_cur, T_tgt):
        r = RW._ref_pose_error(T_cur, T_tgt)
        rec["pose"].append((float(r[0]), float(r[1])))
        return r

    def seg_rec(lab_a, lab_b, organs=VM.MATCH_ORGANS, min_pixels=20):
        v = RW._ref_seg_organ_iou(lab_a, lab_b, organs, min_pixels)
        rec["seg"].append(float(v))
        rec["vis"].append(RW.visible_area(lab_a, lab_b))
        return v

    plug_info = {"slot": "reward", "evaluation": "appearance-term learnability (paper protocol)",
                 "terms": {t: RW.CITATIONS[t] for t in terms},
                 "potentials": {p: RW.CITATIONS[p] for p in pots},
                 "lpips": None, "versions": RW.package_versions(),
                 "smoke": bool(args.smoke), "wrapper": os.path.abspath(__file__)}
    if "lpips" in fns:
        plug_info["lpips"] = fns["lpips"].info()
    orig_collect = P.collect

    def collect_plug(**kw):
        prov = orig_collect(**kw)
        prov["analysis_status"] = RW.ANALYSIS_STATUS
        prov["plug"] = plug_info
        return prov

    VLS = load_landscape_module()
    lineage_report = {}
    orig_check = VLS.check_lineage

    def check_lineage_smoke(paths, eval_cases, allow_unchecked, what):
        try:
            r = orig_check(paths, eval_cases, True, what)
        except SystemExit as e:                          # smoke on TRAIN patients: record, do not refuse
            r = {"SMOKE_LINEAGE_REFUSAL_RECORDED": str(e)}
        lineage_report[what] = r
        return r

    t0 = time.time()
    with RW.patched(VM, "sector_appearance", sector_appearance_rec), \
            RW.patched(VM, "pose_error", pose_error_rec), \
            RW.patched(VM, "seg_organ_iou", seg_rec), \
            RW.patched(P, "collect", collect_plug), \
            RW.patched(VLS, "check_lineage", check_lineage_smoke if args.smoke else orig_check):
        rc = VLS.main(list(rest) + ["--out", base_out])
    wall = time.time() - t0
    if rc not in (0, None):
        raise SystemExit(f"view_match_landscape.main returned {rc}")

    # ---------------------------------------------------------------- align and check
    base = json.load(open(base_out))
    if base.get("seat_mode") != "anatomical":
        raise SystemExit("only the anatomical seat is supported")
    per_view = base["per_view"]
    stages = list(base["stages"])
    arms = list(base["dr_arms"])
    S = int(base["samples_per_target"])
    V, A, K, T = len(per_view), len(arms), len(stages), len(terms)
    weights = [tuple(float(x) for x in w.split("/")) for w in base["weights"]]
    exp = {"pose": V * S, "seg": V * A * S, "app": V * A * S * K}
    got = {k: len(rec[k]) for k in exp}
    if got != exp:
        raise SystemExit(f"recorded call counts {got} != expected {exp}: the landscape loop order changed; "
                         "the alignment would be wrong, nothing written")
    pose = np.asarray(rec["pose"], float).reshape(V, S, 2)
    dist, ang = pose[..., 0], pose[..., 1]
    seg = np.asarray(rec["seg"], float).reshape(V, A, S)
    vis = np.asarray(rec["vis"], float).reshape(V, A, S)
    app = np.asarray(rec["app"], float).reshape(V, A, S, K, T)
    sp = VLS.spearman
    mism = []
    for v in range(V):
        for ai, arm in enumerate(arms):
            c = sp(seg[v, ai], -dist[v])
            if not _nan_eq(c, per_view[v]["terms"][arm]["seg_ct"][stages[0]]):
                mism.append((per_view[v]["view_id"], arm, "seg_ct", c))
            for ki, st in enumerate(stages):
                c = sp(app[v, ai, :, ki, 0], -dist[v])
                if not _nan_eq(c, per_view[v]["terms"][arm]["app"][st]):
                    mism.append((per_view[v]["view_id"], arm, st, c))
    if mism:
        raise SystemExit(f"alignment check failed on {len(mism)} cells, e.g. {mism[:3]}; nothing written")

    BINS = VLS.BINS
    bidx = np.full((V, S), -1, int)
    for v in range(V):
        for s in range(S):
            b = next((i for i, bb in enumerate(BINS) if bb[0] <= dist[v, s] < bb[1]), None)
            bidx[v, s] = -1 if b is None else b
    close2 = -(dist + SECONDARY_ANG_W * ang)

    def summarize(vals):                                 # vals (V, S) for one arm / stage
        pv = np.asarray([sp(vals[v], -dist[v]) for v in range(V)], float)
        pv2 = np.asarray([sp(vals[v], close2[v]) for v in range(V)], float)
        f, f2 = pv[np.isfinite(pv)], pv2[np.isfinite(pv2)]
        by_bin = [float(vals[bidx == b].mean()) if (bidx == b).any() else float("nan") for b in range(len(BINS))]
        return {"per_view_mean": float(f.mean()) if f.size else float("nan"),
                "per_view_std": float(f.std()) if f.size else float("nan"), "n_views": int(f.size),
                "per_view": [float(x) for x in pv], "by_bin": by_bin,
                "pooled": sp(vals.ravel(), -dist.ravel()),
                "closeness_pose_per_view_mean": float(f2.mean()) if f2.size else float("nan")}

    results = {}
    for ai, arm in enumerate(arms):
        results[arm] = {}
        for ki, st in enumerate(stages):
            r = {}
            for ti, t in enumerate(terms):
                a = app[:, ai, :, ki, ti]
                r[t] = {"app": summarize(a)}
                for ws, wa in weights:
                    r[t][f"combined_ct_{ws:g}/{wa:g}"] = summarize(ws * seg[:, ai] + wa * a)
            for p in pots:
                pp = -(dist + RW.ANG_W * ang) / RW.POSE_SCALE
                if p == "pose_area":
                    pp = pp + RW.AREA_COEF * vis[:, ai]
                r[p] = {"potential": summarize(pp)}
            r["seg_ct"] = {"potential": summarize(seg[:, ai])}
            results[arm][st] = r

    def pv_get(arm, st, t, key):
        return lambda v: results[arm][st][t][key]["per_view"][v]

    vids = list(range(V))
    tests = {"term_vs_ncc_lit": {}, "ref_stage_vs_stage": {}}
    ref_stage = "sonospade" if "sonospade" in stages else None
    for arm in arms:
        tests["term_vs_ncc_lit"][arm] = {
            st: {t: {key: VLS.paired_test(vids, pv_get(arm, st, t, key), pv_get(arm, st, "ncc_lit", key))
                     for key in ("app", f"combined_ct_{weights[0][0]:g}/{weights[0][1]:g}")}
                 for t in terms[1:]} for st in stages}
        if ref_stage:
            tests["ref_stage_vs_stage"][arm] = {
                t: {st: VLS.paired_test(vids, pv_get(arm, ref_stage, t, "app"), pv_get(arm, st, t, "app"))
                    for st in stages if st != ref_stage} for t in terms}

    parity = None
    if args.parity_ref:
        pr = json.load(open(args.parity_ref))
        prv = {v["view_id"]: v for v in pr.get("per_view", [])}
        diffs = []
        for v in per_view:
            o = prv.get(v["view_id"])
            if o is None:
                continue
            for arm in arms:
                for st in stages:
                    a = v["terms"][arm]["app"][st]
                    b = ((o.get("terms") or {}).get(arm) or {}).get("app", {}).get(st)
                    if b is not None and a == a and b == b:
                        diffs.append(abs(a - b))
        parity = {"ref": os.path.abspath(args.parity_ref), "n_cells": len(diffs),
                  "max_abs_diff": float(max(diffs)) if diffs else None,
                  "identical": (max(diffs) == 0.0) if diffs else None}

    n_calls = V * A * S * K
    out_d = {
        "analysis_status": RW.ANALYSIS_STATUS,
        "what": "appearance-term learnability of plugged reward terms on the paper's landscape protocol",
        "base_json": base_out, "terms": terms, "potentials": pots, "stages": stages, "dr_arms": arms,
        "cases": base["cases"], "organs": base["organs"], "n_target_views": V, "samples_per_target": S,
        "bins_mm": base["bins_mm"], "weights": base["weights"], "closeness": "negative translation mm "
        "(paper); closeness_pose = -(mm + 2 deg)", "results": results, "tests": tests,
        "view_ids": [v["view_id"] for v in per_view],
        "alignment_check": {"cells": V * A * (K + 1), "mismatches": 0,
                            "rule": "reference term and CT seg correlations recomputed from the recorded "
                                    "calls equal the base output exactly"},
        "parity_with_paper_landscape": parity,
        "cost": {"wall_s": round(wall, 1), "pairs_scored": n_calls,
                 "ms_per_pair": {t: round(1000 * t_cost[t] / max(n_calls, 1), 3) for t in terms}},
        "smoke": bool(args.smoke), "smoke_lineage": lineage_report or None,
    }
    out_d["provenance"] = collect_plug(base_json=base_out, base_provenance_time=(base.get("provenance") or {})
                                       .get("time_utc"), landscape_argv=list(rest))
    np.savez_compressed(npz_out, dist_mm=dist, rot_deg=ang, seg_ct=seg, visible_area=vis, app=app,
                        terms=np.asarray(terms), stages=np.asarray(stages), arms=np.asarray(arms),
                        view_ids=np.asarray([v["view_id"] for v in per_view]))
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out_d, fh, indent=1)
    os.replace(tmp, out)
    _figure(out_d, out.replace(".json", ".png"))
    arm = arms[0]
    print(f"\n[{arm}] appearance-term learnability (per-view Spearman with closeness, mean over {V} views)")
    print("term        " + " ".join(f"{st:>10s}" for st in stages))
    for t in terms:
        print(f"{t:11s} " + " ".join(f"{results[arm][st][t]['app']['per_view_mean']:+10.3f}" for st in stages))
    for p in pots:
        print(f"{p:11s} " + " ".join(f"{results[arm][st][p]['potential']['per_view_mean']:+10.3f}" for st in stages)
              + "   (privileged ceiling)")
    print(f"cost ms/pair: {out_d['cost']['ms_per_pair']}  wall {wall:.0f}s")
    print(f"wrote {out}, {npz_out}, {base_out}")
    return 0


def _figure(d, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    arm = d["dr_arms"][0]
    terms, stages = d["terms"], d["stages"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    w = 0.8 / max(len(stages), 1)
    for ki, st in enumerate(stages):
        m = [d["results"][arm][st][t]["app"]["per_view_mean"] for t in terms]
        s = [d["results"][arm][st][t]["app"]["per_view_std"] for t in terms]
        ax.bar(np.arange(len(terms)) + ki * w, m, w, yerr=s, capsize=2, label=st)
    ax.set_xticks(np.arange(len(terms)) + 0.4 - w / 2)
    ax.set_xticklabels(terms, fontsize=8)
    ax.set_ylabel("learnability (Spearman)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
