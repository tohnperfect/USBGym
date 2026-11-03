#!/usr/bin/env python3
"""plug_reward_agg.py: summary of the reward-slot pluggability runs (v2.1 pluggability, exploratory).

Reads the closed-loop runs of scripts/plug_reward_train.py (<train-root>/<arm>/<stage>_s<seed>.json) and the
landscape outputs of scripts/plug_reward_landscape.py, and writes one summary JSON (with a provenance
block) plus a printed table. Descriptive only: this script runs no hypothesis test.

Closed loop, per stage and reward arm: the final pose error on held-out patients (mm and deg), the keep-best
error and the match score S, as means over seeds of the per-seed means over the 60 crossed episodes; the
error after imitation, checked and reported to be the same in every arm of one stage and seed, since all
arms start from one imitation checkpoint; and each arm minus the
reference arm ("paper"), paired by seed and episode, with a bootstrap interval over patient clusters (patients
resampled with replacement, all seeds and episodes of a drawn patient kept; 2000 resamples, seed 0).

Landscape: per term and stage, the mean per-view learnability of the appearance term and of the combined
0.6/0.4 potential, from the primary DR arm.

Usage:
  PYTHONPATH=src python scripts/plug_reward_agg.py --train-root outputs/v2_1/plug/reward/train \
      --landscape outputs/v2_1/plug/reward/landscape/heldout.json outputs/v2_1/plug/reward/landscape/heldout_lpips.json \
      --out outputs/v2_1/plug/reward/summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

REF_ARM = "paper"
N_BOOT = 2000


def load_runs(root):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "*_s*.json"))):
        m = re.match(r"(.+)_s(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        arm, stage, seed = os.path.basename(os.path.dirname(f)), m.group(1), int(m.group(2))
        try:
            d = json.load(open(f))
        except Exception as e:                                       # pragma: no cover
            print(f"skip unreadable {f}: {e!r}")
            continue
        if "final" not in d:
            continue
        runs[(stage, arm, seed)] = (f, d)
    return runs


def _eps(d, key="final"):
    return {e["ep"]: e for e in (d[key].get("episodes") or [])}


def cluster_boot(diffs_by_patient, n_boot=N_BOOT, seed=0):
    pats = sorted(diffs_by_patient)
    if len(pats) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(len(pats), size=len(pats))
        x = np.concatenate([diffs_by_patient[pats[i]] for i in pick])
        vals.append(float(x.mean()))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarize_train(runs):
    stages = sorted({k[0] for k in runs})
    arms = sorted({k[1] for k in runs}, key=lambda a: (a != REF_ARM, a))
    out = {}
    for st in stages:
        out[st] = {}
        for arm in arms:
            seeds = sorted(s for (s_, a, s) in runs if s_ == st and a == arm)
            if not seeds:
                continue
            rows = {s: runs[(st, arm, s)][1] for s in seeds}
            rec = {"seeds": seeds, "files": [runs[(st, arm, s)][0] for s in seeds]}
            for key, src in (("pos", "pos"), ("ang", "ang"), ("keepbest_pos", "keepbest_pos"), ("match", "match"),
                             ("seg", "seg"), ("start_pos", "start_pos")):
                v = np.asarray([rows[s]["final"][src] for s in seeds], float)
                rec[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None,
                            "per_seed": [float(x) for x in v]}
            rec["after_imitation_pos"] = [float(rows[s]["after_imitation"]["pos"]) for s in seeds]
            pl = [(rows[s].get("plug") or {}).get("train_summary") for s in seeds]
            rec["train_gain"] = {"plug_S": [p and p.get("plug_gain_mean") for p in pl],
                                 "paper_S": [p and p.get("paper_gain_mean") for p in pl]}
            rec["reward"] = ((rows[seeds[0]].get("plug") or {}).get("reward"))
            rec["privileged_reward"] = bool((rows[seeds[0]].get("plug") or {}).get("privileged"))
            rec["smoke"] = any(bool((rows[s].get("plug") or {}).get("smoke")) for s in seeds)
            rec["lpips_random_trunk_smoke"] = any(bool(((rows[s].get("plug") or {}).get("lpips") or {})
                                                       .get("random_trunk_smoke_only")) for s in seeds)
            rec["analysis_status"] = sorted({str(rows[s].get("analysis_status")) for s in seeds})
            if arm != REF_ARM:
                ref_seeds = [s for s in seeds if (st, REF_ARM, s) in runs]
                by_pat, n_pairs, imit_same = {}, 0, []
                for s in ref_seeds:
                    ref = runs[(st, REF_ARM, s)][1]
                    imit_same.append(ref["after_imitation"]["pos"] == rows[s]["after_imitation"]["pos"])
                    ea, eb = _eps(rows[s]), _eps(ref)
                    for ep in sorted(set(ea) & set(eb)):
                        if ea[ep]["patient"] != eb[ep]["patient"]:
                            raise SystemExit(f"episode {ep} patient differs between {arm} and {REF_ARM} ({st} s{s})")
                        by_pat.setdefault(ea[ep]["patient"], []).append(ea[ep]["final_pos"] - eb[ep]["final_pos"])
                        n_pairs += 1
                by_pat = {k: np.asarray(v, float) for k, v in by_pat.items()}
                allv = np.concatenate(list(by_pat.values())) if by_pat else np.zeros(0)
                rec["vs_paper"] = {"paired_seeds": ref_seeds, "n_episode_pairs": n_pairs,
                                   "pos_diff_mean_mm": float(allv.mean()) if allv.size else None,
                                   "pos_diff_ci95_patient_bootstrap": cluster_boot(by_pat),
                                   "post_imitation_identical": imit_same}
            out[st][arm] = rec
    return out


def summarize_landscape(paths):
    res = {}
    for p in paths or []:
        if not os.path.exists(p):
            res[p] = "missing"
            continue
        d = json.load(open(p))
        arm = d["dr_arms"][0]
        w = d["weights"][0]
        tab = {}
        for st in d["stages"]:
            r = d["results"][arm][st]
            tab[st] = {t: {"app": r[t]["app"]["per_view_mean"], "app_sd": r[t]["app"]["per_view_std"],
                           f"combined_ct_{w}": r[t][f"combined_ct_{w}"]["per_view_mean"]} for t in d["terms"]}
            for pot in d.get("potentials") or []:
                tab[st][pot] = {"potential": r[pot]["potential"]["per_view_mean"]}
        res[p] = {"arm": arm, "n_views": d["n_target_views"], "table": tab, "smoke": d.get("smoke"),
                  "parity_with_paper_landscape": d.get("parity_with_paper_landscape"),
                  "cost_ms_per_pair": (d.get("cost") or {}).get("ms_per_pair")}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--landscape", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from usbg import provenance as P
    from usbg.plug.rewards import ANALYSIS_STATUS
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")
    runs = load_runs(args.train_root)
    summary = {"analysis_status": ANALYSIS_STATUS, "reference_arm": REF_ARM,
               "train": summarize_train(runs), "landscape": summarize_landscape(args.landscape),
               "n_runs": len(runs),
               "notes": ["descriptive; this script runs no hypothesis test",
                         "pose and pose_area are privileged training rewards (the policy observes images only)",
                         "every arm of one (stage, seed) starts from the same post-imitation checkpoint"]}
    summary["provenance"] = P.collect(analysis_status=ANALYSIS_STATUS, train_root=os.path.abspath(args.train_root),
                                      landscape=[os.path.abspath(p) for p in args.landscape])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    for st, arms in summary["train"].items():
        print(f"\n[{st}] final held-out pose error (mm), mean over seeds; difference from '{REF_ARM}' (patient bootstrap)")
        for arm, r in arms.items():
            vs = r.get("vs_paper") or {}
            ci = vs.get("pos_diff_ci95_patient_bootstrap")
            print(f"  {arm:12s} n={len(r['seeds'])} pos {r['pos']['mean']:6.2f}  ang {r['ang']['mean']:5.2f}  "
                  f"keep-best {r['keepbest_pos']['mean']:6.2f}  match {r['match']['mean']:.3f}"
                  + (f"  d={vs['pos_diff_mean_mm']:+.2f} mm CI {ci}" if vs.get("pos_diff_mean_mm") is not None else "")
                  + ("  [privileged reward]" if r["privileged_reward"] else ""))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
