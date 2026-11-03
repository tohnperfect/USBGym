#!/usr/bin/env python3
"""v2.1 pluggability, slot "controller": parity check and summary table of the plug_train_sb3.py outputs
(EXPLORATORY, opt-in; new file; reads results, writes two new files, overwrites nothing).

  parity  For every row the paper already reports (v2_final, v2_imit, servo_base, static, pose_oracle, cem),
          compare the episodes evaluated through USBGym/ViewMatch-v0 with the archived episodes of the same
          rung (outputs/v2/table4/... and outputs/v2_1/s22_lookahead/...), matched on (episode, patient,
          target, window): max absolute difference of the final and keep-best translation error, the final
          rotation error and the final S. "exact" < 1e-6, "close" < 0.05 mm and 0.05 deg, else "differs".
          This is the evidence that every plugged row sits on the same episodes and numbers as the oracle-ladder table.
  table   One row per (label, base, condition, weights, contact): mean and SD over seeds of the patient-level means
          (final and keep-best translation and rotation error, success at 5 mm / 5 deg and 10 mm / 10 deg,
          final S, peak force), plus training env steps, train hours and the env-steps-per-second of each run.

  PYTHONPATH=src .venv/bin/python scripts/plug_ctrl_collect.py parity --root outputs/v2_1/plug/ctrl \\
      --out outputs/v2_1/plug/ctrl/parity.json
  PYTHONPATH=src .venv/bin/python scripts/plug_ctrl_collect.py table --root outputs/v2_1/plug/ctrl \\
      --out outputs/v2_1/plug/ctrl/summary.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(REPO, "outputs", "v2", "table4")
LOOK = os.path.join(REPO, "outputs", "v2_1", "s22_lookahead")
PARITY_METHODS = ("v2_policy", "servo_base", "static", "pose_oracle", "cem")
KEYS = ("final_pos", "keepbest_pos", "final_ang", "final_match")
TABLE_METRICS = ("final_pos", "final_ang", "keepbest_pos", "keepbest_ang", "success_5mm_5deg", "success_10mm_10deg",
                 "kb_success_5mm_5deg", "kb_success_10mm_10deg", "final_match", "final_seg", "peak_force")


def load_runs(root):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        if os.path.basename(p) in ("parity.json", "summary.json"):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if d.get("slot") == "controller" and d.get("results"):
            d["_path"] = p
            runs.append(d)
    return runs


def cond_key(d):
    c = d["condition"]
    return f"{c['texture']}_{'ct' if c['mask'] == 'ct' else 'matched'}"


def reference(d):
    """-> (path, episodes) of the archived rung this run reproduces, or (None, None)."""
    c, m = d["condition"], d["method"]
    tex, arm, seed = c["texture"], ("ctmask" if c["mask"] == "ct" else "matched"), c["seed"]
    lad = "ct" if c["mask"] == "ct" else "matched"
    if m == "v2_policy":
        pt = (d.get("artifacts") or {}).get("model") or ""
        p = os.path.join(V2, arm, f"{tex}_deploy_s{seed}.json")
        key = "after_imitation" if pt.endswith("_imit.pt") else "final"
        if os.path.exists(p):
            return f"{p}#{key}", json.load(open(p))[key].get("episodes")
    elif m == "servo_base":
        p = os.path.join(V2, "ladder", f"ibvs_base_{tex}_{lad}_s{seed}.json")
        if os.path.exists(p):
            r = [b for b in json.load(open(p))["results"] if b["contact"] == "on" and b.get("eval_randomize", "off") == "off"]
            return p, r[0].get("episodes") if r else None
    elif m in ("static", "pose_oracle") and c["mask"] == "ct":
        p = os.path.join(V2, "ladder", f"{m}_{tex}_ct.json")
        if os.path.exists(p):
            r = [b for b in json.load(open(p))["results"] if b["contact"] == "on" and b.get("eval_randomize", "off") == "off"]
            return p, r[0].get("episodes") if r else None
    elif m == "cem":
        p = os.path.join(LOOK, f"lookahead_{tex}_ct_s{seed}_n12.json")
        if os.path.exists(p):
            r = [b for b in json.load(open(p))["results"] if b["contact"] == "on"]
            return p, r[0].get("episodes") if r else None
    return None, None


def parity(args):
    from usbg import provenance as PROV
    rows = []
    for d in load_runs(args.root):
        if d["method"] not in PARITY_METHODS:
            continue
        ref_path, ref = reference(d)
        blk = [b for b in d["results"] if b["contact"] == "on" and b.get("bn_mode", "train") == "train"]
        if not blk or not blk[0].get("episodes"):
            continue
        mine = blk[0]["episodes"]
        row = {"run": d["_path"], "method": d["method"], "label": d.get("label"), "condition": cond_key(d),
               "seed": d["condition"]["seed"],
               "reference": ref_path}
        if ref is None:
            row["status"] = "no archived reference"
        else:
            bykey = {(e["ep"], e["patient"], e["target"], e.get("window")): e for e in ref}
            diffs, missing = {k: 0.0 for k in KEYS}, 0
            for e in mine:
                r = bykey.get((e["ep"], e["patient"], e["target"], e.get("window")))
                if r is None:
                    missing += 1
                    continue
                for k in KEYS:
                    diffs[k] = max(diffs[k], abs(float(e[k]) - float(r[k])))
            row.update(max_abs_diff=diffs, n_episodes=len(mine), n_unmatched=missing)
            exact = missing == 0 and all(v < 1e-6 for v in diffs.values())
            close = missing == 0 and diffs["final_pos"] < 0.05 and diffs["keepbest_pos"] < 0.05 and diffs["final_ang"] < 0.05
            row["status"] = "exact" if exact else ("close" if close else "differs")
        rows.append(row)
    out = {"rows": rows, "counts": {s: sum(r["status"] == s for r in rows) for s in
                                    ("exact", "close", "differs", "no archived reference")},
           "provenance": {**PROV.collect(), "analysis_status": "exploratory, v2.1 pluggability", "slot": "controller",
                          "inputs": [r["run"] for r in rows]}}
    write(args.out, out)
    print(json.dumps(out["counts"]))


def table(args):
    from usbg import provenance as PROV
    groups = {}
    for d in load_runs(args.root):
        for b in d["results"]:
            key = (d.get("label") or d["method"], d["condition"]["base"], cond_key(d), b["weights"], b["contact"])
            groups.setdefault(key, []).append((d, b))
    rows = []
    for (m, base, cond, w, contact), L in sorted(groups.items()):
        row = {"label": m, "method": L[0][0]["method"], "base": base, "condition": cond, "weights": w, "contact": contact,
               "seeds": sorted(d["condition"]["seed"] for d, _ in L), "privileged": bool(L[0][0].get("privileged")),
               "citation": L[0][0].get("citation"), "smoke": any(d.get("smoke") for d, _ in L)}
        for k in TABLE_METRICS:
            v = [b["patient_level"].get(k) for _, b in L if b.get("patient_level", {}).get(k) is not None]
            row[k + "_mean"] = float(np.mean(v)) if v else None
            row[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        tr = [d.get("train") for d, _ in L if d.get("train")]
        row["train_env_steps"] = [t["env_steps"] for t in tr] or None
        row["train_hours"] = [round(t["sec"] / 3600.0, 3) for t in tr] or None
        row["env_steps_per_sec"] = [t["env_steps_per_sec"] for t in tr] or None
        rows.append(row)
    out = {"rows": rows, "metric_note": "patient-level means (mean over the 6 held-out patients of per-patient means) "
                                        "per run; mean and SD over seeds",
           "provenance": {**PROV.collect(), "analysis_status": "exploratory, v2.1 pluggability", "slot": "controller",
                          "inputs": sorted({d["_path"] for L in groups.values() for d, _ in L})}}
    write(args.out, out)
    csv_path = args.out[:-5] + ".csv" if args.out.endswith(".json") else args.out + ".csv"
    if os.path.exists(csv_path):
        raise SystemExit(f"refusing to overwrite {csv_path}")
    cols = ["label", "method", "base", "condition", "weights", "contact", "seeds", "privileged"] + \
           [k + s for k in TABLE_METRICS for s in ("_mean", "_sd")] + ["train_hours"]
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for r in rows:
            wr.writerow([r.get(c) for c in cols])
    print(f"{len(rows)} rows -> {args.out}, {csv_path}")


def write(path, obj):
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["parity", "table"])
    ap.add_argument("--root", default=os.path.join(REPO, "outputs", "v2_1", "plug", "ctrl"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    (parity if a.what == "parity" else table)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
