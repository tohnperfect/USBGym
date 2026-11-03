#!/usr/bin/env python3
"""Aggregate the v2 oracle-ladder outputs under OUT into long-format tables for the mixed model.

  OUT/episodes_long.csv      one row per eval episode of every trainer run (after_imitation and final),
                             ladder rung, contact and DR eval (group = the OUT subfolder, so an in-run
                             eval and its contact re-eval are never pooled; arm, rung, stage, mask, seed, phase,
                             contact, eval_randomize, patient, target, window, pose / seg / match / force)
  OUT/registration_long.csv  one row per real-target registration episode
  OUT/cells.json             cell means: mean +/- population std over the per-seed means (the convention of the earlier tables), overall and per target

Refuses to aggregate when the provenance seat modes are mixed or missing (D4 guard, via
usbg.provenance.assert_one_seat), so an earlier canonical-seat output can never leak into a v2 cell.

  PYTHONPATH=src .venv/bin/python scripts/agg_table4_v2.py outputs/v2/table4
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

EP_COLS = ["source", "group", "arm", "rung", "stage", "mask", "seg_md5", "texture_md5", "seat", "seed", "phase",
           "contact", "eval_randomize", "design", "patient", "target", "window", "window_organ", "n_windows",
           "ep", "start_clamped", "clamp_mm", "start_pos", "start_ang", "final_pos", "final_ang",
           "keepbest_pos", "keepbest_ang", "static_seg", "final_seg", "keepbest_seg", "static_match",
           "final_match", "peak_force", "contact_frac"]
REG_COLS = ["source", "arm", "stage", "seed", "seg_md5", "seat", "src", "frame", "organ", "ct", "start", "mask",
            "window", "start_clamped", "static_iou", "final_iou", "keepbest_iou", "static_iou_seg",
            "final_iou_seg", "keepbest_iou_seg", "static_iou_valid", "final_iou_valid", "keepbest_step",
            "static_r", "final_r", "peak_force"]


def _seat(prov):
    return (prov or {}).get("seat_mode") or ((prov or {}).get("seat") or {}).get("resolved")


def result_files(out):
    fs = []
    for f in sorted(glob.glob(os.path.join(out, "**", "*.json"), recursive=True)):
        b = os.path.basename(f)
        if b in ("runner_state.json", "cells.json") or b.endswith(".provenance.json") or ".stale." in b:
            continue
        fs.append(f)
    return fs


def collect_rows(out):
    """-> (episode rows, registration rows, files used). Raises SystemExit on mixed / missing seats."""
    from usbg import provenance as PROV
    files, rows, reg = [], [], []
    for f in result_files(out):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or not ("after_imitation" in d or "rung" in d or
                                          ("results" in d and d.get("targets_n") is not None)):
            continue
        files.append(f)
    try:
        seat = PROV.assert_one_seat(files) if files else None
    except AssertionError as e:
        raise SystemExit(f"MIXED or MISSING SEAT MODES under {out}: {e}. Refusing to aggregate (D4 guard).")
    for f in files:
        d = json.load(open(f)); rel = os.path.relpath(f, out)
        group = rel.split(os.sep)[0] if os.sep in rel else "."   # ctmask/matched/dr_on (trainer), ladder, contact, dr_eval
        arm = group
        prov = d.get("provenance") or {}
        m = prov.get("mask") or {}
        base = dict(source=rel, group=group, seat=_seat(prov), texture_md5=(prov.get("texture") or {}).get("md5"),
                    seg_md5=m.get("seg_md5"))
        if "after_imitation" in d:                                     # trainer output
            for ph in ("after_imitation", "final"):
                res = d[ph]
                for e in res.get("episodes", []):
                    rows.append({**base, "arm": arm, "rung": "policy", "stage": d["translator"],
                                 "mask": "learned" if m.get("learned") else "ct", "seed": d["seed"], "phase": ph,
                                 "contact": res.get("contact"), "eval_randomize": res.get("eval_randomize"),
                                 "design": res.get("design"), **{k: e.get(k) for k in EP_COLS if k in e}})
        elif "rung" in d:                                              # eval_view_ladder output
            for r in d["results"]:
                phase = {"final": "final", "imit": "after_imitation"}.get(r.get("weights"), r.get("weights"))
                for e in r.get("episodes", []):
                    rows.append({**base, "arm": d.get("arm") or arm, "rung": d["rung"], "stage": d["translator"],
                                 "mask": d.get("mask_source"), "seed": d.get("seed", d.get("calib_seed")),
                                 "phase": phase, "contact": r.get("contact"),
                                 "eval_randomize": r.get("eval_randomize"), "design": r.get("design"),
                                 **{k: e.get(k) for k in EP_COLS if k in e}})
        else:                                                         # eval_real_registration output
            for r in d["results"]:
                rp = r.get("provenance") or {}
                for e in r.get("episodes", []):
                    reg.append({"source": rel, "arm": r.get("arm"), "stage": r.get("translator"), "seed": r.get("seed"),
                                "seg_md5": rp.get("seg_md5"), "seat": _seat(rp),
                                **{k: e.get(k) for k in REG_COLS if k in e}})
    return rows, reg, files, seat


def cell_table(rows, by=("group", "arm", "rung", "stage", "mask", "phase", "contact", "eval_randomize"),
               metric="final_pos"):
    cells = {}
    for r in rows:
        if r.get(metric) is None:
            continue
        key = tuple(str(r.get(k)) for k in by)
        cells.setdefault(key, {}).setdefault(r["seed"], []).append(float(r[metric]))
    out = []
    for k in sorted(cells):
        per_seed = [float(np.mean(v)) for v in cells[k].values()]
        out.append({**dict(zip(by, k)), "metric": metric, "mean": float(np.mean(per_seed)),
                    "std_over_seeds": float(np.std(per_seed)), "n_seeds": len(per_seed),
                    "n_episodes": int(sum(len(v) for v in cells[k].values()))})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args(argv)
    rows, reg, files, seat = collect_rows(a.out)
    with open(os.path.join(a.out, "episodes_long.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EP_COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "registration_long.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REG_COLS, extrasaction="ignore"); w.writeheader(); w.writerows(reg)
    cells = {"seat": seat, "n_files": len(files),
             "final_pos": cell_table(rows), "keepbest_pos": cell_table(rows, metric="keepbest_pos"),
             "final_pos_per_target": cell_table(rows, by=("group", "arm", "rung", "stage", "mask", "phase",
                                                         "contact", "eval_randomize", "target")),
             "registration_final_iou": cell_table(reg, by=("arm", "stage", "mask", "src", "organ"), metric="final_iou"),
             "registration_keepbest_iou": cell_table(reg, by=("arm", "stage", "mask", "src", "organ"),
                                                     metric="keepbest_iou")}
    with open(os.path.join(a.out, "cells.json"), "w") as fh:
        json.dump(cells, fh, indent=1)
    print(f"{len(rows)} episode rows, {len(reg)} registration rows from {len(files)} files, seat={seat} -> "
          f"{a.out}/episodes_long.csv, registration_long.csv, cells.json")
    print(f"{'group':9} {'arm':9} {'rung':11} {'stage':9} {'mask':7} {'phase':15} {'contact':7} {'dr':7} "
          f"{'final_pos mm':>16} {'n_seeds':>7} {'n_ep':>5}")
    for c in cells["final_pos"]:
        print(f"{c['group']:9} {c['arm']:9} {c['rung']:11} {c['stage']:9} {c['mask']:7} {c['phase']:15} {c['contact']:7} "
              f"{c['eval_randomize']:7} {c['mean']:8.2f} +- {c['std_over_seeds']:4.2f} {c['n_seeds']:7d} {c['n_episodes']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
