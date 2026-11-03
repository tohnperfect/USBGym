#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): the identical-episode audit.

Each plug row must face the same starts as the reference rungs. The script reads each per-episode block of the
input files and forms its setup tuple
    (ep, patient, organ, window, start translation error, start rotation error)    rounded to 1e-6 mm / deg
which the case, organ, window and start pose of the setup fix (seed eval_seed + ep); neither texture nor mask source,
renderer, controller or reward changes it. The script compares this list with the list of the reference rung under
the same contact setting (on the original six patients the archived static rung, on the cohort its static rung). It
also checks the horizon (len(pos_traj) - 1 in --horizons) and records the render (speckle) seed and the target
speckle offset where the file records them. Any block whose fingerprint differs is REFUSED (exit 1), which keeps
the slot aggregator that waits for it BLOCKED until the file has been inspected.

  episode_audit.py --reference outputs/v2/table4/ladder/static_physics_ct.json \\
      --inputs 'outputs/v2_1/plug/ctrl/*/*_s?.json' --out outputs/v2_1/checks/audit/ctrl.json
  episode_audit.py --reference-job v21:cohort:eval:ladder:static_physics_ct --queue outputs/v2_1/jobs_all.json ...

Files without per-episode records (landscapes, realism, segmentation scores, summaries) are listed as skipped.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

STATUS = "exploratory, v2.1 pluggability"


def blocks(d):
    """[(label, contact, episodes)] of a result file (eval_view_ladder / plug_train_sb3 'results' blocks, or a
    trainer JSON's 'final' and 'after_imitation' blocks)."""
    out = []
    for i, r in enumerate(d.get("results") or []):
        if isinstance(r, dict) and r.get("episodes"):
            out.append((f"results[{i}]:{r.get('weights')}", r.get("contact", "on"), r["episodes"],
                        r.get("eval_randomize", "off"), r.get("target_seed_offset")))
    for k in ("final", "after_imitation"):
        b = d.get(k)
        if isinstance(b, dict) and b.get("episodes"):
            out.append((k, b.get("contact", "on"), b["episodes"], b.get("eval_randomize", "off"), None))
    return out


def fingerprint(eps):
    return [(int(e["ep"]), str(e["patient"]), str(e.get("target_requested") or e["target"]), e.get("window"),
             round(float(e["start_pos"]), 6), round(float(e["start_ang"]), 6)) for e in sorted(eps, key=lambda x: x["ep"])]


def digest(fp):
    return hashlib.sha256(json.dumps(fp).encode()).hexdigest()


def render_seed(d):
    p = d.get("provenance") or {}
    for path in (("condition", "render_seed"), ("provenance", "env", "render_seed")):
        x = d
        for k in path:
            x = x.get(k) if isinstance(x, dict) else None
        if x is not None:
            return x
    if "calib_seed" in d:
        return d["calib_seed"]
    return (d.get("config") or {}).get("seed", p.get("calib_seed"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--reference-job", default=None, help="take the reference file from this job's out in --queue")
    ap.add_argument("--queue", default=None)
    ap.add_argument("--inputs", nargs="+", required=True, help="files or glob patterns")
    ap.add_argument("--horizons", type=int, nargs="+", default=[12])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    ref = a.reference
    if a.reference_job:
        q = {j["name"]: j for j in json.load(open(a.queue))}
        ref = q[a.reference_job]["out"]
    R = json.load(open(ref))
    refs = {}
    for lab, contact, eps, er, _ in blocks(R):
        refs.setdefault(contact, fingerprint(eps))
    if not refs:
        raise SystemExit(f"{ref}: no per-episode block")
    files = []
    for pat in a.inputs:
        files += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    rows, skipped, refused = [], [], []
    for f in sorted(set(files)):
        if not f.endswith(".json") or not os.path.exists(f):
            continue
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            skipped.append({"path": f, "why": "unreadable"})
            continue
        bl = blocks(d)
        if not bl:
            skipped.append({"path": f, "why": "no per-episode records"})
            continue
        for lab, contact, eps, er, tso in bl:
            want = refs.get(contact) or refs.get("on")
            fp = fingerprint(eps)
            n_bad = sum(1 for x, y in zip(fp, want) if x != y) + abs(len(fp) - len(want))
            hz = sorted({len(e["pos_traj"]) - 1 for e in eps if e.get("pos_traj")})
            ok_h = all(h in a.horizons for h in hz)
            ok = n_bad == 0 and ok_h and er == "off"
            rec = {"path": os.path.abspath(f), "block": lab, "contact": contact, "n_episodes": len(fp),
                   "episode_set_sha256": digest(fp), "reference_sha256": digest(want), "n_mismatched": n_bad,
                   "horizons": hz, "eval_randomize": er, "render_seed": render_seed(d), "target_seed_offset": tso,
                   "passed": ok}
            rows.append(rec)
            if not ok:
                refused.append(rec)
    body = {"check": "identical-episode audit", "analysis_status": STATUS, "reference": os.path.abspath(ref),
            "reference_sha256": {c: digest(v) for c, v in refs.items()}, "horizons_allowed": a.horizons,
            "n_files": len({r["path"] for r in rows}), "n_blocks": len(rows), "refused": refused, "skipped": skipped,
            "passed": not refused, "blocks": rows,
            "provenance": {"analysis_status": STATUS, "argv": list(sys.argv),
                           "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    out = a.out if body["passed"] else (a.out[:-5] if a.out.endswith(".json") else a.out) + ".failed.json"
    if os.path.exists(out):
        out = out[:-5] + f".{int(time.time())}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(body, fh, indent=1)
    print(f"[audit] {body['n_files']} files, {body['n_blocks']} blocks, {len(refused)} refused, {len(skipped)} skipped "
          f"-> {out}")
    return 0 if body["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
