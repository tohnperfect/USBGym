#!/usr/bin/env python3
"""Held-out leakage and private-data gate over both repos (exits 1 on any gate finding).

Gate checks (all use the merged usbg.provenance; nothing here re-implements the schema):
  * sidecars: every ``*.provenance.json`` under the roots (default: this repo's data/dataset and
    outputs, plus the segmenter repo's outputs and data/nnunet) is validated and its lineage resolved
    through ``parents``: LEAK (a held-out patient, usbg.benchmark.TEST_PATIENTS, in the transitive
    train/val cases), PRIVATE (a private pool or a known private-pool descendant), STALE (sidecar md5
    != file; skipped with --no-md5), SCHEMA (malformed record or a parent without a sidecar);
  * v2 datasets: every ``*.npz`` under data/dataset/v2 whose meta lists ``cases`` (or whose
    ``sessions`` array holds case ids) must hold no held-out patient;
  * manifest: no non-retired training stream of data/manifest.json lists a held-out patient or a
    private pool;
  * optional: --expect-seat flags sidecars of another seat mode; --results checks that the given
    result JSONs carry one provenance.seat_mode (usbg.provenance.assert_one_seat).

Smoke-test and recipe scratch trees (path components ``_smoke`` and ``_recipes``) are excluded by
default (--include-scratch keeps them). --legacy additionally prints the pre-provenance ledger
(manifest streams, segmenter nnU-Net datasets and training histories, live private descendants);
legacy findings are informational and never change the exit code.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/check_leakage.py                          # leakage gate
  PYTHONPATH=src .venv/bin/python scripts/check_leakage.py --expect-seat=anatomical --json outputs/v2/leakage_gate.json
  PYTHONPATH=src .venv/bin/python scripts/check_leakage.py --legacy --json outputs/v2/leakage_legacy.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_REPO = os.environ.get("US_SEG_REPO", os.environ.get("USBG_SEG_REPO", os.path.join(ROOT, "segtrain")))
SCRATCH_DIRS = ("_smoke", "_recipes")


def default_roots():
    return [os.path.join(ROOT, "data", "dataset"), os.path.join(ROOT, "outputs"),
            os.path.join(SEG_REPO, "outputs"), os.path.join(SEG_REPO, "data", "nnunet"),
            os.path.join(SEG_REPO, "data", "nnunet_v2")]


SKIP_DIRS = (".git", ".venv", "__pycache__", "node_modules")


def walk_files(root, suffix, exclude=SCRATCH_DIRS):
    """Files ending in `suffix` under root, found by an os.walk that never follows symlinked
    directories and never enters scratch trees (`exclude`), private pools or VCS / venv dirs. (A plain
    recursive glob follows the _recipes overlay symlinks back into data/, including the private pool.)"""
    from usbg.provenance import PRIVATE_DIRNAMES
    skip = set(exclude) | set(PRIVATE_DIRNAMES) | set(SKIP_DIRS)
    out = []
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        out += [os.path.join(dirpath, f) for f in sorted(files) if f.endswith(suffix)]
    return out


def gate(roots, check_md5=True, exclude=SCRATCH_DIRS, repo=ROOT, expect_seat=None):
    """usbg.provenance.scan over the existing roots with pruned discovery (no scratch trees, private
    pools or symlinked dirs). A kept artifact whose parent sidecar sits in a scratch tree is reported
    (SCHEMA: parent without a sidecar), because a reported artifact must not descend from a smoke run.
    Returns {roots, records, findings, transitive}."""
    from usbg import provenance as P
    roots = [os.path.expanduser(r) for r in roots if os.path.isdir(os.path.expanduser(r))]
    res = P.scan(roots, repo=repo, check_md5=check_md5, exclude=tuple(exclude))
    findings = list(res["findings"])
    if expect_seat:
        findings += [f"SEATMIX {p}: seat_mode {r.get('seat_mode')!r} != expected {expect_seat!r}"
                     for p, r in sorted(res["records"].items()) if r.get("seat_mode") != expect_seat]
    return {"roots": roots, "records": res["records"], "findings": sorted(set(findings)),
            "transitive": res["transitive"]}


def npz_case_findings(root, test_patients=None, exclude=SCRATCH_DIRS):
    """LEAK findings for *.npz under root whose meta 'cases' or 'sessions' hold a held-out patient.
    Returns (findings, n_checked, unreadable)."""
    import numpy as np
    if test_patients is None:
        from usbg.benchmark import TEST_PATIENTS as test_patients
    test = set(test_patients)
    out, n, bad = [], 0, []
    for f in walk_files(root, ".npz", exclude):
        try:
            with np.load(f, allow_pickle=False) as d:
                meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
                sess = d["sessions"] if "sessions" in d.files else None
        except Exception as e:                       # half-written or not a pairs npz: report, do not fail
            bad.append(f"{f}: {type(e).__name__}")
            continue
        cases = set(map(str, meta.get("cases") or []))
        if sess is not None and sess.dtype.kind in "US":
            cases |= set(map(str, np.unique(sess)))
        if not cases:
            continue
        n += 1
        hit = sorted(cases & test)
        if hit:
            out.append(f"LEAK {f}: held-out {hit} in the npz meta cases/sessions")
    return out, n, bad


def manifest_findings(path=os.path.join(ROOT, "data", "manifest.json")):
    from usbg import provenance as P
    return P.legacy_manifest_findings(path) if os.path.exists(path) else []


def legacy_ledger():
    """Pre-provenance findings (informational): manifest streams including retired ones, segmenter
    nnU-Net datasets and histories, live private-pool descendants."""
    from usbg import provenance as P
    from usbg.benchmark import TEST_PATIENTS
    legacy = []
    man = os.path.join(ROOT, "data", "manifest.json")
    if os.path.exists(man):
        with open(man) as fh:
            m = json.load(fh)
        test = set(TEST_PATIENTS)
        for s in m.get("streams", []):
            hit = sorted(set(s.get("cases") or []) & test)
            tag = " [retired]" if s.get("retired") else ""
            if hit:
                legacy.append(f"LEAK manifest stream {s.get('name')}{tag}: held-out {hit}")
            if P.is_private_path(s.get("file") or s.get("path") or ""):
                legacy.append(f"PRIVATE manifest stream {s.get('name')}{tag}: {s.get('file')}")
    for ds in sorted(glob.glob(os.path.join(SEG_REPO, "data", "nnunet", "Dataset*"))):
        if os.path.exists(os.path.join(ds, "domain.json")):
            legacy += P.legacy_nnunet_findings(ds)
    # legacy histories store nnU-Net frame ids; they are mapped through the default build (Dataset011),
    # because a history does not record which dataset it was trained on
    ds011 = os.path.join(SEG_REPO, "data", "nnunet", "Dataset011_USOrgansSimReal")
    if os.path.exists(os.path.join(ds011, "domain.json")):
        for h in sorted(glob.glob(os.path.join(SEG_REPO, "outputs", "us_seg*_history.json"))):
            legacy += [x + " [dataset assumed Dataset011]" for x in P.legacy_nnunet_findings(ds011, h)]
    for md5, (orig, why) in sorted(P.PRIVATE_DESCENDANT_MD5.items(), key=lambda kv: kv[1][0]):
        p = os.path.join(ROOT, orig)
        if os.path.isfile(p) and P.md5_file(p) == md5:
            legacy.append(f"PRIVATE descendant still on disk (allowed; never referenced by code): {orig} ({why})")
    return legacy


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roots", nargs="+", default=None, help="sidecar roots (default: both repos)")
    ap.add_argument("--v2-data", default=os.path.join(ROOT, "data", "dataset", "v2"),
                    help="tree of v2 npz datasets whose meta cases are checked")
    ap.add_argument("--no-md5", action="store_true", help="skip the STALE md5 re-hash (fast)")
    ap.add_argument("--include-scratch", action="store_true", help="also scan _smoke/_recipes trees")
    ap.add_argument("--expect-seat", choices=["anatomical", "canonical"], default=None)
    ap.add_argument("--results", nargs="*", default=[],
                    help="result JSONs that must share one provenance.seat_mode (one table)")
    ap.add_argument("--legacy", action="store_true", help="also print the pre-provenance ledger")
    ap.add_argument("--json", default=None, help="write the report here")
    args = ap.parse_args(argv)

    from usbg import provenance as P
    from usbg.benchmark import TEST_PATIENTS
    exclude = () if args.include_scratch else SCRATCH_DIRS
    res = gate(args.roots or default_roots(), check_md5=not args.no_md5, exclude=exclude,
               expect_seat=args.expect_seat)
    findings = list(res["findings"])
    npz_f, n_npz, unreadable = ([], 0, [])
    if os.path.isdir(args.v2_data):
        npz_f, n_npz, unreadable = npz_case_findings(args.v2_data, exclude=exclude)
    findings += npz_f
    findings += manifest_findings()
    seat = None
    if args.results:
        try:
            seat = P.assert_one_seat(args.results)
        except AssertionError as e:
            findings.append(f"SEATMIX results: {e}")
    legacy = legacy_ledger() if args.legacy else []

    print(f"held-out: {list(TEST_PATIENTS)}")
    print(f"sidecars scanned: {len(res['records'])} under {res['roots']}"
          + ("" if args.include_scratch else f" (excluding {list(SCRATCH_DIRS)})"))
    print(f"v2 npz with case meta: {n_npz} under {args.v2_data}"
          + (f"; unreadable (skipped): {len(unreadable)}" if unreadable else ""))
    if args.results:
        print(f"result JSONs: {len(args.results)}; seat_mode: {seat}")
    print(f"gate findings: {len(findings)}")
    for f in findings:
        print("  " + f)
    for u in unreadable:
        print("  [skipped] " + u)
    if args.legacy:
        print(f"legacy (pre-provenance, informational) findings: {len(legacy)}")
        for f in legacy:
            print("  " + f)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"test_patients": list(TEST_PATIENTS), "roots": res["roots"],
                       "excluded": list(exclude), "n_records": len(res["records"]),
                       "n_v2_npz": n_npz, "unreadable": unreadable, "results_seat_mode": seat,
                       "findings": findings, "legacy": legacy,
                       "transitive_train_cases": res["transitive"],
                       "provenance": P.collect()}, fh, indent=2)
        print(f"wrote {args.json}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
