#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the article's and the supplement's tables and figures from the public results archive, and compare the
printed text with the expected files.

  PYTHONPATH=src python <this folder>/reproduce.py --results ROOT --out DIR

(see README.md beside this script for the archive layout and requirements-reproduce.txt for the environment)

ROOT is the unpacked results archive: ROOT/outputs/..., ROOT/data/..., ROOT/segmenter/outputs/..., ROOT/MANIFEST.tsv
(path, bytes, sha256 of every result file) and ROOT/EXCLUDED.tsv (the inputs it leaves out). DIR must lie outside ROOT.
First the script reports whether ROOT/MANIFEST.tsv is the archive's own (its sha256 and number of rows are in
public_inputs/results_archive.json; any other manifest, of a result root of one's own, is used as given, the summary
line says so, and the exit code is 3 unless --own-results is given). Then it checks every file that MANIFEST.tsv lists
(bytes and sha256) and stops on the first run of differences. The four steps run in public mode, one process each:

  make_tables.py --public       the tables (DIR/tables) and figures/plug_rows.json
  best_so_far.py table --public  tables/best_so_far.tex
  posthoc_logs.py tables --public  tables/posthoc_*.tex
  make_figures.py --public      the figures and their data files (DIR/figures)

Every step reads the result files under ROOT only (checked against MANIFEST.tsv, results_world.py) and the committed
public_inputs/plan_constants.json; it writes under DIR only. Then each file of public_inputs/expected/ (the printed text
of every table the article and the supplement input, with LaTeX comments removed, and the figure data files) is
compared byte for byte with DIR: one PASS or FAIL line per file. A file whose step needs an input that the archive
leaves out (ROOT/EXCLUDED.tsv) and that was not placed at its path under ROOT is not built; the step records it in
DIR/NEEDS.tsv and its line reads "NEEDS <input> (EXCLUDED.tsv)" instead, unless the file exists in DIR (written by a
later run of its step), which is then compared like any other. A file placed there is checked against its
EXCLUDED.tsv row (bytes and sha256) before the step reads it. Exit codes: 0 when every other file passes and ROOT
holds the archive's MANIFEST.tsv (or --own-results is given); 1 when a step stops, a listed file differs from its
MANIFEST.tsv row or a file fails; 3 when every file passes but MANIFEST.tsv is not the archive's.

Printed numbers depend on the numerical stack (bootstraps, quantiles, renders): the expected files were made with
Python 3.13 with NumPy 2.1.3 and SciPy 1.18.0, PyTorch on the CPU; the versions of this run are printed first. The steps
run with every USBG_* and US_SEG_* variable of the caller's environment removed (they would change what the build
reads).
--compare-only skips the build and compares an existing DIR.
"""
from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
CODE_ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
import results_world as RW  # noqa: E402
import script_dirs as SD  # noqa: E402

EXPECTED = os.path.join(HERE, "public_inputs", "expected")
ARCHIVE = os.path.join(HERE, "public_inputs", "results_archive.json")


def manifest_status(root):
    """(True, text) when ROOT/MANIFEST.tsv is the results archive's (sha256 and rows of public_inputs/
    results_archive.json), else (False, text)."""
    import hashlib
    import json
    with open(ARCHIVE, encoding="utf-8") as fh:
        want = json.load(fh)
    p = os.path.join(root, "MANIFEST.tsv")
    if not os.path.isfile(p):
        return False, "ROOT/MANIFEST.tsv not found"
    with open(p, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    rows = len(RW.load_manifest(p))
    if h == want["manifest_sha256"] and rows == want["manifest_rows"]:
        return True, f"ROOT/MANIFEST.tsv is the results archive's ({rows} rows, sha256 {h[:12]})"
    return False, (f"ROOT/MANIFEST.tsv is not the results archive's ({rows} rows, sha256 {h[:12]}; the archive's has "
                   f"{want['manifest_rows']} rows, sha256 {want['manifest_sha256'][:12]}): every file is checked "
                   "against this MANIFEST.tsv only")


def verify_listed(root):
    """Check every file that ROOT/MANIFEST.tsv lists against its row (bytes and sha256), before any step runs (the
    steps check the files they open or probe; this covers the rest, for example the inputs of an item that NEEDS an
    input the archive leaves out). Returns (number of files checked, [problems])."""
    import hashlib
    rows = RW.load_manifest(os.path.join(root, "MANIFEST.tsv"))
    problems = []
    for rel, (size, sha) in sorted(rows.items()):
        p = os.path.join(root, *rel.split("/"))
        if not os.path.isfile(p):
            problems.append(f"{rel}: missing")
            continue
        h, n = hashlib.sha256(), 0
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
                n += len(b)
        if n != size or h.hexdigest() != sha:
            problems.append(f"{rel}: {n} bytes, sha256 {h.hexdigest()[:12]}; MANIFEST.tsv says {size} bytes, {sha[:12]}")
    return len(rows), problems


def versions():
    out = {"python": sys.version.split()[0]}
    from importlib import metadata as md
    for pkg in ("numpy", "scipy", "torch", "matplotlib", "mujoco", "gymnasium", "stable-baselines3"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def steps(root, out):
    pub = ["--public", "--results", root, "--out", out]
    paper, analysis = SD.paper_dir(), SD.analysis_dir()
    return [("make_tables", [os.path.join(paper, "make_tables.py"), *pub]),
            ("best_so_far", [os.path.join(analysis, "best_so_far.py"), "table", *pub]),
            ("posthoc_logs", [os.path.join(analysis, "posthoc_logs.py"), "tables", *pub]),
            ("make_figures", [os.path.join(paper, "make_figures.py"), *pub])]


def build(root, out):
    logs = os.path.join(out, "logs")
    os.makedirs(logs, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("USBG_", "US_SEG_"))}
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(CODE_ROOT, "src")] + (
        [env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("MPLCONFIGDIR", os.path.join(out, ".mplconfig"))
    if os.path.isfile(os.path.join(out, RW.NEEDS_NAME)):      # written afresh by the steps below
        os.remove(os.path.join(out, RW.NEEDS_NAME))
    ok = True
    for name, cmd in steps(root, out):
        t0 = time.time()
        with open(os.path.join(logs, name + ".out"), "w") as fo, open(os.path.join(logs, name + ".err"), "w") as fe:
            rc = subprocess.run([sys.executable, *cmd], cwd=CODE_ROOT, env=env, stdout=fo, stderr=fe).returncode
        notes = [ln.strip() for ln in open(os.path.join(logs, name + ".err")) if ln.startswith("[public]")]
        print(f"{name}: exit {rc} ({time.time() - t0:.0f} s); {len(notes)} public-mode notices "
              f"(logs/{name}.err)")
        for n in notes:
            if n.startswith("[public] NEEDS"):
                print("  " + n)
        if rc != 0:
            ok = False
            tail = open(os.path.join(logs, name + ".err")).read().strip().splitlines()[-5:]
            print("  " + "\n  ".join(tail))
            break
    return ok


def compare(out, archive_manifest=True):
    n_pass = n_fail = n_needs = 0
    needs = RW.read_needs(out)
    for sub in ("tables", "figures"):
        exp_dir = os.path.join(EXPECTED, sub)
        names = sorted(os.listdir(exp_dir)) if os.path.isdir(exp_dir) else []
        for name in names:
            p = os.path.join(out, sub, name)
            if f"{sub}/{name}" in needs and not os.path.isfile(p):
                n_needs += 1
                print(f"NEEDS {', '.join(needs[f'{sub}/{name}'])} ({RW.EXCLUDED_NAME}): {sub}/{name}")
                continue
            want = open(os.path.join(exp_dir, name), "rb").read()
            got = open(p, "rb").read() if os.path.isfile(p) else None
            if got == want:
                n_pass += 1
                print(f"PASS {sub}/{name}")
                continue
            n_fail += 1
            if got is None:
                print(f"FAIL {sub}/{name}: not written")
                continue
            d = list(difflib.unified_diff(want.decode("utf-8", "replace").splitlines(),
                                          got.decode("utf-8", "replace").splitlines(), "expected", "built", n=0,
                                          lineterm=""))
            print(f"FAIL {sub}/{name}: " + (" | ".join(d[2:5])[:300] if d else "bytes differ (line ends)"))
        built = sorted(os.listdir(os.path.join(out, sub))) if os.path.isdir(os.path.join(out, sub)) else []
        extra = [b for b in built if b not in names]
        if extra:
            print(f"also written to {sub}/ (not compared: not printed by the paper, or not a data file): "
                  + ", ".join(extra))
    print(f"{n_pass} PASS, {n_fail} FAIL" + (f", {n_needs} NEEDS (inputs left out of the results archive, "
                                             f"{RW.EXCLUDED_NAME})" if n_needs else "")
          + ("" if archive_manifest else " (MANIFEST.tsv is not the results archive's)"))
    return n_fail == 0 and n_pass > 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=False, help="the results archive root (outputs/, data/, segmenter/)")
    ap.add_argument("--out", required=True, help="the output folder (tables/, figures/, logs/)")
    ap.add_argument("--compare-only", action="store_true", help="compare an existing --out with expected/")
    ap.add_argument("--own-results", action="store_true",
                    help="ROOT is a result root of one's own: exit 0 (not 3) when its MANIFEST.tsv is not the archive's")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    print("versions: " + ", ".join(f"{k} {v}" for k, v in versions().items()))
    if not a.compare_only and not a.results:
        ap.error("--results ROOT is needed to build")
    if a.results and SD.inside(out, os.path.abspath(a.results)):
        ap.error(f"--out {out} lies inside --results ROOT: the build writes nothing under ROOT, so DIR must be a "
                 "folder outside ROOT")
    archive = True
    if a.results:
        archive, txt = manifest_status(os.path.abspath(a.results))
        print("manifest: " + txt)
    else:
        print("manifest: not checked (--compare-only without --results)")
    if a.results and os.path.isfile(os.path.join(a.results, "MANIFEST.tsv")):
        n, problems = verify_listed(os.path.abspath(a.results))
        if problems:
            print(f"listed files: {len(problems)} of {n} differ from MANIFEST.tsv: " + "; ".join(problems[:10])
                  + (" ..." if len(problems) > 10 else ""))
            return 1
        print(f"listed files: all {n} match MANIFEST.tsv (bytes and sha256)")
    if not a.compare_only and not build(os.path.abspath(a.results), out):
        return 1
    if not compare(out, archive):
        return 1
    if not archive and not a.own_results:
        print("exit 3: ROOT/MANIFEST.tsv is not the results archive's (pass --own-results for a result root of your "
              "own)")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
