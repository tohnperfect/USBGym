#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): merge the per-slot job specs into one queue file and validate it.

  PYTHONPATH=src .venv/bin/python scripts/plug/merge_plug_jobs.py            # check, then write both files
  PYTHONPATH=src .venv/bin/python scripts/plug/merge_plug_jobs.py --check    # check only, write nothing

Inputs (read only):
  outputs/v2_1/jobs/plug_{texture,seg,seg_ext,ctrl,reward,render}.json   the six slot specs
  outputs/v2_1/jobs_all.json                                            the paused v2.1 queue (511 jobs)
Outputs (new files; the script refuses to overwrite them unless --replace-own):
  outputs/v2_1/jobs_plug.json            the 795 plug jobs, one list
  outputs/v2_1/jobs_all_with_plug.json   jobs_all.json followed by the plug jobs, unchanged order

jobs_all.json is never written. Its entries are copied byte-for-byte in meaning (same dicts, same order).

Why one combined file: scripts/run_table4_v2.py takes ONE --extra-jobs file, and it silently ignores a
dependency whose job is not in that file. plug_seg_ext.json depends on plug_seg.json, and 6 renderer
hillclimb jobs read outputs of the v2.1 queue's s23 pilot. The combined file makes every dependency visible.

Integration change (the only one): a plug job that lists, in its `files`, the `out` of another job of the
combined queue gets that job added to its `deps` (recorded under `integration_deps_added`). The runner would
wait for the file anyway (a job starts only when its files exist), so the run order does not change, but the
dry-run simulator models deps and not files, so the makespan is now honest. This touches the 6 renderer
hillclimb jobs (s23 pilot outputs) and no queued v2.1 job.

Checks (any failure aborts before writing):
  * unique names in the combined queue; every plug name starts with v21:plug:<slot>:
  * every dep resolves to a job of the combined queue; no dependency cycle
  * unique outs; every out ends in .json (the runner writes <out minus .json>.log); no out exists yet; no
    job log path equals another job's out
  * every plug out lies under outputs/v2_1/plug/, data/dataset/v2_1/plug/ or segtrain's
    outputs/v2_1/plug/
  * cwd is the repository root; env pins PYTHONPATH=<repository root>/src, USBG_SEAT=anatomical and USBG_WINDOWS
  * every missing input file is the out of a job, is produced by a dep job, or is a known external file
    (listed in the report: the LPIPS AlexNet weights)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

GYM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEG = os.environ.get("USBG_SEG_REPO", os.path.join(GYM, "segtrain"))
SLOTS = [("plug_texture", "v21:plug:texture:"), ("plug_seg", "v21:plug:seg:"), ("plug_seg_ext", "v21:plug:seg:"),
         ("plug_ctrl", "v21:plug:ctrl:"), ("plug_reward", "v21:plug:reward:"), ("plug_render", "v21:plug:renderer:")]
OUT_OK = (f"{GYM}/outputs/v2_1/plug/", f"{GYM}/data/dataset/v2_1/plug/", f"{SEG}/outputs/v2_1/plug/")
KNOWN_EXTERNAL = {
    os.path.join(os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights")), "torch_home", "hub", "checkpoints", "alexnet-owt-7be5be79.pth"):
        "LPIPS AlexNet trunk (torchvision release file), downloaded with the plug-in weights",
}
WINDOWS = f"{GYM}/data/cache/windows/usbg-windows-v1.json"


def _real(p):
    return os.path.realpath(p)


def _under(p, roots):
    rp = _real(p)
    return any(rp.startswith(_real(r.rstrip("/")) + "/") for r in roots)


def load(jobs_dir, queue_path):
    queued = json.load(open(queue_path))
    plug, src = [], {}
    for stem, prefix in SLOTS:
        for j in json.load(open(os.path.join(jobs_dir, stem + ".json"))):
            plug.append(j)
            src[j["name"]] = (stem, prefix)
    return queued, plug, src


# Files a slot driver writes by naming convention (documented in that driver's docstring), not via a path
# argument: scripts/plug_texture.py "tex <stage> <seed>" and "seg <stage> <seed>".
_TEX_DRIVER = f"{GYM}/scripts/plug_texture.py"
_TEX_RULES = {"tex": f"{GYM}/data/dataset/v2_1/plug/texture/{{stage}}_curvi_s{{seed}}.pt",
              "seg": f"{SEG}/outputs/v2_1/plug/texture/us_seg_{{stage}}_s{{seed}}.pt"}


def _ancestors_dep(job, by):
    seen, stack = set(), list(job.get("deps", []))
    while stack:
        d = stack.pop()
        if d in seen or d not in by:
            continue
        seen.add(d)
        stack.extend(by[d].get("deps", []))
    return [by[d] for d in seen]


def produced_by_dep(fp, job, by):
    """True if a (transitive) dep job writes the file: its argv or env names the file, its directory or a
    parent directory inside the plug roots, the file sits next to the dep's out, or a driver naming rule
    (_TEX_RULES) maps the dep's argv to exactly this path."""
    rfp = _real(fp)
    for dj in _ancestors_dep(job, by):
        vals = list(dj["cmd"]) + [v for v in (dj.get("env") or {}).values() if isinstance(v, str)]
        for c in vals:
            if not c or not os.path.isabs(c):
                continue
            rc = _real(c)
            if rc == rfp:
                return True
            if rfp.startswith(rc.rstrip("/") + "/") and _under(rc + "/x", OUT_OK):
                return True
        if _real(os.path.dirname(dj["out"])) == _real(os.path.dirname(fp)):
            return True
        cmd = dj["cmd"]
        if len(cmd) >= 5 and cmd[1] == _TEX_DRIVER and cmd[2] in _TEX_RULES:
            if _real(_TEX_RULES[cmd[2]].format(stage=cmd[3], seed=cmd[4])) == rfp:
                return True
    return False


def check_and_merge(queued, plug, src):
    errs, notes = [], collections.defaultdict(list)
    allj = queued + plug
    names = collections.Counter(j["name"] for j in allj)
    errs += [f"duplicate name {n} x{c}" for n, c in names.items() if c > 1]
    by = {j["name"]: j for j in allj}
    out_of = {}
    for j in allj:
        r = _real(j["out"])
        if r in out_of:
            errs.append(f"duplicate out {j['out']} ({out_of[r]} and {j['name']})")
        out_of[r] = j["name"]
    for j in plug:
        n = j["name"]
        stem, prefix = src[n]
        for k in ("name", "cmd", "out", "est_h", "env", "deps", "files", "expect", "cwd"):
            if k not in j:
                errs.append(f"{n}: missing key {k}")
        if not n.startswith(prefix):
            errs.append(f"{n}: name prefix is not {prefix} ({stem})")
        if not (isinstance(j["cmd"], list) and all(isinstance(c, str) for c in j["cmd"])):
            errs.append(f"{n}: cmd is not an argv list")
        elif not os.path.isabs(j["cmd"][0]) or not os.path.exists(j["cmd"][0]):
            errs.append(f"{n}: interpreter {j['cmd'][0]} missing")
        if not j["out"].endswith(".json"):
            errs.append(f"{n}: out does not end in .json")
        if not _under(j["out"], OUT_OK):
            errs.append(f"{n}: out outside the plug directories: {j['out']}")
        if os.path.exists(j["out"]):
            errs.append(f"{n}: out already exists: {j['out']}")
        log = _real(j["out"])[:-5] + ".log"
        if log in out_of:
            errs.append(f"{n}: its log path is the out of {out_of[log]}")
        if j.get("cwd") != GYM:
            errs.append(f"{n}: cwd {j.get('cwd')}")
        env = j.get("env") or {}
        if env.get("PYTHONPATH") != f"{GYM}/src":
            errs.append(f"{n}: PYTHONPATH {env.get('PYTHONPATH')}")
        if env.get("USBG_SEAT") != "anatomical":
            errs.append(f"{n}: USBG_SEAT {env.get('USBG_SEAT')}")
        if env.get("USBG_WINDOWS", None) not in ("", WINDOWS):
            errs.append(f"{n}: USBG_WINDOWS not pinned ({env.get('USBG_WINDOWS')})")
        if not isinstance(j["est_h"], (int, float)) or j["est_h"] <= 0:
            errs.append(f"{n}: est_h {j['est_h']}")
        for d in j.get("deps", []):
            if d not in by:
                errs.append(f"{n}: dep {d} is not a job of the combined queue")
        added = []
        for fp in j.get("files", []):
            if os.path.exists(fp):
                continue
            r = _real(fp)
            if r in out_of:
                prod = out_of[r]
                if prod not in j["deps"]:
                    added.append(prod)
                notes["file is the out of a job"].append(f"{n} <- {prod}")
            elif produced_by_dep(fp, j, by):
                notes["file produced by a dep job"].append(f"{n}: {fp}")
            elif fp in KNOWN_EXTERNAL:
                notes["known external file (BLOCKED until placed)"].append(f"{n}: {fp}")
            else:
                errs.append(f"{n}: input file has no producer: {fp}")
        if added:
            j["deps"] = list(j["deps"]) + sorted(set(added))
            j["integration_deps_added"] = sorted(set(added))
            notes["deps added from file producers"].append(f"{n} + {sorted(set(added))}")
    state = {}

    def dfs(n, path):
        if state.get(n) == 1:
            errs.append("cycle " + " -> ".join(path + [n]))
            return
        if state.get(n) == 2:
            return
        state[n] = 1
        for d in by[n].get("deps", []):
            if d in by:
                dfs(d, path + [n])
        state[n] = 2

    sys.setrecursionlimit(10000)
    for n in by:
        dfs(n, [])
    return errs, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", default=f"{GYM}/outputs/v2_1/jobs")
    ap.add_argument("--queue", default=f"{GYM}/outputs/v2_1/jobs_all.json")
    ap.add_argument("--out-plug", default=f"{GYM}/outputs/v2_1/jobs_plug.json")
    ap.add_argument("--out-all", default=f"{GYM}/outputs/v2_1/jobs_all_with_plug.json")
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    ap.add_argument("--replace-own", action="store_true", help="allow replacing this script's two output files")
    a = ap.parse_args(argv)
    for p in (a.out_plug, a.out_all):
        if _real(p) == _real(a.queue):
            sys.exit("refusing: an output path equals the paused queue file")
    queued, plug, src = load(a.jobs_dir, a.queue)
    errs, notes = check_and_merge(queued, plug, src)
    per = collections.OrderedDict()
    for j in plug:
        stem = src[j["name"]][0]
        c, h = per.get(stem, (0, 0.0))
        per[stem] = (c + 1, h + j["est_h"])
    print("plug specs:", {k: (c, round(h, 2)) for k, (c, h) in per.items()},
          f"total {len(plug)} jobs, {sum(j['est_h'] for j in plug):.2f} job-h")
    print(f"paused queue: {len(queued)} entries ({len([j for j in queued if re.search(r'^v21:(?!cohort:cpu:)', j['name'])])}"
          " under the resume filter)")
    for k, v in notes.items():
        print(f"{k}: {len(v)}")
        for m in v[:4]:
            print("   ", m)
    if errs:
        print(f"\nFAILED: {len(errs)} problem(s)")
        for e in errs[:60]:
            print("  ", e)
        return 1
    print("\nall checks passed")
    if a.check:
        return 0
    for p in (a.out_plug, a.out_all):
        if os.path.exists(p) and not a.replace_own:
            sys.exit(f"refusing to overwrite {p} (use --replace-own)")
    with open(a.out_plug, "w") as fh:
        json.dump(plug, fh, indent=1)
    with open(a.out_all, "w") as fh:
        json.dump(queued + plug, fh, indent=1)
    print(f"wrote {a.out_plug} ({len(plug)} jobs) and {a.out_all} ({len(queued) + len(plug)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
