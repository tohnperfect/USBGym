#!/usr/bin/env python3
"""v2 oracle-ladder scheduler: replaces scripts/run_view_policy_heldout.sh and run_view_policy_seg.sh.

What the old runners got wrong: OUT hardcoded; an existing output was skipped SILENTLY even if it came
from another seat/segmenter; jobs packed per seed (26.3 h for 52.4 job-h); USBG_LEARNED_SEG defaulted to
1 with no US_SEG_CKPT (so every learned-mask row used the withdrawn shared checkpoint 22a4972c); every
run also did the legacy in-run --real-eval.

This runner:
  * OUT is required; every output lives under it (job logs next to their outputs);
  * one dependency-ordered queue across ALL seeds and arms; a job starts when its job deps are
    DONE/SKIPPED and its file deps exist; among ready jobs the LONGEST estimate starts first;
  * at most K (default 3) jobs at once; the whole run sits under `caffeinate -i -s`;
  * skips are LOUD and checked: an existing output is reused only if its provenance matches the
    expected seat / texture md5 / segmenter md5 / windows md5, else it is STALE and not overwritten
    (--rerun-stale renames it to <out>.stale.<time> and recomputes);
  * missing upstream files (texture stages, segmenters, the Vitale holdout) leave their jobs BLOCKED
    (--wait-files-h keeps polling for them, re-reading both manifests, so the runner can start before
    the texture and segmenter runs finish); the exit code is non-zero if anything FAILED/BLOCKED/STALE;
  * --dry-run prints the queue, every exact command and a simulated K-slot makespan, and writes nothing.

Phases (all under USBG_SEAT=--seat, default anatomical; targets liver kidney):
  ctmask       12 trainer runs, 4 stages x 3 seeds, USBG_LEARNED_SEG=0 (US_SEG_CKPT removed)      -> OUT/ctmask/
  matched      12 trainer runs, USBG_LEARNED_SEG=1, US_SEG_CKPT = seg manifest[<stage>_s<seed>]   -> OUT/matched/
  dr           3 SonoSPADE CT-mask runs with --randomize episode, + 6 evals (dr_on and ctmask,
               weights imit/final, --eval-randomize off episode)                                -> OUT/dr_on/, OUT/dr_eval/
  ladder       pose_oracle / static / scripted per stage (CT mask), ibvs_base per stage x seed under
               the CT mask and under the matched segmenter, each with contact on and off          -> OUT/ladder/
  contact      every ctmask/matched policy, weights imit/final, contact on/off                     -> OUT/contact/
  registration every matched policy against the Vitale group holdout + AUL + KidneyUS subsets      -> OUT/registration/

  PYTHONPATH=src .venv/bin/python scripts/run_table4_v2.py --out outputs/v2/table4 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO = os.environ.get("USBG_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from usbg.benchmark import TRAIN_PATIENTS, TEST_PATIENTS  # noqa: E402

PY = os.path.join(REPO, ".venv/bin/python")
SEG_REPO = os.environ.get("USBG_SEG_REPO", os.path.join(REPO, "segtrain"))
STAGES = ["sonospade", "cut", "cyclegan", "physics"]          # longest first
# Per-run hours at K=3 on this laptop (MPS), used only to order the queue and estimate the makespan.
# Learned-mask arm: the v1 policy_heldout wall_sec (2.49/2.14/2.10/2.00 h). CT-mask arm: x0.85, the
# measured SonoSPADE CT/learned ratio. A 2026-10-13 K=3 timing probe of the v2 trainer (anatomical seat,
# 16-episode RL iters) projects 0.7 h (physics, CT mask) to 1.6 h (SonoSPADE, learned mask), so these are
# conservative; replace them with the dry-run wall_sec.
EST_TRAIN_H = {"sonospade": 2.5, "cut": 2.15, "cyclegan": 2.1, "physics": 2.0}
EST_CT_FACTOR = 0.85
EST_EVAL_H = {"pose_oracle": 0.03, "static": 0.03, "scripted": 0.03, "ibvs_base": 0.08,
              "policy": 0.12, "dr": 0.12, "registration": 0.3}
TRAIN_FLAGS = ["--fix-residual", "--residual-image", "--ibvs-features", "rich", "--w-seg", "0.6",
               "--w-app", "0.4", "--imitation", "4000", "--iters", "300", "--episodes-per-iter", "16",
               "--n-steps", "12", "--eval-episodes", "60", "--eval-seed", "999", "--eval-design", "crossed"]
CT_ENV = {"USBG_LEARNED_SEG": "0", "US_SEG_CKPT": ""}           # "" = remove from the job environment


def windows_path():
    return os.environ.get("USBG_WINDOWS", os.path.join(REPO, "data", "cache", "windows", "usbg-windows-v1.json"))


def md5(p):
    if not p or not os.path.exists(p):
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Job:
    def __init__(self, name, cmd, out, est_h, env=None, deps=(), files=(), expect=None, cwd=REPO):
        self.name, self.cmd, self.out, self.est_h = name, cmd, out, est_h
        self.env, self.deps, self.files, self.expect, self.cwd = env or {}, list(deps), list(files), expect or {}, cwd
        self.state, self.proc, self.t0, self.note = "PENDING", None, None, ""
        self.release_h = 0.0          # dry-run only: hours until an external file dep exists
        self.priority = 0             # higher starts first among ready jobs (then longest first)


def _abs(p):
    return p if (p is None or os.path.isabs(p)) else os.path.join(REPO, p)


def build_jobs(a):
    O = os.path.abspath(a.out)
    tex = json.load(open(a.tex_manifest)) if a.tex_manifest and os.path.exists(a.tex_manifest) else {}
    seg = json.load(open(a.seg_manifest)) if a.seg_manifest and os.path.exists(a.seg_manifest) else {}
    seat = {"USBG_SEAT": a.seat}
    win = windows_path()
    win_md5 = md5(win) if a.seat == "anatomical" else None
    cohort = ["--train-cases", *a.train_cases, "--eval-cases", *a.eval_cases, "--targets", *a.targets,
              "--min-windows", str(a.min_windows)]
    seat_files = [win] if a.seat == "anatomical" else []
    jobs = []

    def tex_ck(t):
        if t == "physics":
            return "none"
        p = tex.get(t)
        return _abs(p) if p else f"<MISSING texture {t} in {a.tex_manifest}>"

    def seg_ck(t, s):
        p = seg.get(f"{t}_s{s}")
        return _abs(p) if p else f"<MISSING segmenter {t}_s{s} in {a.seg_manifest}>"

    def tex_files(t):
        return [] if t == "physics" else [tex_ck(t)]

    def expect(t, seg_path=None, ct=False):
        e = {"seat": a.seat, "texture_md5": None if t == "physics" else md5(tex_ck(t)), "windows_md5": win_md5}
        if ct:
            e["seg_md5"] = None
        elif seg_path:
            e["seg_md5"] = md5(seg_path)
        return e

    def trainer(arm, t, s, extra=(), env=None, files=(), ct=True):
        out = f"{O}/{arm}/{t}_deploy_s{s}.json"
        cmd = [PY, "scripts/train_view_match_rl.py", "--translator", t, "--texture-ckpt", tex_ck(t),
               *TRAIN_FLAGS, *cohort, "--contact", "on", "--seed", str(s), "--ablation", arm, *extra, "--out", out]
        sk = (env or {}).get("US_SEG_CKPT") or None
        est = EST_TRAIN_H[t] * (EST_CT_FACTOR if ct else 1.0)
        return Job(f"train:{arm}:{t}:s{s}", cmd, out, est, {**seat, **(env or {})},
                   files=[*tex_files(t), *files, *seat_files], expect=expect(t, sk, ct=ct))

    for s in a.seeds:
        for t in STAGES:
            if "ctmask" in a.phases:                                    # CT-mask image-servo arm
                jobs.append(trainer("ctmask", t, s, env=CT_ENV))
            if "matched" in a.phases:                                   # matched learned-mask arm
                sk = seg_ck(t, s)
                jobs.append(trainer("matched", t, s, env={"USBG_LEARNED_SEG": "1", "US_SEG_CKPT": sk},
                                    files=[sk], ct=False))
        if "dr" in a.phases:                                            # SonoSPADE, CT mask, DR on
            jobs.append(trainer("dr_on", "sonospade", s, extra=["--randomize", "episode"], env=CT_ENV))

    def ladder(rung, t, mask, s=None):
        env = dict(seat)
        sk = None
        if mask == "ct":
            env.update(CT_ENV)
        else:
            sk = seg_ck(t, s)
            env.update({"USBG_LEARNED_SEG": "1", "US_SEG_CKPT": sk})
        tag = f"{rung}_{t}_{mask}" + (f"_s{s}" if s is not None else "")
        out = f"{O}/ladder/{tag}.json"
        cmd = [PY, "scripts/eval_view_ladder.py", "--rung", rung, "--translator", t, "--texture-ckpt", tex_ck(t),
               *cohort, "--contact", "on", "off", "--eval-episodes", "60", "--eval-seed", "999",
               "--eval-design", "crossed"]
        if s is not None:
            cmd += ["--calib-seed", str(s)]
        cmd += ["--out", out]
        files = tex_files(t) + ([sk] if sk else []) + seat_files
        return Job(f"eval:ladder:{tag}", cmd, out, EST_EVAL_H[rung], env, files=files,
                   expect=expect(t, sk, ct=(mask == "ct")))

    if "ladder" in a.phases:                                            # rungs
        for t in STAGES:
            for rung in ("pose_oracle", "static", "scripted"):
                jobs.append(ladder(rung, t, "ct"))
            for s in a.seeds:
                jobs.append(ladder("ibvs_base", t, "ct", s))
                jobs.append(ladder("ibvs_base", t, "matched", s))

    def policy_eval(kind, arm, t, s, extra, out):
        pj = f"{O}/{arm}/{t}_deploy_s{s}.json"
        sk = None if arm in ("ctmask", "dr_on") else seg_ck(t, s)
        env = dict(seat); env.update(CT_ENV if sk is None else {"USBG_LEARNED_SEG": "1", "US_SEG_CKPT": sk})
        cmd = [PY, "scripts/eval_view_ladder.py", "--rung", "policy", "--policy-json", pj,
               "--weights", "imit", "final", *extra, "--out", out]
        return Job(f"eval:{kind}:{arm}:{t}:s{s}", cmd, out, EST_EVAL_H["policy" if kind == "contact" else "dr"],
                   env, deps=[f"train:{arm}:{t}:s{s}"], files=[pj] + ([sk] if sk else []),
                   expect=expect(t, sk, ct=sk is None))

    if "contact" in a.phases:                                           # imit vs final for every cell
        for arm in ("ctmask", "matched"):                              # needs those runs' outputs (or their jobs)
            for t in STAGES:
                for s in a.seeds:
                    jobs.append(policy_eval("contact", arm, t, s, ["--contact", "on", "off"],
                                            f"{O}/contact/{arm}_{t}_s{s}.json"))
    if "dr" in a.phases:                                                # both arms, default render and DR draws
        for arm in ("dr_on", "ctmask"):
            for s in a.seeds:
                jobs.append(policy_eval("dr", arm, "sonospade", s, ["--eval-randomize", "off", "episode"],
                                        f"{O}/dr_eval/{arm}_sonospade_s{s}.json"))
    if "registration" in a.phases:                                      # D12
        for t in STAGES:
            for s in a.seeds:
                pj = f"{O}/matched/{t}_deploy_s{s}.json"
                out = f"{O}/registration/{t}_s{s}.json"
                sk = seg_ck(t, s)
                cmd = [PY, "scripts/eval_real_registration.py", "--policy-json", pj,
                       "--vitale-npz", a.vitale_npz, "--aul-npz", a.aul_npz, "--aul-n", str(a.aul_n),
                       "--kidneyus-npz", a.kidneyus_npz, "--kidneyus-n", str(a.kidneyus_n),
                       "--organs", *[o for o in a.targets if o in ("liver", "kidney")],
                       "--ct-cases", *a.eval_cases, "--cts-per-target", str(a.cts_per_target), "--out", out]
                jobs.append(Job(f"eval:registration:{t}:s{s}", cmd, out, EST_EVAL_H["registration"],
                                {**seat, "USBG_LEARNED_SEG": "1", "US_SEG_CKPT": sk},
                                deps=[f"train:matched:{t}:s{s}"], files=[pj, a.vitale_npz, sk],
                                expect=expect(t, sk)))
    if not a.no_early_ladder:                                          # cheap CT-mask floor cells first: an
        for j in jobs:                                                 # early warning (seat, clamp, force) in ~1 h
            if j.name.startswith("eval:ladder:") and j.name.endswith(("_ct", "_ct_s0", "_ct_s1", "_ct_s2")):
                j.priority = 1
    seg_paths = {seg_ck(t, s) for t in STAGES for s in a.seeds}
    for j in jobs:                                                     # model the segmenter lead time in the dry run
        if any(f in seg_paths or "MISSING segmenter" in f for f in j.files):
            j.release_h = a.seg_ready_h
    if a.extra_jobs:                                                    # upstream texture and segmenter commands, same K pool
        for e in json.load(open(a.extra_jobs)):
            j = Job(e["name"], e["cmd"], e["out"], e["est_h"], e.get("env"), e.get("deps", ()),
                    e.get("files", ()), e.get("expect"), e.get("cwd", REPO))
            j.priority = int(e.get("priority", 0))                     # opt-in: no field = 0 = previous order
            jobs.append(j)
    if a.only:
        jobs = [j for j in jobs if re.search(a.only, j.name)]
    return jobs


def refresh_pending(jobs, a, log):
    """Re-read the manifests for PENDING jobs whose texture / segmenter path was still a <MISSING ...>
    placeholder (the texture and segmenter runs finished after the runner started): their cmd, env, files and expectation
    are rebuilt from the fresh manifests."""
    stale = [j for j in jobs if j.state == "PENDING" and any(f.startswith("<MISSING") for f in j.files)]
    if not stale:
        return
    try:
        fresh = {j.name: j for j in build_jobs(a)}
    except (OSError, ValueError) as e:                 # a manifest mid-write: try again next poll
        log(f"refresh deferred: {e!r}")
        return
    for j in stale:
        n = fresh.get(j.name)
        if n is not None and not any(f.startswith("<MISSING") for f in n.files):
            j.cmd, j.env, j.files, j.expect = n.cmd, n.env, n.files, n.expect
            log(f"RESOLVED {j.name}: manifest entries now present")


def check_existing(j):
    """-> (ok, reason). An existing output is reused only if its provenance matches the expectation."""
    try:
        d = json.load(open(j.out))
    except Exception as e:
        return False, f"unreadable ({e!r})"
    p = d.get("provenance") or (d.get("results") or [{}])[0].get("provenance") or {}
    if not p:
        return False, "no provenance block (pre-v2 output)"
    got = {"seat": (p.get("seat") or {}).get("resolved") or p.get("seat_mode"),
           "texture_md5": (p.get("texture") or {}).get("md5"),
           "seg_md5": (p.get("mask") or {}).get("seg_md5") or p.get("seg_md5"),
           "windows_md5": (p.get("windows") or {}).get("md5")}
    bad = [f"{k}: expected {v} got {got.get(k)}" for k, v in j.expect.items() if got.get(k) != v]
    return (not bad), ("; ".join(bad) or "provenance matches")


def simulate(jobs, k):
    """List-scheduling estimate of the makespan with K slots, longest-ready-first."""
    by = {j.name: j for j in jobs}
    done_t, running, t, order = {}, [], 0.0, []
    pending = set(by)
    while pending or running:
        ready = [n for n in pending if all(d in done_t or d not in by for d in by[n].deps)
                 and by[n].release_h <= t + 1e-9]
        ready.sort(key=lambda n: (-by[n].priority, -by[n].est_h, n))
        while ready and len(running) < k:
            n = ready.pop(0); pending.discard(n)
            heapq.heappush(running, (t + by[n].est_h, n)); order.append((round(t, 2), n))
        if not running:
            rel = [by[n].release_h for n in pending if by[n].release_h > t]
            if not rel:
                break
            t = min(rel); continue
        t, n = heapq.heappop(running); done_t[n] = t
    return t, order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--phases", nargs="+", default=["ctmask", "matched", "dr", "ladder", "contact", "registration"],
                    choices=["ctmask", "matched", "dr", "ladder", "contact", "registration"])
    ap.add_argument("--seat", default="anatomical", choices=["anatomical", "canonical"])
    ap.add_argument("--train-cases", nargs="+", default=list(TRAIN_PATIENTS))
    ap.add_argument("--eval-cases", nargs="+", default=list(TEST_PATIENTS))
    ap.add_argument("--targets", nargs="+", default=["liver", "kidney"])
    ap.add_argument("--min-windows", type=int, default=2,
                    help="per (case, target) cell; 2 skips exactly s0058/s0522/s0727 kidney on usbg-windows-v1 "
                         "(s0058 and s0522 have ONE left-kidney window, s0727 none)")
    ap.add_argument("--tex-manifest", default=os.path.join(REPO, "outputs/v2/textures/manifest.json"),
                    help='textures: {"physics": null, "cut": path, "cyclegan": path, "sonospade": path}')
    ap.add_argument("--seg-manifest", default=os.path.join(SEG_REPO, "outputs/v2/seg_manifest.json"),
                    help='segmenters: {"sonospade_s0": abs path, ..., "physics_s2": abs path}')
    ap.add_argument("--vitale-npz", default=os.path.join(SEG_REPO, "data/real/vitale_holdout_v2.npz"))
    ap.add_argument("--aul-npz", default=os.path.join(SEG_REPO, "data/real/aul_liver_test.npz"))
    ap.add_argument("--aul-n", type=int, default=60)
    ap.add_argument("--kidneyus-npz", default=os.path.join(SEG_REPO, "data/real/kidneyus_holdout_test.npz"))
    ap.add_argument("--kidneyus-n", type=int, default=60)
    ap.add_argument("--cts-per-target", type=int, default=3)
    ap.add_argument("--no-early-ladder", action="store_true",
                    help="do not start the cheap CT-mask ladder cells first (default: they go first, ~1 h)")
    ap.add_argument("--extra-jobs", default=None)
    ap.add_argument("--only", default=None, help="regex on job names")
    ap.add_argument("--rerun-stale", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--wait-files-h", type=float, default=0.0,
                    help="keep polling this long for missing upstream files (the segmenters) once idle")
    ap.add_argument("--seg-ready-h", type=float, default=0.0, help="dry-run: hours until the segmenters exist")
    a = ap.parse_args(argv)
    logf = None
    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)
        logf = open(os.path.join(a.out, "runner.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    jobs = build_jobs(a)
    by = {j.name: j for j in jobs}
    for j in jobs:                                   # existing outputs: loud, provenance-checked
        if os.path.exists(j.out):
            ok, why = check_existing(j)
            if ok:
                j.state, j.note = "SKIPPED", why
                log(f"SKIP   {j.name}  ({j.out} exists, {why})")
            elif a.rerun_stale and not a.dry_run:
                dst = f"{j.out}.stale.{int(time.time())}"
                shutil.move(j.out, dst)
                log(f"STALE->RERUN {j.name}: {why}; renamed to {dst}")
            else:
                j.state, j.note = "STALE", why
                log(f"!! STALE {j.name}: {why}. Not overwritten (use --rerun-stale).")
    counts = {}
    for j in jobs:
        kind = j.name.split(":")[0] + ":" + j.name.split(":")[1]
        counts[kind] = counts.get(kind, 0) + 1
    produced = {j.out for j in jobs}
    missing = sorted({f for j in jobs for f in j.files if not os.path.exists(f) and f not in produced})
    mk, order = simulate([j for j in jobs if j.state == "PENDING"], a.k)
    log(f"queue: {len(jobs)} jobs {counts}; {sum(j.est_h for j in jobs if j.state == 'PENDING'):.1f} job-h pending; "
        f"simulated makespan at K={a.k}: {mk:.1f} h (x1.3 slack: {1.3 * mk:.1f} h)")
    if missing:
        log(f"missing upstream files now ({len(missing)}): " + "; ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    if a.dry_run:
        for t0, n in order:
            j = by[n]
            env = " ".join(f"{k}={v}" if v != "" else f"-u {k}" for k, v in j.env.items())
            print(f"  t+{t0:5.2f}h  {n:44s} est {j.est_h:4.2f}h  deps={j.deps}")
            print(f"           env {env} {' '.join(shlex.quote(c) for c in j.cmd)}")
        return 0

    caf = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    t_start = time.time()
    running = []
    while True:
        for j in list(running):
            rc = j.proc.poll()
            if rc is None:
                continue
            running.remove(j)
            j.state = "DONE" if (rc == 0 and os.path.exists(j.out)) else "FAILED"
            log(f"{j.state:6s} {j.name}  rc={rc}  {(time.time() - j.t0) / 3600:.2f} h  log={j.out[:-5]}.log")
        for j in jobs:                               # propagate failures loudly
            if j.state == "PENDING" and any(by[d].state in ("FAILED", "BLOCKED", "STALE") for d in j.deps if d in by):
                j.state = "BLOCKED"; log(f"!! BLOCKED {j.name}: upstream job failed/blocked/stale")
        refresh_pending(jobs, a, log)
        ready = [j for j in jobs if j.state == "PENDING"
                 and all(by[d].state in ("DONE", "SKIPPED") for d in j.deps if d in by)
                 and all(os.path.exists(f) for f in j.files)]
        ready.sort(key=lambda j: (-j.priority, -j.est_h, j.name))
        while ready and len(running) < a.k:
            j = ready.pop(0)
            os.makedirs(os.path.dirname(j.out), exist_ok=True)
            env = {**os.environ, "PYTHONPATH": "src", "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1",
                   **{k: v for k, v in j.env.items() if v != ""}}
            for k, v in j.env.items():
                if v == "":
                    env.pop(k, None)                 # e.g. never leak US_SEG_CKPT into a CT-mask job
            env.pop("USBG_CONTACT", None)            # every command passes --contact explicitly
            lf = open(j.out[:-5] + ".log", "w")
            j.proc = subprocess.Popen(j.cmd, cwd=j.cwd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            j.t0, j.state = time.time(), "RUNNING"; running.append(j)
            log(f"START  {j.name}  est {j.est_h:.2f} h  [{len(running)}/{a.k}]")
        pending = [j for j in jobs if j.state == "PENDING"]
        if not running and pending and time.time() - t_start < a.wait_files_h * 3600 and \
                any(not all(os.path.exists(f) for f in j.files) for j in pending):
            time.sleep(a.poll); continue                  # idle but upstream files may still arrive
        if not running and pending:
            for j in pending:
                miss = [f for f in j.files if not os.path.exists(f)]
                j.state = "BLOCKED"; log(f"!! BLOCKED {j.name}: missing {miss or 'deps'}")
            break
        if not running and not pending:
            break
        time.sleep(a.poll)
    caf.terminate()
    summ = {}
    for j in jobs:
        summ.setdefault(j.state, []).append(j.name)
    with open(os.path.join(a.out, "runner_state.json"), "w") as fh:
        json.dump({j.name: {"state": j.state, "out": j.out, "note": j.note} for j in jobs}, fh, indent=1)
    log("SUMMARY " + "  ".join(f"{k}={len(v)}" for k, v in sorted(summ.items())))
    for k in ("FAILED", "BLOCKED", "STALE"):
        for n in summ.get(k, []):
            log(f"   {k}: {n}")
    return 1 if any(k in summ for k in ("FAILED", "BLOCKED", "STALE")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
