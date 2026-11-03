#!/usr/bin/env python3
"""plug_render_jobs.py: write the job spec of the renderer-slot pluggability runs (v2.1 pluggability, exploratory).

Writes outputs/v2_1/jobs/plug_render.json in the schema of the other outputs/v2_1/jobs/*.json (read by
scripts/run_table4_v2.py); refuses to overwrite it. Launches nothing.

Renderers: lotus_reimpl (the in-repo LOTUS reimplementation), lotus_orig (the ORIGINAL LOTUS forward model via
usbg.plug.renderers, rendered on CPU: faster than MPS at this size and the original float64 arithmetic), simus
(SIMUS via PyMUST, CPU, 8 worker processes, evaluation only). The physics renderer is the reference: the paper's
rows are reused, and the same-code physics runs here (bench, realism, landscape) check that the wrapper is a no-op.

Jobs (outputs under outputs/v2_1/plug/renderer/)
  bench       per-frame cost on CPU and MPS (render and slice+render+warp), determinism, LOTUS parity, 5 preview
              frames per renderer (TRAIN patient s0011)
  realism     the v2 sector realism protocol (6 held-out patients x 50 views, the 60 public Vitale TEST frames): W1, KID,
              FID, C2ST per renderer; physics and lotus_reimpl rows must reproduce the realism table (parity block)
  landscape   appearance-term learnability, the paper's held-out landscape protocol (5 liver + 5 kidney windows per
              held-out patient, 40 samples per view, DR off, CT seg term); SIMUS on the nested 2-window x 20-sample
              subset with a physics run of the same subset
  ladder      closed loop on the crossed held-out design (6 held-out patients, eval seed 999, 60 episodes, CT masks):
              ibvs_base   the CT-mask servo base rung, calib seeds 0-2, contact on/off (pose trajectory is
                          renderer-invariant under CT masks by construction; S, S_app and keep-best are not)
              policy      the paper's physics-trained CT-mask residual policies (seeds 0-2) evaluated under the plugged
                          renderer (sim-to-sim renderer gap)
              trained     policies TRAINED under the plugged renderer (the paper's CT-mask recipe: imitation 4000,
                          300 x 16 x 12 REINFORCE), evaluated under their own renderer and under physics
              hillclimb   the non-privileged (1+1)-ES on S, sigma0 picked by the pre-declared rule on the v2.1 TRAIN pilots
                          (waits for those pilot files)
  agg         scripts/plug_render_agg.py summary
"""
from __future__ import annotations

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv", "bin", "python")
S = lambda name: os.path.join(REPO, "scripts", name)          # noqa: E731
WINDOWS = os.path.join(REPO, "data", "cache", "windows", "usbg-windows-v1.json")
WINDOWS_MD5 = "af35f0bddeb57c079981eb0ba8aca5b3"
REAL = os.path.join(REPO, "data", "eval", "kaggle_vitale_test.npz")
TABLE2 = os.path.join(REPO, "outputs", "v2", "realism", "realism_sector.json")
PAPER_LANDSCAPE = os.path.join(REPO, "outputs", "v2", "view_match", "landscape_heldout.json")
V2_POLICY = os.path.join(REPO, "outputs", "v2", "table4", "ctmask", "physics_deploy_s{s}.json")
PILOTS = [os.path.join(REPO, "outputs", "v2_1", "s23_pilot", f"hillclimb_{t}_ct_s0.json")
          for t in ("sonospade", "cut", "cyclegan", "physics")]
OUT = os.path.join(REPO, "outputs", "v2_1", "plug", "renderer")
TRAIN = ["s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0522", "s0703", "s0720", "s0727", "s0777", "s1086",
         "s1373"]
EVAL = ["s0358", "s0513", "s0519", "s0591", "s0649", "s0762"]
SEEDS = (0, 1, 2)
TORCH_R = ("lotus_reimpl", "lotus_orig")
EXPECT_CT = {"seat": "anatomical", "texture_md5": None, "windows_md5": WINDOWS_MD5, "seg_md5": None}
EXPECT_SEAT = {"seat": "anatomical", "windows_md5": WINDOWS_MD5}
# measured in the smoke (machine load ~9): per frame (slice+render+warp)
# physics 14-16 ms, lotus_reimpl 10-11 ms, lotus_orig 13 ms CPU / 24 ms MPS, SIMUS 5.3-6 s (8 workers);
# one SIMUS closed-loop episode (14 frames) 82 s. v2 physics CT-mask training measured 2.19 h per seed.
EST = {"bench": 0.15, "realism_torch": 0.3, "realism_simus": 0.75, "land_full": 0.12, "land_sub": 0.04,
       "land_simus": 1.0, "ibvs": 0.06, "ibvs_simus": 1.5, "policy": 0.15, "policy_simus": 1.5,
       "train": {"lotus_reimpl": 2.3, "lotus_orig": 2.5}, "hill": 0.04, "agg": 0.02}


def env(renderer=None):
    e = {"PYTHONPATH": os.path.join(REPO, "src"), "USBG_SEAT": "anatomical", "USBG_LEARNED_SEG": "0",
         "US_SEG_CKPT": "", "USBG_WINDOWS": "", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    if renderer == "lotus_orig":
        e["USBG_PLUG_RENDER_DEVICE"] = "cpu"
    if renderer in ("simus", "all"):
        e["USBG_SIMUS_WORKERS"] = "8"
    return e


def job(name, cmd, out, est_h, *, renderer=None, deps=(), files=(), expect=None, tier="core", note=None):
    j = {"name": name, "cmd": [PY, *cmd], "out": out, "est_h": round(float(est_h), 3), "env": env(renderer),
         "deps": list(deps), "files": [WINDOWS, *files], "expect": expect or EXPECT_CT, "cwd": REPO, "tier": tier}
    if note:
        j["note"] = note
    return j


def run(renderer, target, args, calib=None):
    return [S("plug_render_run.py"), "--renderer", renderer, *(["--calib-renderer", calib] if calib else []),
            "--", target, *args]


def bench_jobs():
    o = os.path.join(OUT, "bench", "bench.json")
    return [job("v21:plug:renderer:bench",
                [S("plug_render_bench.py"), "--renderers", "physics", "lotus_reimpl", "lotus_orig", "simus",
                 "--case", "s0011", "--window", "0", "--devices", "cpu", "mps", "--n", "200", "--warmup", "20",
                 "--n-simus", "5", "--warmup-simus", "1", "--frames", "5", "--out", o],
                o, EST["bench"], renderer="all", expect=EXPECT_SEAT,
                note="timings: best run with nothing else on the machine (load averages are recorded)")]


def realism_jobs():
    common = ["--cases", *EVAL, "--views-per-case", "50", "--renorm", "both", "--kid-subset", "50", "--c2st-repeats",
              "5", "--c2st-folds", "5", "--c2st-epochs", "40", "--c2st-size", "64", "--seed", "0", "--erode", "3",
              "--device", "cpu", "--parity-ref", TABLE2, "--save-frames"]
    o1 = os.path.join(OUT, "realism", "torch.json")
    o2 = os.path.join(OUT, "realism", "simus.json")
    return [job("v21:plug:renderer:realism:torch",
                [S("plug_render_realism.py"), "--renderers", "physics", "lotus_reimpl", "lotus_orig", *common,
                 "--out", o1], o1, EST["realism_torch"], expect=EXPECT_SEAT, files=[REAL, TABLE2],
                note="physics and lotus_reimpl rows must reproduce the realism table's physics / prior LOTUS rows (parity_with_table2)"),
            job("v21:plug:renderer:realism:simus",
                [S("plug_render_realism.py"), "--renderers", "physics", "simus", *common, "--out", o2],
                o2, EST["realism_simus"], renderer="simus", expect=EXPECT_SEAT, files=[REAL, TABLE2], tier="secondary",
                note="300 SIMUS frames on CPU (8 worker processes)")]


def landscape_jobs():
    common = ["--cases", *EVAL, "--organs", "liver", "kidney", "--translators", "physics=", "--ref", "physics",
              "--dr", "off", "--device", "cpu"]
    full = ["--targets-per-case", "5", "--samples-per-target", "40"]
    sub = ["--targets-per-case", "2", "--samples-per-target", "20"]
    jobs = []
    for r in ("physics", *TORCH_R):
        o = os.path.join(OUT, "landscape", f"{r}_full.json")
        jobs.append(job(f"v21:plug:renderer:landscape:{r}:full", run(r, "view_match_landscape", [*common, *full,
                    "--out", o]), o, EST["land_full"], renderer=r, expect=EXPECT_SEAT, files=[PAPER_LANDSCAPE],
                    note=("same-code reference: must equal the paper's physics stage per view "
                          "(paper_physics_parity in the summary)") if r == "physics" else None))
    for r, est, tier in (("physics", EST["land_sub"], "secondary"), ("simus", EST["land_simus"], "secondary")):
        o = os.path.join(OUT, "landscape", f"{r}_sub2x20.json")
        jobs.append(job(f"v21:plug:renderer:landscape:{r}:sub2x20", run(r, "view_match_landscape",
                    [*common, *sub, "--out", o]), o, est, renderer=r, expect=EXPECT_SEAT, tier=tier,
                    note="nested subset of the full design (first 2 windows per organ, first 20 samples per view)"))
    return jobs


def ladder_common():
    return ["--translator", "physics", "--texture-ckpt", "none", "--train-cases", *TRAIN, "--eval-cases", *EVAL,
            "--targets", "liver", "kidney", "--min-windows", "2", "--eval-episodes", "60", "--eval-seed", "999",
            "--eval-design", "crossed", "--ext-log"]


def ladder_jobs():
    jobs = []
    L = os.path.join(OUT, "ladder")
    for r in TORCH_R:
        for s in SEEDS:
            o = os.path.join(L, f"ibvs_base_{r}_ct_s{s}.json")
            jobs.append(job(f"v21:plug:renderer:ladder:ibvs_base:{r}:s{s}", run(r, "eval_view_ladder",
                        ["--rung", "ibvs_base", *ladder_common(), "--calib-seed", str(s), "--contact", "on", "off",
                         "--out", o]), o, EST["ibvs"], renderer=r))
            o = os.path.join(L, f"policy_v2physics_under_{r}_s{s}.json")
            pj = V2_POLICY.format(s=s)
            jobs.append(job(f"v21:plug:renderer:ladder:policy_v2physics:{r}:s{s}", run(r, "eval_view_ladder",
                        ["--rung", "policy", "--policy-json", pj, "--weights", "imit", "final", "--contact", "on",
                         "off", "--ext-log", "--out", o]), o, EST["policy"], renderer=r,
                        files=[pj, pj.replace(".json", ".pt"), pj.replace(".json", "_imit.pt")],
                        note="sim-to-sim: the CT-mask policy of the paper, trained on physics frames, under the plugged renderer"))
    o = os.path.join(L, "ibvs_base_simus_ct_s0.json")
    jobs.append(job("v21:plug:renderer:ladder:ibvs_base:simus:s0", run("simus", "eval_view_ladder",
                ["--rung", "ibvs_base", *ladder_common(), "--calib-seed", "0", "--contact", "on", "--out", o],
                calib="physics"), o, EST["ibvs_simus"], renderer="simus", tier="secondary",
                note="calibration frames rendered by physics: under CT masks the linear base reads only CT labels, "
                     "so the base is identical (checked in the smoke); 840 SIMUS frames"))
    pj = V2_POLICY.format(s=0)
    o = os.path.join(L, "policy_v2physics_under_simus_s0.json")
    jobs.append(job("v21:plug:renderer:ladder:policy_v2physics:simus:s0", run("simus", "eval_view_ladder",
                ["--rung", "policy", "--policy-json", pj, "--weights", "final", "--contact", "on", "--ext-log",
                 "--out", o]), o, EST["policy_simus"], renderer="simus", tier="secondary",
                files=[pj, pj.replace(".json", ".pt")], note="840 SIMUS frames"))
    return jobs


def train_jobs():
    jobs = []
    T = os.path.join(OUT, "train")
    L = os.path.join(OUT, "ladder")
    for r in TORCH_R:
        for s in SEEDS:
            o = os.path.join(T, f"{r}_ctmask_s{s}.json")
            tname = f"v21:plug:renderer:train:{r}:s{s}"
            jobs.append(job(tname, run(r, "train_view_match_rl",
                        ["--translator", "physics", "--texture-ckpt", "none", "--fix-residual", "--residual-image",
                         "--ibvs-features", "rich", "--w-seg", "0.6", "--w-app", "0.4", "--imitation", "4000",
                         "--iters", "300", "--episodes-per-iter", "16", "--n-steps", "12", "--eval-episodes", "60",
                         "--eval-seed", "999", "--eval-design", "crossed", "--train-cases", *TRAIN, "--eval-cases",
                         *EVAL, "--targets", "liver", "kidney", "--min-windows", "2", "--contact", "on", "--seed",
                         str(s), "--ablation", f"ctmask_{r}", "--out", o]), o, EST["train"][r], renderer=r,
                        tier="core" if r == "lotus_orig" else "secondary",
                        note="the paper's CT-mask recipe (outputs/v2/table4/ctmask/physics_deploy_s*.json argv) with "
                             "the renderer swapped"))
            for ev, tag in ((r, "own"), ("physics", "physics")):
                oe = os.path.join(L, f"policy_{r}trained_under_{tag}_s{s}.json")
                jobs.append(job(f"v21:plug:renderer:ladder:policy_{r}trained:under_{tag}:s{s}",
                            run(ev, "eval_view_ladder", ["--rung", "policy", "--policy-json", o, "--weights", "imit",
                                                         "final", "--contact", "on", "off", "--ext-log", "--out", oe]),
                            oe, EST["policy"], renderer=ev, deps=[tname], tier="core" if r == "lotus_orig" else "secondary"))
    return jobs


def hill_jobs():
    jobs = []
    L = os.path.join(OUT, "ladder")
    for r in TORCH_R:
        for s in SEEDS:
            o = os.path.join(L, f"hillclimb_{r}_ct_s{s}.json")
            jobs.append(job(f"v21:plug:renderer:ladder:hillclimb:{r}:s{s}", run(r, "eval_view_ladder",
                        ["--rung", "hillclimb", *ladder_common(), "--calib-seed", str(s), "--contact", "on",
                         "--es-select", *PILOTS, "--out", o]), o, EST["hill"], renderer=r, files=PILOTS,
                        tier="secondary", note="needs the v2.1 queue's s23 TRAIN pilot outputs (file deps)"))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "v2_1", "jobs", "plug_render.json"))
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")
    jobs = bench_jobs() + realism_jobs() + landscape_jobs() + ladder_jobs() + train_jobs() + hill_jobs()
    agg_out = os.path.join(OUT, "summary.json")
    jobs.append(job("v21:plug:renderer:agg", [S("plug_render_agg.py"), "--out", agg_out], agg_out, EST["agg"],
                    deps=[j["name"] for j in jobs], expect=EXPECT_SEAT))
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)), "duplicate job names"
    outs = [j["out"] for j in jobs]
    assert len(outs) == len(set(outs)), "duplicate outputs"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(jobs, fh, indent=1)
    tot = sum(j["est_h"] for j in jobs)
    core = sum(j["est_h"] for j in jobs if j["tier"] == "core")
    print(f"wrote {args.out}: {len(jobs)} jobs, {tot:.2f} job-hours ({core:.2f} core)")
    if args.print:
        for j in jobs:
            print(f"  {j['est_h']:5.2f} h  {j['tier']:9s} {j['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
