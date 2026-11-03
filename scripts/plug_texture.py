#!/usr/bin/env python3
"""plug_texture.py: v2.1 pluggability, texture slot. EXPLORATORY and opt-in: a new file, no existing
script, default or artifact changes, and every output carries provenance.analysis_status
"exploratory, v2.1 pluggability". Archived v2 numbers stay primary.

Adds published unpaired sim-to-real translators to the texture stage (usbg.plug_texture: FastCUT, DCLGAN,
UNSB, global histogram matching) and runs them through the paper's texture-row pipeline:

  tex  <stage> <seed>   train with EXACTLY the v2 grayscale recipe (scripts/run_textures_v2.sh step 5):
                        train_texture_gan.py --sim-npz data/dataset/v2/pairs_med_curvi.npz (1540 TRAIN-only
                        curvi frames of 8 TRAIN patients) --real-dir data/real/kaggle_vitale/pool (404 public
                        Vitale TRAIN-split frames) --iters 800 --batch 4 --seed <seed>, raw last iterate;
                        histmatch is fitted once on the same frames (seed 0 only). Measures the per-frame
                        inference cost on CPU and MPS.
                        -> data/dataset/v2_1/plug/texture/<stage>_curvi_s<seed>.pt (+ .provenance.json)
                        -> outputs/v2_1/plug/texture/tex/<stage>_s<seed>.json
  seg  <stage> <s>      the stage-matched segmenter through segtrain's v2 pipeline (run_seg_v2.sh per
                        stage+seed without the crossval/consistency steps):
                        build(train, 13 TRAIN patients, 27 poses, seed s), build(TEST sim-eval, 12 poses,
                        seed 1000+s), train 4000 it --kidneyus --vitale-split, eval_seg. Texture seed 0 for
                        every segmenter seed s (the v2 oracle-ladder design).
                        -> segtrain/outputs/v2_1/plug/texture/us_seg_<stage>_s<s>.pt, eval_*.json
                        -> outputs/v2_1/plug/texture/seg/<stage>_s<s>.json
  base <stage> <s>      learned-mask servo base rung, the v2 oracle-ladder cell recipe (eval_view_ladder --rung
                        ibvs_base, 60 crossed held-out episodes, eval seed 999, contact on and off, calib seed s)
                        -> outputs/v2_1/plug/texture/base/ibvs_base_<stage>_matched_s<s>.json
  land                  appearance-term learnability landscape (Table learnability, DR off), the v2 argv with
                        a manifest of the v2 stages plus the plug stages, then agg_table3 contrasts
                        -> outputs/v2_1/plug/texture/landscape/landscape_heldout_plug.json (+ table3_stats_*.json)
  landdr                DR landscape (off, shared, cross, crossseed) with each plug stage's seed-0 segmenter
                        -> outputs/v2_1/plug/texture/landscape/landscape_dr_plug.json
  real                  distance to real frames in matched sector geometry (Table realism), the v2 argv with
                        the plug stages -> outputs/v2_1/plug/texture/realism/realism_sector_plug.json
  agg                   -> outputs/v2_1/plug/texture/plug_texture_summary.json (contrasts vs the archived v2 rows)
  jobs [path]           runner spec -> outputs/v2_1/jobs/plug_texture.json (not launched)
  smoke                 tiny end-to-end run on TRAIN patients only, under $PLUG_SMOKE_DIR

Existing scripts run through scripts/plug_texture_exec.py, which registers the plug-ins and widens their
--variant / --translator choices in that process only. Never overwrites an existing output (set
PLUG_TEXTURE_OVERWRITE=1 to replace this script's OWN plug outputs; never a v2 or v2.1-queue path).
Usage: cd <repository root> && USBG_SEAT=anatomical PYTHONPATH=src .venv/bin/python scripts/plug_texture.py <sub> ...
"""
from __future__ import annotations

import collections
import glob
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
if os.path.join(HERE, "src") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "src"))

PY = os.path.join(HERE, ".venv", "bin", "python")
EXEC = os.path.join(HERE, "scripts", "plug_texture_exec.py")
SELF = os.path.abspath(__file__)
SEG = os.environ.get("USBG_SEG_REPO", os.path.join(HERE, "segtrain"))
STATUS = "exploratory, v2.1 pluggability"
V2D = os.path.join(HERE, "data", "dataset", "v2")
PAIRS_V2 = os.path.join(V2D, "pairs_med_curvi.npz")
POOL = "data/real/kaggle_vitale/pool"                          # relative to the gym repo, as the v2 command
V2LADDER = os.path.join(HERE, "outputs", "v2", "table4", "ladder")
V21TEXBASE = os.path.join(HERE, "outputs", "v2_1", "texture", "base")
V2VM = os.path.join(HERE, "outputs", "v2", "view_match")
V2REAL = os.path.join(HERE, "outputs", "v2", "realism", "realism_sector.json")
WINDOWS = os.path.join(HERE, "data", "cache", "windows", "usbg-windows-v1.json")
WINDOWS_MD5_V2 = "af35f0bddeb57c079981eb0ba8aca5b3"
TRAIN13 = "s0011 s0058 s0344 s0461 s0477 s0511 s0522 s0703 s0720 s0727 s0777 s1086 s1373".split()
TEST6 = "s0358 s0513 s0519 s0591 s0649 s0762".split()
SKIP_PAIRS = "s0058:kidney,s0522:kidney,s0727:kidney"
SPLIT = "data/real/vitale_split_v2.npz"                         # seg repo relative
V2_STAGES = ("physics", "cut", "cyclegan", "sonospade")
PLUG = ("fastcut", "dclgan", "unsb", "histmatch")
STAGE_B = {"fastcut": 1, "dclgan": 2, "unsb": 3, "histmatch": 4}  # nnU-Net ids 700+10b+s (train), 800+10b+s (eval)
SEEDS = (0, 1, 2)

SMOKE = os.environ.get("PLUG_SMOKE", "0") == "1"
SMOKE_DIR = os.environ.get("PLUG_SMOKE_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "usbg_plug_smoke", "texture"))
if SMOKE:
    TEXD, OUT, SEGOUT = (os.path.join(SMOKE_DIR, "data"), os.path.join(SMOKE_DIR, "out"), os.path.join(SMOKE_DIR, "seg"))
    PAIRS = os.path.join(SMOKE_DIR, "data", "pairs_smoke.npz")
    TEX_IT, SEG_IT, NPOSES, EPIS = int(os.environ.get("SMOKE_TEX_IT", "4")), 20, 2, 2
    BASE_TRAIN, BASE_EVAL, SEG_CASES = ["s0011"], ["s0344"], "s0011,s0344"
    LAND_CASES = ["s0720"]                                      # TRAIN patient outside every smoke lineage
else:
    TEXD = os.path.join(HERE, "data", "dataset", "v2_1", "plug", "texture")
    OUT = os.path.join(HERE, "outputs", "v2_1", "plug", "texture")
    SEGOUT = os.path.join(SEG, "outputs", "v2_1", "plug", "texture")
    PAIRS = PAIRS_V2
    TEX_IT, SEG_IT, NPOSES, EPIS = 800, 4000, 27, 60
    BASE_TRAIN, BASE_EVAL, SEG_CASES = TRAIN13, TEST6, "train"
    LAND_CASES = TEST6
NPOSES_EVAL = 12
NN = os.path.join(SEGOUT, "nnunet")
MANIFESTS = os.path.join(OUT, "manifests")


def die(msg, code=2):
    print(f"plug_texture: {msg}", file=sys.stderr)
    raise SystemExit(code)


def log(msg):
    print(f"[plug_texture] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- guards
def guards():
    if os.environ.get("USBG_SEAT") != "anatomical":
        die(f"export USBG_SEAT=anatomical explicitly (got {os.environ.get('USBG_SEAT')!r})")
    if os.environ.get("USBG_WINDOWS"):
        die(f"USBG_WINDOWS is set ({os.environ['USBG_WINDOWS']}): the plug rows use the v1 windows file")
    if os.environ.get("USBG_CACHE_DIR"):
        die("USBG_CACHE_DIR is obsolete; unset it")
    from usbg.benchmark import TRAIN_PATIENTS, TEST_PATIENTS
    from usbg.seatmode import seat_mode
    from usbg import windows as W, provenance as P
    if seat_mode() != "anatomical":
        die(f"seat {seat_mode()}")
    if sorted(TRAIN13) != sorted(TRAIN_PATIENTS) or sorted(TEST6) != sorted(TEST_PATIENTS):
        die("cohort drift: usbg.benchmark TRAIN/TEST differ from the v2 cohort this script pins")
    if W.file_md5() != WINDOWS_MD5_V2:
        die(f"windows file {W.WINDOWS_PATH} md5 {W.file_md5()} != v2 {WINDOWS_MD5_V2}")
    if P.is_private_path(POOL):
        die("private real pool")


def _own(path):
    """True if the path is one of this script's own plug outputs (the only ones it may ever replace)."""
    p = os.path.abspath(path)
    roots = [os.path.abspath(r) + os.sep for r in (TEXD, OUT, SEGOUT)] + ([os.path.abspath(SMOKE_DIR) + os.sep] if SMOKE else [])
    return any(p.startswith(r) for r in roots) and "/outputs/v2/" not in p and "/dataset/v2/" not in p


def no_clobber(*paths):
    for f in paths:
        if not os.path.lexists(f):
            continue
        if not _own(f):
            die(f"refusing a path outside the plug outputs: {f}")
        if SMOKE or os.environ.get("PLUG_TEXTURE_OVERWRITE") == "1":
            shutil.rmtree(f) if os.path.isdir(f) and not os.path.islink(f) else os.remove(f)
        else:
            die(f"refusing to overwrite {f} (PLUG_TEXTURE_OVERWRITE=1 replaces this script's own outputs)")


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(path + ".tmp", path)


def run(cmd, cwd=HERE, env=None, unset=()):
    e = {**os.environ, "PYTHONPATH": os.path.join(HERE, "src"), "PYTORCH_ENABLE_MPS_FALLBACK": "1",
         "PYTHONUNBUFFERED": "1", "USBG_GYM_ROOT": HERE, **(env or {})}
    for k in unset:
        e.pop(k, None)
    log("+ " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    rc = subprocess.call([str(c) for c in cmd], cwd=cwd, env=e)
    if rc:
        die(f"command failed (rc {rc}): {cmd[:4]} ...", rc)
    return time.time() - t0


def tex_of(stage, seed=0):
    if stage == "physics":
        return None
    if stage in V2_STAGES:
        return os.path.join(V2D, f"{stage}_curvi.pt")
    if stage not in PLUG:
        die(f"unknown stage {stage}")
    return os.path.join(TEXD, f"{stage}_curvi_s{seed}.pt")


def seg_of(stage, s):
    if stage in V2_STAGES:
        return os.path.join(SEG, "outputs", "v2", f"us_seg_{stage}_s{s}.pt")
    return os.path.join(SEGOUT, f"us_seg_{stage}_s{s}.pt")


def plug_block(**kw):
    from usbg import plug_texture as PT
    st = kw.get("stage")
    return {"analysis_status": STATUS, "slot": "texture", "smoke": SMOKE,
            **({"method": PT.METHODS[st]} if st in PT.METHODS else {}), **kw}


def stamp_prov(d, block, extra_result_block=None):
    """Stamp a result JSON's provenance with the plug block; add seat/windows from a result_block if the
    script's own provenance lacks them (the runner's check_existing reads seat and windows md5)."""
    from usbg import provenance as P
    prov = d.setdefault("provenance", {})
    prov["analysis_status"] = STATUS
    prov["v21_plug"] = block
    if not (prov.get("seat_mode") or (prov.get("seat") or {}).get("resolved")) or not (prov.get("windows") or {}).get("md5"):
        rb = extra_result_block or P.result_block()
        prov.setdefault("seat_mode", rb["seat_mode"])
        prov["windows"] = prov.get("windows") or rb["windows"]
    return d


# ------------------------------------------------------------------------------------------------ tex
def ensure_smoke_pairs():
    if os.path.exists(PAIRS):
        return
    os.makedirs(os.path.dirname(PAIRS), exist_ok=True)
    d = np.load(PAIRS_V2, allow_pickle=False)
    ses = d["sessions"]
    keep = np.concatenate([np.flatnonzero(ses == c)[:24] for c in ("s0011", "s0344")])   # 2 TRAIN patients
    meta = json.loads(str(d["meta"])); meta["smoke_subset_of"] = PAIRS_V2; meta["cases"] = ["s0011", "s0344"]
    np.savez_compressed(PAIRS, images=d["images"][keep], labels=d["labels"][keep], sessions=ses[keep],
                        meta=np.asarray(json.dumps(meta)))
    log(f"smoke pairs {PAIRS}: {len(keep)} frames from s0011, s0344")


def infer_cost(ck, frames, n):
    import torch
    from usbg import texture_gan as TG, plug_texture as PT
    out = {}
    for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        tr = TG.load_translator(ck, dev)
        if str(getattr(tr, "device", dev)) != dev:
            out[dev] = {"device": str(tr.device), "note": "this stage runs on the CPU (NumPy lookup) on every device"}
            continue
        out[dev] = PT.time_translator(tr, frames, n=n, warmup=5)
    out["note"] = ("batch 1, 128 px curvi frame, translator call only (numpy in/out), after 5 warm-up calls, fresh process; "
                   "measured right after training on the same machine (M5 Pro), other jobs may share it")
    return out


def infer_cost_fresh(ck, n):
    """infer_cost in a fresh process, so the training run's MPS work and threads do not bias the timing."""
    p = subprocess.run([sys.executable, SELF, "_cost", ck, PAIRS, str(n)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.path.join(HERE, "src"), "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    if p.returncode:
        die(f"inference timing failed: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def cmd_cost(ck, pairs, n):
    import torch
    import train_texture_gan as T
    frames = T.load_sim_frames(pairs)[:64]
    out = infer_cost(ck, frames, n)
    out["torch_threads"] = torch.get_num_threads()
    print(json.dumps(out))


def cmd_tex(stage, seed):
    import torch
    import train_texture_gan as T
    from usbg import texture_gan as TG, provenance as P, plug_texture as PT
    if stage not in PLUG:
        die("tex: stage must be one of " + ", ".join(PLUG))
    if stage == "histmatch" and seed != 0:
        die("histmatch is deterministic (no training randomness): seed 0 only")
    if SMOKE:
        ensure_smoke_pairs()
    ck = tex_of(stage, seed)
    res = os.path.join(OUT, "tex", f"{stage}_s{seed}.json")
    prev = os.path.splitext(res)[0] + "_preview.png"
    no_clobber(ck, P.sidecar_path(ck), res, prev)
    os.makedirs(TEXD, exist_ok=True); os.makedirs(os.path.dirname(res), exist_ok=True)
    budget = {"recipe": "v2 grayscale texture recipe (scripts/run_textures_v2.sh step 5)", "iters": TEX_IT,
              "batch": 4, "size": 128, "ngf": 32, "n_blocks": 6, "lr": 2e-4, "sim_frames": PAIRS,
              "augmentation": "none (the 1540 non-augmented TRAIN-only curvi frames, as v2 CUT/CycleGAN)",
              "checkpoint_rule": "raw last iterate (as v2 CUT/CycleGAN); no validation-based selection",
              "real_pool": POOL, "texture_seed": seed}
    blk = plug_block(stage=stage, texture_seed=seed, budget=budget,
                     plan="v2.1 pluggability, texture slot: published unpaired translators at the v2 recipe")
    t0 = time.time()
    if stage in PT.LEARNED_STAGES:
        PT.widen_choices()
        _ws = T.write_sidecar

        def write_sidecar(args, train_cases, parents, real_pool, seat, extra):    # stamp only; recipe untouched
            return _ws(args, train_cases, parents, real_pool, seat, {**extra, "v21_plug": blk})
        T.write_sidecar = write_sidecar
        argv = ["--variant", stage, "--sim-npz", PAIRS, "--real-dir", POOL, "--iters", str(TEX_IT), "--batch", "4",
                "--seed", str(seed), "--out", ck, "--preview", prev]
        sys.argv = ["scripts/train_texture_gan.py(via scripts/plug_texture.py tex)"] + argv
        rc = T.main(argv)
        if rc:
            die(f"train_texture_gan rc {rc}", rc)
    else:                                                           # histmatch: one global fit, same inputs
        why = P.is_private_descendant(PAIRS)
        if why:
            die(f"sim frames are a private-pool descendant: {why}")
        cases, cases_src = T.npz_cases(PAIRS)
        from usbg.benchmark import TEST_PATIENTS
        if set(cases) & set(TEST_PATIENTS):
            die(f"held-out patients {sorted(set(cases) & set(TEST_PATIENTS))} in the sim frames")
        frames = T.load_sim_frames(PAIRS)
        real = T.load_real_dir(POOL, 128)
        lut, stats = PT.fit_histmatch(frames, real)
        PT.save_histmatch(ck, lut, stats)
        sys.argv = ["scripts/plug_texture.py", "tex", stage, str(seed)]
        sc = P.write_provenance(ck, "translator", train_cases=cases, real_pool=T.real_pool_record(POOL, len(real)),
                                iters=0, seed=0, dr_mode="per_frame",
                                dr_notes="sim frames carry the pairs' per-frame render DR; no DR at fit time",
                                parents=[P.parent_entry(PAIRS, "sim_pool")], seat_mode=T.npz_seat(PAIRS),
                                extra={"variant": "histmatch", "size": 128, "sim_frames": len(frames),
                                       "cases_from": cases_src, "fit": stats, "v21_plug": blk})
        log(f"wrote provenance -> {sc}")
        tr = TG.load_translator(ck, "cpu")
        show = frames[:5]
        T._save_before_after(show, [tr(f) for f in show], prev, f"histmatch (real-dir ({len(real)}), fit, cpu)")
    wall = time.time() - t0
    sc = P.sidecar_path(ck)
    if not os.path.exists(sc):
        die(f"no provenance sidecar for {ck}")
    side = json.load(open(sc))
    if not SMOKE:
        rep = P.scan([os.path.dirname(ck), V2D])
        mine = [f for f in rep["findings"] if os.path.basename(ck) in f]
        if mine:
            die("provenance findings:\n  " + "\n  ".join(mine))
    frames = T.load_sim_frames(PAIRS)
    cost = infer_cost_fresh(ck, n=30)
    # one translated frame, kept as a small array summary (the preview PNG holds the picture)
    tr = TG.load_translator(ck, "cpu")
    one = tr(frames[0])
    prov = P.result_block(texture=ck, train_cases=side["train_cases"], target_set=["liver", "kidney"],
                          extra={"v21_plug": blk, "sidecar": sc, "sidecar_md5": P.md5(sc)})
    prov["analysis_status"] = STATUS
    out = {"experiment": "v2_1/plug/texture", "kind": "texture", "stage": stage, "texture_seed": seed, "ckpt": ck,
           "md5": P.md5(ck), "sidecar": sc, "preview": prev, "iters": TEX_IT if stage in PT.LEARNED_STAGES else 0,
           "wall_sec": round(wall, 1),
           "sec_per_iter": round(wall / max(TEX_IT, 1), 4) if stage in PT.LEARNED_STAGES else None,
           "infer_cost_ms": cost,
           "one_frame": {"in_mean": float(frames[0].mean()), "out_mean": float(one.mean()), "shape": list(one.shape),
                         "out_min": float(one.min()), "out_max": float(one.max())},
           "torch": torch.__version__, "provenance": prov}
    from usbg.render_lotus import get_device
    out["train_device"] = get_device()
    write_json(res, out)
    log(f"tex {stage} s{seed}: {ck} ({wall:.0f}s) cpu {cost['cpu']['ms_mean']:.1f} ms"
        + (f", mps {cost['mps']['ms_mean']:.1f} ms" if "ms_mean" in cost.get("mps", {}) else "") + f" -> {res}")


# ------------------------------------------------------------------------------------------------ seg
def cmd_seg(stage, s):
    from usbg import provenance as P
    if stage not in PLUG:
        die("seg: stage must be one of " + ", ".join(PLUG))
    tx = tex_of(stage, 0)
    if not os.path.exists(tx):
        die(f"texture {tx} missing (run: tex {stage} 0)")
    b = STAGE_B[stage]
    tid, eid = 700 + 10 * b + s, 800 + 10 * b + s
    ds = os.path.join(NN, f"Dataset{tid}_USOrgansV21plug_{stage}_s{s}")
    eds = os.path.join(NN, f"Dataset{eid}_USOrgansV21plugEval_{stage}_s{s}")
    ck = seg_of(stage, s)
    hist = os.path.splitext(ck)[0] + "_history.json"
    ev = os.path.join(SEGOUT, f"eval_{stage}_s{s}.json")
    res = os.path.join(OUT, "seg", f"{stage}_s{s}.json")
    no_clobber(ds, eds, ds + ".provenance.json", eds + ".provenance.json", ck, hist, ck + ".provenance.json", ev, res)
    for d in (NN, SEGOUT, os.path.dirname(res)):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(SEG, SPLIT)):
        die(f"{SEG}/{SPLIT} missing")
    env = {"USBG_LEARNED_SEG": "0", "USBG_SEAT": "anatomical"}
    rs = os.path.join(SEG, "run.sh")
    tm = {}
    tm["build_train"] = run([rs, EXEC, "scripts/build_dataset.py", "--out", NN, "--dataset-id", tid,
                             "--name", f"USOrgansV21plug_{stage}_s{s}", "--translator", tx, "--cases", SEG_CASES,
                             "--role", "train", "--seat-organs", "liver,kidney", "--strict-seat",
                             "--skip-pairs", SKIP_PAIRS, "--require-seat", "anatomical", "--n-poses", NPOSES,
                             "--seed", s], cwd=SEG, env=env, unset=("US_SEG_CKPT",))
    if SMOKE:                                       # smoke never renders a held-out patient: no TEST sim-eval build
        tm["build_eval_and_check"] = run([rs, "scripts/check_manifest.py", "--datasets", ds, "--require-seat",
                                          "anatomical"], cwd=SEG, env=env, unset=("US_SEG_CKPT",))
    else:
        t1 = run([rs, EXEC, "scripts/build_dataset.py", "--out", NN, "--dataset-id", eid,
                  "--name", f"USOrgansV21plugEval_{stage}_s{s}", "--translator", tx, "--cases", "test", "--role", "eval",
                  "--seat-organs", "liver,kidney", "--strict-seat", "--skip-pairs", SKIP_PAIRS, "--require-seat",
                  "anatomical", "--n-poses", NPOSES_EVAL, "--seed", 1000 + s, "--no-real-test"],
                 cwd=SEG, env=env, unset=("US_SEG_CKPT",))
        t2 = run([rs, "scripts/check_manifest.py", "--datasets", ds, "--eval-datasets", eds, "--require-seat",
                  "anatomical"], cwd=SEG, env=env, unset=("US_SEG_CKPT",))
        tm["build_eval_and_check"] = t1 + t2
    trk = ["--eval-every", "10"] if SMOKE else []
    tm["train"] = run([rs, "scripts/train.py", "--dataset", ds, "--iters", SEG_IT, "--seed", s, "--kidneyus",
                       "--vitale-split", SPLIT, "--forbid-test-patients", *trk, "--out", ck],
                      cwd=SEG, env=env, unset=("US_SEG_CKPT",))
    t3 = run([rs, "scripts/write_seg_manifest.py", "--dir", SEGOUT], cwd=SEG, env=env, unset=("US_SEG_CKPT",))
    t4 = run([rs, "scripts/check_manifest.py", "--histories", hist, "--vitale-split", SPLIT, "--seg-manifest",
              os.path.join(SEGOUT, "seg_manifest.json"), "--require-seat", "anatomical"], cwd=SEG, env=env,
             unset=("US_SEG_CKPT",))
    if SMOKE:
        t5 = run([rs, "scripts/eval_seg.py", "--ckpt", ck, "--dataset", ds, "--real-npz",
                  "data/real/vitale_holdout_v2.npz", "--out", ev], cwd=SEG, env=env, unset=("US_SEG_CKPT",))
    else:
        t5 = run([rs, "scripts/eval_seg.py", "--ckpt", ck, "--dataset", eds, "--sim-all",
                  "--real-npz", "data/real/vitale_holdout_v2.npz", "--real-npz", "data/real/aul_liver_test.npz",
                  "--real-npz", "data/real/kidneyus_holdout_test.npz", "--out", ev], cwd=SEG, env=env,
                 unset=("US_SEG_CKPT",))
    tm["manifest_and_eval"] = t3 + t4 + t5
    blk = plug_block(stage=stage, texture_seed=0, segmenter_seed=s, texture=tx, texture_md5=P.md5(tx),
                     recipe=("segtrain run_seg_v2.sh per stage+seed WITHOUT the 6-fold crossval and "
                             "consistency steps: build(train, 27 poses, seed s), build(TEST sim-eval, 12 poses, seed "
                             "1000+s), train 4000 it --kidneyus --vitale-split, eval_seg; texture seed 0 for every s "
                             "(v2 oracle-ladder design)"))

    def stamp(path, where):
        if not os.path.exists(path):
            return False
        d = json.load(open(path))
        b2 = d.setdefault(where, {}) if where else d
        b2["analysis_status"] = STATUS
        b2["v21_plug"] = blk
        write_json(path, d)
        return True
    stamp(hist, "provenance"); stamp(ev, "provenance"); stamp(ck + ".provenance.json", "extra")
    for dsd in (ds, eds):
        stamp(dsd + ".provenance.json", "extra")
    evd = json.load(open(ev)) if os.path.exists(ev) else {}

    def dice(block, cls):
        return ((evd.get(block) or {}).get("per_class") or {}).get(cls)
    summary = {"sim_val_mean": (evd.get("sim_val") or {}).get("mean"),
               "sim_val_liver": dice("sim_val", "liver"), "sim_val_kidney": dice("sim_val", "kidney"),
               "vitale_holdout_liver": dice("vitale_holdout_v2.npz", "liver"),
               "aul_liver": dice("aul_liver_test.npz", "liver"), "kidneyus_kidney": dice("kidneyus_holdout_test.npz", "kidney")}
    hd = json.load(open(hist)) if os.path.exists(hist) else {}
    prov = P.result_block(texture=tx, segmenter=ck, train_cases=hd.get("train_source_cases") or [],
                          target_set=["liver", "kidney"], extra={"v21_plug": blk, "dataset": ds, "eval_dataset": eds,
                                                                 "history": hist, "eval_json": ev})
    prov["analysis_status"] = STATUS
    out = {"experiment": "v2_1/plug/texture", "kind": "segmenter", "stage": stage, "texture_seed": 0,
           "segmenter_seed": s, "ckpt": ck, "md5": P.md5(ck), "texture": tx, "eval_json": ev, "dice": summary,
           "timing_sec": {k: round(v, 1) for k, v in tm.items()}, "wall_sec": round(sum(tm.values()), 1),
           "provenance": prov}
    write_json(res, out)
    log(f"seg {stage} s{s}: {ck} dice {summary} -> {res}")


# ----------------------------------------------------------------------------------------------- base
def cmd_base(stage, s):
    from usbg import provenance as P
    if stage not in PLUG:
        die("base: stage must be one of " + ", ".join(PLUG))
    tx, sk = tex_of(stage, 0), seg_of(stage, s)
    for f in (tx, sk):
        if not os.path.exists(f):
            die(f"missing {f}")
    res = os.path.join(OUT, "base", f"ibvs_base_{stage}_matched_s{s}.json")
    raw = os.path.join(OUT, "base", f".unstamped_ibvs_base_{stage}_matched_s{s}.json")
    no_clobber(res)
    os.makedirs(os.path.dirname(res), exist_ok=True)
    if os.path.exists(raw):
        os.remove(raw)                                  # this job's own scratch file from an earlier attempt
    xtra = ["--contact", "on", "--n-steps", "3"] if SMOKE else ["--contact", "on", "off"]
    wall = run([PY, EXEC, os.path.join(HERE, "scripts", "eval_view_ladder.py"), "--rung", "ibvs_base",
                "--translator", stage, "--texture-ckpt", tx, "--train-cases", *BASE_TRAIN, "--eval-cases", *BASE_EVAL,
                "--targets", "liver", "kidney", "--min-windows", "2", *xtra, "--eval-episodes", EPIS,
                "--eval-seed", "999", "--eval-design", "crossed", "--calib-seed", s, "--out", raw],
               env={"USBG_LEARNED_SEG": "1", "US_SEG_CKPT": sk, "USBG_SEAT": "anatomical"})
    d = json.load(open(raw))
    v2cells = {st: os.path.join(V2LADDER, f"ibvs_base_{st}_matched_s{s}.json") for st in V2_STAGES}
    blk = plug_block(stage=stage, texture_seed=0, segmenter_seed=s, calib_seed=s, texture=tx, segmenter=sk,
                     recipe="v2 oracle-ladder learned-mask ibvs_base cell (eval_view_ladder argv identical except stage/ckpts)",
                     comparators_archived_v2={k: v for k, v in v2cells.items() if os.path.exists(v)},
                     unstamped_md5=P.md5(raw), wall_sec=round(wall, 1))
    stamp_prov(d, blk)
    d["v21_plug_stage"] = stage
    write_json(res, d)
    os.remove(raw)
    on = [r for r in d["results"] if r["contact"] == "on"][0]
    log(f"base {stage} s{s}: final {on['pos']:.2f} mm (start {on['start_pos']:.2f}, keep-best "
        f"{on['keepbest_pos']:.2f}) -> {res}")


# ------------------------------------------------------------------------------------------ manifests
def write_manifest(name, man):
    """Write (or check) a texture manifest under OUT/manifests. Never replaces a different one."""
    p = os.path.join(MANIFESTS, name)
    if os.path.exists(p):
        if json.load(open(p)) != man:
            if SMOKE or os.environ.get("PLUG_TEXTURE_OVERWRITE") == "1":
                os.remove(p)
            else:
                die(f"{p} exists with different content (PLUG_TEXTURE_OVERWRITE=1 to replace)")
        else:
            return p
    write_json(p, man)
    return p


def _move_siblings(raw, final):
    """Rename files the script wrote next to its --out (e.g. <stem>.png) from the raw stem to the final one."""
    rs, fs = os.path.splitext(raw)[0], os.path.splitext(final)[0]
    for f in glob.glob(rs + ".*"):
        if f != raw:
            tgt = fs + f[len(rs):]
            no_clobber(tgt)
            os.replace(f, tgt)


def cmd_land():
    from usbg import provenance as P
    man = {"physics": None, **{st: tex_of(st) for st in V2_STAGES if st != "physics"},
           **{st: tex_of(st, 0) for st in PLUG}}
    if SMOKE:
        man = {"physics": None, **{st: tex_of(st, 0) for st in PLUG}}
    for k, v in man.items():
        if v and not os.path.exists(v):
            die(f"texture {k} missing: {v}")
    mp = write_manifest("landscape_manifest.json", man)
    d0 = os.path.join(OUT, "landscape")
    res = os.path.join(d0, "landscape_heldout_plug.json")
    raw = os.path.join(d0, ".unstamped_landscape_heldout_plug.json")
    refs = (["sonospade"] if not SMOKE else []) + list(PLUG)
    no_clobber(res, os.path.splitext(res)[0] + ".png", *[os.path.join(d0, f"table3_stats_ref_{r}.json") for r in refs])
    os.makedirs(d0, exist_ok=True)
    for f in glob.glob(os.path.splitext(raw)[0] + ".*"):
        os.remove(f)
    tps, sps = (1, 4) if SMOKE else (5, 40)
    wall = run([PY, EXEC, os.path.join(HERE, "scripts", "view_match_landscape.py"), "--cases", *LAND_CASES,
                "--organs", "liver", "kidney", "--targets-per-case", tps, "--samples-per-target", sps,
                "--tex-manifest", mp, "--dr", "off", "--device", "cpu", "--out", raw],
               env={"USBG_LEARNED_SEG": "0", "USBG_SEAT": "anatomical"}, unset=("US_SEG_CKPT",))
    d = json.load(open(raw))
    blk = plug_block(stage="landscape", manifest=mp, manifest_md5=P.md5(mp), wall_sec=round(wall, 1),
                     recipe="v2 argv of outputs/v2/view_match/landscape_heldout.json except --tex-manifest/--out",
                     archived_v2=os.path.join(V2VM, "landscape_heldout.json"))
    stamp_prov(d, blk)
    write_json(res, d)
    os.remove(raw)
    _move_siblings(raw, res)
    stats = {}
    for ref in refs:
        sp = os.path.join(d0, f"table3_stats_ref_{ref}.json")
        run([PY, os.path.join(HERE, "scripts", "agg_table3.py"), res, "--ref", ref, "--out", sp],
            env={"USBG_SEAT": "anatomical"})
        sd = json.load(open(sp)) if os.path.exists(sp) else {}
        stamp_prov(sd, plug_block(stage="landscape_stats", ref=ref, source=res, source_md5=P.md5(res)),
                   P.result_block(eval_cases=LAND_CASES, target_set=["liver", "kidney"]))
        write_json(sp, sd)
        stats[ref] = sp
    log(f"land -> {res}; stats {sorted(stats)}")


def cmd_landdr():
    from usbg import provenance as P
    man = {"physics": None, **{st: tex_of(st, 0) for st in PLUG}}
    segs = {"physics": seg_of("physics", 0), **{st: seg_of(st, 0) for st in PLUG}}
    if SMOKE:
        segs = {st: seg_of(st, 0) for st in PLUG if os.path.exists(seg_of(st, 0))}
        man = {"physics": None, **{st: tex_of(st, 0) for st in segs}}
    for k, v in list(man.items()) + list(segs.items()):
        if v and not os.path.exists(v):
            die(f"{k} missing: {v}")
    mp = write_manifest("landscape_dr_manifest.json" if not SMOKE else "landscape_dr_manifest_smoke.json", man)
    d0 = os.path.join(OUT, "landscape")
    res = os.path.join(d0, "landscape_dr_plug.json")
    raw = os.path.join(d0, ".unstamped_landscape_dr_plug.json")
    no_clobber(res, os.path.splitext(res)[0] + ".png")
    os.makedirs(d0, exist_ok=True)
    for f in glob.glob(os.path.splitext(raw)[0] + ".*"):
        os.remove(f)
    tps, sps = (1, 4) if SMOKE else (5, 40)
    wall = run([PY, EXEC, os.path.join(HERE, "scripts", "view_match_landscape.py"), "--cases", *LAND_CASES,
                "--organs", "liver", "kidney", "--targets-per-case", tps, "--samples-per-target", sps,
                "--tex-manifest", mp, "--seg-ckpt", *[f"{k}={v}" for k, v in segs.items()], "--seg-seed", "0",
                "--dr", "off", "shared", "cross", "crossseed", "--weights", "0.6/0.4", "0.5/0.5",
                "--device", "cpu", "--out", raw],
               env={"USBG_LEARNED_SEG": "0", "USBG_SEAT": "anatomical"}, unset=("US_SEG_CKPT",))
    d = json.load(open(raw))
    blk = plug_block(stage="landscape_dr", manifest=mp, manifest_md5=P.md5(mp), seg_ckpts=segs,
                     seg_md5={k: P.md5(v) for k, v in segs.items()}, wall_sec=round(wall, 1),
                     recipe="v2 argv of outputs/v2/view_match/landscape_dr.json except the stages, --seg-ckpt and --out",
                     archived_v2=os.path.join(V2VM, "landscape_dr.json"))
    stamp_prov(d, blk)
    write_json(res, d)
    os.remove(raw)
    _move_siblings(raw, res)
    log(f"landdr -> {res}")


def cmd_real():
    from usbg import provenance as P
    man = {"physics": None, **{st: tex_of(st, 0) for st in PLUG}}
    for k, v in man.items():
        if v and not os.path.exists(v):
            die(f"texture {k} missing: {v}")
    mp = write_manifest("realism_manifest.json", man)
    d0 = os.path.join(OUT, "realism")
    res = os.path.join(d0, "realism_sector_plug.json")
    raw = os.path.join(d0, ".unstamped_realism_sector_plug.json")
    no_clobber(res)
    os.makedirs(d0, exist_ok=True)
    for f in glob.glob(os.path.splitext(raw)[0] + ".*"):
        os.remove(f)
    if SMOKE:
        args = ["--views-per-case", "4", "--dr", "off", "--erode", "3", "--renorm", "none", "--no-fid",
                "--c2st-repeats", "1", "--c2st-folds", "2", "--c2st-epochs", "1", "--c2st-size", "32",
                "--null-repeats", "1"]
    else:
        args = ["--views-per-case", "50", "--dr", "both", "--erode", "3", "--renorm", "both", "--kid-subset", "50",
                "--c2st-repeats", "5", "--c2st-folds", "5", "--c2st-epochs", "40", "--c2st-size", "64"]
    wall = run([PY, EXEC, os.path.join(HERE, "scripts", "eval_realism.py"), "--sector", "--cases", *LAND_CASES,
                "--tex-manifest", mp, *args, "--seed", "0", "--out", raw],
               env={"USBG_LEARNED_SEG": "0", "USBG_SEAT": "anatomical"}, unset=("US_SEG_CKPT",))
    d = json.load(open(raw))
    blk = plug_block(stage="realism", manifest=mp, manifest_md5=P.md5(mp), wall_sec=round(wall, 1),
                     recipe="v2 argv of outputs/v2/realism/realism_sector.json except --tex-manifest/--out",
                     archived_v2=V2REAL, real_frames="data/eval/kaggle_vitale_test.npz (60 public Vitale TEST frames)")
    stamp_prov(d, blk)
    write_json(res, d)
    os.remove(raw)
    _move_siblings(raw, res)
    log(f"real -> {res}")


# ------------------------------------------------------------------------------------------------ agg
METRICS = {"final_pos": lambda e: e["final_pos"], "keepbest_pos": lambda e: e["keepbest_pos"],
           "final_ang": lambda e: e["final_ang"], "final_match": lambda e: e["final_match"],
           "final_seg": lambda e: e["final_seg"],
           "success_5mm_5deg": lambda e: float(e["final_pos"] <= 5 and e["final_ang"] <= 5),
           "success_10mm_10deg": lambda e: float(e["final_pos"] <= 10 and e["final_ang"] <= 10)}


def _result(d, contact):
    rs = [r for r in d["results"] if r["contact"] == contact and r.get("eval_randomize", "off") == "off"
          and r["weights"] == "zero_residual"]
    return rs[0] if rs else None


def _per_patient(r, metric):
    by = collections.defaultdict(list)
    for e in r["episodes"]:
        by[e["patient"]].append(METRICS[metric](e))
    return {p: float(np.mean(v)) for p, v in by.items()}


def _sign_flip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    sims = [abs((d * np.asarray(sg)).mean()) for sg in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([x >= obs - 1e-12 for x in sims]))


def cmd_agg():
    from usbg import provenance as P
    inputs, cells = {}, {}

    def cell_path(arm, s, src):
        if src == "plug":
            return os.path.join(OUT, "base", f"ibvs_base_{arm}_matched_s{s}.json")
        if src == "v2":
            return os.path.join(V2LADDER, f"ibvs_base_{arm}_matched_s{s}.json")
        name = {"physics": "physics", "cut": "cut_v2budget", "cyclegan": "cyclegan_v2budget",
                "sonospade": "sonospade" if s == 0 else "sonospade_v2tex0"}[arm]
        return os.path.join(V21TEXBASE, f"ibvs_base_{name}_matched_s{s}.json")   # v2.1 re-runs of the v2 cells

    for src, arms in (("plug", PLUG), ("v2", V2_STAGES), ("v21rerun", V2_STAGES)):
        for arm in arms:
            for s in SEEDS:
                f = cell_path(arm, s, src)
                if os.path.exists(f):
                    cells[(src, arm, s)] = json.load(open(f)); inputs[f] = P.md5(f)
    patients = sorted({e["patient"] for d in cells.values() for r in d["results"] for e in r["episodes"]})

    def matrix(src, arm, metric="final_pos", contact="on"):
        rows, ss = [], []
        for s in SEEDS:
            d = cells.get((src, arm, s))
            r = _result(d, contact) if d else None
            if r is None:
                continue
            pp = _per_patient(r, metric)
            if set(pp) != set(patients):
                continue
            rows.append([pp[p] for p in patients]); ss.append(s)
        return (np.asarray(rows, float) if rows else None), ss

    def contrast(a, b, srcb="v2", metric="final_pos", contact="on", nboot=10000):
        XA, sa = matrix("plug", a, metric, contact); XB, sb = matrix(srcb, b, metric, contact)
        if XA is None or XB is None or sa != sb:
            return {"a": a, "b": f"{b} ({srcb})", "status": "pending", "seeds_a": sa, "seeds_b": sb}
        nP = len(patients)
        per_p = XA.mean(0) - XB.mean(0)                     # seed-averaged per-patient difference (paired seeds)
        m = float(per_p.mean())
        r2 = np.random.default_rng(0)
        bs = per_p[r2.integers(0, nP, size=(nboot, nP))].mean(1)
        lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
        agree = int((np.sign(per_p) == np.sign(m)).sum())
        return {"a": a, "b": f"{b} ({srcb})", "metric": metric, "contact": contact, "seeds": sa, "n_patients": nP,
                "mean_diff": m, "ci95_patient": [lo, hi], "sign_agree": agree,
                "sign_flip_p_exact": _sign_flip_p(per_p), "per_patient_diff": dict(zip(patients, per_p.round(4).tolist())),
                "holds_v2_rule": bool((lo > 0 or hi < 0) and agree >= 5)}

    contrasts = {}
    for metric, contact in (("final_pos", "on"), ("final_pos", "off"), ("keepbest_pos", "on"), ("final_ang", "on")):
        key = f"{metric}_contact_{contact}"
        contrasts[key] = [contrast(a, b, "v2", metric, contact) for a in PLUG for b in V2_STAGES]
        contrasts[key + "_vs_v21_reruns"] = [contrast(a, b, "v21rerun", metric, contact) for a in PLUG for b in V2_STAGES]

    arms = {}
    for src, names in (("plug", PLUG), ("v2", V2_STAGES)):
        for arm in names:
            row = {"source": src, "seeds": []}
            for metric in METRICS:
                X, ss = matrix(src, arm, metric)
                if X is None:
                    continue
                row["seeds"] = ss
                row[metric] = {"mean": float(X.mean()), "seed_sd": float(X.mean(1).std(ddof=1)) if len(ss) > 1 else None,
                               "per_seed": dict(zip([f"s{s}" for s in ss], X.mean(1).round(4).tolist())),
                               "per_patient": dict(zip(patients, X.mean(0).round(4).tolist()))}
            arms[f"{arm} ({src})"] = row

    def seg_eval(arm, s):
        f = os.path.join(SEGOUT if arm in PLUG else os.path.join(SEG, "outputs", "v2"), f"eval_{arm}_s{s}.json")
        if not os.path.exists(f):
            return None
        d = json.load(open(f)); inputs[f] = P.md5(f)
        g = lambda blk, c: ((d.get(blk) or {}).get("per_class") or {}).get(c)
        return {"file": f, "sim_val_mean": (d.get("sim_val") or {}).get("mean"), "sim_val_liver": g("sim_val", "liver"),
                "sim_val_kidney": g("sim_val", "kidney"), "vitale_holdout_liver": g("vitale_holdout_v2.npz", "liver"),
                "aul_liver": g("aul_liver_test.npz", "liver"), "kidneyus_kidney": g("kidneyus_holdout_test.npz", "kidney")}
    seg_dice = {f"{a}_s{s}": seg_eval(a, s) for a in list(PLUG) + list(V2_STAGES) for s in SEEDS}

    tex = {}
    for f in sorted(glob.glob(os.path.join(OUT, "tex", "*.json"))):
        t = json.load(open(f)); inputs[f] = P.md5(f)
        tex[os.path.basename(f)[:-5]] = {k: t.get(k) for k in ("ckpt", "md5", "iters", "wall_sec", "sec_per_iter",
                                                                "infer_cost_ms", "train_device")}
    segt = {}
    for f in sorted(glob.glob(os.path.join(OUT, "seg", "*.json"))):
        t = json.load(open(f)); inputs[f] = P.md5(f)
        segt[os.path.basename(f)[:-5]] = {"wall_sec": t.get("wall_sec"), "timing_sec": t.get("timing_sec")}

    def pv(res, arm, st, t):
        return (((res.get(arm) or {}).get(st) or {}).get(t) or {}).get("per_view_mean")

    land = {}
    lf = os.path.join(OUT, "landscape", "landscape_heldout_plug.json")
    if os.path.exists(lf):
        L = json.load(open(lf)); inputs[lf] = P.md5(lf)
        res = L.get("results") or {}
        arm0 = next(iter(res), None)
        land["app_per_view_mean"] = {st: pv(res, arm0, st, "app") for st in (res.get(arm0) or {})}
        a2 = os.path.join(V2VM, "landscape_heldout.json")
        if os.path.exists(a2):
            A = json.load(open(a2)).get("results") or {}
            a0 = next(iter(A), None)
            land["repeatability_vs_archived_v2"] = {st: {"archived": pv(A, a0, st, "app"), "rerun": pv(res, arm0, st, "app")}
                                                    for st in V2_STAGES}
        for f in sorted(glob.glob(os.path.join(OUT, "landscape", "table3_stats_ref_*.json"))):
            inputs[f] = P.md5(f)
            land.setdefault("stats_files", []).append(f)
    landdr = {}
    df = os.path.join(OUT, "landscape", "landscape_dr_plug.json")
    if os.path.exists(df):
        L = json.load(open(df)); inputs[df] = P.md5(df)
        res = L.get("results") or {}
        for st in (res.get("off") or {}):
            for mask in ("ct", "learned"):
                comb = f"combined_{mask}_0.6/0.4"
                off, cross = pv(res, "off", st, comb), pv(res, "cross", st, comb)
                ao, ac = pv(res, "off", st, "app"), pv(res, "cross", st, "app")
                if off is None or cross is None or ao is None or ac is None:
                    continue
                keep = cross / off if off else float("nan")
                landdr[f"{st}/{mask}"] = {"combined_retained": keep, "app_drops": bool(ac < ao),
                                          "rule_holds": bool(keep >= 0.8 and ac < ao)}
    realism = {}
    rf = os.path.join(OUT, "realism", "realism_sector_plug.json")
    if os.path.exists(rf):
        R = json.load(open(rf)); inputs[rf] = P.md5(rf)
        for k, v in (R.get("variants") or {}).items():
            realism[k] = {m: v.get(m) for m in ("w1_in_sector", "w1_per_case", "kid", "fid_secondary", "c2st_auc_mean", "c2st_auc_std", "edge_iou_in_sector")
                          if m in v}
        if os.path.exists(V2REAL):
            A = json.load(open(V2REAL)).get("variants") or {}
            realism["_repeatability_vs_archived_v2"] = {k: {"archived_w1": (A.get(k) or {}).get("w1_in_sector"),
                                                            "rerun_w1": (R["variants"].get(k) or {}).get("w1_in_sector")}
                                                        for k in ("prior LOTUS", "NumPy", "physics") if k in R["variants"]}
    from usbg import plug_texture as PT
    prov = P.result_block(train_cases=TRAIN13, eval_cases=TEST6, target_set=["liver", "kidney"],
                          extra={"inputs": inputs, "v21_plug": plug_block(stage="agg")})
    prov["analysis_status"] = STATUS
    out = {"experiment": "v2_1/plug/texture: published unpaired translators in the texture slot",
           "analysis_status": STATUS,
           "note": ("Exploratory v2.1 pluggability rows; archived v2 numbers stay primary. Plug stages use texture seed 0 "
                    "(v2 recipe) with segmenter and calibration seeds 0-2, the design of the archived v2 rows they are "
                    "compared with."),
           "methods": PT.METHODS, "excluded": PT.EXCLUDED,
           "unit": "patient (n = 6); per-patient mean over that patient's crossed episodes; DR off, 12 steps",
           "bootstrap": "v2 method: patient-cluster bootstrap of the seed-averaged per-patient differences, 10,000, seed 0",
           "rule": "v2 rule: 95% CI excludes 0 and at least 5 of 6 patients share the sign",
           "patients": patients, "arms": arms, "contrasts": contrasts, "segmenter_dice": seg_dice,
           "landscape": land, "landscape_dr_verdict": landdr, "realism": realism,
           "texture_training": tex, "segmenter_training": segt, "provenance": prov}
    res = os.path.join(OUT, "plug_texture_summary.json")
    write_json(res, out)
    log(f"agg -> {res} (cells {len(cells)}, patients {patients})")
    for c in contrasts["final_pos_contact_on"]:
        if c.get("status") == "pending":
            continue
        log(f"  {c['a']:10s} - {c['b']:16s}: {c['mean_diff']:+.2f} mm CI [{c['ci95_patient'][0]:+.2f}, "
            f"{c['ci95_patient'][1]:+.2f}] sign {c['sign_agree']}/{c['n_patients']}")


# ----------------------------------------------------------------------------------------------- jobs
# est_h (2026-10-15, idle M5 Pro, scaled to the full budgets, then x1.5 for MPS contention with K=3 runner slots):
#   tex   150-it sanity runs on MPS: FastCUT 0.05 s/it, DCLGAN 0.15 s/it, UNSB 0.19 s/it -> 800 it 0.7 / 2 / 2.5 min,
#         plus loading and the 30-frame CPU+MPS timing; histmatch is one quantile fit (seconds)
#   seg   the v2.1 texture estimate (build 1-2 min + sim-eval build 0.5 min + 4000 it at 4.3-10.6 it/s + eval_seg);
#         translation adds at most about 1000 frames x 21 ms (UNSB on MPS)
#   base  v2 matched cell wall 60 s (CUT, contact on and off); UNSB adds about 2700 frames x 18 ms
#   land  v2 landscape_heldout 777 s for 4 stages on CPU; + FastCUT, DCLGAN 2460 frames x 28 ms each, UNSB x 140 ms
#   landdr v2 landscape_dr 2961 s for 4 stages x 4 arms; this run has physics + 4 plug stages (UNSB 4 x 344 s)
#   real  v2 realism_sector 981 s (C2ST 69 s per variant); 14 variants here instead of 10
EST = {"tex": {"fastcut": 0.05, "dclgan": 0.08, "unsb": 0.12, "histmatch": 0.02},
       "seg": {"fastcut": 0.5, "dclgan": 0.5, "unsb": 0.5, "histmatch": 0.5},
       "base": {"fastcut": 0.06, "dclgan": 0.06, "unsb": 0.08, "histmatch": 0.06},
       "land": 0.6, "landdr": 1.7, "real": 0.6, "agg": 0.02}


def cmd_jobs(path=None):
    path = path or os.path.join(HERE, "outputs", "v2_1", "jobs", "plug_texture.json")
    if os.path.exists(path) and not SMOKE and os.environ.get("PLUG_TEXTURE_OVERWRITE") != "1":
        die(f"refusing to overwrite {path} (PLUG_TEXTURE_OVERWRITE=1)")
    base_env = {"PYTHONPATH": os.path.join(HERE, "src"), "USBG_SEAT": "anatomical", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "USBG_GYM_ROOT": HERE, "USBG_LEARNED_SEG": "0", "US_SEG_CKPT": "",
                "USBG_WINDOWS": ""}                       # "" = removed by the runner: always the v1 windows file
    expect = {"seat": "anatomical", "windows_md5": WINDOWS_MD5_V2}
    jobs = []

    def job(name, args, out, est, deps=(), files=(), env=None):
        jobs.append({"name": f"v21:plug:texture:{name}", "cmd": [PY, SELF, *[str(a) for a in args]], "out": out,
                     "est_h": est, "env": {**base_env, **(env or {})},
                     "deps": [f"v21:plug:texture:{d}" for d in deps], "files": list(files) + [WINDOWS],
                     "expect": expect, "cwd": HERE})

    for st in PLUG:
        job(f"tex:{st}:s0", ["tex", st, 0], os.path.join(OUT, "tex", f"{st}_s0.json"), EST["tex"][st],
            files=[PAIRS_V2, os.path.join(HERE, POOL)])
    for st in PLUG:
        for s in SEEDS:
            job(f"seg:{st}:s{s}", ["seg", st, s], os.path.join(OUT, "seg", f"{st}_s{s}.json"), EST["seg"][st],
                deps=[f"tex:{st}:s0"], files=[tex_of(st, 0)])
    for st in PLUG:
        for s in SEEDS:
            job(f"base:{st}:s{s}", ["base", st, s], os.path.join(OUT, "base", f"ibvs_base_{st}_matched_s{s}.json"),
                EST["base"][st], deps=[f"seg:{st}:s{s}"], files=[tex_of(st, 0), seg_of(st, s)],
                env={"USBG_LEARNED_SEG": "1", "US_SEG_CKPT": seg_of(st, s)})
    v2tex = [tex_of(st) for st in V2_STAGES if st != "physics"]
    job("land", ["land"], os.path.join(OUT, "landscape", "landscape_heldout_plug.json"), EST["land"],
        deps=[f"tex:{st}:s0" for st in PLUG], files=v2tex + [tex_of(st, 0) for st in PLUG])
    job("landdr", ["landdr"], os.path.join(OUT, "landscape", "landscape_dr_plug.json"), EST["landdr"],
        deps=[f"seg:{st}:s0" for st in PLUG],
        files=[tex_of(st, 0) for st in PLUG] + [seg_of(st, 0) for st in PLUG] + [seg_of("physics", 0)])
    job("real", ["real"], os.path.join(OUT, "realism", "realism_sector_plug.json"), EST["real"],
        deps=[f"tex:{st}:s0" for st in PLUG], files=[tex_of(st, 0) for st in PLUG])
    job("agg", ["agg"], os.path.join(OUT, "plug_texture_summary.json"), EST["agg"],
        deps=[j["name"].split("v21:plug:texture:")[1] for j in jobs])
    write_json(path, jobs)
    tot = sum(j["est_h"] for j in jobs)
    log(f"wrote {path}: {len(jobs)} jobs, {tot:.2f} job-h")
    return path


# ---------------------------------------------------------------------------------------------- smoke
def cmd_smoke():
    if not SMOKE:
        env = {**os.environ, "PLUG_SMOKE": "1", "PLUG_SMOKE_DIR": SMOKE_DIR}
        raise SystemExit(subprocess.call([sys.executable, SELF, "smoke"], env=env))
    os.makedirs(SMOKE_DIR, exist_ok=True)
    timing = {}
    which = os.environ.get("SMOKE_STEPS", "tex,seg,base,land,real,agg,jobs").split(",")
    if "tex" in which:
        for st in PLUG:
            t0 = time.time(); cmd_tex(st, 0); timing[f"tex_{st}"] = round(time.time() - t0, 1)
    seg_stage = os.environ.get("SMOKE_SEG_STAGE", "unsb")
    if "seg" in which:
        t0 = time.time(); cmd_seg(seg_stage, 0); timing[f"seg_{seg_stage}"] = round(time.time() - t0, 1)
    if "base" in which:
        t0 = time.time(); cmd_base(seg_stage, 0); timing[f"base_{seg_stage}"] = round(time.time() - t0, 1)
    if "land" in which:
        t0 = time.time(); cmd_land(); timing["land"] = round(time.time() - t0, 1)
    if "landdr" in which:
        t0 = time.time(); cmd_landdr(); timing["landdr"] = round(time.time() - t0, 1)
    if "real" in which:
        t0 = time.time(); cmd_real(); timing["real"] = round(time.time() - t0, 1)
    if "agg" in which:
        cmd_agg()
    if "jobs" in which:
        cmd_jobs(os.path.join(SMOKE_DIR, "jobs_plug_texture.json"))
    write_json(os.path.join(SMOKE_DIR, f"smoke_timing_{int(time.time())}.json"), timing)
    log(f"smoke OK {timing}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__); return 2
    if argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    sub, rest = argv[0], argv[1:]
    if sub == "tex":
        if len(rest) != 2: die("usage: tex <stage> <seed>")
        guards(); cmd_tex(rest[0], int(rest[1]))
    elif sub == "seg":
        if len(rest) != 2: die("usage: seg <stage> <seed>")
        guards(); cmd_seg(rest[0], int(rest[1]))
    elif sub == "base":
        if len(rest) != 2: die("usage: base <stage> <seed>")
        guards(); cmd_base(rest[0], int(rest[1]))
    elif sub == "land":
        guards(); cmd_land()
    elif sub == "landdr":
        guards(); cmd_landdr()
    elif sub == "real":
        guards(); cmd_real()
    elif sub == "_cost":
        cmd_cost(rest[0], rest[1], int(rest[2]))
    elif sub == "agg":
        cmd_agg()
    elif sub == "jobs":
        cmd_jobs(rest[0] if rest else None)
    elif sub == "smoke":
        os.environ.setdefault("USBG_SEAT", "anatomical")
        if SMOKE:
            guards()
        cmd_smoke()
    else:
        print(__doc__); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
