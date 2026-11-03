#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): write the job specs of the mask-source / segmenter slot.

  PYTHONPATH=src .venv/bin/python scripts/plug/make_plug_seg_jobs.py            # writes both files
  outputs/v2_1/jobs/plug_seg.json       core: every method, the paper's matched design
  outputs/v2_1/jobs/plug_seg_ext.json   optional: the 'vitale' OOF side, SAM-family seeds 1-2, nnU-Net 20 epochs

Each job is {name, cmd, out, est_h, env, deps, files, expect, cwd} like outputs/v2_1/jobs/*.json and runs
with scripts/run_table4_v2.py's runner. Nothing is launched here.

plug_seg_ext.json depends on job names that live in plug_seg.json (the train and nnraw jobs). The runner
takes ONE --extra-jobs file and ignores a dependency it cannot see, so either run the ext file only after
the core file has finished (its jobs then find their checkpoints on disk), or concatenate the two lists
into one file and run that.

Per method, stage (sonospade, physics, cut, cyclegan) and seed (0, 1, 2), the core design is
  train    the segmenter on the stage-and-seed v2 sim build with the v2 recipe (learned methods only)
  segeval  its own sim Dice (TEST-patient sim-eval build) and real Dice / consistency (Vitale holdout,
           KidneyUS holdout, AUL), with the unchanged v2 scripts
  ladder   the learned-mask servo base rung (ibvs_base, recalibrated on TRAIN with this mask source,
           calib seed = seed), contact on and off, the crossed held-out design (6 patients, seed 999,
           60 episodes), extended logging
  contact  the final v2 matched policy of that stage and seed evaluated with this mask source (contact
           on and off, --allow-mismatch: the policy keeps its saved servo base)
Methods: lraspp_mbv3, deeplabv3_r50, deeplabv3_mbv3, fcn_r50 (torchvision, from scratch), nnunet2d
(nnU-Net v2 2D, sample-budget schedule), and the SAM family at seed 0 in core (seeds 1-2 in ext):
sam_priv / medsam_priv (PRIVILEGED: box prompts from the CT label in the gym, from the ground truth in
segeval) and sam_unet / medsam_unet (non-privileged: box prompts from the stage-and-seed v2 U-Net mask).
SAM-family contact rungs evaluate the final weights only (the per-frame cost is ~0.6 s).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

GYM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEG = os.environ.get("USBG_SEG_REPO", os.path.join(GYM, "segtrain"))
PY = f"{GYM}/.venv/bin/python"
PPY = os.environ.get("USBG_PLUGIN_PYTHON", PY)
W = os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights"))
SAMW = f"{W}/sam/sam_vit_b_01ec64.pth"
MEDSAMW = f"{W}/medsam/medsam_vit_b.pth"
WEIGHT_MD5 = {SAMW: "01ec64d29a2fca3f0661936605ae66f8", MEDSAMW: "3bb6db55bd0c9ca30b61248bca72f8d6"}
WINDOWS = f"{GYM}/data/cache/windows/usbg-windows-v1.json"
TRAIN = ["s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0522", "s0703", "s0720", "s0727", "s0777",
         "s1086", "s1373"]
EVAL = ["s0358", "s0513", "s0519", "s0591", "s0649", "s0762"]
STAGES = ["sonospade", "physics", "cut", "cyclegan"]
BASE = {"sonospade": 1, "physics": 2, "cut": 3, "cyclegan": 4}      # run_seg_v2.sh dataset-id base
SEEDS = [0, 1, 2]
PLUG_OUT = f"{GYM}/outputs/v2_1/plug/seg"
SEG_OUT = f"{SEG}/outputs/v2_1/plug/seg"
NN_RAW, NN_PRE, NN_RES = (f"{SEG}/outputs/v2_1/plug/nnunet_raw", f"{SEG}/outputs/v2_1/plug/nnunet_preprocessed",
                          f"{SEG}/outputs/v2_1/plug/nnunet_results")
EXT_TRAINERS = f"{SEG}/src/us_seg/nnunet_plug_trainers"
SPLIT = f"{SEG}/data/real/vitale_split_v2.npz"
ANALYSIS_STATUS = "exploratory, v2.1 pluggability"

# measured costs (M5 Pro, MPS; TRAIN smoke tests). seg_ms = per-frame mask cost in
# the gym loop; train_h / cv_h = one training (4000 it) / one 6-fold crossval (6 x 2500 it), with slack.
TV = {"lraspp_mbv3": dict(seg_ms=7.0, train_h=0.06, cv_h=0.15),
      "deeplabv3_r50": dict(seg_ms=11.0, train_h=0.25, cv_h=0.85),
      "deeplabv3_mbv3": dict(seg_ms=7.5, train_h=0.06, cv_h=0.16),
      "fcn_r50": dict(seg_ms=9.0, train_h=0.17, cv_h=0.55)}
NN_SEG_MS = 75.0
SAMF = {"sam_priv": dict(kind="sam", w=SAMW, prompt="ct", seg_ms=620.0),
        "medsam_priv": dict(kind="medsam", w=MEDSAMW, prompt="ct", seg_ms=640.0),
        "sam_unet": dict(kind="sam", w=SAMW, prompt="seg", seg_ms=630.0),
        "medsam_unet": dict(kind="medsam", w=MEDSAMW, prompt="seg", seg_ms=700.0)}
# the existing v2.1 matched-mask estimates (outputs/v2_1/jobs/rungs.json) per stage: (ibvs_base, contact)
BASE_EST = {"sonospade": (0.045, 0.057), "physics": (0.034, 0.044), "cut": (0.038, 0.049), "cyclegan": (0.038, 0.049)}
FRAMES = {"ladder": 2680, "contact": 3360, "contact_final": 1680, "segeval": 1205}
UNET_MS = 4.0
SLACK = 1.2


def md5(p):
    sys.path.insert(0, f"{GYM}/src")
    from usbg.provenance import md5 as _m
    return _m(p)


def tex(t):
    return "none" if t == "physics" else f"{GYM}/data/dataset/v2/{t}_curvi.pt"


def gym_env(nn=False):
    e = {"PYTHONPATH": f"{GYM}/src", "USBG_SEAT": "anatomical", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
         "USBG_LEARNED_SEG": "1", "US_SEG_CKPT": "", "USBG_WINDOWS": ""}
    return dict(e, **NN_ENV) if nn else e


def seg_env(nn=False):
    e = {"PYTHONPATH": f"{GYM}/src", "USBG_GYM_ROOT": GYM, "USBG_SEAT": "anatomical",
         "PYTORCH_ENABLE_MPS_FALLBACK": "1", "USBG_LEARNED_SEG": "0", "US_SEG_CKPT": "", "USBG_WINDOWS": ""}
    return dict(e, **NN_ENV) if nn else e


NN_ENV = {"nnUNet_raw": NN_RAW, "nnUNet_preprocessed": NN_PRE, "nnUNet_results": NN_RES,
          "nnUNet_n_proc_DA": "6", "nnUNet_extTrainer": EXT_TRAINERS}


def job(name, cmd, out, est_h, env, deps=(), files=(), expect=None):
    return {"name": name, "cmd": cmd, "out": out, "est_h": round(float(est_h), 3), "env": env, "deps": list(deps),
            "files": list(files), "expect": expect or {}, "cwd": GYM}


def rung_est(t, kind, seg_ms, frames_key, load_s=5.0):
    base = BASE_EST[t][0 if kind == "ladder" else 1]
    if frames_key == "contact_final":
        base = base / 2.0
    return SLACK * (base + FRAMES[frames_key] * (seg_ms - UNET_MS) / 3.6e6 + load_s / 3600.0)


class Spec:
    def __init__(self):
        self.win_md5 = md5(WINDOWS)
        self.tex_md5 = {t: (None if t == "physics" else md5(tex(t))) for t in STAGES}

    def expect_gym(self, t, seg_md5=None):
        e = {"seat": "anatomical", "texture_md5": self.tex_md5[t], "windows_md5": self.win_md5}
        if seg_md5:
            e["seg_md5"] = seg_md5
        return e

    def expect_seg(self):
        return {"seat": "anatomical", "windows_md5": self.win_md5}

    def tex_files(self, t):
        return [WINDOWS] + ([] if t == "physics" else [tex(t)])

    # ---------------------------------------------------------------- gym rungs
    def ladder(self, key, t, s, spec, seg_ms, deps, files, nn=False, seg_md5=None, tier="core"):
        out = f"{PLUG_OUT}/ladder/ibvs_base_{t}_{key}_s{s}.json"
        cmd = [PPY if nn else PY, f"{GYM}/scripts/plug/eval_plug_mask.py", "--mask-plug", spec, "--method-key", key,
               "--", "--rung", "ibvs_base", "--translator", t, "--texture-ckpt", tex(t),
               "--train-cases", *TRAIN, "--eval-cases", *EVAL, "--targets", "liver", "kidney", "--min-windows", "2",
               "--eval-episodes", "60", "--eval-seed", "999", "--eval-design", "crossed", "--calib-seed", str(s),
               "--contact", "on", "off", "--ext-log", "--analysis-status", ANALYSIS_STATUS, "--out", out]
        return job(f"v21:plug:seg:ladder:{key}:{t}:s{s}", cmd, out, rung_est(t, "ladder", seg_ms, "ladder"),
                   gym_env(nn), deps, self.tex_files(t) + files, self.expect_gym(t, seg_md5))

    def contact(self, key, t, s, spec, seg_ms, deps, files, nn=False, seg_md5=None, final_only=False):
        out = f"{PLUG_OUT}/contact/{key}_{t}_s{s}.json"
        pj = f"{GYM}/outputs/v2/table4/matched/{t}_deploy_s{s}.json"
        w = ["final"] if final_only else ["imit", "final"]
        cmd = [PPY if nn else PY, f"{GYM}/scripts/plug/eval_plug_mask.py", "--mask-plug", spec, "--method-key", key,
               "--", "--rung", "policy", "--policy-json", pj, "--weights", *w, "--contact", "on", "off",
               "--allow-mismatch", "--ext-log", "--analysis-status", ANALYSIS_STATUS, "--out", out]
        fk = "contact_final" if final_only else "contact"
        return job(f"v21:plug:seg:contact:{key}:{t}:s{s}", cmd, out, rung_est(t, "contact", seg_ms, fk),
                   gym_env(nn), deps, self.tex_files(t) + files + [pj, pj[:-5] + ".pt"] +
                   ([] if final_only else [pj[:-5] + "_imit.pt"]), self.expect_gym(t, seg_md5))

    def segeval(self, key, t, s, spec, seg_ms, deps, files, nn=False, train_build=None, sources=None, oof=None):
        b = BASE[t]
        eds = f"{SEG}/data/nnunet_v2/Dataset{200 + b * 10 + s}_USOrgansV2eval_{t}_s{s}"
        od = f"{SEG_OUT}/{key}"
        srcs = sources or ["vitaleho", "kidneyus", "aul"]
        cmd = [PPY if nn else PY, f"{SEG}/scripts/plug/eval_plug_seg.py", "--mask-plug", spec, "--method-key", key,
               "--stage", t, "--seed", str(s), "--dataset", eds, "--out-dir", od, "--sources", *srcs]
        if train_build:
            cmd += ["--train-build", train_build]
        if oof:
            cmd += ["--oof", oof, "--skip-eval-seg"]
            out = f"{od}/segeval_{t}_s{s}_{'-'.join(srcs)}.json"
            name, est = f"v21:plug:seg:segeval_vitale:{key}:{t}:s{s}", 0.02
        else:
            out = f"{od}/segeval_{t}_s{s}.json"
            name, est = f"v21:plug:seg:segeval:{key}:{t}:s{s}", SLACK * (FRAMES["segeval"] * seg_ms / 3.6e6 + 60 / 3600)
        return job(name, cmd, out, est, seg_env(nn), deps, [eds] + files + ([oof] if oof else []), self.expect_seg())


def build(spec_obj):
    core, ext = [], []
    S = spec_obj
    # ---------------------------------------------------------------- torchvision segmenters
    for arch, c in TV.items():
        for t in STAGES:
            for s in SEEDS:
                b = BASE[t]
                ds = f"{SEG}/data/nnunet_v2/Dataset{100 + b * 10 + s}_USOrgansV2_{t}_s{s}"
                ck = f"{SEG_OUT}/{arch}/us_seg_{arch}_{t}_s{s}.pt"
                tr = f"v21:plug:seg:train:{arch}:{t}:s{s}"
                cmd = [PY, f"{SEG}/scripts/plug/train_plug_seg.py", "train", "--arch", arch, "--dataset", ds,
                       "--seed", str(s), "--kidneyus", "--vitale-split", SPLIT, "--forbid-test-patients",
                       "--iters", "4000", "--out", ck]
                core.append(job(tr, cmd, ck[:-3] + "_history.json", c["train_h"], seg_env(), (), [ds, SPLIT],
                                S.expect_seg()))
                spec = f"kind=tv,ckpt={ck}"
                core.append(S.segeval(arch, t, s, spec, c["seg_ms"], [tr], [ck], train_build=ds))
                core.append(S.ladder(arch, t, s, spec, c["seg_ms"], [tr], [ck]))
                core.append(S.contact(arch, t, s, spec, c["seg_ms"], [tr], [ck]))
                # ext: the 'vitale' real side (6-fold OOF + deploy on the holdout), then its consistency
                cv = f"{SEG_OUT}/{arch}/crossval_{arch}_{t}_s{s}.json"
                oof = f"{SEG_OUT}/{arch}/real_oof_{arch}_{t}_s{s}.npz"
                cvn = f"v21:plug:seg:cv:{arch}:{t}:s{s}"
                ccmd = [PY, f"{SEG}/scripts/plug/train_plug_seg.py", "crossval", "--arch", arch, "--dataset", ds,
                        "--seed", str(s), "--kidneyus", "--vitale-split", SPLIT, "--forbid-test-patients",
                        "--folds", "6", "--iters", "2500", "--deploy-ckpt", ck, "--out", cv, "--oof-out", oof]
                ext.append(job(cvn, ccmd, cv, c["cv_h"], seg_env(), [tr], [ds, SPLIT, ck], S.expect_seg()))
                ext.append(S.segeval(arch, t, s, spec, c["seg_ms"], [cvn], [ck], train_build=ds, sources=["vitale"],
                                     oof=oof))
    # ---------------------------------------------------------------- nnU-Net v2 2D
    for trainer, key, tier, seeds in (("nnUNetTrainer_USBGbudget", "nnunet2d", core, SEEDS),
                                      ("nnUNetTrainer_20epochs", "nnunet2d_e20", ext, [0])):
        for t in STAGES:
            for s in seeds:
                b = BASE[t]
                did = 500 + b * 10 + s
                dname = f"Dataset{did}_USOrgansV2plug_{t}_s{s}"
                ds = f"{SEG}/data/nnunet_v2/Dataset{100 + b * 10 + s}_USOrgansV2_{t}_s{s}"
                rawn = f"v21:plug:seg:nnraw:{t}:s{s}"
                if key == "nnunet2d":
                    cmd = [PPY, f"{SEG}/scripts/plug/build_nnunet_plug_dataset.py", "--src-dataset", ds,
                           "--out-raw", NN_RAW, "--dataset-id", str(did), "--name", f"USOrgansV2plug_{t}_s{s}",
                           "--seed", str(s), "--vitale-split", SPLIT, "--kidneyus", "--forbid-test-patients"]
                    core.append(job(rawn, cmd, f"{NN_RAW}/{dname}/usbg_split.json", 0.01, seg_env(True), (),
                                    [ds, SPLIT], S.expect_seg()))
                model = f"{NN_RES}/{dname}/{trainer}__nnUNetPlans__2d"
                ck = f"{model}/fold_0/checkpoint_final.pth"
                trn = f"v21:plug:seg:train:{key}:{t}:s{s}"
                deps = [rawn] if key == "nnunet2d" else [rawn, f"v21:plug:seg:train:nnunet2d:{t}:s{s}"]
                cmd = [PPY, f"{SEG}/scripts/plug/train_nnunet_plug.py", "--dataset-id", str(did), "--raw", NN_RAW,
                       "--preprocessed", NN_PRE, "--results", NN_RES, "--trainer", trainer, "--device", "mps",
                       "--np", "4", "--n-proc-da", "6", "--out-json", f"{SEG_OUT}/{key}/train_{t}_s{s}.json"]
                tier.append(job(trn, cmd, f"{SEG_OUT}/{key}/train_{t}_s{s}.json",
                                0.2 if key == "nnunet2d" else 1.3, seg_env(True), deps,
                                [f"{NN_RAW}/{dname}/usbg_split.json"], S.expect_seg()))
                spec = f"kind=nnunet,model={model},fold=0"
                tier.append(S.segeval(key, t, s, spec, NN_SEG_MS, [trn], [ck], nn=True, train_build=ds))
                tier.append(S.ladder(key, t, s, spec, NN_SEG_MS, [trn], [ck], nn=True))
                tier.append(S.contact(key, t, s, spec, NN_SEG_MS, [trn], [ck], nn=True))
    # ---------------------------------------------------------------- SAM / MedSAM box prompts
    for key, c in SAMF.items():
        for t in STAGES:
            for s in SEEDS:
                tier = core if s == 0 else ext
                files = [c["w"]]
                gym_spec = f"kind={c['kind']},weights={c['w']},prompt={c['prompt']}"
                off_spec = f"kind={c['kind']},weights={c['w']},prompt=gt"
                if c["prompt"] == "seg":
                    pu = f"{SEG}/outputs/v2/us_seg_{t}_s{s}.pt"
                    files = files + [pu]
                    gym_spec = off_spec = f"kind={c['kind']},weights={c['w']},prompt=seg,prompt_ckpt={pu}"
                tier.append(S.segeval(key, t, s, off_spec, c["seg_ms"], (), files))
                tier.append(S.ladder(key, t, s, gym_spec, c["seg_ms"], (), files, seg_md5=WEIGHT_MD5[c["w"]]))
                tier.append(S.contact(key, t, s, gym_spec, c["seg_ms"], (), files, seg_md5=WEIGHT_MD5[c["w"]],
                                      final_only=True))
    return core, ext


def summarize(jobs):
    from collections import defaultdict
    g = defaultdict(lambda: [0, 0.0])
    for j in jobs:
        parts = j["name"].split(":")
        k = (parts[3], parts[4])
        g[k][0] += 1
        g[k][1] += j["est_h"]
    return {f"{a}:{b}": {"n": n, "est_h": round(h, 2)} for (a, b), (n, h) in sorted(g.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{GYM}/outputs/v2_1/jobs")
    ap.add_argument("--replace-own", action="store_true", help="replace plug_seg*.json written by this script")
    args = ap.parse_args(argv)
    core, ext = build(Spec())
    names = [j["name"] for j in core + ext]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate job names")
    outs = [j["out"] for j in core + ext]
    if len(outs) != len(set(outs)):
        raise SystemExit("duplicate outputs")
    have = [j["out"] for j in core + ext if os.path.exists(j["out"])]
    if have:
        print(f"note: {len(have)} outputs already exist (the runner reuses them only if provenance matches)")
    for fn, jobs in (("plug_seg.json", core), ("plug_seg_ext.json", ext)):
        p = os.path.join(args.out_dir, fn)
        if os.path.exists(p):
            old = json.load(open(p))
            if not args.replace_own or not all(j["name"].startswith("v21:plug:seg:") for j in old):
                raise SystemExit(f"{p} exists (pass --replace-own to regenerate this script's own spec)")
        with open(p + ".tmp", "w") as fh:
            json.dump(jobs, fh, indent=1)
        os.replace(p + ".tmp", p)
        tot = sum(j["est_h"] for j in jobs)
        print(f"wrote {p}: {len(jobs)} jobs, {tot:.1f} job-h")
        print(json.dumps(summarize(jobs), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
