#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): write the job spec of the SAMUS mask source (usbg.plug.samus_mask).

  PYTHONPATH=src .venv/bin/python scripts/plug/make_plug_samus_jobs.py     # writes outputs/v2_1/jobs/plug_samus.json

The SAMUS rungs mirror the SAM / MedSAM rungs of scripts/plug/make_plug_seg_jobs.py exactly: this script
imports that generator and calls its own Spec.segeval / Spec.ladder / Spec.contact builders (that file is not
changed), so the stages (sonospade, physics, cut, cyclegan), the seeds (0, 1, 2), the commands, the crossed
held-out design (6 held-out patients, eval seed 999, 60 crossed episodes, contact on and off, --ext-log),
the environments, the cost model and the expect blocks are the SAM-family ones. Only the method differs:
  samus_priv  PRIVILEGED: box prompts from the CT label in the gym (prompt=ct), from the ground truth in segeval
              (prompt=gt), as sam_priv / medsam_priv
  samus_unet  non-privileged: box prompts from the stage-and-seed v2 U-Net mask (prompt=seg), as sam_unet / medsam_unet
Per method, stage and seed: segeval (sim Dice on the TEST-patient sim-eval build; real Dice and consistency on
the Vitale holdout, KidneyUS holdout and AUL), ladder (ibvs_base recalibrated on TRAIN with this mask source,
calib seed = seed) and contact (the final v2 matched policy of that stage and seed under this mask, final
weights only, --allow-mismatch). Seed 0 is tier "core" and seeds 1-2 tier "ext", as the SAM family's split
between plug_seg.json and plug_seg_ext.json.

Extra fields (the runner reads "priority" and ignores the rest): "tier", "priority" (seed 0 is 1 =
survey "should"; seeds 1-2 are 0, as the SAM-family ext jobs), "note". Every SAMUS job depends on the SAMUS
admission gate job v21:checks:gate:mask:samus (scripts/plug/admission_gate_samus.py), which this file also holds.

Costs measured in TRAIN smoke tests (M5 Pro, MPS, torch 2.12.1): CT-box prompts
124 ms median per frame (1.1 boxes per frame), U-Net-box prompts 170 ms median (3.2 boxes per frame, 1014
calls in a full ibvs_base calibration of 219 s), CPU 0.66-0.9 s. seg_ms below adds headroom for more boxes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GYM = os.path.dirname(os.path.dirname(HERE))


def _load_seg_generator():
    spec = importlib.util.spec_from_file_location("_usbg_make_plug_seg_jobs", os.path.join(HERE, "make_plug_seg_jobs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MS = _load_seg_generator()
SAMUSW = f"{MS.W}/samus/SAMUS.pth"
SAMUS_MD5 = "ccf9b05497cae4300a772f9c17995f07"          # md5 of SAMUS.pth (checked against the file below)
SAMUS_CODE = f"{MS.W}/samus_code"
SAMUSF = {"samus_priv": dict(prompt="ct", seg_ms=150.0, note="SAMUS, CT-label boxes (PRIVILEGED); survey: should"),
          "samus_unet": dict(prompt="seg", seg_ms=185.0, note="SAMUS, U-Net boxes (non-privileged); survey: should")}
GATE = "v21:checks:gate:mask:samus"
GATE_OUT = f"{GYM}/outputs/v2_1/checks/gate/mask_samus.json"
OUT = f"{GYM}/outputs/v2_1/jobs/plug_samus.json"


def gate_job(S):
    pu = f"{MS.SEG}/outputs/v2/us_seg_sonospade_s0.pt"
    env = {"PYTHONPATH": f"{GYM}/src", "USBG_SEAT": "anatomical", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
           "USBG_LEARNED_SEG": "0", "US_SEG_CKPT": "", "USBG_WINDOWS": "", "CUDA_VISIBLE_DEVICES": "-1"}
    j = MS.job(GATE, [MS.PY, f"{GYM}/scripts/plug/admission_gate_samus.py", "--out", GATE_OUT], GATE_OUT, 0.03, env,
               (), [MS.WINDOWS, SAMUSW, pu], S.expect_seg())
    j.update(tier="core", priority=3, note="SAMUS admission gate (same protocol as mask:sam): privileged-channel test, "
             "pretraining corpus record, CUDA-free and CPU / MPS parity; out exists only if admitted; measured 45 s")
    return j


def build(S):
    jobs = [gate_job(S)]
    for key, c in SAMUSF.items():
        for t in MS.STAGES:
            for s in MS.SEEDS:
                files = [SAMUSW]
                gym_spec = f"kind=samus,weights={SAMUSW},prompt={c['prompt']}"
                off_spec = f"kind=samus,weights={SAMUSW},prompt=gt"
                if c["prompt"] == "seg":
                    pu = f"{MS.SEG}/outputs/v2/us_seg_{t}_s{s}.pt"
                    files = files + [pu]
                    gym_spec = off_spec = f"kind=samus,weights={SAMUSW},prompt=seg,prompt_ckpt={pu}"
                trio = [S.segeval(key, t, s, off_spec, c["seg_ms"], [GATE], files),
                        S.ladder(key, t, s, gym_spec, c["seg_ms"], [GATE], files, seg_md5=SAMUS_MD5),
                        S.contact(key, t, s, gym_spec, c["seg_ms"], [GATE], files, seg_md5=SAMUS_MD5, final_only=True)]
                for j in trio:
                    j.update(tier="core" if s == 0 else "ext", priority=1 if s == 0 else 0,
                             note=c["note"] + ("" if s == 0 else "; seeds 1-2 as the SAM-family ext jobs"))
                jobs += trio
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    got = MS.md5(SAMUSW)
    if got != SAMUS_MD5:
        raise SystemExit(f"{SAMUSW}: md5 {got} != {SAMUS_MD5}")
    jobs = build(MS.Spec())
    names = [j["name"] for j in jobs]
    outs = [j["out"] for j in jobs]
    if len(names) != len(set(names)) or len(outs) != len(set(outs)):
        raise SystemExit("duplicate job names or outputs")
    have = [o for o in outs if os.path.exists(o)]
    if have:
        raise SystemExit(f"{len(have)} outputs already exist, e.g. {have[0]}")
    other = []
    for fn in sorted(os.listdir(os.path.dirname(args.out))):
        if fn.endswith(".json") and os.path.join(os.path.dirname(args.out), fn) != os.path.abspath(args.out):
            try:
                other += [j["name"] for j in json.load(open(os.path.join(os.path.dirname(args.out), fn)))
                          if isinstance(j, dict) and "name" in j]
            except (ValueError, TypeError):
                pass
    clash = sorted(set(names) & set(other))
    if clash:
        raise SystemExit(f"job names already used by another spec: {clash[:5]}")
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists (never overwritten)")
    with open(args.out + ".tmp", "w") as fh:
        json.dump(jobs, fh, indent=1)
    os.replace(args.out + ".tmp", args.out)
    tot = sum(j["est_h"] for j in jobs)
    print(f"wrote {args.out}: {len(jobs)} jobs, {tot:.2f} job-h")
    print(json.dumps(MS.summarize([j for j in jobs if j["name"].startswith("v21:plug:")]), indent=1))
    for tier in ("core", "ext"):
        sel = [j for j in jobs if j.get("tier") == tier]
        print(f"  {tier}: {len(sel)} jobs, {sum(j['est_h'] for j in sel):.2f} job-h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
