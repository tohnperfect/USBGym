#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): job spec for the torchvision segmenters with ImageNet-pretrained backbones.

  PYTHONPATH=src .venv/bin/python scripts/plug/make_plug_seg_pretrained_jobs.py
  -> outputs/v2_1/jobs/plug_seg_pretrained.json   (refuses to overwrite; --replace-own regenerates its own file)

plug_seg.json trains the four torchvision segmenters from scratch (--backbone-weights none). This spec adds the
same four with the torchvision ImageNet-1K backbones (--backbone-weights imagenet), which were downloaded
on 2026-10-15 and are recorded in <plugin weights>/MANIFEST.json:
  lraspp_mbv3, deeplabv3_mbv3   mobilenet_v3_large-8738ca79.pth  (MobileNet_V3_Large_Weights.IMAGENET1K_V1)
  deeplabv3_r50, fcn_r50        resnet50-0676ba61.pth            (ResNet50_Weights.IMAGENET1K_V1)
Only the backbone is pretrained; the segmentation heads start from the seeded random init, as in torchvision.

Design and command shapes are exactly those of plug_seg.json's torchvision block: scripts/plug/make_plug_seg_jobs.py
is imported (not edited) and its Spec.segeval / Spec.ladder / Spec.contact build the downstream jobs. Per arch,
stage (sonospade, physics, cut, cyclegan) and seed (0, 1, 2):
  train    train_plug_seg.py train --backbone-weights imagenet, otherwise the v2 recipe (4000 iterations)
  segeval  sim Dice (TEST-patient sim-eval build) and real Dice / consistency (Vitale holdout, KidneyUS, AUL)
  ladder   ibvs_base servo base recalibrated on TRAIN with this mask source, contact on/off, crossed held-out
           design (6 held-out patients, eval seed 999, 60 crossed episodes)
  contact  the v2 matched final policy of that stage and seed with this mask source (--allow-mismatch)
Method keys are <arch>_imnet, so every output path differs from the from-scratch ones.

Every job sets TORCH_HOME=<plugin weights>/torch_home. Only training reads the hub file
(usbg.plug.masks refuses rather than downloads when it is missing); a trained checkpoint loads without it. The
train jobs list the hub file in "files", so the runner holds them until it exists. Nothing is launched here.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("make_plug_seg_jobs", os.path.join(HERE, "make_plug_seg_jobs.py"))
MK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MK)

TORCH_HOME = os.path.join(os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights")), "torch_home")
MANIFEST = os.path.join(os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights")), "MANIFEST.json")
HUB = {"mbv3": f"{TORCH_HOME}/hub/checkpoints/mobilenet_v3_large-8738ca79.pth",
       "r50": f"{TORCH_HOME}/hub/checkpoints/resnet50-0676ba61.pth"}
SUFFIX = "_imnet"
OUT_NAME = "plug_seg_pretrained.json"


def hub_file(arch):
    return HUB["mbv3"] if arch.endswith("mbv3") else HUB["r50"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_weights():
    """Each hub file exists, its sha256 starts with the tag in its name and equals the MANIFEST record."""
    M = json.load(open(MANIFEST))
    rec = {os.path.join(os.path.dirname(MANIFEST), e["path"]): e.get("sha256") for e in M.get("files", [])}
    out = {}
    for p in HUB.values():
        if not os.path.isfile(p):
            raise SystemExit(f"{p} missing (download it first)")
        s = sha256(p)
        tag = os.path.basename(p).rsplit("-", 1)[-1].split(".")[0]
        if not s.startswith(tag) or rec.get(p) != s:
            raise SystemExit(f"{p}: sha256 {s[:16]}... does not match its tag {tag} or the MANIFEST record")
        out[p] = s
    return out


def with_torch_home(j):
    j = copy.deepcopy(j)
    j["env"]["TORCH_HOME"] = TORCH_HOME
    return j


def build(S):
    jobs = []
    for arch, c in MK.TV.items():
        key = arch + SUFFIX
        for t in MK.STAGES:
            for s in MK.SEEDS:
                b = MK.BASE[t]
                ds = f"{MK.SEG}/data/nnunet_v2/Dataset{100 + b * 10 + s}_USOrgansV2_{t}_s{s}"
                ck = f"{MK.SEG_OUT}/{key}/us_seg_{key}_{t}_s{s}.pt"
                tr = f"v21:plug:seg:train:{key}:{t}:s{s}"
                cmd = [MK.PY, f"{MK.SEG}/scripts/plug/train_plug_seg.py", "train", "--arch", arch,
                       "--backbone-weights", "imagenet", "--dataset", ds, "--seed", str(s), "--kidneyus",
                       "--vitale-split", MK.SPLIT, "--forbid-test-patients", "--iters", "4000", "--out", ck]
                jobs.append(with_torch_home(MK.job(tr, cmd, ck[:-3] + "_history.json", c["train_h"], MK.seg_env(), (),
                                                   [ds, MK.SPLIT, hub_file(arch)], S.expect_seg())))
                spec = f"kind=tv,ckpt={ck}"
                jobs.append(with_torch_home(S.segeval(key, t, s, spec, c["seg_ms"], [tr], [ck], train_build=ds)))
                jobs.append(with_torch_home(S.ladder(key, t, s, spec, c["seg_ms"], [tr], [ck])))
                jobs.append(with_torch_home(S.contact(key, t, s, spec, c["seg_ms"], [tr], [ck])))
    return jobs


def validate(jobs, out_path):
    names = [j["name"] for j in jobs]
    outs = [j["out"] for j in jobs]
    if len(names) != len(set(names)) or len(outs) != len(set(outs)):
        raise SystemExit("duplicate job names or outputs")
    by = {j["name"]: j for j in jobs}
    produced = {}
    for j in jobs:
        if j["name"].startswith("v21:plug:seg:train:"):
            produced[j["cmd"][j["cmd"].index("--out") + 1]] = j["name"]
    for j in jobs:
        missing_deps = [d for d in j["deps"] if d not in by]
        if missing_deps:
            raise SystemExit(f"{j['name']}: dependency outside this file: {missing_deps}")
        for f in j["files"]:
            if not os.path.exists(f) and produced.get(f) not in j["deps"]:
                raise SystemExit(f"{j['name']}: input {f} neither exists nor comes from a dependency")
        if j["env"].get("TORCH_HOME") != TORCH_HOME:
            raise SystemExit(f"{j['name']}: TORCH_HOME not set")
    # no clash with any other spec or queue file (read only)
    others = sorted(set(glob.glob(os.path.join(os.path.dirname(out_path), "*.json")))
                    - {os.path.abspath(out_path)})
    others += sorted(glob.glob(os.path.join(MK.GYM, "outputs", "v2_1", "jobs_*.json")))
    seen_n, seen_o = set(), set()
    for p in others:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        lst = d if isinstance(d, list) else d.get("jobs", []) if isinstance(d, dict) else []
        for e in lst:
            if isinstance(e, dict):
                seen_n.add(e.get("name"))
                seen_o.add(e.get("out"))
    clash = sorted((set(names) & seen_n) | (set(outs) & seen_o))
    if clash:
        raise SystemExit(f"names or outputs already used by another spec: {clash[:5]}")
    have = [o for o in outs if os.path.exists(o)]
    if have:
        raise SystemExit(f"{len(have)} outputs already exist, e.g. {have[0]}")
    return {"n_other_specs_checked": len(others)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=f"{MK.GYM}/outputs/v2_1/jobs")
    ap.add_argument("--replace-own", action="store_true", help=f"replace {OUT_NAME} written by this script")
    args = ap.parse_args(argv)
    shas = check_weights()
    jobs = build(MK.Spec())
    p = os.path.join(args.out_dir, OUT_NAME)
    if os.path.exists(p):
        old = json.load(open(p))
        if not args.replace_own or not all(SUFFIX + ":" in j["name"] for j in old):
            raise SystemExit(f"{p} exists (pass --replace-own to regenerate this script's own spec)")
    info = validate(jobs, p)
    with open(p + ".tmp", "w") as fh:
        json.dump(jobs, fh, indent=1)
    os.replace(p + ".tmp", p)
    print(f"wrote {p}: {len(jobs)} jobs, {sum(j['est_h'] for j in jobs):.1f} job-h; checked against "
          f"{info['n_other_specs_checked']} other spec/queue files")
    print(json.dumps(MK.summarize(jobs), indent=1))
    print("hub files (sha256):", json.dumps(shas, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
