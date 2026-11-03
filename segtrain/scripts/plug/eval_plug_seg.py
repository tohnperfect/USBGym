"""v2.1 pluggability (EXPLORATORY): a plugged segmenter's own sim and real Dice and its cross-domain
consistency, computed by the UNCHANGED v2 scripts (scripts/eval_seg.py, scripts/consistency.py).

The v2 cascade (scripts/run_seg_v2.sh) runs, per stage and seed, eval_seg.py on the TEST-patient
sim-eval build with the three never-trained real sets, and consistency.py with four real sides
(vitale = OOF + deploy-on-holdout, vitaleho, kidneyus, aul). This script runs the same two scripts
with their ``load_segmenter`` pointed at a usbg.plug.masks adapter (us_seg.plug_seg.patched_loader),
then adds analysis_status, the plug description and a usbg.provenance.result_block to each JSON.

  --mask-plug  a usbg.plug.masks spec, e.g.
                 kind=tv,ckpt=<seg>/outputs/v2_1/plug/seg/lraspp_mbv3/us_seg_lraspp_mbv3_sonospade_s0.pt
                 kind=nnunet,model=<.../nnUNetTrainer_100epochs__nnUNetPlans__2d>,fold=0
                 kind=sam,weights=<sam_vit_b_01ec64.pth>,prompt=gt      (PRIVILEGED: GT boxes)
                 kind=medsam,weights=<medsam_vit_b.pth>,prompt=seg,prompt_ckpt=<us_seg_<stage>_s<seed>.pt>
               prompt=gt registers the ground-truth label of every frame the v2 scripts will score (the
               CT label on sim frames, the manual label on real frames) and boxes come from it.
  --oof        crossval npz for the 'vitale' real side (tv models: train_plug_seg.py crossval); without
               it that side is skipped (logged), and the organ-identity table can use --real-side vitaleho.

Outputs (never overwritten) in --out-dir, named like the v2 ones so scripts/agg_tablev_v2.py can
aggregate a method directory: eval_<stage>_s<seed>.json, consistency_<stage>_s<seed>_<src>.json, and
segeval_<stage>_s<seed>.json (timing and the list of outputs).

--smoke (tiny tests only): a TRAINING build is scored on its own seed's val split (TRAIN patients
only), the real side is a scratch npz of Vitale TRAIN-split frames, and load_real_test is restricted to
the Vitale TRAIN-split frames, so no held-out patient or frame is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from us_seg import gym_bridge, real_data  # noqa: E402
from us_seg import plug_seg as PS  # noqa: E402
from us_seg.nnunet_raw import RawDataset  # noqa: E402

REAL_NPZ = {"vitaleho": "data/real/vitale_holdout_v2.npz", "kidneyus": "data/real/kidneyus_holdout_test.npz",
            "aul": "data/real/aul_liver_test.npz"}


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(PS.SEG_ROOT, p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-plug", required=True)
    ap.add_argument("--method-key", required=True, help="e.g. lraspp_mbv3, nnunet2d, sam_priv, medsam_unet")
    ap.add_argument("--stage", required=True, choices=["sonospade", "physics", "cut", "cyclegan"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", required=True, help="the TEST-patient sim-eval build (--smoke: a TRAIN build)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sources", nargs="+", default=["vitaleho", "kidneyus", "aul"],
                    choices=["vitale", "vitaleho", "kidneyus", "aul"])
    ap.add_argument("--oof", default=None, help="OOF npz for the 'vitale' side")
    ap.add_argument("--train-build", default=None,
                    help="the sim TRAIN build the segmenter was trained on (provenance train_cases); none for "
                         "untrained foundation models")
    ap.add_argument("--skip-eval-seg", action="store_true",
                    help="consistency only (e.g. the 'vitale' side once its OOF exists); the summary is then "
                         "segeval_<stage>_s<seed>_<sources>.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-real-n", type=int, default=5)
    args = ap.parse_args(argv)
    gym_bridge.ensure_usbg_importable()
    from usbg.plug import masks as PM

    od = os.path.abspath(args.out_dir)
    if args.smoke and od.startswith(os.path.join(PS.SEG_ROOT, "outputs") + os.sep):
        raise SystemExit("--smoke outputs must not go under the repo's outputs/")
    tag = f"{args.stage}_s{args.seed}"
    srcs = [s for s in args.sources if not (s == "vitale" and not args.oof)]
    if len(srcs) != len(args.sources):
        print("!! SKIP consistency-vitale: no --oof given (the 'vitale' side needs the K-fold OOF)")
    if args.skip_eval_seg:
        outs = {"summary": os.path.join(od, f"segeval_{tag}_{'-'.join(srcs)}.json")}
    else:
        outs = {"eval": os.path.join(od, f"eval_{tag}.json"), "summary": os.path.join(od, f"segeval_{tag}.json")}
    outs.update({f"consistency_{s}": os.path.join(od, f"consistency_{tag}_{s}.json") for s in srcs})
    PS.refuse_existing(*outs.values())
    os.makedirs(od, exist_ok=True)

    build = json.load(open(os.path.join(args.dataset, "build_summary.json")))
    if not args.smoke and build.get("role") != "eval":
        raise SystemExit(f"{args.dataset} is a {build.get('role')!r} build; pass the TEST-patient sim-eval build")
    if args.smoke and build.get("role") != "train":
        raise SystemExit("--smoke scores a TRAINING build's val split (TRAIN patients only)")

    real_npz = {s: _abs(p) for s, p in REAL_NPZ.items()}
    if args.smoke:                                       # Vitale TRAIN-split frames only
        sp = real_data.load_vitale_split(_abs(real_data.VITALE_SPLIT_V2))
        tr_idx = sp["train_idx"]
        orig_load = real_data.load_real_test

        def _train_only():
            d = orig_load()
            return {"images": d["images"][tr_idx], "label6": d["label6"][tr_idx],
                    "label_repo": d["label_repo"][tr_idx], "sessions": d["sessions"][tr_idx], "n": int(len(tr_idx))}
        real_data.load_real_test = _train_only
        rt = _train_only()
        k = int(args.smoke_real_n)
        sm = os.path.join(od, "smoke_real_vitale_train.npz")
        np.savez_compressed(sm, images=rt["images"][:k], label6=rt["label6"][:k].astype(np.uint8))
        real_npz = {s: sm for s in REAL_NPZ}

    t0 = time.time()
    adapter = PM.load_mask_source(args.mask_plug, cache_size=16384)
    load_sec = time.time() - t0
    if getattr(adapter, "prompt", None) == "gt":       # register every frame the v2 scripts will score
        ds = RawDataset(args.dataset)
        X, Y = ds.load_all(ds.train_cases)
        adapter.register_prompts(X, Y)
        rt = real_data.load_real_test()
        adapter.register_prompts(rt["images"], rt["label6"])
        for p in set(real_npz.values()):
            d = np.load(p, allow_pickle=True)
            adapter.register_prompts(d["images"].astype(np.float32), d["label6"].astype(np.int64))
    if getattr(adapter, "prompt", None) == "ct":
        raise SystemExit("prompt=ct needs the gym render loop; offline evaluation uses prompt=gt")

    def factory(_path):
        return adapter

    plug = adapter.describe()
    ckpt = adapter.ckpt_path
    train_cases = []
    if args.train_build:
        tb = json.load(open(os.path.join(args.train_build, "build_summary.json")))
        train_cases = list(tb.get("cases") or [])
    eval_cases = [] if args.smoke else list(build.get("cases") or [])
    prov_kw = dict(texture=build.get("translator_path"), segmenter=ckpt,
                   mask_source=("privileged box prompts (ground truth)" if adapter.privileged else "learned"),
                   train_cases=train_cases, eval_cases=eval_cases)

    t1 = time.time()
    if not args.skip_eval_seg:
        ev = PS.load_v2_script("eval_seg")
        a = ["--ckpt", ckpt, "--dataset", args.dataset, "--seed", str(args.seed), "--out", outs["eval"]]
        if not args.smoke:
            a.append("--sim-all")
        for s in ("vitaleho", "aul", "kidneyus"):
            a += ["--real-npz", real_npz[s]]
        with PS.patched_loader(ev, factory):
            ev.main(a)
        PS.augment_json(outs["eval"], dict(plug, timing=adapter.timing()), **prov_kw)

    cs = PS.load_v2_script("consistency")
    for s in srcs:
        a = ["--ckpt", ckpt, "--dataset", args.dataset, "--seed", str(args.seed), "--out", outs[f"consistency_{s}"]]
        if not args.smoke:
            a.append("--sim-all")
        a += (["--oof", os.path.abspath(args.oof)] if s == "vitale" else ["--real-npz", real_npz[s]])
        with PS.patched_loader(cs, factory):
            cs.main(a)
        PS.augment_json(outs[f"consistency_{s}"], dict(plug, timing=adapter.timing()), **prov_kw)

    summ = {"method_key": args.method_key, "stage": args.stage, "seed": args.seed, "dataset": args.dataset,
            "outputs": {k: v for k, v in outs.items() if k != "summary"}, "load_sec": round(load_sec, 2),
            "eval_wall_sec": round(time.time() - t1, 1), "timing": adapter.timing(), "smoke": bool(args.smoke),
            "skipped": ([] if len(srcs) == len(args.sources) else ["consistency-vitale (no --oof)"]),
            "provenance": gym_bridge.provenance(argv=sys.argv[1:], ckpt_md5=gym_bridge.file_md5(ckpt),
                                                analysis_status=PS.ANALYSIS_STATUS)}
    with open(outs["summary"], "w") as fh:
        json.dump(summ, fh, indent=2)
    PS.augment_json(outs["summary"], plug, **prov_kw)
    print(f"[plug-segeval] {args.method_key} {tag}: {json.dumps(summ['timing'])} -> {od}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
