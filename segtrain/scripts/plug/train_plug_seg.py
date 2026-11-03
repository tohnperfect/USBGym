"""v2.1 pluggability (EXPLORATORY): train a torchvision segmenter with the UNCHANGED v2 recipe.

  train     scripts/train.py's v2 run with ONE swap: usbg.segmenter.Segmenter -> usbg.plug.masks.TVSegmenter
            (us_seg.plug_seg.tv_segmenter_patch). Same sim build, Vitale split, KidneyUS partial labels,
            real fraction 0.4, rare-class boost, CE + Dice loss, Adam 5e-4, batch 8, 4000 iterations,
            final-iterate checkpoint (no early stopping, no model selection), and the same batch-index
            RNG stream as the v2 U-Net of that seed.
  crossval  scripts/crossval_real.py (6-fold OOF over the Vitale train frames + the deploy model on the
            holdout) with the same swap, for the 'vitale' real side of the consistency table.

Architectures (--arch): lraspp_mbv3, deeplabv3_r50, deeplabv3_mbv3, fcn_r50 (torchvision 0.27). The
backbones are trained from scratch (--backbone-weights none, the default): the ImageNet backbones are an
opt-in download (--backbone-weights imagenet). The torch init is seeded with --seed (the v2 U-Net init is not).

  PYTHONPATH=<gym>/src:<seg>/src USBG_GYM_ROOT=<gym> USBG_SEAT=anatomical USBG_LEARNED_SEG=0 \\
  <gym>/.venv/bin/python scripts/plug/train_plug_seg.py train --arch lraspp_mbv3 \\
      --dataset <seg>/data/nnunet_v2/Dataset110_USOrgansV2_sonospade_s0 --seed 0 --kidneyus \\
      --vitale-split data/real/vitale_split_v2.npz --forbid-test-patients --iters 4000 \\
      --out <seg>/outputs/v2_1/plug/seg/lraspp_mbv3/us_seg_lraspp_mbv3_sonospade_s0.pt

--smoke (tiny tests only): train: the real holdout used for the in-training log is replaced by 4 Vitale
TRAIN frames; crossval: refused with --deploy-ckpt, so only Vitale TRAIN-split frames are predicted. No
held-out frame is touched, and the output must be outside the repo's outputs/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from us_seg import gym_bridge, real_data, training  # noqa: E402
from us_seg import plug_seg as PS  # noqa: E402

ARCHS = ("lraspp_mbv3", "deeplabv3_r50", "deeplabv3_mbv3", "fcn_r50")


def _common(ap):
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--backbone-weights", default="none", choices=["none", "imagenet"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kidneyus", action="store_true")
    ap.add_argument("--vitale-split", default=None)
    ap.add_argument("--forbid-test-patients", action="store_true")
    ap.add_argument("--smoke", action="store_true")


def _check_out(path, smoke):
    path = os.path.abspath(path)
    if smoke and path.startswith(os.path.join(PS.SEG_ROOT, "outputs") + os.sep):
        raise SystemExit("--smoke outputs must not go under the repo's outputs/")
    return path


def cmd_train(args) -> int:
    out = _check_out(args.out, args.smoke)
    report = os.path.splitext(out)[0] + "_history.json"
    PS.refuse_existing(out, report, out + ".provenance.json")
    real_kw = {}
    if args.vitale_split:
        sp = real_data.load_vitale_split(args.vitale_split)
        real_kw = {"real_train_idx": sp["train_idx"].tolist(), "real_holdout_idx": sp["holdout_idx"].tolist()}
        if args.smoke:                                   # never touch a held-out frame in a smoke test
            real_kw["real_holdout_idx"] = sp["train_idx"][:4].tolist()
    forbid = gym_bridge.benchmark_split()[1] if args.forbid_test_patients else None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    with PS.tv_segmenter_patch(args.arch, args.backbone_weights, seed=args.seed):
        result = training.train(
            args.dataset, out, iters=args.iters, batch=args.batch, lr=args.lr, base=32, depth=4,
            dice_weight=args.dice_weight, val_frac=args.val_frac, seed=args.seed, rare_boost=args.rare_boost,
            eval_every=args.eval_every, real_frac=args.real_frac, add_kidneyus=args.kidneyus,
            forbid_sim_cases=forbid, **real_kw)
    wall = time.time() - t0
    bs = os.path.join(args.dataset, "build_summary.json")
    build = json.load(open(bs)) if os.path.exists(bs) else {}
    plug = {"slot": "mask_source", "kind": "tv", "method": args.arch,
            "backbone_weights": args.backbone_weights, "recipe": "v2 us_seg.training.train, network swapped only",
            "train_wall_sec": round(wall, 1), "sec_per_iter": round(wall / max(args.iters, 1), 4),
            "smoke": bool(args.smoke)}
    result["analysis_status"] = PS.ANALYSIS_STATUS
    result["plug"] = plug
    result["provenance"] = gym_bridge.provenance(
        argv=sys.argv[1:], vitale_split=args.vitale_split,
        vitale_split_md5=gym_bridge.file_md5(args.vitale_split) if args.vitale_split else None,
        ckpt_md5=gym_bridge.file_md5(out), dataset_build=build or None, analysis_status=PS.ANALYSIS_STATUS)
    with open(report, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[report] {report}")
    if args.vitale_split and not args.smoke:
        print(f"[prov] {_write_sidecar(args, out, result, build)}")
    if result["history"]:
        print(json.dumps(result["history"][-1], indent=2))
    print(f"[plug] {args.arch} trained in {wall:.0f}s ({wall / max(args.iters, 1) * 1000:.1f} ms/iter incl. evals)")
    return 0


def _write_sidecar(args, out, result, build):
    """<ckpt>.provenance.json through the gym's usbg.provenance (kind segmenter), as scripts/train.py
    writes it for the v2 U-Net, plus the plug fields."""
    from usbg import provenance as GP
    from us_seg import real_data as RD
    n_vit = len(result.get("vitale_trained_idx") or [])
    parents = [GP.parent_entry(os.path.abspath(args.dataset), "sim_dataset")]
    tx = build.get("translator_path")
    if tx and os.path.exists(tx):
        parents.append(GP.parent_entry(tx, "translator_of_sim_dataset"))
    real_pool = {"name": "kaggle_vitale_test[v2 split non-holdout]" + ("+kidneyus_train" if args.kidneyus else ""),
                 "path": gym_bridge.gym_path(gym_bridge.REAL_TEST_NPZ),
                 "n_frames": n_vit + int(result.get("n_kidneyus_train") or 0),
                 "license": "Kaggle ussimandsegm (Vitale et al.)" + ("; KidneyUS CC-BY-NC-SA" if args.kidneyus else ""),
                 "frames_md5": gym_bridge.file_md5(gym_bridge.gym_path(gym_bridge.REAL_TEST_NPZ))}
    return GP.write_provenance(
        out, "segmenter", train_cases=result["train_source_cases"], val_cases=result["val_source_cases"],
        real_pool=real_pool, iters=args.iters, seed=args.seed, dr_mode="per_frame",
        dr_notes="sim frames rendered with per-frame R.domain_randomize (build_dataset)",
        parents=parents, repo=gym_bridge.seg_root(),
        extra={"analysis_status": PS.ANALYSIS_STATUS, "plug": result.get("plug"),
               "vitale_split": os.path.abspath(args.vitale_split),
               "vitale_split_md5": gym_bridge.file_md5(os.path.abspath(args.vitale_split)),
               "kidneyus_split": RD._seg_path(RD._KID_SPLIT) if args.kidneyus else None,
               "kidneyus_split_md5": gym_bridge.file_md5(RD._seg_path(RD._KID_SPLIT)) if args.kidneyus else None,
               "vitale_trained_idx": result["vitale_trained_idx"],
               "n_kidneyus_train": result.get("n_kidneyus_train"),
               "history": os.path.splitext(out)[0] + "_history.json",
               "dataset_role": build.get("role"), "dataset_seat_organs": build.get("seat_organs"),
               "skipped_case_organs": build.get("skipped_case_organs"),
               "inference": "usbg.plug.masks.TorchvisionMaskSource (net.eval(), BatchNorm running stats, argmax)"})


def cmd_crossval(args) -> int:
    out = _check_out(args.out, args.smoke)
    oof = os.path.abspath(args.oof_out)
    PS.refuse_existing(out, oof)
    if args.smoke and args.deploy_ckpt:      # without --deploy-ckpt only Vitale TRAIN-split frames are predicted
        raise SystemExit("--smoke crossval must not pass --deploy-ckpt (it would predict the Vitale holdout)")
    cv = PS.load_v2_script("crossval_real")
    argv = ["--dataset", args.dataset, "--folds", str(args.folds), "--iters", str(args.iters),
            "--real-frac", str(args.real_frac), "--seed", str(args.seed), "--out", out, "--oof-out", oof]
    if args.kidneyus:
        argv.append("--kidneyus")
    if args.vitale_split:
        argv += ["--vitale-split", args.vitale_split]
    if args.forbid_test_patients:
        argv.append("--forbid-test-patients")
    if args.deploy_ckpt:
        argv += ["--deploy-ckpt", os.path.abspath(args.deploy_ckpt)]
    t0 = time.time()
    with PS.tv_segmenter_patch(args.arch, args.backbone_weights, seed=args.seed), \
            PS.patched_loader(cv, PS.tv_adapter_factory(cache_size=0)):
        rc = cv.main(argv)
    plug = {"slot": "mask_source", "kind": "tv", "method": args.arch, "backbone_weights": args.backbone_weights,
            "recipe": "v2 scripts/crossval_real.py, network swapped only", "wall_sec": round(time.time() - t0, 1)}
    PS.augment_json(out, plug, segmenter=os.path.abspath(args.deploy_ckpt) if args.deploy_ckpt else None,
                    mask_source="learned" if args.deploy_ckpt else "learned (K-fold models, discarded)",
                    extra={"oof_out": oof})
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    _common(t)
    t.add_argument("--out", required=True)
    t.add_argument("--iters", type=int, default=3000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--dice-weight", type=float, default=1.0)
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--rare-boost", type=float, default=4.0)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--real-frac", type=float, default=0.4)
    c = sub.add_parser("crossval")
    _common(c)
    c.add_argument("--out", required=True)
    c.add_argument("--oof-out", required=True)
    c.add_argument("--folds", type=int, default=6)
    c.add_argument("--iters", type=int, default=2500)
    c.add_argument("--real-frac", type=float, default=0.4)
    c.add_argument("--deploy-ckpt", default=None)
    args = ap.parse_args(argv)
    gym_bridge.ensure_usbg_importable()
    return cmd_train(args) if args.cmd == "train" else cmd_crossval(args)


if __name__ == "__main__":
    raise SystemExit(main())
