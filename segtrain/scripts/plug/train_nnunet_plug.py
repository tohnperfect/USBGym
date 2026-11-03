"""v2.1 pluggability (EXPLORATORY): plan, preprocess and train nnU-Net v2 (2D) on a dataset written by
build_nnunet_plug_dataset.py, with the v2 sim train/val split as fold 0, on Apple MPS (CUDA-free).

Steps (nnunetv2's own command-line tools, from the plugin venv that runs this script):
  1. nnUNetv2_plan_and_preprocess -d ID -c 2d -np N --verify_dataset_integrity
  2. <nnUNet_preprocessed>/DatasetID_.../splits_final.json = [usbg_split.json fold0] (the v2 split; nnU-Net's
     own 5-fold split is not used)
  3. nnUNetv2_train ID 2d 0 -tr TRAINER -device mps   (nnUNet_extTrainer = us_seg/nnunet_plug_trainers)
  4. a usbg.provenance sidecar next to fold_0/checkpoint_final.pth (the FINAL checkpoint is what the gym
     loads, like the v2 U-Net's final iterate; checkpoint_best is never used), and <out-json>.

Trainers: nnUNetTrainer_USBGbudget (the v2 U-Net's 32,000-frame sample budget at nnU-Net's planned batch,
rounded up to whole 250-iteration epochs: 2 epochs at the planned batch of 93; the primary arm),
nnUNetTrainer_20epochs / nnUNetTrainer_100epochs (nnunetv2 built-ins; schedule-sensitivity arms),
nnUNetTrainer_USBGsmoke / nnUNetTrainer_USBGtiming (smoke and cost measurement only). nnU-Net's default
is 1000 epochs; every shorter schedule is a stated deviation. Measured on the sonospade s0 data (M5 Pro,
MPS, nnUNet_n_proc_DA=4): planned batch 93, patch 128 x 128, about 0.75 s per training iteration, so
about 3.4 min per 250-iteration epoch (100 epochs would be about 5.7 h per model).

The nnUNet_raw / nnUNet_preprocessed / nnUNet_results roots come from --raw/--preprocessed/--results
(exported before nnunetv2 is imported) and must live under segtrain's outputs/v2_1/plug/
(or, for --smoke, outside outputs/).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from us_seg import gym_bridge  # noqa: E402
from us_seg import plug_seg as PS  # noqa: E402

EXT_TRAINERS = os.path.join(PS.SEG_ROOT, "src", "us_seg", "nnunet_plug_trainers")
_ITERS = {"nnUNetTrainer_USBGsmoke": 5, "nnUNetTrainer_USBGtiming": 40}


def _bin(name):
    b = os.path.join(os.path.dirname(sys.executable), name)
    if not os.path.isfile(b):
        raise SystemExit(f"{b} not found: run this with the plugin venv python (nnunetv2 installed)")
    return b


def _run(cmd, env, log):
    print("+ " + " ".join(cmd), flush=True)
    with open(log, "a") as fh:
        fh.write("+ " + " ".join(cmd) + "\n"); fh.flush()
        p = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}; see {log}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--preprocessed", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--trainer", default="nnUNetTrainer_USBGbudget",
                    choices=["nnUNetTrainer_100epochs", "nnUNetTrainer_USBGbudget", "nnUNetTrainer_20epochs", "nnUNetTrainer_USBGsmoke",
                             "nnUNetTrainer_USBGtiming", "nnUNetTrainer_50epochs", "nnUNetTrainer_250epochs"])
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--np", type=int, default=4, help="preprocessing processes")
    ap.add_argument("--n-proc-da", type=int, default=4, help="nnUNet_n_proc_DA (augmentation workers)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    roots = [os.path.abspath(p) for p in (args.raw, args.preprocessed, args.results, args.out_json)]
    outs_root = os.path.join(PS.SEG_ROOT, "outputs") + os.sep
    if args.smoke and any(r.startswith(outs_root) for r in roots):
        raise SystemExit("--smoke outputs must not go under the repo's outputs/")
    if not args.smoke and args.trainer in ("nnUNetTrainer_USBGsmoke", "nnUNetTrainer_USBGtiming"):
        raise SystemExit(f"{args.trainer} is for --smoke only")
    raw, pre, res, out_json = roots
    PS.refuse_existing(out_json)
    names = [d for d in os.listdir(raw) if d.startswith(f"Dataset{args.dataset_id:03d}_")
             and os.path.isdir(os.path.join(raw, d))]
    if len(names) != 1:
        raise SystemExit(f"expected one Dataset{args.dataset_id:03d}_* in {raw}, found {names}")
    dname = names[0]
    model_dir = os.path.join(res, dname, f"{args.trainer}__nnUNetPlans__2d")
    ck = os.path.join(model_dir, "fold_0", "checkpoint_final.pth")
    PS.refuse_existing(ck)
    meta = json.load(open(os.path.join(raw, dname, "usbg_split.json")))

    env = dict(os.environ, nnUNet_raw=raw, nnUNet_preprocessed=pre, nnUNet_results=res,
               nnUNet_n_proc_DA=str(args.n_proc_da), nnUNet_extTrainer=EXT_TRAINERS,
               PYTORCH_ENABLE_MPS_FALLBACK="1")
    os.makedirs(pre, exist_ok=True); os.makedirs(res, exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    # not <out>.log: the v2.1 runner already writes this job's stdout there (opened "w", not append)
    log = os.path.splitext(out_json)[0] + ".nnunet.log"
    timings = {}

    t = time.time()
    pdir = os.path.join(pre, dname)
    if os.path.isfile(os.path.join(pdir, "nnUNetPlans.json")):
        print(f"  !! SKIP plan_and_preprocess (exists: {pdir})")
    else:
        _run([_bin("nnUNetv2_plan_and_preprocess"), "-d", str(args.dataset_id), "-c", "2d", "-np", str(args.np),
              "--verify_dataset_integrity"], env, log)
    timings["plan_preprocess_sec"] = round(time.time() - t, 1)

    splits = [{"train": meta["fold0"]["train"], "val": meta["fold0"]["val"]}]
    sf = os.path.join(pdir, "splits_final.json")
    if os.path.exists(sf):
        if json.load(open(sf)) != splits:
            raise SystemExit(f"{sf} exists and differs from the v2 split; refusing to overwrite")
    else:
        json.dump(splits, open(sf, "w"), indent=1)
    plans = json.load(open(os.path.join(pdir, "nnUNetPlans.json")))
    c2d = plans["configurations"]["2d"]

    t = time.time()
    _run([_bin("nnUNetv2_train"), str(args.dataset_id), "2d", "0", "-tr", args.trainer, "-device", args.device],
         env, log)
    timings["train_sec"] = round(time.time() - t, 1)
    if not os.path.isfile(ck):
        raise SystemExit(f"training finished without {ck}")

    import torch
    ckd = torch.load(ck, map_location="cpu", weights_only=False)
    n_ep = int(ckd.get("current_epoch", 0))
    init = ckd.get("init_args", {})
    logging = ckd.get("logging", {})
    summ = {"dataset": dname, "trainer": args.trainer, "device": args.device, "model_dir": model_dir,
            "checkpoint_final": ck, "checkpoint_md5": gym_bridge.file_md5(ck),
            "epochs_completed": n_ep, "batch_size": c2d.get("batch_size"), "patch_size": c2d.get("patch_size"),
            "iterations_per_epoch": _ITERS.get(args.trainer, 250),
            "frames_seen": n_ep * _ITERS.get(args.trainer, 250) * int(c2d.get("batch_size") or 0),
            "v2_unet_frames_seen": 4000 * 8, "timings": timings,
            "sec_per_epoch": round(timings["train_sec"] / max(n_ep, 1), 1),
            "val_pseudo_dice_last": (logging.get("mean_fg_dice") or [None])[-1],
            "split": {k: v for k, v in meta.items() if k != "fold0"}, "smoke": bool(args.smoke),
            "nnunet_default_epochs": 1000, "deviation": "short schedule (see trainer); final checkpoint; mirroring "
            "TTA off at gym inference unless the spec sets mirror=1", "init_args_fold": init.get("fold"),
            "analysis_status": PS.ANALYSIS_STATUS,
            "provenance": gym_bridge.provenance(argv=sys.argv[1:], analysis_status=PS.ANALYSIS_STATUS,
                                                packages=_pkgs())}
    with open(out_json, "w") as fh:
        json.dump(summ, fh, indent=2, default=str)
    print(f"[nnunet] {dname} {args.trainer}: {n_ep} epochs in {timings['train_sec']:.0f}s -> {ck}")
    if not args.smoke:
        from usbg import provenance as GP
        sc = GP.write_provenance(
            ck, "segmenter", train_cases=meta["sim_train_source_cases"], val_cases=meta["sim_val_source_cases"],
            real_pool={"name": "kaggle_vitale_test[v2 split non-holdout]+kidneyus_train",
                       "path": gym_bridge.gym_path(gym_bridge.REAL_TEST_NPZ),
                       "n_frames": int(meta.get("n_real_unique") or 0), "license": "Kaggle ussimandsegm (Vitale et al.); "
                       "KidneyUS CC-BY-NC-SA", "frames_md5": gym_bridge.file_md5(gym_bridge.gym_path(gym_bridge.REAL_TEST_NPZ))},
            iters=n_ep * 250, seed=int(meta["seed"]), dr_mode="per_frame", dr_notes="v2 sim build frames (per-frame DR)",
            parents=[GP.parent_entry(os.path.join(raw, dname), "dataset")], repo=gym_bridge.seg_root(),
            extra={"analysis_status": PS.ANALYSIS_STATUS, "trainer": args.trainer, "plans_md5":
                   gym_bridge.file_md5(os.path.join(model_dir, "plans.json")), "summary": out_json})
        print(f"[prov] {sc}")
    return 0


def _pkgs():
    out = {}
    for n in ("nnunetv2", "torch", "batchgenerators", "dynamic_network_architectures"):
        try:
            from importlib.metadata import version
            out[n] = version(n)
        except Exception:
            out[n] = None
    return out


if __name__ == "__main__":
    raise SystemExit(main())
