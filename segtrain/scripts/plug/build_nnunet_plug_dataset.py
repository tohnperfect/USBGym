"""v2.1 pluggability (EXPLORATORY): the v2 segmenter training data in nnU-Net v2's own raw format.

The v2 U-Net (us_seg.training.train) trains on a v2 sim TRAIN build plus real frames mixed on the fly:
the Vitale TRAIN-split frames (data/real/vitale_split_v2.npz, never the holdout) and the KidneyUS
partial-label TRAIN frames (kidney supervised, every other pixel ignored), at an expected real fraction
of 0.4 per batch; its sim validation split is split_by_source_case(val_frac=0.2, seed). This script
writes exactly those frames into one nnU-Net raw dataset:

  imagesTr/labelsTr  every sim frame of the build (PNG bytes copied unchanged), the Vitale TRAIN-split
                     frames and the KidneyUS TRAIN frames (PNG uint8; KidneyUS pixels with no label get
                     nnU-Net's ignore label 6, its documented partial-annotation mechanism)
  dataset.json       labels 0..5 as the v2 6-class scheme + "ignore": 6, NaturalImage2DIO
  usbg_split.json    fold 0 = the v2 split: train = sim train cases + real frames, val = sim val cases
                     (scripts/plug/train_nnunet_plug.py writes it as nnU-Net's splits_final.json)
  domain.json        per case: domain, source case / frame index, copy index

nnU-Net samples training cases uniformly, so the real frames are replicated to keep the v2 expected
real fraction (round(real_frac / (1 - real_frac) * n_sim_train) real cases; every real frame at least
once, the remainder to a seeded permutation). The v2 rare-class (gallbladder, vessel) frame boost is
NOT emulated: nnU-Net's own foreground oversampling is used instead (documented deviation).

--smoke (tiny tests only): at most --smoke-sim sim frames per source case and --smoke-real real frames.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from us_seg import gym_bridge, real_data  # noqa: E402
from us_seg import plug_seg as PS  # noqa: E402
from us_seg.nnunet_raw import RawDataset, CHANNEL_SUFFIX, FILE_ENDING  # noqa: E402

IGNORE = 6
LABELS = {"background": 0, "liver": 1, "gallbladder": 2, "vessel": 3, "kidney": 4, "spleen": 5, "ignore": IGNORE}


def _png_img(path, img01):
    a = np.clip(np.asarray(img01, np.float32), 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _png_lab(path, lab):
    lab = np.asarray(lab)
    if lab.min() < 0 or lab.max() > IGNORE:
        raise ValueError(f"label out of range [{lab.min()},{lab.max()}]")
    Image.fromarray(lab.astype(np.uint8), mode="L").save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dataset", required=True, help="a v2 sim TRAIN build (DatasetNNN_USOrgansV2_<stage>_s<seed>)")
    ap.add_argument("--out-raw", required=True, help="nnUNet_raw root for the plug datasets")
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, required=True, help="the v2 seed: sim val split + replication permutation")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--real-frac", type=float, default=0.4)
    ap.add_argument("--vitale-split", default="data/real/vitale_split_v2.npz")
    ap.add_argument("--kidneyus", action="store_true")
    ap.add_argument("--forbid-test-patients", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-sim", type=int, default=6)
    ap.add_argument("--smoke-real", type=int, default=6)
    args = ap.parse_args(argv)
    gym_bridge.ensure_usbg_importable()

    out_raw = os.path.abspath(args.out_raw)
    if args.smoke and out_raw.startswith(os.path.join(PS.SEG_ROOT, "outputs") + os.sep):
        raise SystemExit("--smoke outputs must not go under the repo's outputs/")
    ddir = os.path.join(out_raw, f"Dataset{args.dataset_id:03d}_{args.name}")
    # The v2.1 runner creates the job's output directory (ddir, since the job's out is ddir/usbg_split.json)
    # and opens its log ddir/usbg_split.log before it starts this script, so a ddir holding only that log
    # is not an existing dataset. Anything else in it is, and is never overwritten.
    PS.refuse_existing(ddir + ".provenance.json")
    if os.path.lexists(ddir):
        if not os.path.isdir(ddir):
            raise SystemExit(f"refusing to overwrite existing output(s): {[ddir]}")
        other = sorted(f for f in os.listdir(ddir) if f != "usbg_split.log")
        if other:
            raise SystemExit(f"refusing to overwrite existing dataset {ddir} (contains {other[:5]})")

    src = RawDataset(args.src_dataset)
    srcs = {src.source_case(c) for c in src.train_cases}
    if args.forbid_test_patients:
        leak = sorted(srcs & set(gym_bridge.benchmark_split()[1]))
        if leak:
            raise SystemExit(f"[guard] {args.src_dataset} contains held-out patients {leak}")
    tr_cases, va_cases = src.split_by_source_case(val_frac=args.val_frac, seed=args.seed)   # the v2 split
    if args.smoke:
        def first_k(cases):
            by = {}
            for c in cases:
                by.setdefault(src.source_case(c), []).append(c)
            return sorted(x for v in by.values() for x in sorted(v)[:args.smoke_sim])
        tr_cases, va_cases = first_k(tr_cases), first_k(va_cases)

    # real frames: Vitale TRAIN split (never the holdout) + KidneyUS TRAIN (partial label, ignore elsewhere)
    rt = real_data.load_real_test()
    sp = real_data.load_vitale_split(args.vitale_split)
    vit_idx = [int(i) for i in sp["train_idx"]]
    real_imgs = [rt["images"][i] for i in vit_idx]
    real_labs = [rt["label6"][i].astype(np.int64) for i in vit_idx]
    real_meta = [{"domain": "real", "source": "vitale", "vitale_idx": i, "session": str(rt["sessions"][i])}
                 for i in vit_idx]
    n_kid = 0
    if args.kidneyus:
        kd = real_data.load_kidneyus_train()
        n_kid = int(kd["n"])
        for j in range(n_kid):
            y = kd["label6"][j].astype(np.int64)
            y = np.where(y == 255, IGNORE, y)
            real_imgs.append(kd["images"][j]); real_labs.append(y)
            real_meta.append({"domain": "real", "source": "kidneyus_train", "kidneyus_train_idx": j})
    if args.smoke:
        nv = len(vit_idx)                                # half Vitale, half KidneyUS (exercises the ignore label)
        h = (args.smoke_real + 1) // 2
        keep = list(range(min(h, nv))) + list(range(nv, min(nv + args.smoke_real - h, len(real_imgs))))
        real_imgs = [real_imgs[i] for i in keep]; real_labs = [real_labs[i] for i in keep]
        real_meta = [real_meta[i] for i in keep]
    n_real = len(real_imgs)
    n_sim_tr = len(tr_cases)
    target = int(round(args.real_frac / (1.0 - args.real_frac) * n_sim_tr)) if args.real_frac > 0 else 0
    copies = np.ones(n_real, np.int64)
    if target > n_real:
        base = target // n_real
        copies[:] = base
        extra = target - base * n_real
        perm = np.random.default_rng(args.seed).permutation(n_real)
        copies[perm[:extra]] += 1

    imgs_tr = os.path.join(ddir, "imagesTr"); labs_tr = os.path.join(ddir, "labelsTr")
    os.makedirs(imgs_tr); os.makedirs(labs_tr)
    domain, split_tr, split_va = {}, [], []
    for c in sorted(tr_cases) + sorted(va_cases):
        cid = f"USBG_sim_{c}"
        shutil.copyfile(os.path.join(src.images_tr, f"{c}{CHANNEL_SUFFIX}{FILE_ENDING}"),
                        os.path.join(imgs_tr, f"{cid}{CHANNEL_SUFFIX}{FILE_ENDING}"))
        shutil.copyfile(os.path.join(src.labels_tr, f"{c}{FILE_ENDING}"), os.path.join(labs_tr, f"{cid}{FILE_ENDING}"))
        domain[cid] = {"domain": "sim", "src_case": c, "case": src.source_case(c),
                       "split": "train" if c in set(tr_cases) else "val"}
        (split_tr if c in set(tr_cases) else split_va).append(cid)
    for j in range(n_real):
        for k in range(int(copies[j])):
            cid = f"USBG_real_{j:04d}_c{k}"
            _png_img(os.path.join(imgs_tr, f"{cid}{CHANNEL_SUFFIX}{FILE_ENDING}"), real_imgs[j])
            _png_lab(os.path.join(labs_tr, f"{cid}{FILE_ENDING}"), real_labs[j])
            domain[cid] = dict(real_meta[j], copy=k, split="train")
            split_tr.append(cid)
    n_total = len(domain)
    realized = float(sum(copies)) / max(n_sim_tr + float(sum(copies)), 1.0)
    dj = {"channel_names": {"0": "US"}, "labels": LABELS, "numTraining": n_total, "file_ending": FILE_ENDING,
          "overwrite_image_reader_writer": "NaturalImage2DIO",
          "description": "USB Gym v2.1 pluggability (exploratory): v2 segmenter training data for nnU-Net v2 "
                         "(sim TRAIN build + Vitale TRAIN split + KidneyUS TRAIN, ignore label 6)."}
    json.dump(dj, open(os.path.join(ddir, "dataset.json"), "w"), indent=2)
    json.dump(domain, open(os.path.join(ddir, "domain.json"), "w"), indent=1)
    meta = {"fold0": {"train": split_tr, "val": split_va}, "seed": args.seed, "val_frac": args.val_frac,
            "src_dataset": os.path.abspath(args.src_dataset),
            "sim_train_source_cases": sorted({src.source_case(c) for c in tr_cases}),
            "sim_val_source_cases": sorted({src.source_case(c) for c in va_cases}),
            "n_sim_train": n_sim_tr, "n_sim_val": len(va_cases), "n_real_unique": n_real,
            "n_vitale_train": len(vit_idx) if not args.smoke else None, "n_kidneyus_train": n_kid,
            "real_copies": {"target": target, "total": int(copies.sum()), "min": int(copies.min()) if n_real else 0,
                            "max": int(copies.max()) if n_real else 0},
            "realized_real_fraction": round(realized, 4), "target_real_fraction": args.real_frac,
            "rare_class_boost": "not emulated (nnU-Net foreground oversampling instead)",
            "vitale_split": real_data._seg_path(args.vitale_split),
            "smoke": bool(args.smoke), "analysis_status": PS.ANALYSIS_STATUS}
    json.dump(dict(meta, provenance=gym_bridge.provenance(argv=sys.argv[1:], analysis_status=PS.ANALYSIS_STATUS)),
              open(os.path.join(ddir, "usbg_split.json"), "w"), indent=1)
    print(f"[nnunet-raw] {ddir}: {n_sim_tr} sim train + {len(va_cases)} sim val + {int(copies.sum())} real "
          f"({n_real} unique; realized real fraction {realized:.3f}, target {args.real_frac})")
    if not args.smoke:
        from usbg import provenance as GP
        sc = GP.write_provenance(
            ddir, "dataset", train_cases=meta["sim_train_source_cases"], val_cases=meta["sim_val_source_cases"],
            real_pool={"name": "kaggle_vitale_test[v2 split non-holdout]" + ("+kidneyus_train" if args.kidneyus else ""),
                       "path": gym_bridge.gym_path(gym_bridge.REAL_TEST_NPZ), "n_frames": n_real,
                       "license": "Kaggle ussimandsegm (Vitale et al.)" + ("; KidneyUS CC-BY-NC-SA" if args.kidneyus else ""),
                       "frames_md5": gym_bridge.file_md5(gym_bridge.gym_path(gym_bridge.REAL_TEST_NPZ))},
            iters=0, seed=args.seed, dr_mode="per_frame", dr_notes="sim frames copied from the v2 build (per-frame DR)",
            parents=[GP.parent_entry(os.path.abspath(args.src_dataset), "sim_dataset")], repo=gym_bridge.seg_root(),
            extra={"analysis_status": PS.ANALYSIS_STATUS, "usbg_split": dict(meta, fold0=None)})
        print(f"[prov] {sc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
