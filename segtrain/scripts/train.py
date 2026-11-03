"""Train the 6-class MPS U-Net and report per-domain (sim-val / real-test) Dice.

  ./run.sh scripts/train.py --dataset data/nnunet/Dataset011_USOrgansSimReal --iters 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from us_seg import gym_bridge, real_data, training


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/nnunet/Dataset011_USOrgansSimReal")
    ap.add_argument("--out", default="outputs/us_seg_6cls.pt")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--rare-boost", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--real-frac", type=float, default=0.4,
                    help="target fraction of each batch drawn from real Vitale frames (0 = sim only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kidneyus", action="store_true",
                    help="fold the leakage-free KidneyUS partial-label TRAIN frames into training "
                         "(kidney supervised, non-kidney pixels ignored) to resolve real kidney")
    ap.add_argument("--vitale-split", default=None,
                    help="v2: fixed Vitale split npz (scripts/make_vitale_split.py). Trains only on its "
                         "non-holdout frames and reports real Dice on the holdout (never trained on).")
    ap.add_argument("--forbid-test-patients", action="store_true",
                    help="v2 guard: fail if the sim dataset contains any usbg.benchmark TEST_PATIENTS")
    args = ap.parse_args(argv)

    real_kw = {}
    if args.vitale_split:
        sp = real_data.load_vitale_split(args.vitale_split)
        real_kw = {"real_train_idx": sp["train_idx"].tolist(), "real_holdout_idx": sp["holdout_idx"].tolist()}
    forbid = gym_bridge.benchmark_split()[1] if args.forbid_test_patients else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result = training.train(
        args.dataset, args.out,
        iters=args.iters, batch=args.batch, lr=args.lr, base=args.base, depth=args.depth,
        dice_weight=args.dice_weight, val_frac=args.val_frac, seed=args.seed,
        rare_boost=args.rare_boost, eval_every=args.eval_every, real_frac=args.real_frac,
        add_kidneyus=args.kidneyus, forbid_sim_cases=forbid, **real_kw,
    )
    import json as _json
    bs = os.path.join(args.dataset, "build_summary.json")
    result["provenance"] = gym_bridge.provenance(
        argv=sys.argv[1:], vitale_split=args.vitale_split,
        vitale_split_md5=gym_bridge.file_md5(args.vitale_split) if args.vitale_split else None,
        ckpt_md5=gym_bridge.file_md5(args.out),
        dataset_build=(_json.load(open(bs)) if os.path.exists(bs) else None),
    )
    report = os.path.splitext(args.out)[0] + "_history.json"
    with open(report, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[report] {report}")
    if args.vitale_split:      # v2 runs: <ckpt>.provenance.json through the gym's usbg.provenance
        sc = _write_sidecar(args, result, result["provenance"].get("dataset_build") or {})
        print(f"[prov] {sc}")
    if result["history"]:
        print(json.dumps(result["history"][-1], indent=2))
    return 0


def _write_sidecar(args, result, build):
    """Segmenter checkpoint sidecar (kind segmenter). train_cases / val_cases are the CT source cases
    of the sim train / val split (write_provenance refuses a TEST patient in either); the real pool is
    the Vitale non-holdout frames + KidneyUS train frames; parents are the sim dataset (whose own
    sidecar names the translator), the Vitale split and the KidneyUS split."""
    from usbg import provenance as GP
    from us_seg import real_data as RD
    n_vit = len(result.get("vitale_trained_idx") or [])
    # parents = trained / rendered artifacts only (each has its own sidecar, so usbg.provenance.scan can
    # walk the lineage); the real-data split files are inputs, recorded by md5 in real_pool / extra.
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
        args.out, "segmenter", train_cases=result["train_source_cases"], val_cases=result["val_source_cases"],
        real_pool=real_pool, iters=args.iters, seed=args.seed, dr_mode="per_frame",
        dr_notes="sim frames rendered with per-frame R.domain_randomize (build_dataset)",
        parents=parents, repo=gym_bridge.seg_root(),
        extra={"vitale_split": os.path.abspath(args.vitale_split),
               "vitale_split_md5": gym_bridge.file_md5(os.path.abspath(args.vitale_split)),
               "kidneyus_split": RD._seg_path(RD._KID_SPLIT) if args.kidneyus else None,
               "kidneyus_split_md5": gym_bridge.file_md5(RD._seg_path(RD._KID_SPLIT)) if args.kidneyus else None,
               "vitale_trained_idx": result["vitale_trained_idx"],
               "n_kidneyus_train": result.get("n_kidneyus_train"), "history": os.path.splitext(args.out)[0] + "_history.json",
               "dataset_role": build.get("role"), "dataset_seat_organs": build.get("seat_organs"),
               "skipped_case_organs": build.get("skipped_case_organs"),
               "inference": "usbg.segmenter.Segmenter.predict (net.eval(), BatchNorm running stats), unchanged from v1"})


if __name__ == "__main__":
    raise SystemExit(main())
