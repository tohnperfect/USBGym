"""Evaluate a trained checkpoint: per-class Dice on held-out sim and on real.

NOTE: the real column here is a FIT if the checkpoint trained on all 60 real frames (the deployment
model does): it is NOT leakage-free. Use scripts/crossval_real.py for the honest real Dice. Sim is
clean (held-out CT cases). Emits a JSON report + LaTeX rows for Table~\\ref{tab:seg-dice}.

  ./run.sh scripts/eval_seg.py --ckpt outputs/us_seg_6cls.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from us_seg import gym_bridge, ontology as ONT, real_data
from us_seg.infer import load_segmenter
from us_seg.nnunet_raw import RawDataset
from us_seg.training import per_class_dice, mean_fg_dice


def _fmt(pcd: dict, name: str) -> str:
    v = pcd.get(name)
    return f"{v[0]:.3f}" if v else r"\blankcell"


def latex_dice_table(sim_pcd: dict, real_pcd: dict) -> str:
    rows = []
    for name in ONT.CLASS_NAMES[1:]:
        rows.append(f"{name.capitalize():<12} & {_fmt(sim_pcd, name)} & {_fmt(real_pcd, name)} \\\\")
    rows.append(r"\midrule")
    rows.append(f"{'Mean':<12} & {mean_fg_dice(sim_pcd):.3f} & {mean_fg_dice(real_pcd):.3f} \\\\")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="outputs/us_seg_6cls.pt")
    ap.add_argument("--dataset", default="data/nnunet/Dataset011_USOrgansSimReal")
    ap.add_argument("--val-frac", type=float, default=0.2, help="match training to reproduce sim val split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sim-all", action="store_true",
                    help="v2: score every frame of --dataset (a TEST-patient sim-eval build)")
    ap.add_argument("--real-npz", action="append", default=[],
                    help="v2: extra real sets (images,label6), never trained on; repeatable. Partial-label "
                         "sets (AUL liver, KidneyUS kidney) only report their labeled organ.")
    args = ap.parse_args(argv)

    seg = load_segmenter(args.ckpt)
    print(f"[ckpt] {args.ckpt} on {seg.device}")

    # sim held-out val (reproduce the training split by seed/val_frac)
    ds = RawDataset(args.dataset)
    bs_path = os.path.join(args.dataset, "build_summary.json")
    build = json.load(open(bs_path)) if os.path.exists(bs_path) else {}
    if args.sim_all and build.get("role") == "train":
        raise SystemExit(f"[guard] --sim-all on a TRAINING build ({args.dataset}) is in-sample")
    if args.sim_all:
        va_cases = ds.train_cases
    else:
        _, va_cases = ds.split_by_source_case(val_frac=args.val_frac, seed=args.seed)
    Xva, Yva = ds.load_all(va_cases)
    sim_pred = np.stack([seg.segment6(im) for im in Xva])
    sim_pcd = per_class_dice(sim_pred, Yva)

    # real (fit if the checkpoint trained on these frames; see crossval_real.py for leakage-free)
    rt = real_data.load_real_test()
    real_pred = np.stack([seg.segment6(im) for im in rt["images"]])
    real_pcd = per_class_dice(real_pred, rt["label6"])

    report = {
        "checkpoint": args.ckpt,
        "sim_val": {"cases": len(va_cases), "frames": len(Xva),
                    "per_class": {k: round(v[0], 4) for k, v in sim_pcd.items()},
                    "support": {k: v[1] for k, v in sim_pcd.items()},
                    "mean": round(mean_fg_dice(sim_pcd), 4)},
        "real_test": {"frames": rt["n"],
                      "per_class": {k: round(v[0], 4) for k, v in real_pcd.items()},
                      "support": {k: v[1] for k, v in real_pcd.items()},
                      "mean": round(mean_fg_dice(real_pcd), 4)},
    }
    for rn in args.real_npz:
        d = np.load(rn, allow_pickle=True)
        X = d["images"].astype(np.float32)
        Y = d["label6"].astype(np.int64)
        valid = [int(v) for v in d["valid_classes"]] if "valid_classes" in d.files else list(range(ONT.N_CLASSES))
        P = np.stack([seg.segment6(im) for im in X])
        pcd = {k: v for k, v in per_class_dice(P, Y).items() if ONT.NNUNET_LABELS[k] in valid}
        iou = {}
        for k in pcd:
            c = ONT.NNUNET_LABELS[k]
            vals = [float(np.logical_and(p == c, t == c).sum()) / max(float(np.logical_or(p == c, t == c).sum()), 1.0)
                    for p, t in zip(P, Y) if (t == c).any()]
            iou[k] = round(float(np.mean(vals)), 4)
        report[os.path.basename(rn)] = {"frames": int(len(X)),
                                        "per_class": {k: round(v[0], 4) for k, v in pcd.items()},
                                        "per_class_iou": iou,
                                        "support": {k: v[1] for k, v in pcd.items()}}
    print(json.dumps(report, indent=2))
    print("\n% ---- paste into Table~\\ref{tab:seg-dice} ----")
    print(latex_dice_table(sim_pcd, real_pcd))

    if args.sim_all:
        report["real_test"]["note"] = ("all 60 Vitale frames incl. the frames a v2 model trained on (a FIT); "
                                       "report only the --real-npz blocks")
    report["sim_build"] = {"dataset": args.dataset, "role": build.get("role"), "cases": build.get("cases"),
                           "translator_md5": build.get("translator_md5")}
    report["provenance"] = gym_bridge.provenance(
        argv=sys.argv[1:], ckpt_md5=gym_bridge.file_md5(args.ckpt),
        real_npz_md5={rn: gym_bridge.file_md5(rn) for rn in args.real_npz})
    out = args.out or (os.path.splitext(args.ckpt)[0] + "_eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[report] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
