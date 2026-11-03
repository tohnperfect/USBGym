"""Cross-domain class-assignment consistency: does the segmenter label the same organ region with
the same class in sim and in real?  (The controller's real requirement, not per-frame Dice.)

Sim side = deployment model on held-out sim cases (clean). Real side = leakage-free out-of-fold
predictions from scripts/crossval_real.py (outputs/real_oof.npz) if present, else the deployment
model on real (a fit, flagged as optimistic).

  ./run.sh scripts/consistency.py --ckpt outputs/us_seg_6cls.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from us_seg import gym_bridge, ontology as ONT, real_data, consistency as C
from us_seg.infer import load_segmenter
from us_seg.nnunet_raw import RawDataset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="outputs/us_seg_6cls.pt")
    ap.add_argument("--dataset", default="data/nnunet/Dataset011_USOrgansSimReal")
    ap.add_argument("--oof", default=None,
                    help="out-of-fold real predictions from crossval_real.py for THIS texture+seed. "
                         "No default on purpose: it used to point at a shared outputs/real_oof.npz, "
                         "so a run that omitted it scored against whichever texture crossvalled last "
                         "(cyclegan seed 2's predictions reached every row of a published table).")
    ap.add_argument("--real-npz", default=None,
                    help="external real npz (images,label6) to use as the real source, e.g. AUL; "
                         "these frames were never trained on, so the eval is leakage-free")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/consistency.json")
    ap.add_argument("--sim-all", action="store_true",
                    help="v2: use EVERY frame of --dataset as the sim side (a sim-eval build on the "
                         "held-out TEST patients) instead of the seed's val split of a training build")
    args = ap.parse_args(argv)

    seg = load_segmenter(args.ckpt)

    # sim: clean held-out cases
    ds = RawDataset(args.dataset)
    bs_path = os.path.join(args.dataset, "build_summary.json")
    build = json.load(open(bs_path)) if os.path.exists(bs_path) else {}
    if args.sim_all:
        if build.get("role") == "train":
            raise SystemExit(f"[guard] --sim-all on a TRAINING build ({args.dataset}) would score the "
                             "segmenter in-sample; pass the TEST-patient sim-eval build")
        va = ds.train_cases
    else:
        _, va = ds.split_by_source_case(val_frac=0.2, seed=args.seed)
    Xs, Ys = ds.load_all(va)
    sim_src = sorted({ds.source_case(c) for c in va})
    sim_pred = np.stack([seg.segment6(im) for im in Xs])

    # real source (priority): external npz > out-of-fold > deployment-fit
    if args.real_npz:
        d = np.load(args.real_npz)
        imgs = d["images"].astype(np.float32)
        real_gt = d["label6"].astype(np.int64)
        real_pred = np.stack([seg.segment6(im) for im in imgs])
        real_kind = f"external real npz, never trained on (leakage-free): {args.real_npz}"
    elif args.oof and os.path.exists(args.oof):
        d = np.load(args.oof, allow_pickle=True)
        real_pred = d["pred6"].astype(np.int64)
        real_gt = d["label6"].astype(np.int64)
        if "covered" in d.files:   # v2 OOF: score only frames that actually received a prediction
            cov = np.asarray(d["covered"], bool)
            real_pred, real_gt = real_pred[cov], real_gt[cov]
        real_kind = f"leakage-free out-of-fold ({args.oof}, n={len(real_pred)})"
    else:
        rt = real_data.load_real_test()
        real_pred = np.stack([seg.segment6(im) for im in rt["images"]])
        real_gt = rt["label6"].astype(np.int64)
        real_kind = "deployment model fit on all real (OPTIMISTIC - run crossval_real.py for leakage-free)"

    rep = C.consistency_report(sim_pred, Ys, real_pred, real_gt)
    rep["real_prediction_source"] = real_kind
    rep["n_sim_frames"] = int(len(Xs))
    rep["sim_source_cases"] = sim_src
    rep["n_real_frames"] = int(len(real_pred))
    rep["sim_build"] = {"dataset": args.dataset, "role": build.get("role"), "cases": build.get("cases"),
                        "translator_path": build.get("translator_path"),
                        "translator_md5": build.get("translator_md5"),
                        "seat_mode": (build.get("provenance") or {}).get("seat_mode")}
    rep["provenance"] = gym_bridge.provenance(
        argv=sys.argv[1:], ckpt_md5=gym_bridge.file_md5(args.ckpt),
        oof_md5=gym_bridge.file_md5(args.oof) if args.oof else None,
        real_npz_md5=gym_bridge.file_md5(args.real_npz) if args.real_npz else None)

    print(f"[real predictions] {real_kind}\n")
    print(f"MACRO cross-domain consistency (hist-intersection of P(pred|organ), sim vs real): "
          f"{rep['macro_consistency']:.3f}")
    print(f"Modal-class match rate (organ called the same in both domains): {rep['modal_match_rate']:.3f}\n")
    hdr = f"{'organ':<12}{'sim->self':>10}{'real->self':>11}{'modal sim/real':>18}{'consistency':>13}{'JSD':>7}"
    print(hdr); print("-" * len(hdr))
    for name, m in rep["per_class"].items():
        modal = f"{m['modal_class_sim']}/{m['modal_class_real']}"
        flag = "" if m["modal_is_correct"] else ("  <-- mismatch" if not m["modal_match"] else "  <-- both wrong-but-consistent")
        print(f"{name:<12}{m['sim_self_rate']:>10.3f}{m['real_self_rate']:>11.3f}{modal:>18}"
              f"{m['hist_intersection']:>13.3f}{m['js_divergence']:>7.3f}{flag}")

    print("\n% ---- Table~\\ref{tab:seg-agree}: cross-domain class-assignment consistency ----")
    for name in ONT.CLASS_NAMES[1:]:
        m = rep["per_class"].get(name)
        cell = f"{m['hist_intersection']:.3f}" if m else r"\blankcell"
        print(f"{name.capitalize():<12} & {cell} \\\\")
    macro = rep["macro_consistency"]
    print(r"\midrule")
    print(f"{'Macro':<12} & {macro:.3f} \\\\")

    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\n[report] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
