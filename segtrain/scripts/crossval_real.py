"""Leakage-free real Dice via K-fold cross-validation over the real Vitale frames.

Real frames are used for TRAINING (mixed with sim), so a clean generalization number needs CV: for
each of K folds, train on sim + the other K-1 real folds, evaluate on the held-out real fold; then
aggregate per-class Dice (mean +/- std across folds). This is the honest real number for the paper.
The deployment model (trained on sim + ALL real) is produced separately by scripts/train.py.

  ./run.sh scripts/crossval_real.py --folds 6 --iters 2500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from us_seg import gym_bridge, ontology as ONT, real_data, training
from us_seg.infer import load_segmenter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/nnunet/Dataset011_USOrgansSimReal")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--real-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/crossval_real.json")
    ap.add_argument("--kidneyus", action="store_true",
                    help="fold the leakage-free KidneyUS partial-label TRAIN frames into each fold's "
                         "training set (kidney supervised, non-kidney pixels ignored), matching "
                         "train.py --kidneyus. Required for the OOF to correspond to a "
                         "kidney-augmented deployment model.")
    ap.add_argument("--oof-out", default=None,
                    help="where to write the out-of-fold real predictions. Defaults to a path derived "
                         "from --out rather than a shared outputs/real_oof.npz: one shared path lets a "
                         "later texture's crossval silently hand its real predictions to an earlier "
                         "texture's consistency run.")
    ap.add_argument("--vitale-split", default=None,
                    help="v2: fixed Vitale split npz. Folds partition ONLY its non-holdout frames; the "
                         "holdout frames are never trained on by any fold.")
    ap.add_argument("--deploy-ckpt", default=None,
                    help="v2: deployment checkpoint (train.py with the same --vitale-split). Its "
                         "predictions fill the holdout frames, so the OOF npz holds a leakage-free "
                         "prediction for all 60 frames (covered mask + pred_source recorded).")
    ap.add_argument("--forbid-test-patients", action="store_true")
    args = ap.parse_args(argv)

    # A single hardcoded outputs/real_oof.npz is what let cyclegan seed 2's real predictions end up
    # scoring every texture's consistency: whoever ran last won, and consistency.py's --oof default
    # picked it up. Derive the path from --out so each run owns its own file.
    oof_out = args.oof_out or (os.path.splitext(args.out)[0].replace("crossval", "real_oof") + ".npz")

    real = real_data.load_real_test()
    n_real = real["n"]
    rng = np.random.default_rng(args.seed)
    # split by SESSION (patient), not raw frame, so folds stay leakage-free even if a session has
    # multiple frames. (Vitale currently has 1 frame/session, so this reduces to a frame split.)
    sessions = np.asarray(real["sessions"]).astype(str)
    pool = np.arange(n_real)
    split = None
    if args.vitale_split:
        split = real_data.load_vitale_split(args.vitale_split)
        pool = split["train_idx"]
    uniq = sorted(set(sessions[pool].tolist()))
    rng.shuffle(uniq)
    sess_folds = np.array_split(np.array(uniq, dtype=object), args.folds)
    folds = [np.intersect1d(np.where(np.isin(sessions, list(sf)))[0], pool) for sf in sess_folds]

    # out-of-fold real predictions: each frame predicted by a model that never trained on it
    oof_pred = np.zeros_like(real["label6"], dtype=np.uint8)
    covered = np.zeros(n_real, dtype=bool)
    pred_source = np.array(["none"] * n_real, dtype=object)
    forbid = gym_bridge.benchmark_split()[1] if args.forbid_test_patients else None
    fold_stem = os.path.splitext(args.out)[0]   # per-run fold ckpts: parallel crossvals must not collide

    per_class = defaultdict(list)   # class -> [fold dice]
    fold_means = []
    fold_records = []
    for k, ho in enumerate(folds):
        ckpt = f"{fold_stem}_fold{k}.pt"
        print(f"\n===== fold {k + 1}/{args.folds}: hold out {len(ho)} real frames =====", flush=True)
        res = training.train(
            args.dataset, ckpt,
            iters=args.iters, val_frac=0.2, seed=args.seed,
            real_frac=args.real_frac, real_holdout_idx=ho.tolist(),
            eval_every=max(args.iters // 2, 1),
            add_kidneyus=args.kidneyus,
            real_train_idx=(pool.tolist() if split is not None else None),
            forbid_sim_cases=forbid,
        )
        last = res["history"][-1]
        fold_records.append({"fold": k, "held_out": ho.tolist(), "real": last["real"],
                             "real_mean": last["real_mean"], "sim_val_mean": last["sim_val_mean"]})
        fold_means.append(last["real_mean"])
        for cls, d in last["real"].items():
            per_class[cls].append(d)
        # save this fold's held-out predictions (leakage-free) for the consistency metric
        seg = load_segmenter(ckpt)
        for i in ho.tolist():
            oof_pred[i] = seg.segment6(real["images"][i]).astype(np.uint8)
            covered[i] = True
            pred_source[i] = f"oof_fold{k}"
        os.remove(ckpt)  # fold models are throwaway; deployment model is trained separately

    if args.deploy_ckpt and split is not None:
        # the deploy model fills the holdout ONLY if it never trained on a holdout frame
        dh = os.path.splitext(args.deploy_ckpt)[0] + "_history.json"
        if not os.path.exists(dh):
            raise SystemExit(f"[guard] {dh} missing: cannot verify the deploy model's Vitale frames")
        dtr = set(json.load(open(dh)).get("vitale_trained_idx") or [])
        bad = sorted(dtr & set(split["holdout_idx"].tolist()))
        if not dtr or bad:
            raise SystemExit(f"[guard] deploy model {args.deploy_ckpt} trained on Vitale holdout frames {bad} "
                             f"(or has no vitale_trained_idx record)")
        dep = load_segmenter(args.deploy_ckpt)
        for i in split["holdout_idx"].tolist():
            oof_pred[i] = dep.segment6(real["images"][i]).astype(np.uint8)
            covered[i] = True
            pred_source[i] = "deploy_holdout"
    if args.deploy_ckpt and split is not None and not covered.all():
        raise SystemExit(f"[guard] OOF covers {int(covered.sum())}/{n_real} Vitale frames; expected all")
    np.savez_compressed(oof_out, pred6=oof_pred,
                        label6=real["label6"].astype(np.uint8),
                        covered=covered, pred_source=pred_source.astype(str),
                        sessions=sessions,
                        is_holdout=(split["is_holdout"] if split is not None else np.zeros(n_real, bool)))
    print(f"[oof] wrote {oof_out} (leakage-free real predictions for consistency)", flush=True)

    agg = {cls: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4),
                 "folds": len(v)} for cls, v in per_class.items()}
    summary = {
        "folds": args.folds, "iters": args.iters, "real_frac": args.real_frac,
        "per_class_real_dice": agg,
        "macro_mean_real_dice": round(float(np.mean(fold_means)), 4),
        "macro_std_real_dice": round(float(np.std(fold_means)), 4),
        "fold_records": fold_records,
        "oof_out": oof_out, "n_covered": int(covered.sum()),
        "n_oof": int(sum(str(x).startswith("oof_") for x in pred_source)),
        "n_deploy_fill": int(sum(str(x) == "deploy_holdout" for x in pred_source)),
        "fold_sizes": [int(len(f)) for f in folds],
        "provenance": gym_bridge.provenance(
            argv=sys.argv[1:], vitale_split=args.vitale_split,
            deploy_ckpt=args.deploy_ckpt, deploy_ckpt_md5=gym_bridge.file_md5(args.deploy_ckpt),
            vitale_split_md5=gym_bridge.file_md5(args.vitale_split) if args.vitale_split else None,
            oof_md5=gym_bridge.file_md5(oof_out)),
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== CROSS-VALIDATED REAL DICE (leakage-free) =====", flush=True)
    print(json.dumps({k: summary[k] for k in ("per_class_real_dice", "macro_mean_real_dice",
                                              "macro_std_real_dice")}, indent=2))
    print("\n% ---- paste into Table~\\ref{tab:seg-dice} (real column) ----")
    for name in ONT.CLASS_NAMES[1:]:
        a = agg.get(name)
        cell = f"{a['mean']:.3f}$\\pm${a['std']:.3f}" if a else r"\blankcell"
        print(f"{name.capitalize():<12} & {cell} \\\\")
    print(f"[report] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
