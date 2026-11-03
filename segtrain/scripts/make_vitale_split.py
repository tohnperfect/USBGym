"""v2: make the ONE fixed Vitale train/holdout split used by every v2 segmenter (all stages, all seeds).

The 60 annotated Vitale frames come from the Kaggle RUS *test* letters {a,b,g,h}; the 404-frame
unlabeled pool the translators train on is the RUS *train* letters {c,d,e,f,i,j,k}, so the labeled
frames are already letter-group-disjoint from every translator. Inside the 60, two split modes:

  --mode group      (default) hold out whole letter groups (default g,b -> 18 frames; train a,h -> 42).
                    Letters are the dataset's own acquisition grouping (the Kaggle train/test split is
                    by letter), so this is the strongest disjointness available; it is NOT a proven
                    patient split.
  --mode stratified hold out --n-holdout frames (default 20), stratified so each organ keeps about
                    n_holdout/60 of its frames; frame-disjoint only.

Writes data/real/vitale_split_v2.npz (is_holdout, sessions, mode) and data/real/vitale_holdout_v2.npz
(images float32 [0,1], label6, sessions, valid_classes) for consistency.py / eval_seg.py --real-npz,
plus a JSON summary of per-organ counts. Refuses to overwrite an existing split unless --force.

  ./run.sh scripts/make_vitale_split.py --mode group --holdout-groups g,b
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from us_seg import gym_bridge, ontology as ONT, real_data  # noqa: E402


def organ_sets(label6: np.ndarray) -> list[set]:
    return [set(int(k) for k in np.unique(l) if k) for l in label6]


def stratified(label6: np.ndarray, n_holdout: int, seed: int) -> np.ndarray:
    n = len(label6)
    sets = organ_sets(label6)
    freq = {c: sum(c in s for s in sets) for c in range(1, ONT.N_CLASSES)}
    target = {c: round(freq[c] * n_holdout / n) for c in freq}
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    # rarest organ first, so spleen / GB get their share before liver fills the quota
    order = sorted(order, key=lambda i: min([freq[c] for c in sets[i]] or [n]))
    ho = np.zeros(n, bool)
    got = {c: 0 for c in freq}
    for i in order:
        if ho.sum() >= n_holdout:
            break
        if sets[i] and all(got[c] < target[c] + (1 if freq[c] > 10 else 0) for c in sets[i]):
            ho[i] = True
            for c in sets[i]:
                got[c] += 1
    for i in order:                      # top up to exactly n_holdout
        if ho.sum() >= n_holdout:
            break
        ho[i] = True
    return ho


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["group", "stratified"], default="group")
    ap.add_argument("--holdout-groups", default="g,b")
    ap.add_argument("--n-holdout", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-out", default=real_data.VITALE_SPLIT_V2)
    ap.add_argument("--holdout-out", default=real_data.VITALE_HOLDOUT_V2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if os.path.exists(args.split_out) and not args.force:
        raise SystemExit(f"{args.split_out} exists; the split is fixed once for all v2 runs (use --force "
                         "only before any v2 model has been trained)")
    rt = real_data.load_real_test()
    sessions = np.asarray(rt["sessions"]).astype(str)
    if args.mode == "group":
        groups = {g.strip() for g in args.holdout_groups.split(",") if g.strip()}
        is_ho = np.array([s[0] in groups for s in sessions], bool)
    else:
        is_ho = stratified(rt["label6"], args.n_holdout, args.seed)

    os.makedirs(os.path.dirname(args.split_out) or ".", exist_ok=True)
    np.savez_compressed(args.split_out, is_holdout=is_ho, sessions=sessions,
                        mode=np.array(f"{args.mode}:{args.holdout_groups if args.mode == 'group' else args.n_holdout}:seed{args.seed}"))
    np.savez_compressed(args.holdout_out, images=rt["images"][is_ho].astype(np.float32),
                        label6=rt["label6"][is_ho].astype(np.uint8), sessions=sessions[is_ho],
                        valid_classes=np.arange(ONT.N_CLASSES))
    sets = organ_sets(rt["label6"])
    summ = {"mode": args.mode, "n_train": int((~is_ho).sum()), "n_holdout": int(is_ho.sum()),
            "holdout_sessions": sessions[is_ho].tolist(),
            "per_organ_frames": {ONT.CLASS_NAMES[c]: {"train": int(sum(c in sets[i] for i in np.where(~is_ho)[0])),
                                                     "holdout": int(sum(c in sets[i] for i in np.where(is_ho)[0]))}
                                 for c in range(1, ONT.N_CLASSES)}}
    src_npz = gym_bridge.gym_path(gym_bridge.REAL_TEST_NPZ)
    summ["provenance"] = {
        "source_npz": src_npz, "source_md5": gym_bridge.file_md5(src_npz),
        "split_npz": os.path.abspath(args.split_out), "split_md5": gym_bridge.file_md5(args.split_out),
        "holdout_npz": os.path.abspath(args.holdout_out), "holdout_md5": gym_bridge.file_md5(args.holdout_out),
        "argv": sys.argv[1:], "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seg_git": gym_bridge.git_rev(gym_bridge.seg_root()),
        "note": "letter groups are the Kaggle RUS acquisition grouping; patient identity is unknown, so "
                "this is a group-disjoint split, not a proven patient-disjoint one",
    }
    with open(os.path.splitext(args.split_out)[0] + ".json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
