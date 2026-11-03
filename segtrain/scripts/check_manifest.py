"""v2 leakage guard (segmenter side): fail if a held-out benchmark TEST patient appears in any
segmenter TRAINING manifest, or if a v2 checkpoint trained on a Vitale holdout frame.

Checks
  * every --datasets dir (training builds): no TEST patient in domain.json split=train source cases;
    build_summary role is not 'eval'; with --require-seat: the gym-reported seat_mode matches, every
    case was loaded from the matching cache variant (anatomical -> the bladder-free cache_anat), and the
    translator is not a D3-leaky v1 checkpoint
  * every --eval-datasets dir (TEST-patient sim-eval builds): only TEST patients, role 'eval', same seat
  * every *_history.json: train_source_cases / val_source_cases hold no TEST patient, a
    vitale_trained_idx record exists and misses the Vitale holdout, and the history's dataset is not an
    eval build
  * --seg-manifest (outputs/v2/seg_manifest.json): every entry exists, its history passes the checks
    above, and its .provenance.json sidecar md5 matches the checkpoint

  ./run.sh scripts/check_manifest.py --datasets data/nnunet_v2/Dataset1[1-4]?_* \
      --eval-datasets data/nnunet_v2/Dataset2[1-4]?_* \
      --histories outputs/v2/us_seg_*_history.json --vitale-split data/real/vitale_split_v2.npz \
      --seg-manifest outputs/v2/seg_manifest.json --require-seat anatomical
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from us_seg import gym_bridge, real_data  # noqa: E402


def _build(d):
    bs = os.path.join(d, "build_summary.json")
    return json.load(open(bs)) if os.path.exists(bs) else None


def _seat_checks(d, b, require_seat):
    bad = []
    if b is None:
        return [f"{d}: no build_summary.json"]
    seat = (b.get("provenance") or {}).get("seat_mode")
    if seat != require_seat:
        bad.append(f"{d}: seat_mode {seat!r} != {require_seat!r}")
    want = gym_bridge.expected_cache_variant(require_seat)
    cc = b.get("case_cache")
    if not cc:
        bad.append(f"{d}: no per-case cache record (pre-v2 build?)")
    else:
        wrong = {c: m.get("cache_variant") for c, m in cc.items() if m.get("cache_variant") != want}
        if wrong:
            bad.append(f"{d}: cases loaded from the wrong cache variant (want {want!r}): {wrong}")
    why = gym_bridge.is_legacy_translator(b.get("translator"), b.get("translator_path")) \
        if b.get("translator") not in (None, "none") else None
    if why and require_seat == "anatomical":
        bad.append(f"{d}: legacy translator in a v2 build ({why})")
    return bad


def check_history(h, test_p, ho):
    bad = []
    r = json.load(open(h))
    srcs = set(r.get("train_source_cases", [])) | set(r.get("val_source_cases", []))
    if not srcs:
        bad.append(f"{h}: no source-case record (pre-v2 history?)")
    leak = sorted(srcs & test_p)
    if leak:
        bad.append(f"{h}: trained/validated on TEST patients {leak}")
    if ho is not None:
        tr = set(r.get("vitale_trained_idx", []))
        if not tr:
            bad.append(f"{h}: no vitale_trained_idx record")
        elif tr & ho:
            bad.append(f"{h}: trained on Vitale holdout frames {sorted(tr & ho)}")
    dd = r.get("dataset_dir")
    if dd and os.path.isdir(dd):
        b = _build(dd)
        if b and b.get("role") == "eval":
            bad.append(f"{h}: trained on a sim-EVAL build {dd}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--eval-datasets", nargs="*", default=[])
    ap.add_argument("--histories", nargs="*", default=[])
    ap.add_argument("--vitale-split", default=None)
    ap.add_argument("--seg-manifest", default=None)
    ap.add_argument("--require-seat", default=None)
    args = ap.parse_args(argv)

    _, test_p = gym_bridge.benchmark_split()
    test_p = set(test_p)
    bad = []
    # a shell glob like Dataset1??_* also matches the <dataset>.provenance.json sidecars: keep dirs only
    for name in ("datasets", "eval_datasets"):
        keep = [d for d in getattr(args, name) if os.path.isdir(d)]
        extra = [d for d in getattr(args, name) if not os.path.isdir(d) and not d.endswith(".provenance.json")]
        bad += [f"{d}: not a dataset directory" for d in extra]
        setattr(args, name, keep)
    for d in args.datasets:
        dom = json.load(open(os.path.join(d, "domain.json")))
        srcs = {v.get("case") for v in dom.values() if v.get("split") == "train"}
        leak = sorted(srcs & test_p)
        if leak:
            bad.append(f"{d}: TEST patients in training manifest {leak}")
        b = _build(d)
        if b is not None and b.get("role") == "eval":
            bad.append(f"{d}: is a sim-eval build; pass it via --eval-datasets")
        if args.require_seat:
            bad += _seat_checks(d, b, args.require_seat)
    for d in args.eval_datasets:
        dom = json.load(open(os.path.join(d, "domain.json")))
        srcs = {v.get("case") for v in dom.values() if v.get("split") == "train"}
        extra = sorted(srcs - test_p)
        if extra:
            bad.append(f"{d}: sim-eval build holds non-TEST patients {extra}")
        b = _build(d)
        if b is None or b.get("role") != "eval":
            bad.append(f"{d}: role {None if b is None else b.get('role')!r}, expected 'eval'")
        if args.require_seat:
            bad += _seat_checks(d, b, args.require_seat)
    ho = None
    if args.vitale_split:
        ho = set(real_data.load_vitale_split(args.vitale_split)["holdout_idx"].tolist())
    for h in args.histories:
        bad += check_history(h, test_p, ho)
    n_man = 0
    if args.seg_manifest:
        man = json.load(open(args.seg_manifest))
        hist_abs = {os.path.abspath(h) for h in args.histories}
        for key, ck in man.items():
            n_man += 1
            if not os.path.isabs(ck) or not os.path.exists(ck):
                bad.append(f"{args.seg_manifest}: {key} -> {ck} missing or not absolute")
                continue
            h = os.path.splitext(ck)[0] + "_history.json"
            if not os.path.exists(h):
                bad.append(f"{args.seg_manifest}: {key}: no history {h}")
            elif os.path.abspath(h) not in hist_abs:
                bad += check_history(h, test_p, ho)
            sc = ck + ".provenance.json"
            if not os.path.exists(sc):
                bad.append(f"{args.seg_manifest}: {key}: no provenance sidecar {sc}")
            else:
                rec = json.load(open(sc))
                if rec.get("artifact", {}).get("md5") != gym_bridge.file_md5(ck):
                    bad.append(f"{args.seg_manifest}: {key}: sidecar md5 != checkpoint md5 (STALE)")
                if args.require_seat and rec.get("seat_mode") != args.require_seat:
                    bad.append(f"{args.seg_manifest}: {key}: sidecar seat {rec.get('seat_mode')!r}")
    if bad:
        print("MANIFEST CHECK FAILED:")
        for b in bad:
            print("  -", b)
        return 1
    print(f"manifest check clean: {len(args.datasets)} train dataset(s), {len(args.eval_datasets)} eval "
          f"dataset(s), {len(args.histories)} history file(s), {n_man} seg-manifest entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
