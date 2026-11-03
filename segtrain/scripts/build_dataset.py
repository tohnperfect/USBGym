"""Build the cross-domain segmentation dataset in nnU-Net v2 raw format.

Renders SonoSPADE-textured, domain-randomized simulated (image, 6-class label) pairs into
imagesTr/labelsTr, writes the 60 real test frames into imagesTs, and saves a real-eval npz plus a
domain sidecar. See segtrain/README.md.

Run (from repo root):
  ./run.sh scripts/build_dataset.py --smoke                # tiny, ~1-2 cases
  ./run.sh scripts/build_dataset.py --n-poses 12           # full build
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

# make `us_seg` importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from us_seg import gym_bridge, ontology as ONT, real_data
from us_seg.nnunet_raw import RawWriter
from us_seg.render_pairs import default_seat_organs, iter_sim_frames, load_translator


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/nnunet", help="root under which DatasetXXX_... is written")
    ap.add_argument("--dataset-id", type=int, default=11)
    ap.add_argument("--name", default="USOrgansSimReal")
    ap.add_argument("--translator", default="sonospade",
                    help="sonospade|cut|cyclegan|cyclegan_sc|none|<ckpt path>")
    ap.add_argument("--n-poses", type=int, default=12, help="pose draws per (case, seat-organ)")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all cached cases")
    ap.add_argument("--no-start-pose", action="store_true", help="only target views (half the frames)")
    ap.add_argument("--no-real-test", action="store_true", help="skip writing real test frames")
    ap.add_argument("--lateral-m", type=float, default=0.02)
    ap.add_argument("--axial-m", type=float, default=0.008)
    ap.add_argument("--fan-deg", type=float, default=15.0)
    ap.add_argument("--roll-deg", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true", help="tiny build: 2 cases, 2 poses")
    ap.add_argument("--cases", default="all",
                    help="all | train | test | comma list of stems. train/test = the gym's frozen "
                         "usbg.benchmark TRAIN_PATIENTS / TEST_PATIENTS. v2 training builds MUST use train.")
    ap.add_argument("--seat-organs", default=None,
                    help="comma list of seat organs. Default by seat: anatomical liver,kidney (spleen is "
                         "not a v2 target; it stays a class); canonical liver,kidney,spleen (v1 builds)")
    ap.add_argument("--skip-pairs", default="",
                    help="explicit case:organ pairs to skip (logged), e.g. s0058:kidney,s0522:kidney. "
                         "Applied before any rng draw, like a gym NoAcceptedWindow.")
    ap.add_argument("--role", choices=["auto", "train", "eval", "legacy"], default="auto",
                    help="train: no TEST patient may be rendered; eval: only TEST patients (a sim-eval "
                         "build for the sim side of the organ-identity table); legacy: the v1 all-case canonical "
                         "builds (no guard, no sidecar). auto: eval for --cases test, legacy for --cases "
                         "all, else train.")
    ap.add_argument("--strict-seat", action="store_true",
                    help="skip a (case, organ) with no label / no accepted window instead of the legacy "
                         "silent liver fallback; skips are logged in build_summary.json")
    ap.add_argument("--max-skip-frac", type=float, default=0.5,
                    help="fail if more than this fraction of (case, organ) pairs had no accepted window")
    ap.add_argument("--require-seat", default=None,
                    help="fail unless the gym reports this seat mode (e.g. anatomical)")
    args = ap.parse_args(argv)

    # Training data MUST use CT-oracle labels (texture-independent ground truth), never the learned
    # segmenter's own output. If the gym's curvi_render is later flag-gated on USBG_LEARNED_SEG
    # (usbg.view_match), forcing it off here keeps labels CT-derived: this is the
    # control the per-texture segmenter comparison depends on.
    os.environ["USBG_LEARNED_SEG"] = "0"

    if args.smoke:
        args.max_cases = args.max_cases or 2
        args.n_poses = 2

    gym_bridge.ensure_usbg_importable()

    train_p, test_p = gym_bridge.benchmark_split()
    if args.cases == "all":
        want = None
    elif args.cases == "train":
        want = train_p
    elif args.cases == "test":
        want = test_p
    else:
        want = [c.strip() for c in args.cases.split(",") if c.strip()]
    cases = gym_bridge.cache_cases(want)
    role = args.role if args.role != "auto" else \
        {"test": "eval", "all": "legacy"}.get(args.cases, "train")
    seat = gym_bridge.seat_mode()
    if role == "legacy" and seat == "anatomical":
        raise SystemExit("[guard] an anatomical (v2) build must say what it is: --cases train (or a TRAIN "
                         "list) for training, --cases test for the sim-eval build")
    if args.require_seat and seat != args.require_seat:
        raise SystemExit(f"[guard] gym seat mode is {seat!r}, required {args.require_seat!r} "
                         "(export USBG_SEAT=anatomical)")
    print(f"[seat] {seat}   [cache] {gym_bridge.cache_dir()} -> variant "
          f"{gym_bridge.expected_cache_variant(seat)!r}   [role] {role}")
    if args.max_cases:
        cases = cases[: args.max_cases]
    stems = [os.path.splitext(os.path.basename(c))[0] for c in cases]
    if role == "train":
        leak = sorted(set(stems) & set(test_p))
        if leak:
            raise SystemExit(f"[guard] held-out TEST patients in a training build: {leak}")
    elif role == "eval":
        extra = sorted(set(stems) - set(test_p))
        if extra:
            raise SystemExit(f"[guard] a sim-eval build may hold only TEST patients; got {extra}")
    print(f"[cases] {len(cases)} CT case(s): {stems}")

    print(f"[translator] loading {args.translator!r}")
    tx_path = gym_bridge.translator_ckpt(args.translator)
    if seat == "anatomical":
        why = gym_bridge.is_legacy_translator(args.translator, tx_path)
        if why:
            raise SystemExit(f"[guard] anatomical (v2) build with a D3-leaky v1 translator: {why}. "
                             "Pass the abs path from the gym's outputs/v2/textures/manifest.json.")
    translator = load_translator(args.translator)
    print(f"            -> {type(translator).__name__ if translator is not None else None}")

    writer = RawWriter(args.out, dataset_id=args.dataset_id, name=args.name)
    setup_kw = dict(lateral_m=args.lateral_m, axial_m=args.axial_m,
                    fan_deg=args.fan_deg, roll_deg=args.roll_deg)

    # ---- simulated training frames --------------------------------------------------------------
    t0 = time.time()
    n = 0
    organ_frames = Counter()   # 6-class organ name -> #frames containing it
    per_case = Counter()
    seat_organs = (tuple(o.strip() for o in args.seat_organs.split(",") if o.strip())
                   if args.seat_organs else default_seat_organs(seat))
    skip_pairs = []
    for tok in (t.strip() for t in args.skip_pairs.split(",") if t.strip()):
        c, _, o = tok.partition(":")
        if o not in seat_organs and o:
            print(f"[seat] note: --skip-pairs {tok} names an organ that is not seated here")
        skip_pairs.append((c, o))
    skipped: list = []
    case_meta: dict = {}
    seat_counts = Counter()      # (organ, window side) -> frames, anatomical seat
    clamp_n = 0
    clamp_mm: list = []
    for fr in iter_sim_frames(
        cases, translator,
        n_poses=args.n_poses, size=args.size, seed=args.seed,
        include_start=not args.no_start_pose, setup_kw=setup_kw,
        seat_organs=seat_organs, strict_seat=args.strict_seat, skipped=skipped,
        skip_pairs=skip_pairs, case_meta=case_meta,
    ):
        writer.add_training(
            fr.image, fr.label6,
            provenance={
                "domain": "sim", "case": fr.case, "seat_organ": fr.seat_organ,
                "pose": fr.pose, "organs_present": fr.organs_present,
                "translator": args.translator, "seat_mode": seat,
                "seat_meta": fr.seat,
            },
        )
        n += 1
        seat_counts[f"{fr.seat_organ}:{fr.seat.get('window_organ', '-')}"] += 1
        clamp_n += int(bool(fr.seat.get("start_clamped")))
        if fr.pose == "start":
            clamp_mm.append(float(fr.seat.get("clamp_mm") or 0.0))
        per_case[fr.case] += 1
        for org in fr.organs_present:
            organ_frames[org] += 1
        if n % 100 == 0:
            print(f"  ...{n} sim frames ({n / (time.time() - t0):.1f}/s)")
    dt = time.time() - t0
    print(f"[sim] wrote {n} training frames in {dt:.1f}s ({n / max(dt, 1e-9):.1f}/s)")
    n_pairs = len(cases) * len(seat_organs)
    if skipped:
        print(f"[seat] SKIPPED {len(skipped)}/{n_pairs} (case, organ): "
              + ", ".join(f"{x['case']}:{x['organ']}" for x in skipped))
    if n == 0 or len(skipped) > args.max_skip_frac * n_pairs:
        raise SystemExit(f"[guard] {n} frames, {len(skipped)}/{n_pairs} (case, organ) skipped "
                         f"(> --max-skip-frac {args.max_skip_frac}); the seat/window API is not working")
    print(f"[sim] per-organ frame coverage: {dict(organ_frames)}")
    print(f"[seat] frames per seat organ:window side: {dict(seat_counts)}; "
          f"start poses lifted outside the skin: {clamp_n}/{len(clamp_mm)}"
          + (f" (lift mm median {np.median(clamp_mm):.1f}, max {np.max(clamp_mm):.1f})" if clamp_mm else ""))
    bad_var = {c: m["cache_variant"] for c, m in case_meta.items()
               if m["cache_variant"] != gym_bridge.expected_cache_variant(seat)}
    if bad_var:
        raise SystemExit(f"[guard] cache variant mismatch for seat {seat!r}: {bad_var}")

    # ---- real test frames -----------------------------------------------------------------------
    if not args.no_real_test:
        rt = real_data.load_real_test()
        for i in range(rt["n"]):
            writer.add_test_image(
                rt["images"][i],
                provenance={"domain": "real", "session": str(rt["sessions"][i]),
                            "source": "kaggle_vitale_test"},
            )
        eval_npz = os.path.join(writer.dir, "real_eval.npz")
        np.savez_compressed(
            eval_npz,
            images=rt["images"].astype(np.float32),
            label6=rt["label6"].astype(np.uint8),
            label_repo=rt["label_repo"].astype(np.uint8),
            sessions=rt["sessions"].astype(str),
        )
        real_organs = Counter()
        for i in range(rt["n"]):
            for k in np.unique(rt["label6"][i]).tolist():
                if k:
                    real_organs[ONT.CLASS_NAMES[k]] += 1
        print(f"[real] wrote {rt['n']} test images + real_eval.npz; per-organ coverage: {dict(real_organs)}")

    out_dir = writer.finalize()
    summary = {
        "dataset_dir": out_dir,
        "n_train_sim": n,
        "n_test_real": 0 if args.no_real_test else rt["n"],
        "translator": args.translator,
        "size": args.size,
        "sim_organ_frame_coverage": dict(organ_frames),
        "labels": ONT.NNUNET_LABELS,
        "six_to_repo": ONT.SIX_TO_REPO,
        "cases": stems,
        "cases_arg": args.cases,
        "per_case_frames": dict(per_case),
        "seat_organs": list(seat_organs),
        "strict_seat": bool(args.strict_seat),
        "skipped_case_organs": skipped,
        "n_poses": args.n_poses, "seed": args.seed, "setup_kw": setup_kw,
        "include_start": not args.no_start_pose,
        "domain_randomization": "per-frame R.domain_randomize(R.RenderConfig(), rng)",
        "translator_path": tx_path,
        "translator_md5": gym_bridge.file_md5(tx_path),
        "translator_provenance_json": _tx_sidecar(tx_path),
        "role": role,
        "skip_pairs": [f"{c}:{o}" for c, o in skip_pairs],
        "n_skipped_case_organs": len(skipped),
        "seat_frames": dict(seat_counts),
        "n_start_clamped": clamp_n, "n_start_poses": len(clamp_mm),
        "start_clamp_mm": ({"median": float(np.median(clamp_mm)), "max": float(np.max(clamp_mm))}
                           if clamp_mm else None),
        "case_cache": case_meta,
        "provenance": gym_bridge.provenance(),
    }
    with open(os.path.join(out_dir, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if role != "legacy":
        _write_sidecar(out_dir, summary, role, stems, tx_path, seat)
    print(f"\n[done] {out_dir}")
    print(json.dumps(summary, indent=2))
    return 0


def _tx_sidecar(tx_path):
    """The translator's provenance sidecar (the gym writes <ckpt>.provenance.json), if present."""
    if not tx_path:
        return None
    for c in (tx_path + ".provenance.json", os.path.splitext(tx_path)[0] + ".provenance.json"):
        if os.path.exists(c):
            return {"path": c, "md5": gym_bridge.file_md5(c)}
    return None


def _write_sidecar(out_dir, summary, role, stems, tx_path, seat):
    """<dataset dir>.provenance.json via the gym's usbg.provenance (kind dataset / eval). A train
    build lists its CT cases as train_cases (write_provenance refuses a TEST patient there); an eval
    build lists none (its TEST cases go in extra.eval_cases)."""
    try:
        from usbg import provenance as GP
    except Exception as e:  # noqa: BLE001
        print(f"[prov] usbg.provenance unavailable ({e!r}); no sidecar written")
        return None
    parents = [GP.parent_entry(tx_path, "translator")] if tx_path else []
    kind = "dataset" if role == "train" else "eval"
    sc = GP.write_provenance(
        out_dir, kind, train_cases=(stems if role == "train" else []), val_cases=[],
        iters=0, seed=int(summary["seed"]), dr_mode="per_frame",
        dr_notes=summary["domain_randomization"], parents=parents, repo=gym_bridge.seg_root(),
        seat_mode=seat if seat in GP.SEAT_MODES else None,
        extra={"role": role, "eval_cases": (stems if role == "eval" else []),
               "seat_organs": summary["seat_organs"], "skipped_case_organs": summary["skipped_case_organs"],
               "n_frames": summary["n_train_sim"], "translator": summary["translator"],
               "windows": summary["provenance"].get("windows"),
               "gym_git": summary["provenance"].get("gym_git_full")})
    print(f"[prov] {sc}")
    return sc


if __name__ == "__main__":
    raise SystemExit(main())
