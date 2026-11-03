#!/usr/bin/env python3
"""Phase 4 end-to-end: drive the env with the Repo B demonstrator (clean parametric + noisy
PPO) across all maneuvers and cases, render the B-mode each frame, and write the (image, pose,
twist, maneuver, next, ttnb, instruction) dataset (npz + LeRobot-style) plus a contact sheet.

Each maneuver's RELATIVE motion is replayed on a base pose seated over the target organ, so the
body twist (hence the labels) is preserved while the render intersects anatomy.

Usage:
  python scripts/make_dataset.py --cases s1086 s0777 --n-clean 3 --n-noisy 3 \\
      --target gallbladder --image-size 128 --renderer lotus \\
      --out data/dataset/usbg.npz --lerobot data/dataset/lerobot \\
      --contact-sheet outputs/dataset_contact_sheet.png
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from usbg import volume as V
from usbg import slicer as S
from usbg import render_lotus as R
from usbg import dataset as D
from usbg.demonstrator import Demonstrator
from usbg.mujoco_env import USProbeEnv
from usbg.placement import seat_volume_under_probe
from usbg.contracts_bridge import invert_T

RAW_ROOT = "data/raw/totalseg_small"
CACHE_DIR = "data/cache"


def resolve_case(case_id: str) -> V.LabeledVolume:
    cache = os.path.join(CACHE_DIR, f"{case_id}.npz")
    if os.path.exists(cache):
        return V.load_cache(cache)
    case_dir = os.path.join(RAW_ROOT, case_id)
    ct = os.path.join(case_dir, "ct.nii.gz")
    seg = os.path.join(case_dir, "segmentations")
    print(f"  building volume for {case_id} (not cached) ...")
    lv = V.load_labeled_ct(ct, seg)
    V.save_cache(lv, cache)
    return lv


def _resize_float(img01, size):
    from PIL import Image
    a = np.clip(np.asarray(img01) * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), np.float32) / 255.0


def replay_trajectory(traj, lv, world_from_volume, T_base, env, kind, device,
                      image_size, case, traj_id, texture=None, randomize=False, rng=None):
    """Render a demonstrator trajectory as a ClipSample (relative motion on T_base).

    Every frame goes through the sim-to-real texture translator (when given) so the whole dataset is
    textured; `randomize` additionally draws a per-frame physics domain-randomization config, so the
    policy keys on structure rather than a fixed appearance.
    """
    A = T_base @ invert_T(traj.poses[0])              # align first demo pose to the base
    # one appearance (domain-randomization) draw AND one speckle seed per CLIP, held fixed across
    # its frames (the per-episode rule): within a clip only the anatomy moves, so the frame history
    # carries a clean motion cue; appearance and speckle vary across clips
    if kind == "physics":
        clip_cfg = (R.domain_randomize(R.RenderConfig(), rng) if randomize
                    else R.RenderConfig().with_seed(int(rng.integers(2 ** 31 - 1))))
    else:
        clip_cfg = None
    rend = R.Renderer(kind=kind, cfg=clip_cfg, translator=texture, device=device)
    imgs, poses, forces = [], [], []
    for k in range(len(traj)):
        T_wp = A @ traj.poses[k]
        lab = S.slice_at_pose(lv, T_wp, world_from_volume)
        img = _resize_float(rend.render(lab), image_size)
        imgs.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        poses.append(T_wp)
        forces.append(env.force_at_pose(T_wp) if env is not None else 0.0)
    return D.ClipSample(
        images=np.asarray(imgs, dtype=np.uint8), poses=np.asarray(poses, dtype=np.float64),
        twist=traj.twist.astype(np.float32), rel6=traj.rel6.astype(np.float32),
        valid=traj.valid, label=traj.label, instruction=traj.instruction,
        source=traj.source, case=case, traj_id=traj_id,
        forces=np.asarray(forces, dtype=np.float32))


def _mild_randomize(base, rng):
    """A gentler domain randomization for S_US pretraining: vary gain, TGC, PSF, and speckle (so
    S_US is appearance-robust) while leaving the per-tissue acoustic constants (z/mu/backscatter)
    at their defaults, so acoustically distinct tissues keep distinct brightness and stay separable.
    Full domain_randomize scrambles backscatter, which collapses the soft-tissue classes together."""
    from dataclasses import replace
    u = rng.uniform
    return replace(base, gain=float(u(0.85, 1.15)), tgc_slope=float(u(0.7, 1.3)),
                   psf_sigma_lat0=float(u(0.9, 1.4)), psf_focus_depth=float(u(0.03, 0.08)),
                   speckle_strength=float(u(0.7, 1.1)), noise_sigma=float(u(0.0, 0.03)),
                   dyn_range_db=float(u(48, 56)), seed=int(rng.integers(2 ** 31 - 1)))


def _pair_cfg(randomize, rng):
    """Per-frame render config of the aligned pairs (rng draw order is part of the recipe)."""
    if randomize == "full" or randomize is True:
        return R.domain_randomize(R.RenderConfig(), rng)
    if randomize == "mild":
        return _mild_randomize(R.RenderConfig(), rng)
    return R.RenderConfig().with_seed(int(rng.integers(2 ** 31 - 1)))   # "off": speckle seed only


def _render_pair(lv, T_wp, world_from_volume, cfg, image_size, device):
    from usbg.segmenter import align_label_to_image
    lab = S.slice_at_pose(lv, T_wp, world_from_volume)                  # (N_LINES, N_SAMPLES)
    img = R.render(lab, "physics", device=device, cfg=cfg)              # (N_SAMPLES, N_LINES)
    img_sq = _resize_float(img, image_size)                             # (size, size) float [0,1]
    lab_sq = align_label_to_image(lab, (image_size, image_size))
    return (np.clip(img_sq, 0, 1) * 255).astype(np.uint8), lab_sq.astype(np.uint8)


# Anatomical-seat pair perturbation (the canonical ranges, re-expressed on the skin): slides in the
# skin's tangent plane and tilts about the contact point; the tz draw only sets a PRESS depth
# (|tz| / PAIR_TZ_RANGE * press_max_m, into the body), never a lift-off, and before the press the pose
# is clamped outside the skin (placement.clamp_start_outside_skin), so a tilt never digs the probe in.
PAIR_RANGES = dict(tx=0.03, ty=0.03, tz=0.02, rx=12.0, ry=12.0, rz=20.0)


def contact_perturb(T_base, tx, ty, tz, rx, ry, rz, tz_range, press_max_m):
    """(T, lift_m, press_m): perturb a seated window pose (face on the skin, beam along world -z)
    without leaving or penetrating the skin except by an explicit press of at most press_max_m."""
    from usbg.contracts_bridge import pose_from_rpy
    from usbg.placement import clamp_start_outside_skin
    T, lift = clamp_start_outside_skin(T_base @ pose_from_rpy(tx=tx, ty=ty, tz=0.0, rx=rx, ry=ry, rz=rz))
    press = (abs(tz) / tz_range) * press_max_m if tz_range > 0 else 0.0
    T = T.copy()
    T[2, 3] -= press
    return T, float(lift), float(press)


def _gen_pairs_anatomical(cases, organs, n_per_case, image_size, device, seed, randomize, press_max_m,
                          min_windows=1):
    """Anatomical seat: per case and organ, n_per_case frames cycling round-robin through that
    organ's accepted windows (usbg.windows; KIDNEY interleaves right/left). A (case, organ) with fewer
    than min_windows screened windows is SKIPPED and logged, never replaced by another organ."""
    from usbg import windows as W
    from usbg.placement import seat_volume_under_probe

    name_to_label = {v: k for k, v in V.USBG_LABEL_NAMES.items()}
    rng = np.random.default_rng(seed)
    r = PAIR_RANGES
    imgs, labs, sess, fwin = [], [], [], []
    info = {"seat_mode": "anatomical", "organs": list(organs), "n_per_case_per_organ": int(n_per_case),
            "perturbation": {"slide_m": r["tx"], "tilt_deg": [r["rx"], r["ry"]], "roll_deg": r["rz"],
                             "press_max_m": float(press_max_m),
                             "rule": "slide+tilt about the contact point, clamp outside the skin, then "
                                     "press |tz|/0.02*press_max_m along world -z (never lift-off)"},
            "min_windows": int(min_windows), "per_case": {}, "skipped": []}
    lifts, presses = [], []
    for case in cases:
        lv = resolve_case(case)
        info["per_case"][case] = {}
        for organ in organs:
            label = name_to_label[organ]
            try:
                metas = W.windows_meta(lv, label)
            except W.NoAcceptedWindow as e:
                metas, why = [], str(e)
            else:
                why = "no accepted window for this organ"
            if metas and len(metas) < min_windows:
                why = f"only {len(metas)} accepted window(s) < min_windows={min_windows}"
                metas = []
            if not metas:
                info["skipped"].append({"case": case, "organ": organ, "reason": why})
                print(f"  [skip] {case} {organ}: {why} (0 frames; no substitute organ)")
                continue
            seats = {}
            used = []
            for i in range(n_per_case):
                wi = i % len(metas)
                if wi not in seats:
                    seats[wi] = seat_volume_under_probe(lv, label, window=wi)
                Wfv, T_base = seats[wi]
                d = rng.uniform                      # same draw order as the canonical recipe
                tx, ty, tz = float(d(-r["tx"], r["tx"])), float(d(-r["ty"], r["ty"])), float(d(-r["tz"], r["tz"]))
                rx, ry, rz = float(d(-r["rx"], r["rx"])), float(d(-r["ry"], r["ry"])), float(d(-r["rz"], r["rz"]))
                T_wp, lift, press = contact_perturb(T_base, tx, ty, tz, rx, ry, rz, r["tz"], press_max_m)
                im, lb = _render_pair(lv, T_wp, Wfv, _pair_cfg(randomize, rng), image_size, device)
                imgs.append(im); labs.append(lb); sess.append(case)
                wid = f"{metas[wi]['organ']}/w{metas[wi]['window_id']}"
                fwin.append(wid); used.append(wid)
                lifts.append(lift); presses.append(press)
            info["per_case"][case][organ] = {"n_windows": len(metas), "n_frames": int(n_per_case),
                                             "windows_used": sorted(set(used))}
            print(f"  {case} {organ}: {n_per_case} frames over {len(set(used))}/{len(metas)} windows")
    li = np.asarray(lifts, float)
    info["clamp"] = {"n_frames": int(li.size), "clamped_frac": float((li > 1e-4).mean()) if li.size else 0.0,
                     "lift_mm_mean": float(li.mean() * 1e3) if li.size else 0.0,
                     "lift_mm_max": float(li.max() * 1e3) if li.size else 0.0,
                     "press_mm_mean": float(np.mean(presses) * 1e3) if presses else 0.0}
    info["frame_window"] = fwin
    print(f"  clamp outside skin: {info['clamp']['clamped_frac']*100:.1f}% of {li.size} frames lifted "
          f"(mean {info['clamp']['lift_mm_mean']:.2f} mm, max {info['clamp']['lift_mm_max']:.2f} mm); "
          f"skipped (case, organ): {[(s['case'], s['organ']) for s in info['skipped']]}")
    return np.asarray(imgs, np.uint8), np.asarray(labs, np.uint8), np.asarray(sess), info


def gen_labeled_pairs(cases, target_name, n_per_case, image_size, device, seed=0,
                      randomize="full", standoff_m=None, depth_m=None, organs=None,
                      press_max_m=0.0, min_windows=1, return_info=False):
    """Render aligned (physics image, canonical label) pairs by sweeping poses around the seated
    base pose on each case. These are SonoSPADE's free supervision: the generator's content input
    and the exact labels that pretrain S_US. Returns (images uint8 (N,H,W), labels uint8 (N,H,W),
    sessions (N,)) [+ info dict with return_info=True].

    Each pose is a small random se(3) perturbation of the base pose (so views vary while anatomy is
    intersected), and with randomize=True each frame draws its own physics domain-randomization
    config so S_US keys on structure rather than a fixed appearance. The label is transposed and
    resized to the render orientation with segmenter.align_label_to_image so image and label align.

    USBG_SEAT=canonical: the v1 recipe byte-for-byte (one seat over target_name per case, standoff
    0.0 and depth 0.05 by default). USBG_SEAT=anatomical: n_per_case frames per (case, organ in
    `organs`, default [target_name]) at that organ's screened windows (see _gen_pairs_anatomical);
    standoff_m/depth_m are canonical-only and raise here.
    """
    from usbg.placement import seat_volume_under_probe, seat_mode
    from usbg.contracts_bridge import pose_from_rpy

    if seat_mode() == "anatomical":
        if depth_m is not None or standoff_m is not None:
            raise ValueError("standoff_m/depth_m are canonical-only; the anatomical seat takes the "
                             "depth from the screened window")
        out = _gen_pairs_anatomical(cases, list(organs or [target_name]), n_per_case, image_size,
                                    device, seed, randomize, press_max_m, min_windows)
        return out if return_info else out[:3]

    standoff_m = 0.0 if standoff_m is None else standoff_m
    depth_m = 0.05 if depth_m is None else depth_m
    name_to_label = {v: k for k, v in V.USBG_LABEL_NAMES.items()}
    target = name_to_label.get(target_name, V.GALLBLADDER)
    rng = np.random.default_rng(seed)
    imgs, labs, sess = [], [], []
    for case in cases:
        lv = resolve_case(case)
        tgt = target if lv.has(target) else (V.LIVER if lv.has(V.LIVER) else V.SOFT_TISSUE)
        world_from_volume, T_base = seat_volume_under_probe(lv, tgt, standoff_m=standoff_m,
                                                            depth_m=depth_m)
        for _ in range(n_per_case):
            perturb = pose_from_rpy(tx=float(rng.uniform(-0.03, 0.03)),
                                    ty=float(rng.uniform(-0.03, 0.03)),
                                    tz=float(rng.uniform(-0.02, 0.02)),
                                    rx=float(rng.uniform(-12, 12)),
                                    ry=float(rng.uniform(-12, 12)),
                                    rz=float(rng.uniform(-20, 20)))
            T_wp = T_base @ perturb
            im, lb = _render_pair(lv, T_wp, world_from_volume, _pair_cfg(randomize, rng),
                                  image_size, device)
            imgs.append(im); labs.append(lb); sess.append(case)
    out = (np.asarray(imgs, np.uint8), np.asarray(labs, np.uint8), np.asarray(sess))
    if return_info:
        return out + ({"seat_mode": "canonical", "organs": [target_name], "skipped": []},)
    return out


def _pairs_provenance(path, sessions, info, args, organs):
    """<pairs>.provenance.json (usbg.provenance). A pairs set with a held-out patient is never blessed:
    the sidecar is refused and the reason printed (legacy reproductions of the v1 leaky pairs)."""
    from usbg import provenance as P
    cases = sorted(set(np.asarray(sessions).tolist()))
    dr = {"full": "full", "mild": "mild", "off": "off"}[args.pairs_dr]
    caches = {}
    for c in cases:
        p = os.path.join(CACHE_DIR, f"{c}.npz")
        if info.get("seat_mode") == "anatomical":
            p = V.anat_cache_path(p)
        caches[c] = {"path": p, "md5": P.md5(p)}
    extra = {"organs": organs, "image_size": args.image_size, "renderer": "physics",
             "pairs_n_per_case_per_organ": args.pairs_n, "img_depth_m": float(R.IMG_DEPTH_M),
             "ct_caches": caches, "windows": P.windows_ref(),
             "skipped": info.get("skipped", []), "clamp": info.get("clamp"),
             "perturbation": info.get("perturbation")}
    try:
        out = P.write_provenance(path, "pairs", train_cases=cases, iters=0, seed=args.seed, dr_mode=dr,
                                 dr_notes=f"one {args.pairs_dr} draw per frame (make_dataset --pairs-dr)",
                                 seat_mode=info.get("seat_mode"), extra=extra)
        print(f"wrote provenance -> {out}")
    except ValueError as e:
        print(f"WARNING: no provenance sidecar for {path}:\n  {e}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["s1086", "s0777"])
    ap.add_argument("--maneuvers", nargs="+", default=None,
                    help="restrict to a subset of maneuvers (slide rock tilt rotate compress); "
                         "default: all five. Pass one to generate a single-maneuver dataset.")
    ap.add_argument("--target", default="gallbladder")
    ap.add_argument("--n-clean", type=int, default=3, help="clean clips per maneuver per case")
    ap.add_argument("--n-noisy", type=int, default=3, help="noisy clips per maneuver per case")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--renderer", choices=["physics", "lotus", "numpy"], default="physics")
    ap.add_argument("--texture-gan", default=None,
                    help="trained sim-to-real texture generator checkpoint, applied to EVERY frame")
    ap.add_argument("--randomize", action="store_true",
                    help="per-frame physics domain randomization (recommended with --texture-gan)")
    ap.add_argument("--no-force", action="store_true", help="skip the MuJoCo QC force column")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/dataset/usbg.npz")
    ap.add_argument("--lerobot", default="data/dataset/lerobot")
    ap.add_argument("--contact-sheet", default="outputs/dataset_contact_sheet.png")
    # SonoSPADE aligned (image, label) pairs (free supervision for S_US and the generator content)
    ap.add_argument("--pairs-out", default=None,
                    help="also emit aligned (physics image, label) pairs to this .npz")
    ap.add_argument("--pairs-n", type=int, default=200, help="aligned pairs per case")
    ap.add_argument("--pairs-dr", choices=["full", "mild", "off"], default="full",
                    help="domain randomization for the aligned pairs (mild keeps per-tissue acoustics)")
    ap.add_argument("--pairs-only", action="store_true",
                    help="emit only the aligned pairs (skip the demonstrator trajectory dataset)")
    ap.add_argument("--pairs-organs", nargs="+", default=None, choices=["liver", "kidney", "gallbladder"],
                    help="anatomical seat only: organs whose accepted windows the pairs cycle through "
                         "(default: --target); --pairs-n frames per (case, organ); a (case, organ) "
                         "without a window is skipped and logged")
    ap.add_argument("--pairs-press-mm", type=float, default=0.0,
                    help="anatomical seat only: maximum press into the skin (the tz draw sets the "
                         "depth); 0 = every frame in skin contact, never lifted off nor inside")
    ap.add_argument("--pairs-min-windows", type=int, default=1,
                    help="anatomical seat only: skip (and log) a (case, organ) with fewer screened windows "
                         "than this, instead of sampling all of its frames from one or two windows")
    ap.add_argument("--no-provenance", action="store_true",
                    help="skip the <pairs-out>.provenance.json sidecar")
    args = ap.parse_args(argv)

    from usbg.placement import seat_mode
    if seat_mode() == "canonical" and args.pairs_organs:
        raise SystemExit("--pairs-organs is anatomical-seat only (USBG_SEAT=canonical seats one --target)")
    device_pairs = R.get_device() if args.renderer in ("physics", "lotus") else "cpu"
    if args.pairs_out:
        organs = args.pairs_organs or [args.target]
        print(f"rendering aligned (image, label) pairs [{seat_mode()} seat]: {args.pairs_n} per case"
              f"{' per organ ' + str(organs) if seat_mode() == 'anatomical' else ''} x {len(args.cases)} cases")
        pimgs, plabs, psess, pinfo = gen_labeled_pairs(
            args.cases, args.target, args.pairs_n, args.image_size, device_pairs, seed=args.seed,
            randomize=args.pairs_dr, organs=organs if seat_mode() == "anatomical" else None,
            press_max_m=args.pairs_press_mm / 1000.0, min_windows=args.pairs_min_windows, return_info=True)
        extra = {"cases": args.cases, "target": args.target}
        if seat_mode() == "anatomical":                 # canonical meta stays exactly the v1 one
            from usbg import provenance as P
            extra.update(seat_mode="anatomical", organs=organs, pairs_dr=args.pairs_dr, seed=args.seed,
                         frame_window=pinfo.pop("frame_window"), pairs_info=pinfo,
                         windows=P.windows_ref(), git_hash=P.git_info()["hash"])
        ppath, pmeta = D.write_pairs_npz(args.pairs_out, pimgs, plabs, sessions=psess, extra_meta=extra)
        print(f"wrote {pmeta['n']} aligned pairs -> {ppath} (shape {pmeta['shape']})")
        if not args.no_provenance:
            _pairs_provenance(ppath, psess, pinfo, args, organs)
        if args.pairs_only:
            return 0

    name_to_label = {v: k for k, v in V.USBG_LABEL_NAMES.items()}
    target = name_to_label.get(args.target, V.GALLBLADDER)
    device = R.get_device() if args.renderer in ("physics", "lotus") else "cpu"
    rng = np.random.default_rng(args.seed)

    texture = None
    if args.texture_gan:
        from usbg.texture_gan import load_translator
        texture = load_translator(args.texture_gan, device=device)   # dual-input aware (SonoSPADE)
        kind = "SonoSPADE (dual input)" if getattr(texture, "wants_label", False) else "grayscale"
        print(f"sim-to-real texture ({kind}) applied to EVERY frame via {args.texture_gan}"
              f"{' + per-frame domain randomization' if args.randomize else ''}")

    print("building demonstrator (Repo B env + PPO policy) ...")
    demo = Demonstrator()
    maneuvers = args.maneuvers or demo.MOVEMENTS
    unknown = [m for m in maneuvers if m not in demo.MOVEMENTS]
    if unknown:
        raise SystemExit(f"unknown maneuver(s) {unknown}; choose from {demo.MOVEMENTS}")
    print(f"maneuvers: {maneuvers}")
    t0 = time.time()
    clips, sid = [], 0
    for case in args.cases:
        print(f"[case {case}] loading volume + seating over {args.target}")
        lv = resolve_case(case)
        tgt = target if lv.has(target) else (V.LIVER if lv.has(V.LIVER) else V.SOFT_TISSUE)
        world_from_volume, T_base = (seat_volume_under_probe(lv, tgt, standoff_m=0.0, depth_m=0.05)
                                     if seat_mode() == "canonical"
                                     else seat_volume_under_probe(lv, tgt, standoff_m=0.0))
        env = None if args.no_force else USProbeEnv(lv, world_from_volume, T_base,
                                                    renderer="numpy")
        for action in maneuvers:
            for _ in range(args.n_clean):
                tr = demo.clean(action, rng)
                clips.append(replay_trajectory(tr, lv, world_from_volume, T_base, env,
                                               args.renderer, device, args.image_size, case, sid,
                                               texture=texture, randomize=args.randomize, rng=rng))
                sid += 1
            for _ in range(args.n_noisy):
                tr = demo.noisy(action, rng, seed=900_000 + sid)
                clips.append(replay_trajectory(tr, lv, world_from_volume, T_base, env,
                                               args.renderer, device, args.image_size, case, sid,
                                               texture=texture, randomize=args.randomize, rng=rng))
                sid += 1
            print(f"  {action}: {args.n_clean} clean + {args.n_noisy} noisy done "
                  f"({sum(len(c.images) for c in clips)} frames so far)")
        if env is not None:
            env.close()
    demo.close()

    nframes = sum(len(c.images) for c in clips)
    print(f"generated {len(clips)} clips, {nframes} frames in {time.time()-t0:.0f}s")
    path, meta = D.write_npz(args.out, clips, args.image_size,
                             extra_meta={"target": args.target, "cases": args.cases})
    print(f"wrote npz -> {path}  (clean={meta['n_clean']} noisy={meta['n_noisy']})")
    if args.lerobot:
        outd, info = D.write_lerobot(args.lerobot, clips, args.image_size)
        print(f"wrote LeRobot-style -> {outd}  ({info['total_frames']} frames)")
    if args.contact_sheet:
        D.contact_sheet(clips, args.contact_sheet)
        print(f"wrote contact sheet -> {args.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
