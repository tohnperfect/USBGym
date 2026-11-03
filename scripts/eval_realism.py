#!/usr/bin/env python3
"""Realism evaluation (the realism table): renderers and texture stages scored against real B-mode frames.

--sector (the v2 realism protocol, anatomical seat; the reported protocol):
  * views: --views-per-case per patient (default: the 6 held-out benchmark patients), at the patient's
    screened acoustic windows (usbg.windows): organs alternate (--organs, default liver kidney), windows
    round-robin within an organ; a (case, organ) without a window is skipped and logged. Jitter: slide
    +/-2 cm in the skin's tangent plane, tilt rx,ry +/-15 deg and roll rz +/-10 deg about the contact
    point, clamp outside the skin (placement.clamp_start_outside_skin, fraction logged), then a press of
    0 to 5 mm (never lift-off).
  * every variant renders the SAME label slice per view: plain LOTUS (seed per view), NumPy (rng per
    view), physics (DR off, speckle seed per view), physics + each texture stage, and with --dr full|both
    the physics rows under one full domain_randomize draw per view. Pipeline: linear render ->
    view_match._resize01 -> align_label_to_image -> view_match.warp_to_sector -> texture stage
    (translate_aligned for SonoSPADE), i.e. the geometry the curvi stages were trained in.
  * real reference: the 60 annotated Vitale TEST frames (data/eval/kaggle_vitale_test.npz), which are
    disjoint from the 404-frame TRAIN pool the stages were trained toward.
  * scoring on in-sector pixels only: the measured sector mask eroded by --erode px is applied to BOTH
    domains. Metrics: intensity W1 (pooled and per patient), KID (primary; subset --kid-subset) and FID
    (secondary; biased at n_real=60), a CNN classifier two-sample test (C2ST, grouped k-fold, balanced;
    0.5 = indistinguishable) with a label-permutation null on the physics row, and edge IoU of each
    translation vs its own physics frame. --renorm robust also scores sim frames after the real ingest's
    per-frame 1-99 percentile normalization (primary = as rendered).
  * refuses USBG_SEAT=canonical unless --allow-canonical.

Without --sector: the withdrawn v1 protocol (linear frames, one case, logistic AUC), kept ONLY under
USBG_SEAT=canonical so the ledger entry stays reproducible. It no longer has private defaults: pass
--real-dir and the translator checkpoints explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np


def resolve_case(cid):
    from usbg import volume as V
    c = os.path.join("data/cache", f"{cid}.npz")
    return V.load_cache(c) if os.path.exists(c) else V.load_labeled_ct(
        os.path.join("data/raw/totalseg_small", cid, "ct.nii.gz"),
        os.path.join("data/raw/totalseg_small", cid, "segmentations"))


def _feats(imgs, size=24):
    """Standardizable low-res features for the two-sample classifier."""
    from PIL import Image
    out = []
    for im in imgs:
        a = np.asarray(Image.fromarray((np.clip(im, 0, 1) * 255).astype("uint8")).resize((size, size)),
                       np.float32) / 255.0
        out.append(a.ravel())
    return np.asarray(out, np.float32)


def _auc_two_sample(fake, real, seed=0):
    """Train a tiny logistic (numpy) to separate real (1) from fake (0); return test ROC-AUC.
    0.5 means indistinguishable (most realistic)."""
    rng = np.random.default_rng(seed)
    n = min(len(fake), len(real))
    Xf, Xr = _feats(fake)[:n], _feats(real)[:n]
    X = np.concatenate([Xf, Xr], 0); y = np.concatenate([np.zeros(n), np.ones(n)])
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    ntr = int(0.7 * len(X))
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    w = np.zeros(X.shape[1]); b = 0.0; lr = 0.1
    for _ in range(300):
        z = Xtr @ w + b; p = 1 / (1 + np.exp(-z)); g = p - ytr
        w -= lr * (Xtr.T @ g / len(Xtr) + 1e-3 * w); b -= lr * g.mean()
    s = Xte @ w + b
    pos, neg = s[yte == 1], s[yte == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


# ============================================================================ legacy (canonical only)
def run_legacy(args):
    """The withdrawn v1 realism protocol (canonical seat only, explicit inputs)."""
    from usbg import render_lotus as R, slicer as S
    from usbg.placement import seat_volume_under_probe
    from usbg.goal_rl import _target_of
    from usbg.texture_gan import intensity_w1, edge_iou, TextureTranslator
    from usbg.contracts_bridge import pose_from_rpy
    from PIL import Image

    dev = "cpu"                                                  # CPU to avoid MPS contention
    lv = resolve_case(args.case)
    Wfv, T0 = seat_volume_under_probe(lv, _target_of(lv), standoff_m=0.0, depth_m=0.05)
    rng = np.random.default_rng(args.seed)
    labs = []
    for _ in range(args.n):
        d = rng.uniform
        T = T0 @ pose_from_rpy(tx=d(-.02, .02), ty=d(-.02, .02), tz=d(-.01, .01),
                               rx=d(-18, 18), ry=d(-18, 18), rz=d(-8, 8))
        labs.append(S.slice_at_pose(lv, T, Wfv))

    real = np.asarray([np.asarray(Image.open(os.path.join(args.real_dir, f)).convert("L")
                                  .resize((128, 128)), np.float32) / 255.0
                       for f in sorted(os.listdir(args.real_dir)) if f.endswith((".png", ".jpg"))])

    def pool(kind, tex=None):
        out = []
        for i, lab in enumerate(labs):
            cfg = R.domain_randomize(R.RenderConfig(), np.random.default_rng(1000 + i)) \
                if kind == "physics" else None
            img = R.render(lab, kind if kind != "physics" else "physics", device=dev, cfg=cfg,
                           seed=i) if kind != "physics" else R.render(lab, "physics", device=dev, cfg=cfg)
            if tex is not None:
                img = tex(img)
            out.append(np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8"))
                                  .resize((128, 128)), np.float32) / 255.0)
        return np.asarray(out)

    def tx(path):
        return TextureTranslator(path, device=dev).translate if path and os.path.exists(path) else None

    variants = {"prior LOTUS": pool("lotus"), "NumPy": pool("numpy"), "physics": pool("physics")}
    for name, path in (("physics+CycleGAN", args.cyclegan), ("physics+CUT", args.cut),
                       ("physics+CycleGAN-SC", args.cyclegan_sc)):
        t = tx(path)
        if t is not None:
            variants[name] = pool("physics", t)
    res = {"real_mean": float(real.mean()), "n_real": int(len(real)), "n_sim": args.n,
           "case": args.case, "struct_thresh": 0.15,
           "protocol": "W1, edge IoU and AUC all use the same sim views of one held-out case, against the "
                       "full real pool. edge IoU is null for variants "
                       "that are not translations of the physics render.",
           "variants": {}}
    src = variants["physics"]   # the untranslated frames the translators consumed
    print(f"realism on {args.case}: {args.n} sim views vs {len(real)} real crops "
          f"(W1, edge IoU and AUC all use these same views)")
    print(f"{'variant':22s} {'mean':>6s} {'W1':>7s} {'edgeIoU':>8s} {'AUC':>6s}")
    for name, p in variants.items():
        if p is None:
            continue
        w1 = intensity_w1(p, real); auc = _auc_two_sample(p, real, args.seed); mn = float(p.mean())
        # edge IoU is a TRANSLATION-fidelity metric: it only means anything for a variant that is a
        # translation of the physics render. The other renderers are not, so they get null, which is
        # the paper's "--" cell.
        if name.startswith("physics+"):
            iou = float(np.mean([edge_iou(src[i], p[i]) for i in range(len(p))]))
        elif name == "physics":
            iou = 1.0            # identity: the source compared against itself
        else:
            iou = None
        res["variants"][name] = {"mean": mn, "w1_to_real": w1, "auc_real_vs_fake": auc,
                                 "struct_iou": iou}
        ious = f"{iou:8.4f}" if iou is not None else f"{'--':>8s}"
        print(f"{name:22s} {mn:6.3f} {w1:7.4f} {ious} {auc:6.3f}")
    return res


# ============================================================================ sector v2
def _cnn(device):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 1)).to(device)


def _auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    w = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(w / (len(pos) * len(neg)))


def c2st_cnn(fake, fake_groups, real, real_groups, *, repeats=5, folds=5, epochs=40, size=64,
             seed=0, device="cpu", permute=False):
    """Balanced grouped k-fold CNN C2ST. fake/real: (N,H,W) in [0,1], already sector-masked. Each
    repeat subsamples n = min(n_real, n_fake) frames per domain; folds are assigned by group (sim:
    patient x window, so near-duplicate views never straddle train/test; real: frame). Returns the
    held-out AUCs (up to repeats*folds; a fold without both classes is skipped). 0.5 = indistinguishable."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    def prep(x):
        t = torch.as_tensor(np.asarray(x, np.float32))[:, None]
        return F.interpolate(t, size=(size, size), mode="area")
    real = np.asarray(real); fake = np.asarray(fake)
    n = min(len(real), len(fake))
    fg = np.asarray(fake_groups); rg = np.asarray(real_groups)
    aucs = []
    for _ in range(repeats):
        idx = rng.permutation(len(fake))[:n]
        ridx = rng.permutation(len(real))[:n]
        X = torch.cat([prep(fake[idx]), prep(real[ridx])])
        y = np.r_[np.zeros(n), np.ones(n)]
        if permute:
            y = rng.permutation(y)
        groups = np.r_[fg[idx], np.array([f"real:{g}" for g in rg[ridx]])]
        ug = rng.permutation(np.unique(groups))
        fold_of = {g: i % folds for i, g in enumerate(ug)}
        fid = np.array([fold_of[g] for g in groups])
        for k in range(folds):
            tr, te = fid != k, fid == k
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            net = _cnn(device)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            Xtr = X[tr].to(device); ytr = torch.as_tensor(y[tr], dtype=torch.float32, device=device)
            for _ep in range(epochs):
                perm = torch.randperm(len(Xtr), device=device)
                for b in range(0, len(Xtr), 32):
                    bi = perm[b:b + 32]
                    xb = Xtr[bi]
                    if torch.rand(1).item() < 0.5:
                        xb = torch.flip(xb, dims=[3])        # the sector is left-right symmetric
                    loss = F.binary_cross_entropy_with_logits(net(xb)[:, 0], ytr[bi])
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                s = net(X[te].to(device))[:, 0].cpu().numpy()
            aucs.append(_auc(s, y[te]))
    return aucs


def robust_norm(a, lo=1.0, hi=99.0):
    """The real ingest's per-frame normalization (scripts/ingest_kaggle_us.py::_robust_norm)."""
    p1, p99 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / (p99 - p1 + 1e-6), 0, 1).astype(np.float32)


def window_views(lv, case, organs, n_views, seed, ranges, press_max_m):
    """n_views jittered poses at the case's screened windows. Organs alternate, windows round-robin
    within an organ. Returns (views, skipped, counts): views are dicts with label slice and ids."""
    from usbg import volume as V, slicer as S, windows as W
    from usbg.placement import seat_volume_under_probe, clamp_start_outside_skin
    from usbg.contracts_bridge import pose_from_rpy
    lab_of = {"liver": V.LIVER, "kidney": V.KIDNEY, "gallbladder": V.GALLBLADDER}
    per, skipped, counts = [], [], {}
    for organ in organs:
        try:
            metas = W.windows_meta(lv, lab_of[organ])
            why = "no accepted window for this organ"
        except W.NoAcceptedWindow as e:
            metas, why = [], str(e)
        counts[organ] = len(metas)
        if metas:
            per.append((organ, metas))
        else:
            skipped.append({"case": case, "organ": organ, "reason": why})
            print(f"  [skip] {case} {organ}: {why}")
    if not per:
        return [], skipped, counts
    rng = np.random.default_rng([seed, zlib.crc32(case.encode())])
    seats, views = {}, []
    for i in range(n_views):
        organ, metas = per[i % len(per)]
        wi = (i // len(per)) % len(metas)
        if (organ, wi) not in seats:
            seats[(organ, wi)] = seat_volume_under_probe(lv, lab_of[organ], window=wi)
        Wfv, T0 = seats[(organ, wi)]
        d = rng.uniform
        tx, ty = d(-ranges["slide"], ranges["slide"]), d(-ranges["slide"], ranges["slide"])
        rx, ry = d(-ranges["tilt"], ranges["tilt"]), d(-ranges["tilt"], ranges["tilt"])
        rz = d(-ranges["roll"], ranges["roll"])
        press = d(0.0, press_max_m)
        T, lift = clamp_start_outside_skin(T0 @ pose_from_rpy(tx=tx, ty=ty, tz=0.0, rx=rx, ry=ry, rz=rz))
        T = T.copy(); T[2, 3] -= press
        m = metas[wi]
        views.append({"lab": S.slice_at_pose(lv, T, Wfv), "T": T, "Wfv": Wfv, "case": case, "organ": organ,
                      "window": f"{m['organ']}/w{m['window_id']}", "lift": float(lift), "press": float(press),
                      "vseed": int(zlib.crc32(f"{seed}:{case}:{i}".encode()) & 0x7FFFFFFF)})
    return views, skipped, counts


def run_sector(args):
    from scipy.ndimage import binary_erosion
    from usbg import render_lotus as R, view_match as VM
    from usbg.segmenter import align_label_to_image
    from usbg.texture_gan import intensity_w1, edge_iou, load_translator
    from usbg import provenance as P

    t0 = time.time()
    dev = args.device
    _, _, M = VM._sector_coords(128, 128, VM.SECTOR)
    Me = binary_erosion(M, iterations=args.erode) if args.erode > 0 else M
    if P.is_private_path(args.real):
        raise SystemExit(f"--real {args.real} is a private pool")
    real_npz = np.load(args.real)
    real_raw = real_npz["images"].astype(np.float32) / 255.0
    real = real_raw * Me
    real_groups = [str(s) for s in real_npz["sessions"]]

    stage_paths = {}
    if args.tex_manifest:
        with open(args.tex_manifest) as fh:
            stage_paths.update({k: v for k, v in json.load(fh).items() if v and not k.startswith("_")})
    for k, v in (("cut", args.cut), ("cyclegan", args.cyclegan), ("cyclegan_sc", args.cyclegan_sc),
                 ("sonospade", args.sonospade)):
        if v:
            stage_paths[k] = v
    nice = {"cut": "CUT", "cyclegan": "CycleGAN", "cyclegan_sc": "CycleGAN-SC", "sonospade": "SonoSPADE"}
    trs = {nice.get(k, k): load_translator(p, dev) for k, p in stage_paths.items()}

    def warp(img_lin, lab_lin):
        return VM.warp_to_sector(VM._resize01(img_lin, 128), align_label_to_image(lab_lin, (128, 128)))

    ranges = dict(slide=0.02, tilt=15.0, roll=10.0)
    pools, groups, cases_of, src, skipped, wcount, lifts, presses = {}, [], [], {}, [], {}, [], []
    dr_modes = ["off", "full"] if args.dr == "both" else [args.dr]
    t_render = time.time()
    for case in args.cases:
        lv = resolve_case(case)
        views, sk, cnt = window_views(lv, case, args.organs, args.views_per_case, args.seed, ranges,
                                      args.press_max_mm / 1000.0)
        skipped += sk; wcount[case] = cnt
        for v in views:
            lab, vseed = v["lab"], v["vseed"]
            groups.append(f"{case}:{v['window']}"); cases_of.append(case)
            lifts.append(v["lift"]); presses.append(v["press"])
            img, _ = warp(R.render(lab, "lotus", device=dev, seed=vseed), lab)
            pools.setdefault("prior LOTUS", []).append(img)
            img, _ = warp(R.render(lab, "numpy", rng=np.random.default_rng(vseed)), lab)
            pools.setdefault("NumPy", []).append(img)
            for drm in dr_modes:
                cfg = R.RenderConfig(seed=vseed) if drm == "off" else \
                    R.domain_randomize(R.RenderConfig(), np.random.default_rng(vseed))
                tag = "" if drm == "off" else " [DR full]"
                img, lsec = warp(R.render(lab, "physics", device=dev, cfg=cfg), lab)
                pools.setdefault("physics" + tag, []).append(img)
                src.setdefault(tag, []).append(img)
                for name, tr in trs.items():
                    out = tr.translate_aligned(img, lsec) if getattr(tr, "wants_label", False) else tr(img)
                    pools.setdefault(f"physics+{name}{tag}", []).append(np.asarray(out, np.float32))
        print(f"  {case}: {len(views)} views ({cnt})", flush=True)
    t_render = time.time() - t_render
    groups = np.asarray(groups); cases_arr = np.asarray(cases_of)
    if not len(groups):
        raise SystemExit("no views rendered (no accepted windows for the requested cases/organs)")

    li = np.asarray(lifts)
    res = {"protocol": "sector v2: anatomical windows, in-sector eroded mask on both domains, "
                       "60 annotated Vitale TEST frames as the reference",
           "seat_mode": P.current_seat_mode(), "cases": list(args.cases), "organs": list(args.organs),
           "views_per_case": args.views_per_case, "n_views": int(len(groups)), "window_counts": wcount,
           "skipped": skipped, "jitter": {**ranges, "press_max_mm": args.press_max_mm,
                                          "rule": "slide+tilt about the contact point, clamp outside the "
                                                  "skin, press U(0, press_max)"},
           "clamp": {"clamped_frac": float((li > 1e-4).mean()), "lift_mm_mean": float(li.mean() * 1e3),
                     "press_mm_mean": float(np.mean(presses) * 1e3)},
           "n_real": int(len(real)), "real": args.real, "real_md5": P.md5(args.real),
           "erode_px": args.erode, "dr": args.dr, "renorm": args.renorm, "render_device": dev,
           "stages": stage_paths, "real_mean_in_sector": float(real[:, Me].mean()),
           "c2st": {"repeats": args.c2st_repeats, "folds": args.c2st_folds, "epochs": args.c2st_epochs,
                    "size": args.c2st_size, "groups": "sim: case x window; real: frame"},
           "variants": {}, "timing_s": {"render": round(t_render, 1)}}
    fr = inc = None
    if not args.no_fid:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from eval_fid_kid import _inception, _features, _fid, _kid
        fdev = R.get_device()
        inc = _inception(fdev)
        fr = _features(inc, list(real), fdev)
    t_c2st = t_inc = 0.0
    renorms = {"none": ["none"], "robust": ["robust"], "both": ["none", "robust"]}[args.renorm]
    for name, Pl in pools.items():
        Praw = np.asarray(Pl, np.float32)
        Pm = Praw * Me
        v = {"n": int(len(Pm)), "mean_in_sector": float(Pm[:, Me].mean()),
             "w1_in_sector": intensity_w1(Pm[:, Me], real[:, Me]),
             "w1_per_case": {c: intensity_w1(Pm[cases_arr == c][:, Me], real[:, Me])
                             for c in args.cases if (cases_arr == c).any()}}
        if "robust" in renorms:
            Pr = np.stack([robust_norm(x) for x in Praw]) * Me
            v["w1_in_sector_robust_renorm"] = intensity_w1(Pr[:, Me], real[:, Me])
            v["w1_per_case_robust_renorm"] = {c: intensity_w1(Pr[cases_arr == c][:, Me], real[:, Me])
                                              for c in args.cases if (cases_arr == c).any()}
        tag = " [DR full]" if name.endswith("[DR full]") else ""
        if name.startswith("physics+"):
            v["edge_iou_in_sector"] = float(np.mean([edge_iou(src[tag][i] * Me, Pm[i]) for i in range(len(Pm))]))
        if fr is not None:
            ti = time.time()
            fg = _features(inc, list(Pm), R.get_device())
            v["fid_secondary"] = _fid(fr, fg)
            km, ks = _kid(fr, fg, subset=min(args.kid_subset, len(fr), len(fg)))
            v["kid"], v["kid_std"] = km, ks
            v["kid_per_case"] = {c: _kid(fr, fg[cases_arr == c],
                                         subset=min(args.kid_subset, len(fr), int((cases_arr == c).sum())))[0]
                                 for c in args.cases if (cases_arr == c).sum() >= 2}
            t_inc += time.time() - ti
        tc = time.time()
        a = c2st_cnn(Pm, groups, real, real_groups, repeats=args.c2st_repeats, folds=args.c2st_folds,
                     epochs=args.c2st_epochs, size=args.c2st_size, seed=args.seed, device=args.c2st_device)
        t_c2st += time.time() - tc
        a = np.asarray(a, float)
        v["c2st_auc_mean"] = float(np.nanmean(a)) if a.size else float("nan")
        v["c2st_auc_std"] = float(np.nanstd(a)) if a.size else float("nan")
        v["c2st_n"] = int(np.isfinite(a).sum())
        res["variants"][name] = v
        print(f"{name:28s} W1 {v['w1_in_sector']:.4f} KID {v.get('kid', float('nan')):.4f} "
              f"FID {v.get('fid_secondary', float('nan')):7.1f} C2ST {v['c2st_auc_mean']:.3f}+-{v['c2st_auc_std']:.3f} "
              f"(n={v['c2st_n']})", flush=True)
    tc = time.time()
    null = np.asarray(c2st_cnn(np.asarray(pools["physics"]) * Me, groups, real, real_groups,
                               repeats=args.null_repeats, folds=args.c2st_folds, epochs=args.c2st_epochs,
                               size=args.c2st_size, seed=args.seed + 99, permute=True,
                               device=args.c2st_device), float)
    t_c2st += time.time() - tc
    res["c2st_permutation_null_physics"] = {"mean": float(np.nanmean(null)) if null.size else float("nan"),
                                            "std": float(np.nanstd(null)) if null.size else float("nan"),
                                            "n": int(np.isfinite(null).sum())}
    res["c2st_all_separable"] = bool(all(v["c2st_auc_mean"] >= 0.99 for v in res["variants"].values()))
    ph, lo = res["variants"].get("physics"), res["variants"].get("prior LOTUS")
    if ph and lo:
        res["physics_beats_lotus_all_six"] = {
            "w1": bool(all(ph["w1_per_case"][c] < lo["w1_per_case"][c] for c in ph["w1_per_case"])),
            "w1_per_case": {c: [ph["w1_per_case"][c], lo["w1_per_case"][c]] for c in ph["w1_per_case"]},
            "n_cases": len(ph["w1_per_case"])}
        if "kid_per_case" in ph:
            res["physics_beats_lotus_all_six"]["kid"] = bool(
                all(ph["kid_per_case"][c] < lo["kid_per_case"][c] for c in ph["kid_per_case"]))
    res["timing_s"].update(inception=round(t_inc, 1), c2st=round(t_c2st, 1), total=round(time.time() - t0, 1))
    res["provenance"] = P.collect(
        stages={k: P.texture_info(k, p) for k, p in stage_paths.items()},
        stage_sidecars={k: (P.sidecar_path(p) if os.path.exists(P.sidecar_path(p)) else None)
                        for k, p in stage_paths.items()},
        real={"path": args.real, "md5": P.md5(args.real)}, cohort={"eval": list(args.cases)},
        stage_blocks={k: P.result_block(texture=p, mask_source=None, dr_mode=args.dr, eval_cases=args.cases,
                                        target_set=list(args.organs)) for k, p in stage_paths.items()})
    print(json.dumps(res["timing_s"]), "null", res["c2st_permutation_null_physics"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sector", action="store_true", help="the v2 realism protocol (see module doc)")
    ap.add_argument("--allow-canonical", action="store_true", help="run --sector under USBG_SEAT=canonical")
    # legacy (canonical only)
    ap.add_argument("--case", default="s0519", help="legacy: held-out case to render sim views on")
    ap.add_argument("--real-dir", default=None, help="legacy: real frame directory (no default)")
    ap.add_argument("--n", type=int, default=40, help="legacy: sim views per variant")
    # stages (both modes)
    ap.add_argument("--cut", default=None)
    ap.add_argument("--cyclegan", default=None)
    ap.add_argument("--cyclegan-sc", default=None)
    ap.add_argument("--sonospade", default=None, help="sector mode only")
    ap.add_argument("--tex-manifest", default=None, help="sector: JSON {stage: ckpt|null}")
    # sector v2
    ap.add_argument("--cases", nargs="+", default=None, help="sector: default benchmark.TEST_PATIENTS")
    ap.add_argument("--organs", nargs="+", default=["liver", "kidney"], choices=["liver", "kidney", "gallbladder"])
    ap.add_argument("--views-per-case", type=int, default=50)
    ap.add_argument("--press-max-mm", type=float, default=5.0)
    ap.add_argument("--real", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--dr", choices=["off", "full", "both"], default="off")
    ap.add_argument("--erode", type=int, default=3)
    ap.add_argument("--renorm", choices=["none", "robust", "both"], default="none")
    ap.add_argument("--kid-subset", type=int, default=50)
    ap.add_argument("--no-fid", action="store_true", help="skip Inception KID/FID")
    ap.add_argument("--c2st-repeats", type=int, default=5)
    ap.add_argument("--c2st-folds", type=int, default=5)
    ap.add_argument("--c2st-epochs", type=int, default=40)
    ap.add_argument("--c2st-size", type=int, default=64)
    ap.add_argument("--c2st-device", default="cpu")
    ap.add_argument("--null-repeats", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="sector: render + translator device")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/realism.json")
    args = ap.parse_args(argv)

    from usbg.placement import seat_mode
    from usbg import provenance as P
    if args.sector:
        if seat_mode() == "canonical" and not args.allow_canonical:
            raise SystemExit("--sector is the anatomical-seat protocol; USBG_SEAT=canonical needs --allow-canonical")
        if args.cases is None:
            from usbg.benchmark import TEST_PATIENTS
            args.cases = list(TEST_PATIENTS)
        res = run_sector(args)
    else:
        if seat_mode() != "canonical":
            raise SystemExit("the legacy (linear, one-case) protocol is canonical-seat only; use --sector "
                             "or USBG_SEAT=canonical to reproduce the withdrawn v1 realism table")
        if not args.real_dir:
            raise SystemExit("legacy protocol: pass --real-dir (no default: the old default was a private pool)")
        for p in [args.real_dir, args.cut, args.cyclegan, args.cyclegan_sc]:
            if p and P.is_private_path(p):
                raise SystemExit(f"{p} is a private clinical pool")
        res = run_legacy(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
