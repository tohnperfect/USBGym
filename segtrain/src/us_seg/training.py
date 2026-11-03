"""Trainer: train the 6-class MPS U-Net on simulated frames, evaluate per-domain.

Reuses `usbg.segmenter` (the MPS-native U-Net + Dice+CE). Adds the two things the imbalanced small
classes (gallbladder, vessel) need: inverse-frequency **class weighting** in the CE
term and rare-class **frame oversampling**. Evaluates per-class Dice on held-out sim (val) and on the
60 real frames (the zero-shot cross-domain / TSTR headline).
"""
from __future__ import annotations

import functools
import time

import numpy as np

_log = functools.partial(print, flush=True)  # unbuffered progress when piped to a file

from . import gym_bridge
from . import ontology as ONT
from .nnunet_raw import RawDataset
from . import real_data


# ---- metrics (numpy, per class) -----------------------------------------------------------------

def per_class_dice(pred: np.ndarray, target: np.ndarray, n_classes: int = ONT.N_CLASSES,
                   eps: float = 1e-6) -> dict:
    """Per-class Dice averaged over frames where the class is present in the target.

    Returns {class_name: (dice, support_frames)} for classes 1..n-1 (foreground) that appear.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    out = {}
    for c in range(1, n_classes):
        dices = []
        for p, t in zip(pred, target):
            tc = t == c
            if not tc.any():
                continue
            pc = p == c
            inter = np.logical_and(pc, tc).sum()
            dices.append((2.0 * inter + eps) / (pc.sum() + tc.sum() + eps))
        if dices:
            out[ONT.CLASS_NAMES[c]] = (float(np.mean(dices)), len(dices))
    return out


def mean_fg_dice(pcd: dict) -> float:
    return float(np.mean([v[0] for v in pcd.values()])) if pcd else 0.0


# ---- sampling / weighting -----------------------------------------------------------------------

def class_pixel_freq(labels: np.ndarray, n_classes: int = ONT.N_CLASSES) -> np.ndarray:
    vals = labels.reshape(-1)
    vals = vals[vals < n_classes]                          # drop ignore_index (255) pixels from stats
    counts = np.bincount(vals, minlength=n_classes).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def inverse_freq_weights(labels: np.ndarray, n_classes: int = ONT.N_CLASSES,
                         cap: float = 20.0) -> np.ndarray:
    """Normalized inverse-sqrt-frequency class weights, capped, for the weighted CE term."""
    freq = class_pixel_freq(labels, n_classes)
    w = 1.0 / np.sqrt(freq + 1e-6)
    w = np.clip(w, None, w.min() * cap)
    w = w / w.mean()
    return w.astype(np.float32)


def frame_sampling_weights(labels: np.ndarray, rare_classes=(2, 3),
                           boost: float = 4.0) -> np.ndarray:
    """Per-frame sampling probability, upweighting frames that contain a rare class."""
    w = np.ones(len(labels), dtype=np.float64)
    for i, t in enumerate(labels):
        present = set(np.unique(t).tolist())
        if present & set(rare_classes):
            w[i] *= boost
    return w / w.sum()


# ---- train --------------------------------------------------------------------------------------

def train(
    dataset_dir: str,
    out_ckpt: str,
    *,
    iters: int = 3000,
    batch: int = 8,
    lr: float = 5e-4,
    base: int = 32,
    depth: int = 4,
    dice_weight: float = 1.0,
    val_frac: float = 0.2,
    seed: int = 0,
    rare_boost: float = 4.0,
    eval_every: int = 500,
    real_frac: float = 0.4,
    real_holdout_idx=None,
    add_kidneyus: bool = False,
    real_train_idx=None,
    forbid_sim_cases=None,
    log=_log,
) -> dict:
    """Train the 6-class U-Net on mixed simulated + real frames.

    Real Vitale frames are folded into training (domain-balanced to ~`real_frac` of each batch) so the
    net directly learns the shared sim/real class mapping. If `real_holdout_idx` is given, those real
    frame indices are excluded from training and used as a leakage-free real eval set (cross-val);
    otherwise all real frames train and the "real" number is reported on the training frames (fit, not
    generalization). Set `real_frac=0` for sim-only.

    v2: `real_train_idx` restricts the Vitale frames that may EVER be trained on (the fixed v2 split's
    non-holdout frames); frames outside it are never trained on, whatever `real_holdout_idx` says.
    `forbid_sim_cases` (e.g. the benchmark TEST_PATIENTS) raises if any sim source case is one of them.
    """
    gym_bridge.ensure_usbg_importable()
    import torch
    import torch.nn.functional as F
    from usbg import segmenter as SEG
    from usbg import render_lotus as R

    device = R.get_device()
    log(f"[device] {device}")

    ds = RawDataset(dataset_dir)
    if forbid_sim_cases:
        srcs = {ds.source_case(c) for c in ds.train_cases}
        leak = sorted(srcs & set(forbid_sim_cases))
        if leak:
            raise ValueError(f"[guard] sim dataset {dataset_dir} contains forbidden cases {leak}")
    tr_cases, va_cases = ds.split_by_source_case(val_frac=val_frac, seed=seed)
    Xsim, Ysim = ds.load_all(tr_cases)
    Xva, Yva = ds.load_all(va_cases)
    log(f"[sim ] {len(ds)} frames; train {len(tr_cases)} cases / val {len(va_cases)} cases; "
        f"Xsim {Xsim.shape}")

    # real: split into train / held-out-eval
    real = real_data.load_real_test()
    Xr_all = real["images"]
    Yr_all = real["label6"].astype(np.int64)
    n_real = real["n"]
    allowed = np.ones(n_real, dtype=bool)
    if real_train_idx is not None:
        allowed[:] = False
        allowed[np.asarray(real_train_idx, dtype=int)] = True
    if real_holdout_idx is not None and len(real_holdout_idx):
        ho = np.zeros(n_real, dtype=bool)
        ho[np.asarray(real_holdout_idx, dtype=int)] = True
        trm = allowed & ~ho
        Xr_tr, Yr_tr = Xr_all[trm], Yr_all[trm]
        Xr_ev, Yr_ev = Xr_all[ho], Yr_all[ho]
        real_eval_kind = f"held-out {int(ho.sum())} real"
    else:
        trm = allowed
        Xr_tr, Yr_tr = Xr_all[trm], Yr_all[trm]
        Xr_ev, Yr_ev = Xr_tr, Yr_tr
        real_eval_kind = f"train-set {int(trm.sum())} real (fit)"
    real_trained_idx = np.where(trm)[0].tolist()
    log(f"[real] train on {len(Xr_tr)} real, eval on {len(Xr_ev)} ({real_eval_kind}); "
        f"target real fraction/batch = {real_frac}")

    n_kidneyus = 0
    if add_kidneyus:
        kd = real_data.load_kidneyus_train()
        if not kd["n"]:
            raise FileNotFoundError("--kidneyus requested but no KidneyUS train frames were found")
        n_kidneyus = int(kd["n"])
        if kd["n"]:
            Xr_tr = np.concatenate([Xr_tr, kd["images"].astype(np.float32)], axis=0)
            Yr_tr = np.concatenate([Yr_tr, kd["label6"].astype(np.int64)], axis=0)
            log(f"[real] + {kd['n']} kidneyUS partial-label frames "
                f"(kidney supervised; non-kidney pixels ignore=255)")

    # combined pool: sim-train + real-train
    if real_frac > 0 and len(Xr_tr):
        Xall = np.concatenate([Xsim, Xr_tr], axis=0)
        Yall = np.concatenate([Ysim, Yr_tr], axis=0)
        domain_real = np.concatenate([np.zeros(len(Xsim), bool), np.ones(len(Xr_tr), bool)])
    else:
        Xall, Yall = Xsim, Ysim
        domain_real = np.zeros(len(Xsim), bool)
    n_sim = int((~domain_real).sum())
    n_rl = int(domain_real.sum())

    # per-frame sampling weights: apply the rare-class boost WITHIN each domain, then renormalize
    # per-domain to the target sim/real split. (A single global renorm after the boost would let a
    # domain with more rare-class frames drift above its target fraction.)
    rare = frame_sampling_weights(Yall, rare_classes=(ONT.NNUNET_LABELS['gallbladder'],
                                                      ONT.NNUNET_LABELS['vessel']), boost=rare_boost)
    samp_w = np.array(rare, dtype=np.float64)  # per-frame rare multiplier (already >0 everywhere)
    if n_rl and real_frac > 0:
        sim_m, real_m = ~domain_real, domain_real
        samp_w[sim_m] *= (1.0 - real_frac) / samp_w[sim_m].sum()
        samp_w[real_m] *= real_frac / samp_w[real_m].sum()
    else:
        samp_w = samp_w / samp_w.sum()
    realized_real = float(samp_w[domain_real].sum()) if n_rl else 0.0
    log(f"[sample] realized real fraction/batch = {realized_real:.3f} (target {real_frac})")

    cls_w = inverse_freq_weights(Yall)
    log(f"[weights] class weights {np.round(cls_w, 2).tolist()} ({list(ONT.CLASS_NAMES)})")

    seg = SEG.Segmenter(device=device, n_classes=ONT.N_CLASSES, base=base, depth=depth, group_map=None)
    opt = torch.optim.Adam(seg.net.parameters(), lr=lr)
    w_t = torch.tensor(cls_w, device=device)
    rng = np.random.default_rng(seed)

    def loss_fn(logits, target):
        # ignore_index=255 marks partial-label (single-organ) pixels whose true class is unknown:
        # they are excluded from both CE and Dice so those frames only supervise their labeled organ.
        ce = F.cross_entropy(logits, target, weight=w_t, ignore_index=255)
        probs = F.softmax(logits, dim=1)
        valid = (target != 255).unsqueeze(1).float()                        # (B,1,H,W)
        tgt = F.one_hot(target.clamp(0, ONT.N_CLASSES - 1), ONT.N_CLASSES).permute(0, 3, 1, 2).float()
        tgt = tgt * valid
        probs = probs * valid
        dims = (0, 2, 3)
        inter = (probs * tgt).sum(dims)
        denom = probs.sum(dims) + tgt.sum(dims)
        dice = 1.0 - ((2 * inter + 1e-6) / (denom + 1e-6)).mean()
        return ce + dice_weight * dice

    history = []
    t0 = time.time()
    seg.net.train()
    for it in range(1, iters + 1):
        idx = rng.choice(len(Xall), size=batch, replace=True, p=samp_w)
        x = torch.from_numpy(Xall[idx][:, None, :, :]).float().to(device)
        y = torch.from_numpy(Yall[idx].astype(np.int64)).to(device)
        opt.zero_grad()
        logits = seg.net(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if it % eval_every == 0 or it == iters:
            sim_pcd = _eval(seg, Xva, Yva)
            real_pcd = _eval(seg, Xr_ev, Yr_ev)
            rec = {
                "iter": it, "loss": float(loss.item()),
                "sim_val": {k: round(v[0], 4) for k, v in sim_pcd.items()},
                "sim_val_mean": round(mean_fg_dice(sim_pcd), 4),
                "real": {k: round(v[0], 4) for k, v in real_pcd.items()},
                "real_mean": round(mean_fg_dice(real_pcd), 4),
            }
            history.append(rec)
            log(f"[{it:5d}] loss {rec['loss']:.3f} | sim-val mDice {rec['sim_val_mean']:.3f} "
                f"{rec['sim_val']} | real[{real_eval_kind}] mDice {rec['real_mean']:.3f} {rec['real']} "
                f"| {it / (time.time() - t0):.1f} it/s")
            seg.net.train()

    seg.save(out_ckpt)
    log(f"[done] saved {out_ckpt} in {time.time() - t0:.0f}s")
    return {"checkpoint": out_ckpt, "history": history, "class_weights": cls_w.tolist(),
            "train_cases": tr_cases, "val_cases": va_cases,
            "train_source_cases": sorted({ds.source_case(c) for c in tr_cases}),
            "val_source_cases": sorted({ds.source_case(c) for c in va_cases}),
            "n_sim_train": n_sim, "n_real_train": n_rl, "real_frac": real_frac,
            "vitale_trained_idx": real_trained_idx, "n_kidneyus_train": n_kidneyus,
            "dataset_dir": dataset_dir, "seed": seed, "iters": iters,
            "real_eval_kind": real_eval_kind}


def _eval(seg, images: np.ndarray, labels6: np.ndarray) -> dict:
    seg.net.eval()
    preds = np.stack([seg.predict(im) for im in images])
    return per_class_dice(preds, labels6)
