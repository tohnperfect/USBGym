#!/usr/bin/env python3
"""Pretrain S_US, the SonoSPADE segmentation-consistency segmenter, on FREE aligned sim pairs.

The simulator emits perfectly aligned (physics image, canonical label) pairs (make_dataset
--pairs-out), so S_US is supervised with exact labels at no annotation cost. Train it here, hold
out whole sessions (CT cases) for validation so the Dice number is honest (no frame leakage), and
save the checkpoint that the sonospade translator later freezes. Domain-randomized renders teach
S_US to key on structure, so it still segments the generator's fakes and the real pool.

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  python scripts/train_segmenter.py --pairs data/dataset/pairs_smoke.npz \\
      --iters 800 --out data/dataset/s_us.pt --preview outputs/s_us_preview.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def _frame_split(n, val_frac, rng):
    idx = rng.permutation(n)
    cut = max(1, int(n * (1 - val_frac)))
    return idx[:cut], idx[cut:]


def _session_split(sessions, val_frac, rng, mode="session"):
    """Split into (train_idx, val_idx). mode='session' holds out whole sessions (CT cases), the
    honest, no-leakage contract; mode='frame' holds out random poses of seen cases, the literal
    'held-out sim slices' reading. Falls back to a frame split when sessions are unavailable or a
    single session is present (so a smoke run still has a val set)."""
    n = len(sessions) if sessions is not None else 0
    if sessions is None or mode == "frame":
        return _frame_split(len(sessions) if sessions is not None else n, val_frac, rng)
    uniq = sorted(set(sessions.tolist()))
    if len(uniq) == 1:
        return _frame_split(n, val_frac, rng)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_sessions = set(uniq[:n_val])
    val = np.array([i for i in range(n) if sessions[i] in val_sessions], dtype=np.int64)
    train = np.array([i for i in range(n) if sessions[i] not in val_sessions], dtype=np.int64)
    return train, val


def _write_sidecar(args, tr_cases, va_cases, meta, final_dice, hp):
    """<out>.provenance.json (usbg.provenance): train/val CT cases from the pairs sessions, the pairs as
    parent, the seat the pairs were rendered in (pairs meta, else the pairs sidecar)."""
    from usbg import provenance as P
    psc = P.sidecar_path(args.pairs)
    parent = P.load_provenance(psc) if os.path.exists(psc) else {}
    seat = (meta or {}).get("seat_mode") or parent.get("seat_mode") or P.current_seat_mode()
    dr = (parent.get("dr") or {}).get("mode", "off")
    try:
        out = P.write_provenance(args.out, "s_us", train_cases=tr_cases, val_cases=va_cases,
                                 iters=hp["iters"], seed=args.seed, dr_mode=dr,
                                 dr_notes="inherited from the pairs (per-frame render DR)",
                                 parents=[P.parent_entry(args.pairs, "pairs")], seat_mode=seat,
                                 extra={**hp, "config": {"path": args.config, "md5": P.md5(args.config)},
                                        "final_val_dice": final_dice})
        print(f"wrote provenance -> {out}")
    except ValueError as e:
        print(f"WARNING: no provenance sidecar for {args.out}:\n  {e}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/dataset/usbg_labeled.npz",
                    help="aligned (image, label) pairs npz from make_dataset --pairs-out")
    ap.add_argument("--config", default="configs/sonospade.yaml")
    ap.add_argument("--base", type=int, default=None, help="U-Net base width (overrides config)")
    ap.add_argument("--depth", type=int, default=None, help="U-Net depth (overrides config)")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=0.34)
    ap.add_argument("--split", choices=["session", "frame"], default="session",
                    help="session: hold out whole cases (no leakage); frame: hold out poses of seen cases")
    ap.add_argument("--group", action="store_true",
                    help="merge acoustically degenerate soft tissues into one class (grouped scheme, "
                         "9 classes) so S_US is not asked to separate the inseparable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/dataset/s_us.pt")
    ap.add_argument("--preview", default="outputs/s_us_preview.png")
    ap.add_argument("--no-provenance", action="store_true", help="skip the <out>.provenance.json sidecar")
    ap.add_argument("--allow-heldout-leak", action="store_true",
                    help="train even if a held-out benchmark patient is in the pairs (reproducing a "
                         "withdrawn artifact only; no provenance sidecar is written)")
    args = ap.parse_args(argv)

    import torch
    from usbg import texture_gan as TG
    from usbg import segmenter as SEG
    from usbg.dataset import load_pairs_npz
    from usbg.render_lotus import get_device
    from usbg.volume import N_LABELS, USBG_LABEL_NAMES

    cfg = TG.load_config(args.config)
    scfg = cfg.get("segmenter", {})
    base = args.base if args.base is not None else scfg.get("base", 32)
    depth = args.depth if args.depth is not None else scfg.get("depth", 4)
    iters = args.iters if args.iters is not None else scfg.get("iters", 3000)
    batch = args.batch if args.batch is not None else scfg.get("batch", 8)
    lr = args.lr if args.lr is not None else scfg.get("lr", 5e-4)

    device = get_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    imgs, labs, sessions, meta = load_pairs_npz(args.pairs)
    imgs = imgs.astype(np.float32) / 255.0
    labs = labs.astype(np.int64)
    # grouped scheme merges the acoustically degenerate soft tissues (improvement 1)
    if args.group:
        gmap = SEG.grouped_map_array()
        labs = SEG.remap_labels(labs, gmap)
        nc = SEG.N_GROUPED
    else:
        gmap = None
        nc = N_LABELS
    tr_idx, va_idx = _session_split(sessions, args.val_frac, rng, mode=args.split)
    from usbg.benchmark import TEST_PATIENTS
    sess_s = np.asarray(sessions).astype(str) if sessions is not None else None
    tr_cases = sorted(set(sess_s[tr_idx].tolist())) if sess_s is not None else []
    va_cases = sorted(set(sess_s[va_idx].tolist()) - set(tr_cases)) if sess_s is not None else []
    leak = sorted((set(tr_cases) | set(va_cases)) & set(TEST_PATIENTS))
    if leak and not args.allow_heldout_leak:
        raise SystemExit(f"refusing to train: held-out benchmark patients {leak} are in {args.pairs} (D3); "
                         f"--allow-heldout-leak reproduces a withdrawn artifact without provenance")
    print(f"train cases {tr_cases} | val cases {va_cases}")
    print(f"pairs {len(imgs)} ({imgs.shape[1:]}), split by {args.split}, "
          f"scheme {'grouped(' + str(nc) + ')' if args.group else 'canonical(13)'}, "
          f"train {len(tr_idx)} / val {len(va_idx)}, "
          f"device {device}, base {base} depth {depth}, {iters} iters")

    seg = SEG.Segmenter(device=device, n_classes=nc, base=base, depth=depth, group_map=gmap)
    opt = torch.optim.Adam(seg.net.parameters(), lr=lr)

    def batch_tensors(idx_pool):
        bi = rng.choice(idx_pool, size=min(batch, len(idx_pool)), replace=len(idx_pool) < batch)
        x = torch.from_numpy(imgs[bi][:, None]).to(device)
        y = torch.from_numpy(labs[bi]).to(device)
        return x, y

    seg.net.train()
    for it in range(iters):
        x, y = batch_tensors(tr_idx)
        opt.zero_grad()
        logits = seg.net(x)
        loss = SEG.seg_ce_dice_loss(logits, y, n_classes=nc,
                                    dice_weight=cfg.get("loss", {}).get("lambda_seg_dice", 1.0))
        loss.backward(); opt.step()
        if (it + 1) % max(1, iters // 10) == 0:
            seg.net.eval()
            with torch.no_grad():
                dices = []
                for j in range(0, len(va_idx), batch):
                    vb = va_idx[j:j + batch]
                    xv = torch.from_numpy(imgs[vb][:, None]).to(device)
                    yv = torch.from_numpy(labs[vb]).to(device)
                    dices.append(SEG.dice_score(seg.net(xv), yv, n_classes=nc))
            val_dice = float(np.mean(dices)) if dices else float("nan")
            print(f"  iter {it+1:5d}  loss {float(loss.detach()):.4f}  val Dice {val_dice:.4f}")
            seg.net.train()

    seg.save(args.out)
    print(f"saved S_US -> {args.out}")

    # final held-out Dice (the acceptance number)
    seg.net.eval()
    with torch.no_grad():
        dices = []
        for j in range(0, len(va_idx), batch):
            vb = va_idx[j:j + batch]
            xv = torch.from_numpy(imgs[vb][:, None]).to(device)
            yv = torch.from_numpy(labs[vb]).to(device)
            dices.append(SEG.dice_score(seg.net(xv), yv, n_classes=nc))
    final_dice = float(np.mean(dices)) if dices else float("nan")
    print(f"held-out sim Dice: {final_dice:.4f}  (accept: > 0.8)")
    if not args.no_provenance and not leak:
        _write_sidecar(args, tr_cases, va_cases, meta, final_dice,
                       dict(base=base, depth=depth, iters=iters, batch=batch, lr=lr, n_classes=nc,
                            grouped=bool(args.group), split=args.split, val_frac=args.val_frac,
                            n_train_frames=int(len(tr_idx)), n_val_frames=int(len(va_idx))))

    if args.preview and len(va_idx):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = va_idx[:5]
        with torch.no_grad():
            preds = [seg.predict(imgs[i]) for i in show]
        fig, axes = plt.subplots(3, len(show), figsize=(2.0 * len(show), 6.2))
        for i, s in enumerate(show):
            axes[0][i].imshow(imgs[s], cmap="gray", vmin=0, vmax=1); axes[0][i].axis("off")
            axes[1][i].imshow(labs[s], cmap="tab20", vmin=0, vmax=nc); axes[1][i].axis("off")
            axes[2][i].imshow(preds[i], cmap="tab20", vmin=0, vmax=nc); axes[2][i].axis("off")
        axes[0][0].set_title("physics image", fontsize=9, loc="left")
        axes[1][0].set_title("true label", fontsize=9, loc="left")
        axes[2][0].set_title("S_US pred", fontsize=9, loc="left")
        os.makedirs(os.path.dirname(os.path.abspath(args.preview)), exist_ok=True)
        fig.suptitle(f"S_US on held-out sim (Dice {final_dice:.3f}, {device})", fontsize=10)
        fig.tight_layout(); fig.savefig(args.preview, dpi=130); plt.close(fig)
        print(f"wrote preview -> {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
