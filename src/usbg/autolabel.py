"""autolabel.py: run the MS-TCN recognizer over each trajectory's kinematics as a QC pass.

The maneuver label is KNOWN in sim (we commanded it), so running par.eval.predict_record over
the same twist/rel6 the dataset stored is a consistency check: does the calibrated recognizer,
given only the kinematics, agree with the commanded maneuver? High agreement on clean
trajectories means the demonstrator motion is on-distribution for the recognizer; where it is
low, the report flags it. This is the exact pipeline that will auto-label real scans later.

Reuses, verbatim: par.baselines.load_teacher (the calibrated MS-TCN++ teacher, runs/best_model.pt),
par.data.ClipRecord / _valid_eff (build a record from twist+rel6+valid), par.eval.predict_record
(per-frame class + anticipation + latent), and par.eval confusion/PRF metrics.
"""
from __future__ import annotations

import os

import numpy as np

DEFAULT_CKPT = os.environ.get("USBG_RECOGNIZER_CKPT")   # v1 only: the probe-action recognizer checkpoint


def get_device(prefer: str = "mps") -> str:
    import torch
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_recognizer(ckpt: str | None = None, device: str | None = None):
    """Return (model, norm, device). Uses the calibrated teacher checkpoint from Repo A."""
    from par.baselines import load_teacher
    ckpt = ckpt or DEFAULT_CKPT
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"recognizer checkpoint not found: {ckpt}")
    device = device or get_device()
    model, norm, _ck = load_teacher(ckpt, device)
    return model, norm, device


def _record(twist, rel6, valid, label, source="clean"):
    from par.data import ClipRecord, _valid_eff
    tw = np.asarray(twist, dtype=np.float64)
    rl = np.asarray(rel6, dtype=np.float64)
    vl = np.asarray(valid, dtype=bool)
    return ClipRecord(twist=tw, rel6=rl, valid_eff=_valid_eff(tw, vl), label=int(label),
                      source=str(source))


def predict_clip(model, norm, twist, rel6, valid, label, device, source="clean") -> dict:
    """Recognizer outputs for one trajectory: per-frame pred/probs, anticipation, latent."""
    from par.eval import predict_record
    rec = _record(twist, rel6, valid, label, source)
    return predict_record(model, rec, norm, device)


def agreement_report(clips: list[dict], model, norm, device, low_thresh: float = 0.5) -> dict:
    """Recognizer-vs-commanded agreement per maneuver and per source, with flags.

    clips : list of dicts with keys twist, rel6, valid, label, source, case, traj_id
            (exactly dataset.load_npz output; 'rel6' or 'seqs' accepted).
    """
    from par.eval import confusion_matrix, per_class_prf
    from .contracts_bridge import CLASSES_5, SESSION_CLASSES_6, N_CLASSES

    C = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)   # commanded (row) x predicted (col)
    per_man = {m: {"clips": 0, "frame_hits": 0, "frame_tot": 0, "clip_hits": 0} for m in CLASSES_5}
    per_src = {"clean": {"hits": 0, "tot": 0}, "noisy": {"hits": 0, "tot": 0}}
    flagged = []
    tot_hits = tot_frames = 0

    for c in clips:
        rel6 = c.get("rel6", c.get("seqs"))
        res = predict_clip(model, norm, c["twist"], rel6, c["valid"], c["label"],
                           device, c.get("source", "clean"))
        pred, valid = res["pred"], np.asarray(res["valid"], bool)
        commanded = int(c["label"])
        man = CLASSES_5[commanded]
        pv = pred[valid]
        if pv.size == 0:
            continue
        hits = int((pv == commanded).sum())
        frame_agreement = hits / pv.size
        # confusion over valid frames (predicted may be 0..5 incl hold)
        for p in pv:
            C[commanded, int(p)] += 1
        # clip-level majority vote
        maj = int(np.bincount(pv, minlength=N_CLASSES).argmax())
        clip_hit = int(maj == commanded)

        per_man[man]["clips"] += 1
        per_man[man]["frame_hits"] += hits
        per_man[man]["frame_tot"] += pv.size
        per_man[man]["clip_hits"] += clip_hit
        src = c.get("source", "clean")
        per_src.setdefault(src, {"hits": 0, "tot": 0})
        per_src[src]["hits"] += hits
        per_src[src]["tot"] += pv.size
        tot_hits += hits
        tot_frames += pv.size

        if frame_agreement < low_thresh:
            flagged.append({
                "traj_id": int(c.get("traj_id", -1)), "case": c.get("case", ""),
                "maneuver": man, "source": src,
                "frame_agreement": round(frame_agreement, 3),
                "majority_pred": SESSION_CLASSES_6[maj]})

    prec, rec, f1, macro = per_class_prf(C)
    report = {
        "n_clips": len(clips), "n_valid_frames": int(tot_frames),
        "overall_frame_agreement": round(tot_hits / max(tot_frames, 1), 4),
        "macro_f1": round(float(macro), 4),
        "per_maneuver": {
            m: {"clips": d["clips"], "frames": int(d["frame_tot"]),
                "frame_agreement": round(d["frame_hits"] / max(d["frame_tot"], 1), 4),
                "clip_agreement": round(d["clip_hits"] / max(d["clips"], 1), 4)}
            for m, d in per_man.items()},
        "per_source": {
            s: {"frames": int(d["tot"]),
                "frame_agreement": round(d["hits"] / max(d["tot"], 1), 4)}
            for s, d in per_src.items() if d["tot"] > 0},
        "confusion_commanded_x_predicted": C.tolist(),
        "classes_row": list(CLASSES_5), "classes_col": list(SESSION_CLASSES_6),
        "flagged_low_agreement": sorted(flagged, key=lambda x: x["frame_agreement"]),
        "low_thresh": low_thresh,
    }
    return report


def plot_confusion(report: dict, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = np.asarray(report["confusion_commanded_x_predicted"], dtype=np.float64)
    Cn = C / np.clip(C.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Cn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(report["classes_col"])), report["classes_col"], rotation=45, ha="right")
    ax.set_yticks(range(len(report["classes_row"])), report["classes_row"])
    ax.set_xlabel("recognizer prediction"); ax.set_ylabel("commanded maneuver")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] > 0:
                ax.text(j, i, int(C[i, j]), ha="center", va="center",
                        color="w" if Cn[i, j] < 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, label="row-normalized")
    ax.set_title(f"recognizer vs commanded (overall {report['overall_frame_agreement']:.2f})")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png
