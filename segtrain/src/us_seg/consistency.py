"""Cross-domain class-assignment consistency metrics.

The controller does not need per-frame segmentation accuracy; it needs the segmenter to *label the
same anatomical region with the same class in sim and in real*. These metrics measure that.

Core idea: for pixels whose ground-truth is organ X, look at the predicted-class distribution
P(pred | true=X) in each domain. If the segmenter is cross-domain consistent, that distribution is
the same in sim and real (ideally concentrated on X, but crucially *matched* across domains, including
the direction of any errors). No sim<->real pixel correspondence is required: each domain uses its own
ground-truth to define "the region that is organ X".
"""
from __future__ import annotations

import numpy as np

from . import ontology as ONT


def confusion_counts(preds: np.ndarray, gts: np.ndarray, n_classes: int = ONT.N_CLASSES) -> np.ndarray:
    """Rows = true class, cols = predicted class; counts over all pixels. (N,H,W) int arrays."""
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, g in zip(np.asarray(preds), np.asarray(gts)):
        p = p.reshape(-1)
        g = g.reshape(-1)
        for t in range(n_classes):
            m = g == t
            if m.any():
                C[t] += np.bincount(p[m], minlength=n_classes)[:n_classes]
    return C


def row_normalize(C: np.ndarray) -> np.ndarray:
    """P(pred | true) per row; zero rows -> zeros."""
    C = C.astype(np.float64)
    s = C.sum(axis=1, keepdims=True)
    out = np.zeros_like(C)
    nz = s[:, 0] > 0
    out[nz] = C[nz] / s[nz]
    return out


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2), in [0,1]; 0 = identical distributions."""
    p = np.clip(p, 1e-12, None); p = p / p.sum()
    q = np.clip(q, 1e-12, None); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * (np.log2(a) - np.log2(b))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def consistency_report(preds_sim, gts_sim, preds_real, gts_real,
                       n_classes: int = ONT.N_CLASSES) -> dict:
    """Compare P(pred|true=X) between sim and real for each foreground organ X.

    Per organ: self-assignment rate in each domain (diagonal), the modal predicted class in each
    domain and whether they match, histogram intersection (in [0,1], 1 = identical predicted-class
    distribution across domains), and JS divergence (0 = identical). Macro = mean over organs.
    """
    Cs = confusion_counts(preds_sim, gts_sim, n_classes)
    Cr = confusion_counts(preds_real, gts_real, n_classes)
    Rs = row_normalize(Cs)
    Rr = row_normalize(Cr)

    per_class = {}
    inter_list, modal_match_list = [], []
    for t in range(1, n_classes):
        if Cs[t].sum() == 0 or Cr[t].sum() == 0:
            continue  # organ absent in one domain's GT
        ds, dr = Rs[t], Rr[t]
        intersection = float(np.minimum(ds, dr).sum())          # = 1 - 0.5*L1
        jsd = _js_divergence(ds, dr)
        modal_s, modal_r = int(ds.argmax()), int(dr.argmax())
        per_class[ONT.CLASS_NAMES[t]] = {
            "sim_self_rate": round(float(ds[t]), 4),            # frac of true-X pixels called X in sim
            "real_self_rate": round(float(dr[t]), 4),           # ... in real
            "modal_class_sim": ONT.CLASS_NAMES[modal_s],
            "modal_class_real": ONT.CLASS_NAMES[modal_r],
            "modal_match": bool(modal_s == modal_r),
            "modal_is_correct": bool(modal_s == t and modal_r == t),
            "hist_intersection": round(intersection, 4),        # cross-domain distribution overlap
            "js_divergence": round(jsd, 4),
            "pred_dist_sim": {ONT.CLASS_NAMES[j]: round(float(ds[j]), 3) for j in range(n_classes) if ds[j] > 0.01},
            "pred_dist_real": {ONT.CLASS_NAMES[j]: round(float(dr[j]), 3) for j in range(n_classes) if dr[j] > 0.01},
        }
        inter_list.append(intersection)
        modal_match_list.append(modal_s == modal_r)

    return {
        "per_class": per_class,
        "macro_consistency": round(float(np.mean(inter_list)), 4) if inter_list else 0.0,
        "modal_match_rate": round(float(np.mean(modal_match_list)), 4) if modal_match_list else 0.0,
        "confusion_sim": Cs.tolist(),
        "confusion_real": Cr.tolist(),
    }
