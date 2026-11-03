"""anticipation.py: a compact causal next-maneuver head trained on multi-maneuver sessions.

The reused MS-TCN recognizer already has next-maneuver and time-to-boundary heads, but they are
trained on the recognizer program's own corpus. This module trains a small, causal GRU
specialized to our maneuver vocabulary and session distribution, so the anticipation used in the
guided overlay is sharper for this setup. Input is the per-frame 12-channel kinematics (body
twist plus start-relative pose, z-normalized); outputs are per-frame next-maneuver logits (over
the 6 session classes, including hold) and a time-to-next-boundary regression (in seconds). The
GRU is causal (left-to-right), so it is a real-time anticipator.

Anticipation for a randomly ordered maneuver sequence is fundamentally bounded before a
transition begins (the next maneuver is a free choice), so the useful signal is a fast, sharp
call once the transition starts plus a good time-to-boundary estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts_bridge import DT_C, SESSION_CLASSES_6, NEXT_ID_NONE, TARGET_CHANNELS


@dataclass
class Norm12:
    mean: np.ndarray      # (12,)
    std: np.ndarray       # (12,)

    @classmethod
    def fit(cls, sessions):
        rows = []
        for s in sessions:
            x = np.concatenate([np.asarray(s["twist"], np.float64),
                                np.asarray(s["rel6"], np.float64)], axis=1)
            rows.append(x[np.asarray(s["valid"], bool)])
        X = np.concatenate(rows, 0)
        mean = X.mean(0)
        std = X.std(0)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def apply(self, twist, rel6):
        x = np.concatenate([np.asarray(twist, np.float64), np.asarray(rel6, np.float64)], axis=1)
        z = (x - self.mean) / self.std
        return np.nan_to_num(z, nan=0.0).astype(np.float32)      # NaN twist frame-0 -> 0

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]))


def load_sessions(path):
    import json
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    out = []
    n = len(d["twist"])
    for i in range(n):
        out.append({"twist": d["twist"][i], "rel6": d["seqs"][i], "valid": d["valid"][i],
                    "frame_labels": d["frame_labels"][i], "next_id": d["next_id"][i],
                    "ttnb": d["ttnb"][i]})
    return out, meta


def build_model(in_ch=12, hidden=64, n_classes=len(SESSION_CLASSES_6)):
    import torch.nn as nn

    class _Antic(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(in_ch, hidden, num_layers=1, batch_first=True)
            self.head_next = nn.Linear(hidden, n_classes)          # next maneuver (class)
            self.head_ttnb = nn.Linear(hidden, 1)                  # time to boundary
            self.head_twist = nn.Linear(hidden, TARGET_CHANNELS)   # next-move body twist (direction+amplitude)

        def forward(self, x):                                     # x (B,T,12)
            h, _ = self.gru(x)
            return (self.head_next(h), self.head_ttnb(h).squeeze(-1),
                    self.head_twist(h))                            # (B,T,C), (B,T), (B,T,6)

    return _Antic()


def supervised_mask(session):
    """Frames with a defined upcoming boundary (valid and next_id is a real class)."""
    v = np.asarray(session["valid"], bool)
    nid = np.asarray(session["next_id"], np.int64)
    return v & (nid != NEXT_ID_NONE)


def next_segment_twist(session):
    """(N,6) per-frame target: the mean body twist of the UPCOMING segment (the next maneuver), so
    the anticipation head can predict the next move's direction and amplitude, not just its class.
    NaN where there is no next segment (masked out in training)."""
    tw = np.asarray(session["twist"], np.float64)
    fl = np.asarray(session["frame_labels"], np.int64)
    n = len(fl)
    out = np.full((n, TARGET_CHANNELS), np.nan)
    segs, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and fl[j + 1] == fl[i]:
            j += 1
        segs.append((i, j))
        i = j + 1
    means = [np.nanmean(tw[a:b + 1], axis=0) for a, b in segs]
    for k, (a, b) in enumerate(segs):
        if k + 1 < len(segs):
            out[a:b + 1] = means[k + 1]
    return out


class Anticipator:
    """Inference wrapper: load a trained GRU + norm, predict per-frame next maneuver and ttnb."""

    def __init__(self, ckpt_path, device=None):
        import torch
        from .render_lotus import get_device
        self.device = device or get_device()
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.norm = Norm12.from_dict(ck["norm"])
        self.model = build_model(hidden=ck.get("hidden", 64)).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def predict(self, twist, rel6):
        """Return (next_id, ttnb_s, next_twist) per frame for a full session sequence: the predicted
        next maneuver class, its time-to-boundary in seconds, and its body twist (from which
        guidance.move_from_twist reads the next move's direction and amplitude)."""
        import torch
        x = self.norm.apply(twist, rel6)[None]                   # (1,T,12)
        with torch.no_grad():
            logits, tt, ntw = self.model(torch.from_numpy(x).to(self.device))
        next_id = logits[0].argmax(-1).cpu().numpy()
        ttnb_s = np.clip(tt[0].cpu().numpy(), 0, None) * DT_C     # frames -> seconds (head is in frames)
        next_twist = ntw[0].cpu().numpy()
        return next_id, ttnb_s, next_twist

    def next_move(self, twist, rel6, frame, horizon_s=2.0):
        """Structured next move (guidance.Move) at `frame`: maneuver + direction + amplitude."""
        from .guidance import move_from_twist
        next_id, _tt, next_twist = self.predict(twist, rel6)
        man = SESSION_CLASSES_6[int(next_id[frame])]
        return move_from_twist(next_twist[frame], man, horizon_s)
