"""policy.py: a compact, MPS-trainable image-conditioned acquisition policy.

Maps a B-mode image (plus a short history and a maneuver/instruction id, the language channel
for a future VLA) to a 6-DOF twist action and a next-maneuver head. Deliberately small so it
trains on the M5 Pro. The dataset/API is kept LeRobot-compatible (observation.images.bmode +
observation.state + action) so Octo/OpenVLA can replace this policy later without touching the
generator or the env.

Twist targets are z-normalized per channel (channels range from ~0.01 m/s to ~0.4 rad/s), so
the model predicts normalized twist; TwistNorm.denormalize recovers SI units for env rollout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts_bridge import CLASSES_5, N_CLASSES, TARGET_CHANNELS


@dataclass
class TwistNorm:
    mean: np.ndarray      # (6,)
    std: np.ndarray       # (6,)

    @classmethod
    def fit(cls, clips: list[dict]) -> "TwistNorm":
        rows = []
        for c in clips:
            tw = np.asarray(c["twist"], dtype=np.float64)
            v = np.asarray(c["valid"], dtype=bool)
            rows.append(tw[v])
        X = np.concatenate(rows, axis=0)
        mean = X.mean(0)
        std = X.std(0)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def normalize(self, tw):
        return (np.asarray(tw, np.float64) - self.mean) / self.std

    def denormalize(self, z):
        return np.asarray(z, np.float64) * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]))


class BModeFrameDataset:
    """Frame-level (image, maneuver_id) -> twist samples, with a short frame history.

    history h stacks the current and previous h-1 frames as channels (padded with the first
    frame at a clip's start). Only frames with valid twist are emitted.
    """

    def __init__(self, clips: list[dict], twist_norm: TwistNorm, image_size: int = 128,
                 history: int = 1):
        self.tn = twist_norm
        self.image_size = image_size
        self.history = history
        self.index = []        # (clip_idx, frame_idx)
        self.clips = clips
        for ci, c in enumerate(clips):
            v = np.asarray(c["valid"], bool)
            for fi in np.nonzero(v)[0]:
                self.index.append((ci, int(fi)))

    def __len__(self):
        return len(self.index)

    def _stack(self, imgs, fi):
        h = self.history
        idxs = [max(0, fi - k) for k in range(h - 1, -1, -1)]      # oldest..current
        stack = np.stack([imgs[j] for j in idxs], axis=0).astype(np.float32) / 255.0
        return stack                                               # (h, H, W)

    def __getitem__(self, i):
        ci, fi = self.index[i]
        c = self.clips[ci]
        x = self._stack(c["images"], fi)
        man = int(c["label"])
        tw = self.tn.normalize(c["twist"][fi]).astype(np.float32)
        return x, man, tw

    def torch_loader(self, batch_size=64, shuffle=True):
        import torch
        from torch.utils.data import Dataset, DataLoader

        outer = self

        class _DS(Dataset):
            def __len__(self):
                return len(outer)

            def __getitem__(self, i):
                x, man, tw = outer[i]
                return (torch.from_numpy(x), torch.tensor(man, dtype=torch.long),
                        torch.from_numpy(tw))

        return DataLoader(_DS(), batch_size=batch_size, shuffle=shuffle)


def _conv_block(cin, cout):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BModePolicy:
    """Factory for the nn.Module (kept import-light so `import usbg.policy` needs no torch)."""

    @staticmethod
    def build(history: int = 1, n_maneuvers: int = len(CLASSES_5), emb_dim: int = 32,
              feat_dim: int = 128):
        import torch.nn as nn
        import torch

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    _conv_block(history, 32), _conv_block(32, 64),
                    _conv_block(64, 128), _conv_block(128, feat_dim),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten())
                self.man_emb = nn.Embedding(n_maneuvers, emb_dim)
                self.trunk = nn.Sequential(
                    nn.Linear(feat_dim + emb_dim, 128), nn.ReLU(inplace=True))
                self.head_twist = nn.Linear(128, TARGET_CHANNELS)
                self.head_next = nn.Linear(128, N_CLASSES)

            def forward(self, img, man):
                z = self.encoder(img)
                e = self.man_emb(man)
                h = self.trunk(torch.cat([z, e], dim=1))
                return {"twist": self.head_twist(h), "next_logits": self.head_next(h)}

        return _Net()


def _encoder(in_ch, feat_dim):
    import torch.nn as nn
    return nn.Sequential(_conv_block(in_ch, 32), _conv_block(32, 64), _conv_block(64, 128),
                         _conv_block(128, feat_dim), nn.AdaptiveAvgPool2d(1), nn.Flatten())


class GoalPolicy:
    """Goal-conditioned policy for reaching a target shot: (current B-mode history, target B-mode)
    -> 6-DoF twist. No pose input; the target is specified by its IMAGE, exactly what a real
    deployment sees. The current stack and the target are stacked as INPUT CHANNELS into one
    encoder, so the conv layers can compare them spatially and infer the displacement direction
    (two separately global-pooled encoders cannot represent where the target is offset)."""

    @staticmethod
    def build(history: int = 4, feat_dim: int = 128):
        import torch
        import torch.nn as nn

        class _Goal(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = _encoder(history + 1, feat_dim)     # current history + target channel
                self.trunk = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(inplace=True))
                self.head = nn.Linear(128, TARGET_CHANNELS)
                nn.init.zeros_(self.head.weight)      # start at zero twist (~stay put), then learn
                nn.init.zeros_(self.head.bias)        # corrections; avoids drifting off at init

            def forward(self, cur, tgt):
                x = torch.cat([cur, tgt], dim=1)              # (B, history+1, H, W)
                return self.head(self.trunk(self.enc(x)))

        return _Goal()


class GoalMDNPolicy:
    """Multimodal goal policy (#7, lightest form): same CNN backbone, but the head is a K-component
    Gaussian MIXTURE over the 6-DoF twist, trained by negative log-likelihood instead of MSE. An MSE
    (single-Gaussian) head regresses the CONDITIONAL MEAN of the corrective twists, which for
    multi-valued / sign-ambiguous targets is a small, averaged, damped action (the under-movement
    signature). A mixture can place a mode on the real large corrective instead of averaging it."""

    @staticmethod
    def build(history: int = 4, feat_dim: int = 128, n_mix: int = 5, max_twist=None):
        import numpy as np
        import torch
        import torch.nn as nn
        K, D = int(n_mix), TARGET_CHANNELS
        mt = np.array([0.03, 0.03, 0.03, 0.5, 0.5, 0.5]) if max_twist is None else np.asarray(max_twist)

        class _MDN(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = _encoder(history + 1, feat_dim)
                self.trunk = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(inplace=True))
                self.head_mu = nn.Linear(128, K * D)
                self.head_ls = nn.Linear(128, K * D)     # per-component log-std
                self.head_pi = nn.Linear(128, K)         # mixture logits
                self.register_buffer("max_twist", torch.as_tensor(mt, dtype=torch.float32))
                self.K, self.D = K, D
                nn.init.zeros_(self.head_mu.weight); nn.init.zeros_(self.head_mu.bias)

            def _params(self, cur, tgt):
                h = self.trunk(self.enc(torch.cat([cur, tgt], dim=1)))
                mu = self.max_twist * torch.tanh(self.head_mu(h).view(-1, self.K, self.D))
                ls = self.head_ls(h).view(-1, self.K, self.D).clamp(-6.0, 1.0)
                logpi = torch.log_softmax(self.head_pi(h), dim=-1)
                return mu, ls, logpi

            def _logprob(self, a, mu, ls, logpi):        # mixture log-prob of a (B,D)
                var = (2.0 * ls).exp()
                comp = (-0.5 * (((a.unsqueeze(1) - mu) ** 2) / var + 2.0 * ls + np.log(2 * np.pi))).sum(-1)
                return torch.logsumexp(logpi + comp, dim=-1)

            def mean(self, cur, tgt):                    # deterministic action = top-weight component mean
                mu, _ls, logpi = self._params(cur, tgt)
                k = logpi.argmax(-1)
                return mu[torch.arange(mu.shape[0], device=mu.device), k]

            def act(self, cur, tgt, deterministic=False):
                mu, ls, logpi = self._params(cur, tgt)
                idx = torch.arange(mu.shape[0], device=mu.device)
                if deterministic:
                    return mu[idx, logpi.argmax(-1)], None
                k = torch.distributions.Categorical(logits=logpi).sample()
                a = mu[idx, k] + ls[idx, k].exp() * torch.randn_like(mu[idx, k])
                a = torch.clamp(a, -self.max_twist, self.max_twist).detach()
                return a, self._logprob(a, mu, ls, logpi)

            def imitation_loss(self, cur, tgt, target):  # NLL of the target twist under the mixture
                mu, ls, logpi = self._params(cur, tgt)
                return -self._logprob(target, mu, ls, logpi).mean()

        return _MDN()
