# Vendored from the authors' probe-action recognizer repository, src/par/contracts.py at commit
# 6d5fe23 (file last changed in 73d3893); the usbg authors' own code, distributed under the usbg license.
# Kept verbatim: the channel / class / timing constants, GEOMETRY_SHA256, the se3 parity check, NormStats
# and class_names (everything usbg.contracts_bridge re-exports). Dropped because usbg never used them:
# the sibling-repo data paths (SIM_*, RIG_*), the real-CSV schema, load_meta, load_P / apply_P,
# load_norm_stats and the recognizer-only constants (NORM_POLICY, SWAP_ROCK_FAN, ANTICIPATION_TAUS_S,
# the F-score k list, ECE_*, KD_TEMP_RANGE, H_FRAMES). tests/test_vendor_regression.py compares every kept name with
# par.contracts when that package is installed.
"""contracts.py: the single source of truth for the cross-repo bridge.

Pins every interface value in one no-torch module so the loaders, model, training, and
eval all speak the same representation.

Hard rules encoded structurally here (the probe-action recognizer design):
  HR1  never freeze normalization on sim   -> NormStats.source
  HR5  tilt == fan; 5 train / 6 session     -> CLASSES_5 / SESSION_CLASSES_6, DOMINANT_AXIS['tilt']
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ======================================================================================
# Channels, classes, timing
# ======================================================================================
CHANNELS_TWIST = ["vx", "vy", "vz", "wx", "wy", "wz"]   # frozen 6-ch target order
CHANNELS_REL6 = ["x", "y", "z", "rx", "ry", "rz"]       # aux input (npz key `seqs`)
CHANNELS_INPUT = CHANNELS_TWIST + CHANNELS_REL6         # 12-ch model input order
IN_CHANNELS = 12                                        # z(twist 6) (+) z(rel6 6)
TARGET_CHANNELS = 6                                     # twist only

CLASSES_5 = ["slide", "rock", "tilt", "rotate", "compress"]   # tilt == fan; single-action npz
SESSION_CLASSES_6 = CLASSES_5 + ["hold"]                       # sessions / recognizer output
N_CLASSES = 6                                                  # model is ALWAYS 6-way
HOLD_IDX = 5

DT_C = 0.1                                              # 10 Hz canonical step (seconds)
OUT_RATE_HZ = 10.0

# dominant_axis keyed by literal 'tilt' (NOT 'fan'); matches the on-disk meta exactly (HR5)
DOMINANT_AXIS = {"slide": "vx", "rock": "wy", "tilt": "wx", "rotate": "wz", "compress": "vz"}

# sha256 of the vendored se3.py; equals the sim data's meta.geometry_hash (the sim vendored
# probe_pose/geometry.py verbatim and stored its file hash). Guards copy-drift AND provenance.
GEOMETRY_SHA256 = "bc784f6ec5edeaaae5a480ca4518f9587bebab4a36edda21dd274bb43cb7cb72"

CLIP_SIGMA = 5.0                                        # clamp normalized inputs to +/- this

# Anticipation / eval constants
TTNB_UNITS = "frames"                                   # multiply by DT_C for seconds
NEXT_ID_NONE = -1                                       # next_maneuver_id sentinel (no next)

# Teacher / Phase-2 export (heads exist from day one; export itself is deferred)
LATENT_D = 64                                           # penultimate latent dim (in [64,128])


def _expand(p: str) -> str:
    return os.path.expanduser(str(p))


# ======================================================================================
# se3 parity / geometry hash
# ======================================================================================
def se3_source_sha256() -> str:
    """sha256 of the vendored se3.py on disk."""
    path = Path(__file__).with_name("se3.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_geometry_parity() -> None:
    """Guard against se3 copy-drift: sha256(se3.py) must equal GEOMETRY_SHA256."""
    got = se3_source_sha256()
    if got != GEOMETRY_SHA256:
        raise AssertionError(
            f"se3.py sha256 {got} != GEOMETRY_SHA256 {GEOMETRY_SHA256}; vendored copy drifted."
        )


# ======================================================================================
# Normalization stats (provisional sim now; frozen real median/MAD later)
# ======================================================================================
@dataclass
class NormStats:
    """Per-channel normalization over the 12 input channels [twist(6) (+) rel6(6)].

    Tolerates BOTH conventions so the frozen real file (robust median/MAD) loads cleanly:
      method == 'zscore' -> center=mean,   scale=std
      method == 'mad'    -> center=median, scale=1.4826*MAD  (robust z-score)
    """
    center: np.ndarray                 # (12,)
    scale: np.ndarray                  # (12,)  (already includes the 1.4826 factor for MAD)
    method: str = "zscore"             # 'zscore' | 'mad'
    source: str = "sim_provisional"    # provenance; the rig freezes source='real_frozen'
    clip_sigma: float = CLIP_SIGMA
    channels: list = field(default_factory=lambda: list(CHANNELS_INPUT))

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(IN_CHANNELS)
        self.scale = np.asarray(self.scale, dtype=np.float64).reshape(IN_CHANNELS)
        # guard against divide-by-zero on a dead channel
        self.scale = np.where(np.abs(self.scale) < 1e-8, 1.0, self.scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """z-normalize x (.., 12); NaN passes through as NaN (the loader masks/imputes it)."""
        return (np.asarray(x, dtype=np.float64) - self.center) / self.scale

    def invert(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.scale + self.center

    def to_json(self, path: str | None = None) -> dict:
        d = {
            "method": self.method,
            "source": self.source,
            "clip_sigma": float(self.clip_sigma),
            "channels": list(self.channels),
        }
        if self.method == "mad":
            d["median"] = self.center.tolist()
            d["mad"] = (self.scale / 1.4826).tolist()
        else:
            d["mean"] = self.center.tolist()
            d["std"] = self.scale.tolist()
        if path is not None:
            with open(_expand(path), "w") as f:
                json.dump(d, f, indent=2)
        return d

    @classmethod
    def from_json(cls, src) -> "NormStats":
        """Parse a stats dict/path tolerant of mean/std OR median/MAD (the real file)."""
        if isinstance(src, (str, os.PathLike)):
            with open(_expand(src)) as f:
                d = json.load(f)
        else:
            d = dict(src)
        method = d.get("method")
        if "median" in d and "mad" in d:
            center = np.asarray(d["median"], dtype=np.float64)
            scale = np.asarray(d["mad"], dtype=np.float64) * 1.4826
            method = method or "mad"
        elif "mean" in d and "std" in d:
            center = np.asarray(d["mean"], dtype=np.float64)
            scale = np.asarray(d["std"], dtype=np.float64)
            method = method or "zscore"
        elif "center" in d and "scale" in d:
            center = np.asarray(d["center"], dtype=np.float64)
            scale = np.asarray(d["scale"], dtype=np.float64)
            method = method or "zscore"
        else:
            raise ValueError(f"norm stats dict has no recognized keys: {list(d)}")
        return cls(
            center=center, scale=scale, method=method,
            source=d.get("source", "unknown"),
            clip_sigma=float(d.get("clip_sigma", CLIP_SIGMA)),
            channels=d.get("channels", list(CHANNELS_INPUT)),
        )


# ======================================================================================
# Convenience
# ======================================================================================
def class_names(domain: str = "session") -> list:
    """'clip'/'train' -> the 5 single-action classes; 'session'/'recognizer' -> 6 (incl hold)."""
    if domain in ("clip", "train", "single"):
        return list(CLASSES_5)
    return list(SESSION_CLASSES_6)
