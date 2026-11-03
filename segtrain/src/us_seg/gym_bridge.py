"""Locate and import the USB Gym repository (the `usbg` package).

This trainer renders its simulated training frames with `usbg`. It ships inside the USB Gym repository
(folder segtrain/). Resolution order for the gym repository root:
  1. env var `USBG_GYM_ROOT`
  2. the repository that contains this trainer (two folders above segtrain/src/us_seg/)
The gym's `src/` is put on `sys.path` so `import usbg` works.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

_DEFAULT_GYM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@lru_cache(maxsize=1)
def gym_root() -> str:
    root = os.environ.get("USBG_GYM_ROOT", _DEFAULT_GYM)
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(os.path.join(root, "src", "usbg")):
        raise FileNotFoundError(
            f"USB Gym repository not found at {root!r} (no src/usbg). "
            "Set USBG_GYM_ROOT to the gym repo root."
        )
    return root


def ensure_usbg_importable() -> str:
    """Put the gym's src/ on sys.path (idempotent). Returns the gym root."""
    root = gym_root()
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return root


def gym_path(*parts: str) -> str:
    """Absolute path inside the gym repo, e.g. gym_path('data', 'cache')."""
    return os.path.join(gym_root(), *parts)


# --- common gym asset locations ------------------------------------------------------------------

def cache_dir() -> str:
    """The gym's CT label-volume cache dir, ALWAYS <gym>/data/cache.

    Cases are loaded through usbg.volume.load_cache (see load_case), which under USBG_SEAT=anatomical
    transparently reads the bladder-free twin <gym>/data/cache_anat/<id>.npz (KIDNEY = kidneys only)
    and under USBG_SEAT=canonical reads data/cache itself (the v1 cache, bladder in KIDNEY). The
    seat switch therefore selects the cache; there is no separate cache env var. USBG_CACHE_DIR (an
    earlier draft of this contract) is refused so a stale export can not point a build elsewhere."""
    if os.environ.get("USBG_CACHE_DIR", "").strip():
        raise RuntimeError("USBG_CACHE_DIR is obsolete: the gym's volume.load_cache redirects data/cache "
                           "-> data/cache_anat under USBG_SEAT=anatomical. Unset it.")
    return gym_path("data", "cache")


def expected_cache_variant(seat: str | None = None) -> str:
    """The lv.meta['cache_variant'] volume.load_cache must report for the seat ('anat'|'canonical')."""
    return "anat" if (seat or seat_mode()) == "anatomical" else "canonical"


def load_case(path: str):
    """Load one cached case through the gym (usbg.volume.load_cache, so the seat's cache redirect
    applies) and verify the variant matches the seat: an anatomical build must read the bladder-free
    cache_anat twin, a canonical build the v1 cache. Returns the LabeledVolume."""
    ensure_usbg_importable()
    from usbg import volume as V
    lv = V.load_cache(path)
    want = expected_cache_variant()
    got = (lv.meta or {}).get("cache_variant")
    if got != want:
        raise RuntimeError(f"{path}: gym loaded cache_variant={got!r} ({(lv.meta or {}).get('cache_path')}), "
                           f"seat {seat_mode()!r} needs {want!r}")
    return lv


def benchmark_split() -> tuple[list[str], list[str]]:
    """(TRAIN_PATIENTS, TEST_PATIENTS) from the gym's frozen benchmark split (single source of truth)."""
    ensure_usbg_importable()
    from usbg import benchmark as B
    return list(B.TRAIN_PATIENTS), list(B.TEST_PATIENTS)


def cache_cases(cases: list[str] | None = None) -> list[str]:
    """Sorted cached CT volume .npz paths under cache_dir(). If `cases` (stems, e.g. "s0011") is given,
    return exactly those and fail loudly if any is missing (never silently drop a patient)."""
    d = cache_dir()
    allp = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz"))
    if cases is None:
        return allp
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in allp}
    missing = [c for c in cases if c not in by_stem]
    if missing:
        raise FileNotFoundError(f"cases {missing} not in cache dir {d!r}")
    return [by_stem[c] for c in sorted(cases)]


def seat_mode() -> str:
    """The seat the gym WILL use in seat_volume_under_probe, as reported by the gym itself.

    Contract with the gym: usbg.placement exposes seat_mode() -> "anatomical" | "canonical".
    If the loaded gym has no such accessor it has no seat switch, so whatever USBG_SEAT says is
    ignored; report that explicitly (never echo the env var), so --require-seat anatomical fails
    instead of passing on a gym checkout that still seats canonically."""
    ensure_usbg_importable()
    try:
        from usbg import placement as P
    except Exception as e:  # noqa: BLE001
        return f"<usbg.placement import failed: {e}>"
    fn = getattr(P, "seat_mode", None)
    if callable(fn):
        return str(fn())
    return f"<gym {gym_root()} has no placement.seat_mode(): canonical seat; USBG_SEAT={os.environ.get('USBG_SEAT')!r} ignored>"


def file_md5(path: str | None) -> str | None:
    import hashlib
    if not path or not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(root: str) -> dict:
    """{'rev': short hash, 'dirty': bool} for a repo root (provenance; None on failure)."""
    import subprocess
    try:
        rev = subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain",
                                              "--untracked-files=no"], text=True,
                                             stderr=subprocess.DEVNULL).strip())
        return {"rev": rev, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"rev": None, "dirty": None}


def seg_root() -> str:
    """This (segmenter) repo's root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def provenance(**extra) -> dict:
    """Shared provenance block for every JSON this repo writes (segmenter side).
    seat_mode is what the GYM reports (never the env echo), plus the windows file, gel coupling,
    USBG_LEARNED_SEG and both repos' git state (the gym's with its working-tree diff md5)."""
    ensure_usbg_importable()
    out = {
        "seat_mode": seat_mode(),
        "cache_dir": gym_path("data", "cache"),
        "cache_variant_expected": expected_cache_variant(),
        "usbg_learned_seg": os.environ.get("USBG_LEARNED_SEG"),
        "gym_git": git_rev(gym_root()),
        "seg_git": git_rev(seg_root()),
    }
    try:   # the gym's own provenance helpers (usbg.provenance); never import view_match here
        from usbg import provenance as GP
        out["seat"] = GP.seat_info()
        out["windows"] = GP.windows_ref()
        out["gym_git_full"] = GP.git_info(gym_root())
        out["seg_git_full"] = GP.git_info(seg_root())
    except Exception as e:  # noqa: BLE001  (an old gym without usbg.provenance)
        out["gym_provenance_error"] = repr(e)
    out.update(extra)
    return out


# translator checkpoints, keyed by their config aliases. LEGACY (v1, canonical
# seat): these exp_real_*_curvi.pt files were trained on pairs that include held-out patients (D3),
# so a v2 (anatomical) build refuses them; v2 takes explicit paths from the gym's texture manifest.
TRANSLATOR_CKPTS = {
    "sonospade": "data/dataset/exp_real_sonospade_curvi.pt",
    "cut": "data/dataset/exp_real_cut_curvi.pt",
    "cyclegan": "data/dataset/exp_real_cyclegan_curvi.pt",
    "cyclegan_sc": "data/dataset/exp_real_cyclegan_sc_curvi.pt",
}
LEGACY_TRANSLATOR_MD5 = {   # the D3-leaky v1 translators (gym data/dataset/exp_real_*_curvi.pt)
    "3edb0ee0ff32d5b72fb4e4a8c5bda693", "d2f6e37d1b5dbd53f3203e7d5241e518", "ea9e275f4a6974eb1c3a6279656d0253",
}
V2_TEXTURE_MANIFEST = "outputs/v2/textures/manifest.json"   # gym: {physics: null, cut, cyclegan, sonospade}
V2_STAGES = ("sonospade", "physics", "cut", "cyclegan")

REAL_TEST_NPZ = "data/eval/kaggle_vitale_test.npz"
REAL_POOL_DIR = "data/real/kaggle_vitale/pool"


def translator_ckpt(name: str) -> str | None:
    """Resolve a translator alias to an absolute checkpoint path. 'none' -> None."""
    if name in ("none", None):
        return None
    if name in TRANSLATOR_CKPTS:
        return gym_path(TRANSLATOR_CKPTS[name])
    # allow an explicit path (absolute, or relative to the gym repo)
    if os.path.isabs(name) and os.path.exists(name):
        return name
    cand = gym_path(name)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"unknown translator {name!r}; not an alias or existing checkpoint path")


def is_legacy_translator(name: str | None, path: str | None) -> str | None:
    """Reason string if a translator is one of the D3-leaky v1 checkpoints (alias, name or md5)."""
    if name in TRANSLATOR_CKPTS:
        return f"alias {name!r} -> legacy {TRANSLATOR_CKPTS[name]}"
    if path:
        b = os.path.basename(path)
        if b.startswith("exp_real_") and b.endswith("_curvi.pt"):
            return f"legacy checkpoint name {b}"
        m = file_md5(path)
        if m in LEGACY_TRANSLATOR_MD5:
            return f"md5 {m} of {path} is a legacy v1 translator"
    return None


def v2_texture_manifest(path: str | None = None) -> dict:
    """The gym's v2 texture manifest {stage: abs ckpt path | None}. Relative paths resolve against the
    gym root. Raises if it is missing or names a legacy translator."""
    p = path or gym_path(V2_TEXTURE_MANIFEST)
    import json
    with open(p) as f:
        m = json.load(f)
    out = {}
    for k, v in m.items():
        if k not in V2_STAGES:
            continue
        if v is None:
            out[k] = None
            continue
        a = v if os.path.isabs(v) else gym_path(v)
        why = is_legacy_translator(None, a)
        if why:
            raise ValueError(f"{p}: stage {k}: {why}")
        out[k] = a
    return out
