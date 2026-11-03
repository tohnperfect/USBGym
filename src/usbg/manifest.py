"""manifest.py: the data-provenance and normalization contract for SonoSPADE.

Every SonoSPADE stream (free aligned sim pairs, the unlabeled own real pool, and open labeled
sets for TSTR) records its counts, split policy, license, and the exact normalization params in
one data/manifest.json, so a run is reproducible and a reader can see where every frame came
from. This module is the single source of truth for that schema.

The normalization contract: every frame is float32 in [0,1], stored at the
render shape (N_SAMPLES, N_LINES) or a square resize of it, split by session (never by frame), and
the normalization params (percentile low/high) are recorded so evaluation is byte-reproducible.
"""
from __future__ import annotations

import json
import os

# The canonical normalization contract. A stream may override low_pct / high_pct but must keep the
# dtype, range, and split policy.
NORMALIZATION_CONTRACT = {
    "dtype": "float32",
    "range": [0.0, 1.0],
    "low_pct": 1.0,          # robust black level (percentile)
    "high_pct": 99.0,        # robust white level (percentile)
    "split_by": "session",   # never "frame" (avoids leakage across train/val)
}

# Required keys for each entry in manifest["streams"].
_STREAM_REQUIRED = ("name", "role", "count", "labeled", "license", "split_by")

# A stream's role in the pipeline. Kept small and explicit so eval code can select by role.
STREAM_ROLES = ("sim_pairs", "real_pool", "eval_labeled")


def default_manifest() -> dict:
    """An empty but schema-valid manifest, ready to have streams appended as data is prepared."""
    return {
        "schema_version": "sonospade-manifest-1.0",
        "normalization": dict(NORMALIZATION_CONTRACT),
        "shape": None,          # [N_SAMPLES, N_LINES] once known, else a square [S, S]
        "streams": [],          # list of stream dicts (see stream_entry)
        "notes": "Verify open-data licenses and own-data ethics before publication.",
    }


def stream_entry(name: str, role: str, count: int, labeled: bool, license: str,
                 split_by: str = "session", **extra) -> dict:
    """Build one stream record. `extra` carries free-form provenance (source, cases, url, ...)."""
    if role not in STREAM_ROLES:
        raise ValueError(f"unknown stream role {role!r} (expected one of {STREAM_ROLES})")
    entry = {"name": name, "role": role, "count": int(count), "labeled": bool(labeled),
             "license": str(license), "split_by": split_by}
    entry.update(extra)
    return entry


def validate_manifest(m: dict) -> list[str]:
    """Return a list of human-readable problems (empty list means the manifest is valid)."""
    problems: list[str] = []
    if not isinstance(m, dict):
        return ["manifest is not a dict"]
    if "schema_version" not in m:
        problems.append("missing schema_version")
    norm = m.get("normalization")
    if not isinstance(norm, dict):
        problems.append("missing or malformed normalization block")
    else:
        if norm.get("dtype") != "float32":
            problems.append("normalization.dtype must be 'float32'")
        if list(norm.get("range", [])) != [0.0, 1.0]:
            problems.append("normalization.range must be [0.0, 1.0]")
        if norm.get("split_by") != "session":
            problems.append("normalization.split_by must be 'session' (never 'frame')")
    streams = m.get("streams")
    if not isinstance(streams, list):
        problems.append("streams must be a list")
    else:
        for i, s in enumerate(streams):
            if not isinstance(s, dict):
                problems.append(f"stream {i} is not a dict")
                continue
            for k in _STREAM_REQUIRED:
                if k not in s:
                    problems.append(f"stream {i} ({s.get('name', '?')}) missing key {k!r}")
            if s.get("role") not in STREAM_ROLES:
                problems.append(f"stream {i} has unknown role {s.get('role')!r}")
            if s.get("split_by") == "frame":
                problems.append(f"stream {i} ({s.get('name', '?')}) splits by frame (leakage)")
    return problems


def write_manifest(path: str, m: dict) -> str:
    """Validate then write a manifest to disk (pretty JSON). Raises on an invalid manifest."""
    problems = validate_manifest(m)
    if problems:
        raise ValueError("invalid manifest:\n  " + "\n  ".join(problems))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(m, fh, indent=2)
    return path


def load_manifest(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


__all__ = ["NORMALIZATION_CONTRACT", "STREAM_ROLES", "default_manifest", "stream_entry",
           "validate_manifest", "write_manifest", "load_manifest"]
