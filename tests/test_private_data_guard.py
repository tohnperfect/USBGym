"""D5 guard: no code path reads a private clinical pool at import time, at run time or through a default
argument; retired private-derived checkpoint NAMES are not referenced by live code; the benchmark
reference pipeline is physics-only (tests/test_benchmark.py). Runs in the fast suite without any data.

The private pool and the private-derived artifacts stay where the user keeps them on disk: only code
references are forbidden. So the on-disk check of known private descendants is REPORT-ONLY (it skips
with the list); it never fails and never moves anything.
"""
from __future__ import annotations

import ast
import importlib
import os
import pkgutil
import re
import subprocess

import pytest

from usbg import provenance as P

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PRIVATE_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(n) for n in P.PRIVATE_DIRNAMES)
                         + r")(?![A-Za-z0-9_])")
_RETIRED_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(n) for n in P.RETIRED_NAMES) + r")\b")

# Files that legitimately NAME the private pools / retired checkpoints: the guard and its inventory.
GUARD_FILES = {
    "src/usbg/provenance.py",          # PRIVATE_DIRNAMES / PRIVATE_DESCENDANT_MD5 / RETIRED_NAMES
    "tests/conftest.py",
    "tests/test_private_data_guard.py",
    "tests/test_heldout_leakage.py",
    "scripts/check_leakage.py",
}
# LEGACY allowlist: canonical-seat, as-published research records that carry a LEGACY header and are never
# used for a v2 number. None of them is part of this repository, so both sets are empty here.
LEGACY_FILES = set()
LEGACY_DIRS = set()
LEGACY_MARKER = "LEGACY (canonical seat, as-published record)"


def _rel(p):
    return os.path.relpath(p, _ROOT).replace(os.sep, "/")


def _allowed(rel):
    return rel in GUARD_FILES or rel in LEGACY_FILES or any(rel.startswith(d + "/") for d in LEGACY_DIRS)


def _files(subdirs, exts):
    for sd in subdirs:
        for root, dirs, files in os.walk(os.path.join(_ROOT, sd)):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", "third_party")]
            for f in sorted(files):
                p = os.path.join(root, f)
                if f.endswith(exts) and not _allowed(_rel(p)):
                    yield p


def _docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = getattr(node, "body", None)
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                ids.add(id(b[0].value))
    return ids


def _armed():
    return os.environ.get("USBG_ALLOW_PRIVATE") != "1"


# ------------------------------------------------------------------------------------- the hook
def test_private_path_classifier():
    for p in ("data/real_us_crop", "data/real_us_crop/00001.png", "data/real_us/site_a/a.jpg",
              "/tmp/x/data/_private_derived/dataset/texture_cut.pt", "~/copy/real_us_crop/f.png"):
        assert P.is_private_path(p), p
    for p in ("data/real/kaggle_vitale/pool", "data/real/kaggle_vitale/pool/0001.png",
              "scripts/prepare_real_us.py", "data/real_us_cropped_public", "data/dataset/v2/cut_curvi.pt"):
        assert not P.is_private_path(p), p


def test_retired_name_matcher():
    for s in ("data/dataset/texture_cut.pt", "embed:data/dataset/bmode_embed.pt", "exp_real_cyclegan_sc.pt",
              "data/dataset/v2/texture_sonospade.pt"):                # a retired NAME is never reused
        assert _RETIRED_RE.search(s), s
    for s in ("data/dataset/v2/cut_curvi.pt", "data/dataset/exp_real_sonospade.pt",
              "data/dataset/v2/bmode_embed_v2.pt", "exp_real_cut_curvi.pt", "my_texture_cut.pt"):
        assert not _RETIRED_RE.search(s), s


@pytest.mark.skipif(not _armed(), reason="USBG_ALLOW_PRIVATE=1: the guard is deliberately unarmed")
def test_hook_blocks_private_reads(tmp_path):
    """Self-test: the session audit hook (tests/conftest.py) is armed and fires on a private path, even
    one outside this repo (a copied pool is still private) and even through a symlink."""
    import glob
    d = tmp_path / "data" / "real_us_crop"
    q = tmp_path / "data" / "_private_derived"
    link = tmp_path / "innocent_pool"
    d.mkdir(parents=True)                                  # creating a directory is not a read
    q.mkdir()
    try:
        with pytest.raises(PermissionError):
            os.listdir(d)
        with pytest.raises(PermissionError):
            open(d / "00000.png", "wb")
        with pytest.raises(PermissionError):
            glob.glob(str(d / "*.png"))

        def _reraise(e):
            raise e
        with pytest.raises(PermissionError):
            list(os.walk(d, onerror=_reraise))             # os.walk swallows errors by default
        cwd = os.getcwd()
        with pytest.raises(PermissionError):
            os.chdir(d)
        assert os.getcwd() == cwd
        with pytest.raises(PermissionError):
            os.listdir(q)
        # a link made outside this process (subprocesses do not inherit the hook) still resolves as private
        subprocess.run(["ln", "-s", str(d), str(link)], check=True)
        with pytest.raises(PermissionError):
            os.listdir(link)
        os.listdir(tmp_path)                               # a public path still works
        assert os.path.isdir(d)                            # stat / existence checks are not reads
    finally:
        # remove the private-named entries here: pytest's own tmp cleanup lists directories, which the
        # armed hook (correctly) blocks for these names. unlink / rmdir of an empty dir is not a read.
        if os.path.islink(link):
            os.unlink(link)
        for x in (d, q):
            if os.path.isdir(x):
                os.rmdir(x)


# ------------------------------------------------------------------------------ static references
def test_no_private_or_retired_literals_in_python():
    """No non-docstring string constant in src/, scripts/, tests/ or examples/ names a private pool or
    a retired private-derived checkpoint (argparse defaults, module constants, env-var fallbacks).
    Docstrings are exempt (usage notes are documentation, not code paths)."""
    bad = []
    for p in _files(("src", "scripts", "tests", "examples"), (".py",)):
        with open(p) as fh:
            tree = ast.parse(fh.read(), p)
        docs = _docstring_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                m = _PRIVATE_RE.search(node.value) or _RETIRED_RE.search(node.value)
                if m:
                    bad.append(f"{_rel(p)}:{node.lineno}: {m.group(0)!r} in {node.value!r}")
    assert not bad, "private / retired data referenced in code:\n  " + "\n  ".join(bad)


def test_no_private_or_retired_paths_in_shell_and_configs():
    bad = []
    for p in _files(("scripts", "configs", "examples"), (".sh", ".yaml", ".yml", ".json", ".toml", ".cfg")):
        with open(p, errors="ignore") as fh:
            for i, line in enumerate(fh, 1):
                code = line.split("#", 1)[0]
                m = _PRIVATE_RE.search(code) or _RETIRED_RE.search(code)
                if m:
                    bad.append(f"{_rel(p)}:{i}: {line.strip()}")
    assert not bad, "private / retired data referenced in shell/config:\n  " + "\n  ".join(bad)


def test_legacy_allowlist_is_marked():
    """Every LEGACY file carries the header that says it is a canonical-seat, as-published record."""
    missing = []
    paths = [os.path.join(_ROOT, f) for f in sorted(LEGACY_FILES)]
    for d in sorted(LEGACY_DIRS):
        for root, _dirs, files in os.walk(os.path.join(_ROOT, d)):
            paths += [os.path.join(root, f) for f in sorted(files) if f.endswith((".py", ".sh"))]
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, errors="ignore") as fh:
            head = fh.read(4000)
        if LEGACY_MARKER not in head:
            missing.append(_rel(p))
    assert not missing, f"LEGACY files without the {LEGACY_MARKER!r} header:\n  " + "\n  ".join(missing)


def test_import_all_usbg_modules_reads_nothing_private():
    """Import-time reads: every usbg submodule imports under the armed audit hook."""
    import usbg
    for m in pkgutil.iter_modules(usbg.__path__):
        importlib.import_module(f"usbg.{m.name}")


# ------------------------------------------------------------------------ on disk (REPORT ONLY)
def test_private_descendants_report_only():
    """REPORT ONLY. Lists known private-pool descendants (by md5) still live under data/dataset. They
    stay where the user keeps them (no move, no delete); the guarantee is that no code references
    them (the two static tests above) and no test reads the pool (the audit hook)."""
    live = []
    for md5, (orig, why) in sorted(P.PRIVATE_DESCENDANT_MD5.items(), key=lambda kv: kv[1][0]):
        p = os.path.join(_ROOT, orig)
        if os.path.isfile(p) and P.md5_file(p) == md5:
            live.append(f"{orig} ({why})")
    for d in P.PRIVATE_DESCENDANT_DIRS:
        p = os.path.join(_ROOT, d)
        if os.path.isdir(p) and not os.path.exists(P.sidecar_path(p)):
            live.append(f"{d}/ (whole directory is a descendant)")
    if live:
        pytest.skip(f"REPORT ONLY: {len(live)} known private-pool descendants are still on disk "
                    "(allowed; code must not reference them):\n  " + "\n  ".join(live))
