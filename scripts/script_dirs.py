#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where the code tree, the analysis scripts, the paper build and the result files are, without assuming the name of
any folder under scripts/.

The table and figure scripts run in one of two modes:

  development (the default): the result files live in the repository's own outputs/ and data/ folders and in the
      sibling segmenter repository, and the scripts may use the inputs that only the development repository holds
      (plan_private.py);
  public (--public --results ROOT): the result files live under ROOT, laid out as ROOT/outputs/..., ROOT/data/...
      and ROOT/segmenter/outputs/... (the segmenter repository's outputs), and the plan inputs come from the
      committed public_inputs/plan_constants.json (plan_inputs.py). Nothing outside the code tree and ROOT is read.

apply_cli(argv) switches a process to public mode before any module computes its data paths; it sets the
environment variables USBG_PUBLIC=1 and USBG_RESULTS_ROOT, which child processes inherit. Only --public on the command
line switches a script to public mode: without it, apply_cli clears those variables, so an inherited environment never
does. The public mirror holds the marker file scripts/PUBLIC_MIRROR (mirror()); there every script runs in public mode
only, and the development backend (plan_private.py) is never imported.
"""
from __future__ import annotations

import glob
import os
import re

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(SCRIPTS)
PUBLIC_ENV = "USBG_PUBLIC"
RESULTS_ENV = "USBG_RESULTS_ROOT"
OUT_ENV = "USBG_PUBLIC_OUT"
# the marker file that only the public mirror holds (published from public_inputs/ by the export list)
MIRROR_MARKER = os.path.join(SCRIPTS, "PUBLIC_MIRROR")
# environment variables that a public run keeps; every other USBG_* and US_SEG_* variable is dropped, since it would
# change which files the build reads or how the testbed renders (USBG_WINDOWS is set to the result root's copy)
KEEP_ENV = (PUBLIC_ENV, RESULTS_ENV, OUT_ENV, "USBG_RECORD_PLAN_CALLS")
# how a figure data file of the public build gives the md5 of a result file: the deposit's MANIFEST.tsv lists its
# sha256 (a deposit may hold other bytes than the development files whose md5 the development build writes)
DIGEST_NOTE = "see MANIFEST.tsv"
# ... and of an input that the deposit leaves out: its EXCLUDED.tsv row gives the sha256 a placed file is checked against
DIGEST_NOTE_EXCLUDED = "see EXCLUDED.tsv"


class DirsError(RuntimeError):
    """A folder or input of the build could not be located."""


def code_root() -> str:
    return CODE_ROOT


def _single(name: str) -> str:
    hits = sorted(glob.glob(os.path.join(SCRIPTS, "*", name)))
    if len(hits) != 1:
        raise DirsError(f"expected exactly one scripts/*/{name}, found {len(hits)}")
    return os.path.dirname(hits[0])


def analysis_dir() -> str:
    """The folder that holds plan_stats.py (the analysis scripts)."""
    return _single("plan_stats.py")


def paper_dir() -> str:
    """The folder that holds make_tables.py (the paper's table and figure build)."""
    return _single("make_tables.py")


def public() -> bool:
    return os.environ.get(PUBLIC_ENV) == "1"


def results_root() -> str:
    """The root of the result files: ROOT of --results in public mode, the code tree otherwise."""
    if public():
        r = os.environ.get(RESULTS_ENV)
        if not r:
            raise DirsError(f"public mode needs {RESULTS_ENV} (the --results folder)")
        return os.path.abspath(r)
    return CODE_ROOT


def mirror() -> bool:
    """True in the public mirror (the marker file scripts/PUBLIC_MIRROR exists): only public mode runs there."""
    return os.path.isfile(MIRROR_MARKER)


def private_module():
    """The development backend plan_private.py, or None in public mode and in the public mirror. In the development
    repository outside public mode the import is not guarded: a missing or broken plan_private.py stops the run."""
    if public() or mirror():
        return None
    import plan_private  # development repository only
    return plan_private


def segmenter_root(required: bool = True):
    """The segmenter repository whose outputs/ the build reads: ROOT/segmenter in public mode, the sibling repository
    of the development checkout otherwise. In the public mirror outside public mode only --help runs: the placeholder
    ROOT/segmenter then names it in the printed defaults (None for an optional caller)."""
    if public():
        return os.path.join(results_root(), "segmenter")
    pp = private_module()
    if pp is None:
        if not required:
            return None
        if _help_only():
            return os.path.join("ROOT", "segmenter")
        raise DirsError("the segmenter outputs are read in public mode only (--public --results ROOT)")
    return pp.SEGMENTER_REPO


def segmenter_repo_name():
    """The folder name of the segmenter repository, which the absolute paths recorded in the development result files
    name (from the development backend), or None in public mode and in the public mirror: there the recorded paths
    are relative to the result root (segmenter/...) and no folder name of the development machine is needed."""
    pp = private_module()
    return os.path.basename(os.path.normpath(pp.SEGMENTER_REPO)) if pp is not None else None


def _help_only() -> bool:
    import sys
    return any(a in ("-h", "--help") for a in sys.argv[1:])


def _has_opt(argv, name):
    return any(a == name or a.startswith(name + "=") for a in argv)


def apply_cli(argv) -> bool:
    """Public mode from the command line: --public with --results ROOT. Returns True in public mode. Call it before
    importing any module that computes data paths at import time. Without --public it clears the public-mode
    variables of an inherited environment, refuses --results (a public-mode option), and in the public mirror refuses
    to run (only --help is answered there)."""
    import sys
    argv = list(argv)
    if "--public" not in argv:
        for k in (PUBLIC_ENV, RESULTS_ENV, OUT_ENV):
            os.environ.pop(k, None)
        if _has_opt(argv, "--results"):
            raise SystemExit("--results ROOT is an option of public mode: add --public (and --out DIR)")
        if mirror() and not any(a in ("-h", "--help") for a in argv):
            raise SystemExit("this is the public mirror of the code: run with --public --results ROOT --out DIR "
                             "(ROOT: the unpacked results archive; see the README next to reproduce.py)")
        return False
    root = out = None
    for i, a in enumerate(argv):
        if a == "--results" and i + 1 < len(argv):
            root = argv[i + 1]
        elif a.startswith("--results="):
            root = a.split("=", 1)[1]
        elif a == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
        elif a.startswith("--out="):
            out = a.split("=", 1)[1]
    if not root:
        raise SystemExit("--public needs --results ROOT (the folder that holds outputs/, data/ and segmenter/)")
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise SystemExit(f"--results {root}: not a folder")
    if out and inside(out, root):
        raise SystemExit(f"--out {os.path.abspath(out)} lies inside --results {root}: the build writes nothing under "
                         "ROOT (only the files MANIFEST.tsv lists exist there for the build), so DIR must be a folder "
                         "outside ROOT")
    dropped = sorted(k for k in os.environ if k.startswith(("USBG_", "US_SEG_")) and k not in KEEP_ENV)
    for k in dropped:
        del os.environ[k]
    if dropped:
        print("[public] ignored inherited environment variables: " + ", ".join(dropped), file=sys.stderr)
    os.environ[PUBLIC_ENV] = "1"
    os.environ[RESULTS_ENV] = root
    if out:
        os.environ[OUT_ENV] = os.path.abspath(out)
    else:
        os.environ.pop(OUT_ENV, None)
    # the screened acoustic windows of the result root (usbg.windows reads this variable at import)
    os.environ["USBG_WINDOWS"] = os.path.join(root, "data", "cache", "windows", "usbg-windows-v1.json")
    return True


def public_out() -> str:
    """The output root DIR of a public run (--out DIR): tables in DIR/tables, figures in DIR/figures."""
    d = os.environ.get(OUT_ENV)
    if not d:
        raise DirsError("public mode needs --out DIR")
    return d


def inside(p, root) -> bool:
    """True when the path p is root or lies below it, as written or with links resolved."""
    for a, r in ((os.path.abspath(p), os.path.abspath(root)), (os.path.realpath(p), os.path.realpath(root))):
        if a == r or a.startswith(r if r.endswith(os.sep) else r + os.sep):
            return True
    return False


def guard_forms(path):
    """The two forms of a path that the build's refusal checks match words against: the absolute and the resolved
    path. In public mode the part of either form below the result root replaces it (with a leading separator, so a
    pattern such as '/outputs/v2_1/...' still matches): the location of ROOT itself never matches a refused word.
    Development mode: the two forms unchanged."""
    a, r = os.path.abspath(str(path)), os.path.realpath(str(path))
    if not public():
        return a, r
    root = results_root()
    bases = sorted({os.path.abspath(root).rstrip(os.sep), os.path.realpath(root).rstrip(os.sep)}, key=len,
                   reverse=True)

    def below(s):
        for b in bases:
            if b and (s == b or s.startswith(b + os.sep)):
                return s[len(b):] or os.sep
        return s
    return below(a), below(r)


def _slash(s):
    """s with the platform's path separator written as '/' (unchanged where the separator is '/')."""
    return s.replace(os.sep, "/") if os.sep != "/" else s


def _is_abs(s):
    return s.startswith("/") or os.path.isabs(s)


def _join(root, rel):
    """root joined with a '/'-separated relative path, in the platform's separators."""
    return os.path.join(root, rel) if os.sep == "/" else os.path.join(root, *rel.split("/"))


def _excluded_inputs():
    """Public mode: the inputs that ROOT/EXCLUDED.tsv lists (left out, or placed under the root and checked against
    their row), as paths relative to the result root; empty otherwise."""
    if not public():
        return frozenset()
    import results_world
    W = results_world.world()
    return frozenset(W.left_out.values()) | frozenset(W.supplied) if W is not None else frozenset()


def public_json_view(obj, text_map=None, figures_prefix=None, excluded=None, key_map=None):
    """A figure data payload as the public build writes it: the keys of every object in sorted order, every absolute
    path relative to the result root's layout (outputs/..., data/..., segmenter/outputs/...), '<figures_prefix><x>'
    (the development build's figures folder) as 'figures/<x>', the strings of text_map and the keys of key_map
    replaced. The expected
    figure data of public_inputs/expected/ are the development build's files passed through the same function. The
    md5 of a file (a key 'md5' or '..._md5' holding 32 hex digits) is written as DIGEST_NOTE, or as
    DIGEST_NOTE_EXCLUDED when a string of the same object names an input that the results archive leaves out
    (excluded: paths relative to the result root; by default the inputs that ROOT/EXCLUDED.tsv lists, in public mode).
    In public mode an absolute path that maps to none of these stops the run (DirsError), so no machine path reaches a
    written file."""
    if excluded is None:
        excluded = _excluded_inputs()
    root = results_root() if public() else None
    name = segmenter_repo_name()
    seg = "/" + name + "/outputs/" if name else None

    def one(s):
        if text_map and s in text_map:
            return text_map[s]
        if figures_prefix and s.startswith(figures_prefix):
            return "figures/" + s[len(figures_prefix):]
        if not _is_abs(s):
            return s
        if root and (_slash(s) == _slash(root) or _slash(s).startswith(_slash(root).rstrip("/") + "/")):
            return _slash(os.path.relpath(s, root))
        t = _slash(s)
        i = t.find(seg) if seg else -1
        if i >= 0:
            return "segmenter/outputs/" + t[i + len(seg):]
        for part in ("outputs", "data"):
            i = t.find("/" + part + "/")
            if i >= 0:
                return t[i + 1:]
        if public():
            raise DirsError(f"an absolute path that maps to no result-root folder would be written: {s[:80]}")
        return s

    def walk(o, key=None, left_out=False):
        if isinstance(o, dict):
            here = bool(excluded) and any(isinstance(v, str) and one(v) in excluded for v in o.values())
            km = key_map or {}
            return {km.get(k, k): walk(v, k, here) for k, v in sorted(o.items(), key=lambda kv: str(km.get(kv[0], kv[0])))}
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        if isinstance(o, str) and isinstance(key, str) and (key == "md5" or key.endswith("_md5")) \
                and re.fullmatch(r"[0-9a-f]{32}", o):
            return DIGEST_NOTE_EXCLUDED if left_out else DIGEST_NOTE
        return one(o) if isinstance(o, str) else o
    return walk(obj)


def map_recorded(p):
    """A path recorded inside a result file, mapped to the result root in public mode. The development files record
    absolute paths of the machine that wrote them: .../<segmenter repo>/outputs/X -> ROOT/segmenter/outputs/X, any
    other .../outputs/X or .../data/X -> ROOT/outputs/X or ROOT/data/X. The results deposit records them relative to
    the repository roots: segmenter/X -> ROOT/segmenter/X, outputs/X and data/X -> ROOT/outputs/X and ROOT/data/X.
    Unchanged in development mode and when nothing matches."""
    if not public() or p is None:
        return p
    s = str(p).replace("\\", "/")
    root = results_root()
    r = _slash(root)
    if s == r or s.startswith(r.rstrip("/") + "/"):
        return p
    if s.startswith("segmenter/"):
        return _join(root, s)
    name = segmenter_repo_name()
    i = s.find("/" + name + "/outputs/") if name else -1
    if i >= 0:
        seg = "/" + name + "/outputs/"
        return _join(os.path.join(root, "segmenter", "outputs"), s[i + len(seg):])
    for part in ("outputs", "data"):
        if s.startswith(part + "/"):
            return _join(root, s)
        i = s.find("/" + part + "/")
        if i >= 0:
            return _join(root, s[i + 1:])
    return p


def display(p) -> str:
    """A result path as the public build prints it: relative to the result root (segmenter files as segmenter/...),
    with '/' as the separator."""
    s = os.path.abspath(str(p))
    root = results_root()
    if s == root or s.startswith(root + os.sep):
        return _slash(os.path.relpath(s, root))
    if s == CODE_ROOT or s.startswith(CODE_ROOT + os.sep):
        return _slash(os.path.relpath(s, CODE_ROOT))
    return os.path.basename(s)
