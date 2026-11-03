#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A closed world over the result root of a public build: only the files ROOT/MANIFEST.tsv lists exist under ROOT, each
is checked against its sha256 the first time it is opened or probed, and the withheld files (plan_inputs.withheld())
report as existing without being read. The inputs of the build that the results archive leaves out (third-party
terms; the rows of ROOT/EXCLUDED.tsv with a sha256) do not exist for the build unless a file is placed at its path
under ROOT; a placed file is checked against its EXCLUDED.tsv row (bytes and sha256) like a listed file. A step
whose output needs an absent one builds everything else, names the input in one notice and records the output in
DIR/NEEDS.tsv (needs()), which reproduce.py reports.

install(root) wraps os.stat, os.lstat, os.access, os.listdir, os.scandir and the predicates os.path.exists, lexists,
isfile, isdir and islink (hence getsize, glob and pathlib too, on Windows as well) for paths under root, and adds an
audit hook on open that refuses an unlisted file and checks a listed file's bytes. Nothing changes outside root.
Public mode needs ROOT/MANIFEST.tsv and ROOT/EXCLUDED.tsv: without either, or when a file MANIFEST.tsv lists (other
than a withheld one) is missing or has another size, install() stops the run and names the files.
MANIFEST.tsv is the byte authority of a public build: the
digests that the development build pins in its code (the records ledger, the pinned late records and the Version 4
combination result) name the development files' bytes and are replaced by the MANIFEST.tsv rows (listed();
plan_inputs.manifest_instead), while the digests that one result file records of another cite the archived file and
are checked as the development build checks them.

MANIFEST.tsv: lines '<path>\\t<bytes>\\t<sha256>' with path relative to ROOT ('/'-separated); lines starting with '#'
and the header line 'path\\t...' are skipped; further columns are ignored. For a result root of one's own:

  python results_world.py manifest ROOT > ROOT/MANIFEST.tsv
  python results_world.py excluded > ROOT/EXCLUDED.tsv      (its header line: nothing left out)

EXCLUDED.tsv (required in public mode): a header line, then '<path>\\t<bytes>\\t<sha256>\\t...' per left-out input;
rows whose sha256 is not 64 hex digits (files the archive withholds and the build never opens) are skipped. Without
it, an input that the archive leaves out would read as a missing file, not as an item the build cannot make.
"""
from __future__ import annotations

import hashlib
import os
import stat as _st
import sys
import threading

_STATE = {}
_LOCAL = threading.local()
EXCLUDED_NAME = "EXCLUDED.tsv"
EXCLUDED_HEADER = "path\tbytes\tsha256\treason\thow_to_obtain_or_regenerate\n"
NEEDS_NAME = "NEEDS.tsv"      # in DIR: '<output>\t<input>' per output a step did not build (reproduce.py reads it)


class ResultMismatch(RuntimeError):
    """A result file differs from its MANIFEST.tsv line."""


def load_manifest(path):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#") or ln.startswith("path\t"):
                continue
            f = ln.split("\t")
            if len(f) < 3:
                raise ResultMismatch(f"{path}: line is not '<path>\\t<bytes>\\t<sha256>': {ln[:80]!r}")
            rows[f[0]] = (int(f[1]), f[2])
    return rows


def load_excluded(path):
    """{path: (bytes, sha256)} of the rows of an EXCLUDED.tsv that give a sha256 (64 hex digits)."""
    import re
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if ln.startswith("#") or f[0] == "path" or len(f) < 3:
                continue
            if re.fullmatch(r"[0-9a-f]{64}", f[2]) and f[1].isdigit():
                rows[f[0]] = (int(f[1]), f[2])
    return rows


def _busy():
    return getattr(_LOCAL, "busy", False)


class _World:
    def __init__(self, root, rows, withheld, supplied=(), left_out=()):
        self.supplied = set(supplied)     # EXCLUDED.tsv inputs placed under ROOT (checked against EXCLUDED.tsv)
        self.left_out = {os.path.join(os.path.abspath(root), *r.split("/")): r for r in left_out}
        self.roots = tuple(sorted({os.path.abspath(root), os.path.realpath(root)}))
        self.root = os.path.abspath(root)
        self.rows = rows
        self.files = {os.path.join(self.root, *r.split("/")): r for r in rows}
        self.withheld = {os.path.join(self.root, *r.split("/")) for r in withheld}
        self.dirs = {self.root}
        for p in list(self.files) + list(self.withheld):
            d = os.path.dirname(p)
            while d.startswith(self.root) and d not in self.dirs:
                self.dirs.add(d)
                d = os.path.dirname(d)
        self.checked = {}
        self.counts = {"checked": 0, "placed": 0, "hidden": 0, "withheld": 0, "left_out": 0}

    def norm(self, p):
        """The path under the result root in its canonical form, or None for a path outside it."""
        if p is None or isinstance(p, int):
            return None
        try:
            s = os.path.abspath(os.fsdecode(os.fspath(p)))
        except TypeError:
            return None
        for r in self.roots:
            if s == r:
                return self.root
            if s.startswith(r + os.sep):
                return os.path.join(self.root, s[len(r) + 1:])
        return None

    def check(self, p):
        """Verify a listed file's size and sha256 once."""
        if p in self.checked:
            return
        rel = self.files[p]
        size, sha = self.rows[rel]
        _LOCAL.busy = True
        try:
            h = hashlib.sha256()
            n = 0
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
                    n += len(b)
        except FileNotFoundError:
            raise FileNotFoundError(2, f"listed in {self.source(rel)} but missing from the result root: {rel}", p)
        finally:
            _LOCAL.busy = False
        if n != size or h.hexdigest() != sha:
            raise ResultMismatch(f"{rel}: {n} bytes, sha256 {h.hexdigest()[:12]}; {self.source(rel)} says {size} "
                                 f"bytes, {sha[:12]}")
        self.checked[p] = True
        self.counts["placed" if rel in self.supplied else "checked"] += 1

    def source(self, rel):
        return EXCLUDED_NAME if rel in self.supplied else "MANIFEST.tsv"

    def kind(self, p):
        """'file', 'dir', 'withheld', 'left_out' (an absent EXCLUDED.tsv input), 'hidden' or None (outside the root)."""
        q = self.norm(p)
        if q is None:
            return None, None
        if q in self.files:
            return "file", q
        if q in self.withheld:
            return "withheld", q
        if q in self.dirs:
            return "dir", q
        if q in self.left_out:
            return "left_out", q
        return "hidden", q


def _fake_stat():
    return os.stat_result((_st.S_IFREG | 0o444, 0, 0, 1, 0, 0, 0, 0, 0, 0))


def install(root, withheld=()):
    """Install the closed world over root (once per process) and return it. SystemExit without ROOT/MANIFEST.tsv, or
    when a listed file other than a withheld one is missing or has another size than its line says."""
    if "world" in _STATE:
        return _STATE["world"]
    man = os.path.join(root, "MANIFEST.tsv")
    if not os.path.isfile(man):
        raise SystemExit(f"{man} not found: public mode checks every result file against the results archive's "
                         "MANIFEST.tsv (for a result root of your own: python results_world.py manifest ROOT > "
                         "ROOT/MANIFEST.tsv)")
    rows = load_manifest(man)
    held = set(withheld)
    exc = os.path.join(root, EXCLUDED_NAME)
    if not os.path.isfile(exc):
        raise SystemExit(f"{exc} not found: public mode needs the results archive's {EXCLUDED_NAME}, which names the "
                         "inputs it leaves out (for a result root of your own that leaves nothing out: python "
                         f"results_world.py excluded > ROOT/{EXCLUDED_NAME})")
    excluded = {r: v for r, v in load_excluded(exc).items() if r not in rows and r not in held}
    supplied = {r: v for r, v in excluded.items() if os.path.isfile(os.path.join(root, *r.split("/")))}
    left_out = sorted(set(excluded) - set(supplied))
    missing, resized = [], []
    for rel, (size, _) in sorted(rows.items()):
        if rel in held:
            continue
        p = os.path.join(root, *rel.split("/"))
        try:
            st = os.stat(p)
        except FileNotFoundError:
            missing.append(rel)
            continue
        if st.st_size != size:
            resized.append(rel)
    resized += [f"{r} (its {EXCLUDED_NAME} row)" for r, (size, _) in sorted(supplied.items())
                if os.path.getsize(os.path.join(root, *r.split("/"))) != size]
    if missing or resized:
        msg = []
        if missing:
            msg.append(f"{len(missing)} listed file(s) missing from the result root: " + ", ".join(missing[:10])
                       + (" ..." if len(missing) > 10 else ""))
        if resized:
            msg.append(f"{len(resized)} listed file(s) with another size: " + ", ".join(resized[:10])
                       + (" ..." if len(resized) > 10 else ""))
        raise SystemExit(f"{man}: " + "; ".join(msg))
    W = _World(root, {**rows, **supplied}, withheld, supplied, left_out)
    _STATE["world"] = W
    o_stat, o_lstat, o_access, o_listdir, o_scandir = os.stat, os.lstat, os.access, os.listdir, os.scandir

    def wrap_stat(fn):
        def w(path, *a, **k):
            if _busy() or k.get("dir_fd") is not None:
                return fn(path, *a, **k)
            kind, q = W.kind(path)
            if kind in ("hidden", "left_out"):
                W.counts["hidden" if kind == "hidden" else "left_out"] += 1
                raise FileNotFoundError(2, "not in MANIFEST.tsv" if kind == "hidden" else
                                        f"left out of the results archive ({EXCLUDED_NAME})", os.fspath(path))
            if kind == "withheld":
                W.counts["withheld"] += 1
                return _fake_stat()
            if kind == "file":
                W.check(q)
            return fn(path, *a, **k)
        return w

    def w_access(path, mode, *a, **k):
        if not _busy():
            kind, q = W.kind(path)
            if kind in ("hidden", "left_out"):
                return False
            if kind == "withheld":
                return not (mode & (os.W_OK | os.X_OK))
            if kind == "file":
                W.check(q)
        return o_access(path, mode, *a, **k)

    def _keep(base, name):
        kind, _ = W.kind(os.path.join(base, name))
        return kind not in ("hidden", "left_out")

    def w_listdir(path="."):
        names = o_listdir(path)
        if _busy() or isinstance(path, int):
            return names
        kind, q = W.kind(path)
        if kind is None:
            return names
        names = [n for n in names if _keep(q, os.fsdecode(n))]
        extra = [os.path.basename(p) for p in W.withheld if os.path.dirname(p) == q]
        return names + [n for n in extra if n not in names]

    class _ScanList(list):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def close(self):
            pass

    def w_scandir(path="."):
        it = o_scandir(path)
        if _busy() or isinstance(path, int):
            return it
        kind, q = W.kind(path)
        if kind is None:
            return it
        with it:
            ents = [e for e in it if _keep(q, e.name)]
        return _ScanList(ents)

    def hook(ev, args):
        if ev != "open" or _busy():
            return
        p, mode, flags = args
        if p is None or isinstance(p, int):
            return
        kind, q = W.kind(p)
        if kind is None or kind == "dir":
            return
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
        if writing:
            raise PermissionError(f"the result root is read-only for the build: {p}")
        if kind == "hidden":
            W.counts["hidden"] += 1
            raise FileNotFoundError(2, "not in MANIFEST.tsv", os.fsdecode(os.fspath(p)))
        if kind == "left_out":
            W.counts["left_out"] += 1
            raise FileNotFoundError(2, f"left out of the results archive ({EXCLUDED_NAME}); place the file at this "
                                    "path to build what needs it", os.fsdecode(os.fspath(p)))
        if kind == "withheld":
            raise FileNotFoundError(2, "withheld from the results archive (existence only)", os.fsdecode(os.fspath(p)))
        W.check(q)

    def wrap_pred(fn, follow, test):
        """An os.path predicate that answers for paths under root through the wrapped os.stat (follow) or os.lstat,
        as the predicates of genericpath and posixpath do; on Windows, Python 3.12 and later implement exists, isfile,
        isdir and islink without calling os.stat. Paths outside root keep the original predicate."""
        def w(path):
            if _busy() or W.kind(path)[0] is None:
                return fn(path)
            try:
                st = (os.stat if follow else os.lstat)(path)
            except (OSError, ValueError):
                return False
            return test is None or test(st.st_mode)
        return w

    os.stat = wrap_stat(o_stat)
    os.lstat = wrap_stat(o_lstat)
    os.access = w_access
    os.listdir = w_listdir
    os.scandir = w_scandir
    for name, follow, test in (("exists", True, None), ("lexists", False, None), ("isfile", True, _st.S_ISREG),
                               ("isdir", True, _st.S_ISDIR), ("islink", False, _st.S_ISLNK)):
        setattr(os.path, name, wrap_pred(getattr(os.path, name), follow, test))
    sys.addaudithook(hook)
    import atexit
    atexit.register(lambda: print(f"[public] closed world: {W.counts['checked']} result files checked against "
                                  f"MANIFEST.tsv, {W.counts['hidden']} probes of unlisted paths hidden, "
                                  f"{W.counts['withheld']} probes of withheld files"
                                  + (f", {W.counts['placed']} placed inputs checked against {EXCLUDED_NAME}"
                                     if W.supplied else "")
                                  + (f", {W.counts['left_out']} probes of left-out inputs" if W.left_out else ""),
                                  file=sys.stderr))
    print(f"[public] closed world over {W.root}: {len(rows)} files of MANIFEST.tsv, {len(W.withheld)} withheld"
          + (f", {len(excluded)} inputs left out ({EXCLUDED_NAME}): {len(supplied)} placed under the root"
             if excluded else ""), file=sys.stderr)
    return W


def world():
    return _STATE.get("world")


def left_out(rel):
    """True for an input of the build that ROOT/EXCLUDED.tsv lists and that is not under ROOT (public mode)."""
    W = world()
    return W is not None and rel in W.left_out.values()


def needs(inputs, outputs, what):
    """Public mode: True when one of inputs (paths relative to the result root) is left out of the results archive and
    absent (left_out). Prints one notice naming the inputs and EXCLUDED.tsv, and appends '<output>\\t<input>' lines for
    outputs (paths relative to DIR) to DIR/NEEDS.tsv, which reproduce.py reports as NEEDS instead of comparing them."""
    miss = sorted({r for r in inputs if left_out(r)})
    if not miss:
        return False
    import script_dirs as SD
    print(f"[public] NEEDS {', '.join(miss)} ({EXCLUDED_NAME}): {what} not built ({', '.join(outputs)})",
          file=sys.stderr)
    out = SD.public_out()
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, NEEDS_NAME), "a", encoding="utf-8") as fh:
        fh.writelines(f"{o}\t{r}\n" for o in outputs for r in miss)
    return True


def clear_needs(prefix):
    """Remove the DIR/NEEDS.tsv lines whose output starts with prefix (a step that builds that output again)."""
    import script_dirs as SD
    p = os.path.join(SD.public_out(), NEEDS_NAME)
    if not os.path.isfile(p):
        return
    with open(p, encoding="utf-8") as fh:
        lines = fh.readlines()
    keep = [ln for ln in lines if not ln.startswith(prefix)]
    if len(keep) != len(lines):
        with open(p, "w", encoding="utf-8") as fh:
            fh.writelines(keep)


def read_needs(out):
    """{output: [inputs]} of DIR/NEEDS.tsv (empty when the file does not exist)."""
    res = {}
    p = os.path.join(out, NEEDS_NAME)
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                o, _, r = ln.rstrip("\n").partition("\t")
                if o and r and r not in res.setdefault(o, []):
                    res[o].append(r)
    return res


def listed(rel_or_path):
    """The sha256 that MANIFEST.tsv lists for a result file (a path relative to the result root, or a path under it),
    after the file's bytes were checked against it; None for a file it does not list (an input placed under the root
    from EXCLUDED.tsv included) or without a closed world."""
    W = world()
    if W is None:
        return None
    s = os.fspath(rel_or_path)
    q = W.norm(s) if os.path.isabs(s) else os.path.join(W.root, *s.split("/"))
    if q not in W.files or W.files[q] in W.supplied:
        return None
    W.check(q)
    return W.rows[W.files[q]][1]


def start(argv):
    """Public mode from a script's command line (--public --results ROOT): set the mode for this process and its
    children (script_dirs.apply_cli), then install the closed world over ROOT with the withheld files of the plan
    inputs. Returns True in public mode. Call it before any module computes a data path."""
    import script_dirs as SD
    if not SD.apply_cli(argv):
        return False
    import plan_inputs as PI
    install(SD.results_root(), PI.withheld())
    return True


def summary():
    W = world()
    return None if W is None else dict(W.counts, listed=len(W.files) - len(W.supplied))


def write_manifest(root, out=sys.stdout):
    """MANIFEST.tsv of every file under root (except MANIFEST.tsv itself), sorted by path."""
    out.write("# Result files: path relative to this folder, bytes, sha256.\npath\tbytes\tsha256\n")
    rows = []
    for dp, dn, fs in os.walk(root):
        dn.sort()
        for f in fs:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if rel == "MANIFEST.tsv" or not os.path.isfile(p):
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            rows.append(f"{rel}\t{os.path.getsize(p)}\t{h.hexdigest()}\n")
    out.write("".join(sorted(rows)))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "excluded":
        sys.stdout.write(EXCLUDED_HEADER)
    elif len(sys.argv) == 3 and sys.argv[1] == "manifest":
        write_manifest(sys.argv[2])
    else:
        raise SystemExit("usage: python results_world.py manifest ROOT > ROOT/MANIFEST.tsv\n"
                         f"       python results_world.py excluded > ROOT/{EXCLUDED_NAME}")


