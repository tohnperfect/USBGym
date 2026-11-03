"""Shared pytest setup.

Seat pinning: the pre-existing suite tests the v1 geometry on the synthetic phantom, which has no
skin, no case id and no screened windows, so every test runs under USBG_SEAT=canonical unless it is
marked `anatomical` (tests/test_placement_anatomical.py), which runs under the anatomical seat.

Private-data guard (D5): a CPython audit hook armed for the whole session. Any open / os.open /
listdir / scandir / glob / chdir / copy / rename / link whose path has a component named real_us_crop,
real_us or _private_derived (usbg.provenance.PRIVATE_DIRNAMES), either as written or after resolving
symlinks, raises PermissionError. So any test that reads a private clinical pool, directly or through
library code, at import time or at run time, fails, and a clone without the pool behaves the same.
Blocked accesses are listed in the terminal summary. Audit hooks cannot be removed, so the hook stays
armed for the whole process. USBG_ALLOW_PRIVATE=1 leaves it unarmed (local, never-published checks
only). Limits: subprocesses do not inherit the hook, and C-level readers that bypass Python's open
(cv2.imread, SimpleITK) are not seen; the static scan in tests/test_private_data_guard.py covers those.
"""
import os
import sys

import pytest

from usbg.provenance import is_private_path

# audit event -> indices of its path arguments (CPython audit events table)
_GUARDED_EVENTS = {
    "open": (0,),                 # builtins.open, io.open, os.open, io.open_code
    "os.listdir": (0,),
    "os.scandir": (0,),           # also os.walk, pathlib iterdir/glob
    "glob.glob": (0,),
    "glob.glob/2": (0, 2),        # (pathname, recursive, root_dir, dir_fd)
    "os.chdir": (0,),             # chdir into a pool, then relative opens would hide the component
    "shutil.copyfile": (0, 1),
    "shutil.copytree": (0, 1),
    "os.rename": (0, 1),          # also os.replace and shutil.move
    "os.link": (0, 1),
    "os.symlink": (0, 1),
}
_HITS: list[str] = []


def _as_str(p):
    if p is None or isinstance(p, int):            # open(fd) / listdir(fd) / dir_fd
        return None
    try:
        return os.fsdecode(os.fspath(p))
    except TypeError:
        return None


def _private_guard(event, args):
    idx = _GUARDED_EVENTS.get(event)
    if idx is None:
        return
    for i in idx:
        if i >= len(args):
            continue
        s = _as_str(args[i])
        if not s:
            continue
        hit = is_private_path(s)
        if not hit:
            try:
                hit = is_private_path(os.path.realpath(s))     # a symlink into a pool is still private
            except (OSError, ValueError):
                hit = False
        if hit:
            _HITS.append(f"{event}: {s}")
            raise PermissionError(f"[usbg private-data guard] {event} on a private clinical pool: {s} "
                                  "(D5; set USBG_ALLOW_PRIVATE=1 only for a local, unpublished check)")


def private_guard_armed() -> bool:
    return os.environ.get("USBG_ALLOW_PRIVATE") != "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "anatomical: run under USBG_SEAT=anatomical (needs data/cache_anat and the windows file)")
    # session default, set before any module-scoped fixture builds a seat
    os.environ["USBG_SEAT"] = "canonical"
    os.environ.pop("USBG_GEL", None)
    # private-data guard: armed before collection, so import-time reads of test modules are covered
    if private_guard_armed() and not getattr(sys, "_usbg_private_guard", False):
        sys.addaudithook(_private_guard)
        sys._usbg_private_guard = True


@pytest.fixture(autouse=True)
def _pin_seat(request, monkeypatch):
    if request.node.get_closest_marker("anatomical"):
        monkeypatch.setenv("USBG_SEAT", "anatomical")
    yield


def pytest_terminal_summary(terminalreporter):
    selftest = [h for h in _HITS if "test_hook_blocks_private_reads" in h]     # intended, in tmp_path
    real = sorted(set(h for h in _HITS if h not in selftest))
    if real:
        terminalreporter.write_sep("=", "usbg private-data guard: blocked accesses")
        for h in real:
            terminalreporter.write_line(h)
    if selftest:
        terminalreporter.write_line(f"usbg private-data guard: armed ({len(selftest)} intended blocks "
                                    "in the self-test)")
    if not private_guard_armed():
        terminalreporter.write_line("usbg private-data guard NOT armed (USBG_ALLOW_PRIVATE=1)")
