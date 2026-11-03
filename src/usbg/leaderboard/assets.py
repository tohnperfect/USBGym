"""usbg.leaderboard.assets: the reference artifacts released with the leaderboard, and their download.

data/assets.json lists each artifact a submitter-run slot condition needs (name, sha256, size, license, the slot,
what it is, its provenance). The files are attached to the GitHub release; ``python -m usbg.leaderboard fetch``
downloads them into a folder (default assets/) and accepts a file only if its sha256 matches.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def manifest():
    with open(os.path.join(HERE, "data", "assets.json"), encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(dest="assets", names=None, from_dir=None, quiet=False):
    """Download (or copy from from_dir) the listed assets into dest, verifying each sha256. -> {name: path}."""
    m = manifest()
    os.makedirs(dest, exist_ok=True)
    got = {}
    for a in m["assets"]:
        if names and a["name"] not in names:
            continue
        p = os.path.join(dest, a["name"])
        if os.path.exists(p) and sha256_file(p) == a["sha256"]:
            got[a["name"]] = p
            continue
        tmp = p + ".part"
        if from_dir:
            shutil.copyfile(os.path.join(from_dir, a["name"]), tmp)
        else:
            url = m["url_pattern"].format(name=a["name"])
            if not quiet:
                print(f"[fetch] {url}")
            try:
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                    shutil.copyfileobj(r, fh)
            except (urllib.error.URLError, OSError) as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                why = f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else str(e)
                raise SystemExit(f"{a['name']}: download from {url} failed ({why}); the asset may not be uploaded to "
                                 "the release yet, or the network is unreachable (--from-dir copies from a folder)")
        sha = sha256_file(tmp)
        if sha != a["sha256"] or os.path.getsize(tmp) != a["size"]:
            os.remove(tmp)
            raise SystemExit(f"{a['name']}: sha256 {sha} or size differs from the manifest; file removed")
        os.replace(tmp, p)
        got[a["name"]] = p
        if not quiet:
            print(f"[fetch] {p} ok (sha256 {sha[:12]})")
    return got


def verify(folder, names):
    """Every named asset in folder has the sha256 and size of the manifest (the runner checks them before a run, so
    a swapped file is never loaded)."""
    m = {a["name"]: a for a in manifest()["assets"]}
    for n in names:
        p = os.path.join(folder, n)
        if n not in m:
            raise SystemExit(f"{n}: not a release asset")
        if not os.path.exists(p) or os.path.getsize(p) != m[n]["size"] or sha256_file(p) != m[n]["sha256"]:
            raise SystemExit(f"{p}: missing, or its sha256 or size differs from the manifest; fetch it again "
                             "(python -m usbg.leaderboard fetch)")
    return True


__all__ = ["manifest", "fetch", "sha256_file", "verify"]
