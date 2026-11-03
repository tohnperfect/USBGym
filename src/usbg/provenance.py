"""provenance.py: one sidecar JSON per trained artifact, a provenance block per result JSON, and the
held-out leakage and private-data checks (merged from the D5 and oracle-ladder drafts).

Every checkpoint or dataset that feeds a reported number gets ``<artifact>.provenance.json`` written
next to it by ``write_provenance``. ``scan`` walks one or more roots, resolves each artifact's
lineage through its ``parents`` (a texture stage trained on pairs, a segmenter trained on a dataset
rendered through a texture stage, a policy trained against a segmenter), and reports:

  * LEAK      a held-out benchmark patient (benchmark.TEST_PATIENTS) appears in the transitive
              training cases of any artifact;
  * PRIVATE   any real pool or parent resolves under a private clinical pool (data/real_us_crop,
              data/real_us) or is a known private-pool descendant;
  * STALE     the sidecar md5 no longer matches the file on disk;
  * SCHEMA    a required field is missing or malformed;
  * SEATMIX   artifacts feeding one table disagree on seat mode (checked by ``assert_one_seat``).

Schema (``usbg-provenance-1.0``), all keys required unless marked optional:

  schema_version  "usbg-provenance-1.0"
  artifact        {"path": repo-relative path, "md5": hex, "bytes": int, "kind": one of KINDS}
  train_cases     [CT case ids actually used for gradient updates, AFTER any internal train/val split]
  val_cases       [CT case ids used only for model selection] (may be [])
  real_pool       {"name": str, "path": str|None, "n_frames": int, "license": str,
                   "frames_md5": md5 of the sorted (name,size) list | None,
                   "frame_ids": optional list of frame ids actually used (e.g. the 40 Vitale frames)}
  iters           int (optimizer steps; 0 for a non-trained artifact such as a pairs npz)
  seed            int
  seat_mode       "anatomical" | "canonical"
  dr              {"mode": "off" | "full" | "mild" | "per_frame", "notes": str}
  parents         [{"path": str, "md5": str, "role": str}]   (pairs, S_US, translator, dataset, ...)
  git             {"repo": abs path, "hash": str, "dirty": bool}
  created         ISO-8601 UTC timestamp
  command         the exact argv that produced the artifact
  benchmark       {"version": str, "test_patients": [...]}   (snapshot, so a later split change is visible)
  extra           optional free-form dict (target set, IMG_DEPTH_M, bladder-removed flag, US_SEG_CKPT, ...)
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

SCHEMA_VERSION = "usbg-provenance-1.0"
SIDECAR_SUFFIX = ".provenance.json"
KINDS = ("translator", "s_us", "segmenter", "policy", "embedding", "pairs", "dataset", "eval")
SEAT_MODES = ("anatomical", "canonical")
DR_MODES = ("off", "full", "mild", "per_frame")

# Tier 1: raw private clinical pools (the private clinical frames), plus a reserved quarantine folder name
# (nothing is moved there automatically; the user's files stay in place and code may not reference them). Any path whose components include one of these names is private, wherever
# the repo is cloned or copied; the pytest audit hook blocks open/listdir/scandir on it.
PRIVATE_DIRNAMES = ("real_us_crop", "real_us", "_private_derived")

# Tier 2: artifacts trained on, or rendered through a model trained on, the tier-1 pool. Keyed by md5
# (exact bytes), so a renamed copy is still caught and a clean regeneration under the same name is not.
# "verified" = producing command recovered from the run log; "inferred" = mtimes/logs only.
PRIVATE_DESCENDANT_MD5 = {
    "68abc2fc0e195f61bf46e723c1bf1e8f": ("data/dataset/texture_cut.pt", "verified: compare_translators.py --real-dir data/real_us_crop"),
    "6dd1e6ba85d09f703bcf84bbe7881fa8": ("data/dataset/texture_cyclegan.pt", "verified: same compare_translators.py call"),
    "e0c1cb57a0cce6a17ed2e5df961c5acb": ("data/dataset/texture_cyclegan_sc.pt", "verified: same compare_translators.py call"),
    "956a211d52b54406d34bce41e8c145da": ("data/dataset/texture_sonospade.pt", "verified: train_texture_gan.py --real-dir data/real_us_crop"),
    "407bbfc2da764459609ec2d4ca11432b": ("data/dataset/texture_gan_real.pt", "inferred: README sec. 9(b) --real-dir data/real_us_crop; mtime after the crop"),
    "a3c36bd89d00374c382b9985e3ee8993": ("data/dataset/exp_cut.pt", "verified: run_sonospade_experiment.sh"),
    "cd2366f778d8bcce821d21e81732dc00": ("data/dataset/exp_cyclegan.pt", "verified: run_sonospade_experiment.sh"),
    "ccd5a6406629437897f2a0b5672ed9a6": ("data/dataset/exp_cyclegan_sc.pt", "verified: run_sonospade_experiment.sh"),
    "4646bea288de0b51f8483b9e8e6d4b18": ("data/dataset/exp_sonospade.pt", "verified: run_sonospade_experiment.sh"),
    "751f793a513edabe3716968a06b4000a": ("data/dataset/usbg.npz", "verified: make_dataset.py --texture-gan data/dataset/texture_cut.pt"),
    "841f6966ffbdb43db570d63f73339371": ("data/dataset/lerobot/meta/info.json", "verified: same make_dataset.py call (--lerobot)"),
    "1298a393ca6a5c83f51f0bcfc918e1cb": ("data/dataset/exp_real_cut.pt", "verified: run_sonospade_experiment_real.sh --sim-npz data/dataset/usbg.npz"),
    "3f57bf0496a82de14b883697a8e0235c": ("data/dataset/exp_real_cyclegan.pt", "verified: run_sonospade_experiment_real.sh --sim-npz data/dataset/usbg.npz"),
    "7a95354f5dd2256584c89b8dcca50537": ("data/dataset/exp_real_cyclegan_sc.pt", "verified: run_sonospade_experiment_real.sh --sim-npz data/dataset/usbg.npz"),
    "bcbf38b5a02c5190e33d5c56431d184a": ("data/dataset/bmode_embed.pt", "inferred: run_benchmark_study.sh step 1 --translator texture_cut.pt (mtime 1 min before bench/usbg_val.npz)"),
    "ff10cec325c0c315b904e483c4a698d2": ("data/dataset/bench/usbg_val.npz", "verified: run_benchmark_study.sh TEX=texture_cut.pt (whole bench/ dir)"),
    "1df66237bee08febfe9cee568bdeadcb": ("data/dataset/bench/bc_0.pt", "verified: run_benchmark_study.sh (whole bench/ dir)"),
    "09e4bf1642b4ea7ab2861723a7519d83": ("data/dataset/scale/usbg_tr_0.npz", "inferred: _scale.log 'translator on' 2026-07-02 (whole scale/ dir)"),
    "2e34cc48a94b09eade0c31e3b18ed767": ("data/dataset/scale/bc_0.pt", "inferred: scale run (whole scale/ dir)"),
    "3d953c14eda6f984ebdd9254d3cc6b58": ("data/dataset/bc_policy.pt", "inferred: mtime 1 min after usbg.npz"),
    "71df370645f2a91d461ea9aa853f24ad": ("data/dataset/rl_policy.pt", "inferred: mtime after usbg.npz / bc_policy.pt"),
}
# Directories quarantined as a whole (every file inside is a descendant).
PRIVATE_DESCENDANT_DIRS = ("data/dataset/bench", "data/dataset/scale", "data/dataset/lerobot")
# Legacy checkpoint NAMES that are retired: no code, default or runner may reference them again.
RETIRED_NAMES = ("texture_cut.pt", "texture_cyclegan.pt", "texture_cyclegan_sc.pt", "texture_sonospade.pt",
                 "texture_gan_real.pt", "exp_cut.pt", "exp_cyclegan.pt", "exp_cyclegan_sc.pt",
                 "exp_sonospade.pt", "exp_real_cut.pt", "exp_real_cyclegan.pt", "exp_real_cyclegan_sc.pt",
                 "bmode_embed.pt")

_REQUIRED = ("schema_version", "artifact", "train_cases", "val_cases", "real_pool", "iters", "seed",
             "seat_mode", "dr", "parents", "git", "created", "command", "benchmark")


# ----------------------------------------------------------------------------------------- helpers
def _relpath_or_abs(path: str, start: str) -> str:
    """path relative to start, or the absolute path when they share no root (on Windows, paths on two
    drives have no relative path and os.path.relpath raises ValueError)."""
    p = os.path.abspath(path)
    try:
        return os.path.relpath(p, start)
    except ValueError:
        return p


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    """md5 of a file, or of a directory as the md5 of its sorted (relpath, size, md5) listing."""
    if os.path.isdir(path):
        h = hashlib.md5()
        for root, _dirs, files in sorted(os.walk(path)):
            for f in sorted(files):
                p = os.path.join(root, f)
                h.update(f"{os.path.relpath(p, path)}\0{os.path.getsize(p)}\0{md5_file(p)}\n".encode())
        return h.hexdigest()
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def pool_fingerprint(pool_dir: str, exts=(".png", ".jpg", ".jpeg")) -> tuple[int, str]:
    """(n_frames, md5 of the sorted (name, size) list) of an image pool directory."""
    fs = sorted(f for f in os.listdir(pool_dir) if f.lower().endswith(exts))
    h = hashlib.md5("".join(f"{f}\0{os.path.getsize(os.path.join(pool_dir, f))}\n" for f in fs).encode())
    return len(fs), h.hexdigest()


def is_private_path(p) -> bool:
    """True if a path (str/bytes/PathLike) points into a tier-1 private pool, by path component."""
    try:
        s = os.fsdecode(os.fspath(p))
    except TypeError:
        return False
    if "real_us" not in s and "_private_derived" not in s:   # cheap prefilter (audit hook stays fast)
        return False
    parts = os.path.normpath(os.path.expanduser(s)).split(os.sep)
    return any(d in parts for d in PRIVATE_DIRNAMES)


def is_private_descendant(path: str | None, md5: str | None = None) -> str | None:
    """Reason string if the artifact is a known tier-2 private-pool descendant (by md5), or sits in
    the quarantine folder, else None. A path without an md5 is hashed if it exists."""
    if path and is_private_path(path):
        return f"{path}: inside a private / quarantine directory"
    if md5 is None and path and os.path.isfile(os.path.expanduser(path)):
        md5 = md5_file(os.path.expanduser(path))
    if md5 and md5 in PRIVATE_DESCENDANT_MD5:
        orig, why = PRIVATE_DESCENDANT_MD5[md5]
        return f"md5 {md5} = {orig} ({why})"
    return None


_GYM_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_info(repo: str | None = None) -> dict:
    """hash, dirty flag, dirty tracked files and md5 of the working-tree diff (so an uncommitted run
    is still identifiable)."""
    repo = os.path.abspath(repo or _GYM_REPO)

    def _git(*a):
        try:
            return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    diff = _git("diff", "HEAD")
    return {"repo": repo, "hash": _git("rev-parse", "HEAD"), "dirty": bool(dirty),
            "dirty_files": dirty.splitlines() if dirty else [],
            "diff_md5": hashlib.md5(diff.encode()).hexdigest() if diff else None}


def current_seat_mode() -> str:
    """The seat switch (usbg.seatmode): env USBG_SEAT, default anatomical."""
    from .seatmode import seat_mode
    return seat_mode()


def sidecar_path(artifact_path: str) -> str:
    return artifact_path.rstrip(os.sep) + SIDECAR_SUFFIX


# ------------------------------------------------------------------------------------ write / load
def write_provenance(artifact_path: str, kind: str, *, train_cases, val_cases=(), real_pool=None,
                     iters: int = 0, seed: int = 0, dr_mode: str = "off", dr_notes: str = "",
                     parents=(), repo: str | None = None, seat_mode: str | None = None,
                     extra: dict | None = None) -> str:
    """Write ``<artifact>.provenance.json``. Raises if the record would already fail the checks
    (held-out case in train_cases, private real pool), so a leaky artifact is never blessed."""
    from .benchmark import BENCHMARK_VERSION, TEST_PATIENTS
    repo = repo or _GYM_REPO
    real_pool = dict(real_pool or {"name": "none", "path": None, "n_frames": 0,
                                   "license": "n/a", "frames_md5": None})
    rec = {
        "schema_version": SCHEMA_VERSION,
        "artifact": {"path": _relpath_or_abs(artifact_path, repo),
                     "md5": md5_file(artifact_path),
                     "bytes": os.path.getsize(artifact_path) if os.path.isfile(artifact_path) else None,
                     "kind": kind},
        "train_cases": sorted(set(train_cases)),
        "val_cases": sorted(set(val_cases)),
        "real_pool": real_pool,
        "iters": int(iters), "seed": int(seed),
        "seat_mode": seat_mode or current_seat_mode(),
        "dr": {"mode": dr_mode, "notes": dr_notes},
        "parents": [dict(p) for p in parents],
        "git": git_info(repo),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
        "benchmark": {"version": BENCHMARK_VERSION, "test_patients": list(TEST_PATIENTS)},
        "extra": extra or {},
    }
    problems = validate_record(rec) + [p for p in _record_findings(rec, set(TEST_PATIENTS))
                                       if not p.startswith("STALE")]
    if problems:
        raise ValueError("refusing to write provenance:\n  " + "\n  ".join(problems))
    out = sidecar_path(artifact_path)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2)
    return out


def parent_entry(path: str, role: str) -> dict:
    return {"path": path, "md5": md5_file(path), "role": role}


def load_provenance(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------- checks
def validate_record(rec: dict) -> list[str]:
    p = []
    for k in _REQUIRED:
        if k not in rec:
            p.append(f"SCHEMA missing key {k!r}")
    if rec.get("schema_version") != SCHEMA_VERSION:
        p.append(f"SCHEMA schema_version {rec.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    a = rec.get("artifact") or {}
    for k in ("path", "md5", "kind"):
        if not a.get(k):
            p.append(f"SCHEMA artifact.{k} missing")
    if a.get("kind") not in KINDS:
        p.append(f"SCHEMA artifact.kind {a.get('kind')!r} not in {KINDS}")
    if rec.get("seat_mode") not in SEAT_MODES:
        p.append(f"SCHEMA seat_mode {rec.get('seat_mode')!r} not in {SEAT_MODES}")
    if (rec.get("dr") or {}).get("mode") not in DR_MODES:
        p.append(f"SCHEMA dr.mode {(rec.get('dr') or {}).get('mode')!r} not in {DR_MODES}")
    if not isinstance(rec.get("train_cases"), list):
        p.append("SCHEMA train_cases must be a list of CT case ids")
    elif a.get("kind") in ("translator", "s_us", "segmenter", "pairs", "dataset", "policy") \
            and not rec["train_cases"] and not rec.get("parents"):
        p.append("SCHEMA train_cases empty with no parents (lineage unrecoverable)")
    return [f"{a.get('path', '?')}: {x}" for x in p]


def _record_findings(rec: dict, test: set) -> list[str]:
    a = rec.get("artifact") or {}
    path = a.get("path", "?")
    out = []
    hit = sorted(set(rec.get("train_cases") or []) & test)
    if hit:
        out.append(f"LEAK {path}: held-out {hit} in train_cases")
    vhit = sorted(set(rec.get("val_cases") or []) & test)
    if vhit:
        out.append(f"LEAK {path}: held-out {vhit} in val_cases (model selection on test patients)")
    rp = rec.get("real_pool") or {}
    if rp.get("path") and is_private_path(rp["path"]):
        out.append(f"PRIVATE {path}: real_pool.path {rp['path']} is a private clinical pool")
    for par in rec.get("parents") or []:
        if is_private_path(par.get("path", "")):
            out.append(f"PRIVATE {path}: parent {par['path']} is a private clinical pool")
        why = is_private_descendant(par.get("path"), par.get("md5"))
        if why:
            out.append(f"PRIVATE {path}: parent is a private-pool descendant ({why})")
    return out


def _walk_sidecars(root, exclude=("_smoke", "_recipes")):
    """Sidecars under root, without following symlinked dirs, entering private dirs or excluded dirs."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in exclude and d not in PRIVATE_DIRNAMES
                       and not os.path.islink(os.path.join(dirpath, d))]
        for f in filenames:
            if f.endswith(SIDECAR_SUFFIX):
                yield os.path.join(dirpath, f)


def scan(roots, test_patients=None, repo: str | None = None, check_md5: bool = True,
         exclude=("_smoke", "_recipes")) -> dict:
    """Walk roots for *.provenance.json, resolve lineage through parents, return findings.

    Returns {"records": {artifact_path: rec}, "findings": [str], "transitive": {path: [cases]}}."""
    if test_patients is None:
        from .benchmark import TEST_PATIENTS as test_patients
    test = set(test_patients)
    repo = repo or _GYM_REPO
    recs, bad = {}, []
    for r in roots:
        for f in _walk_sidecars(os.path.expanduser(r), exclude):
            try:
                rec = load_provenance(f)
            except (OSError, ValueError) as e:
                bad.append(f"SCHEMA {f}: unreadable sidecar ({e})")
                continue
            if not isinstance(rec, dict) or not isinstance(rec.get("artifact"), dict):
                bad.append(f"SCHEMA {f}: not a usbg-provenance record")
                continue
            art = os.path.join(os.path.dirname(f), os.path.basename(f)[:-len(SIDECAR_SUFFIX)])
            recs[os.path.abspath(art)] = rec
    by_md5 = {rec.get("artifact", {}).get("md5"): p for p, rec in recs.items()}
    findings = list(bad)
    for p, rec in recs.items():
        findings += validate_record(rec)
        findings += _record_findings(rec, test)
        if check_md5 and os.path.exists(p) and md5_file(p) != rec.get("artifact", {}).get("md5"):
            findings.append(f"STALE {p}: sidecar md5 != file md5 (artifact changed after provenance)")

    def closure(p, seen):
        if p in seen:
            return set()
        seen.add(p)
        rec = recs[p]
        cases = set(rec.get("train_cases") or []) | set(rec.get("val_cases") or [])
        for par in rec.get("parents") or []:
            q = by_md5.get(par.get("md5")) or os.path.abspath(os.path.join(repo, par.get("path", "")))
            if q in recs:
                cases |= closure(q, seen)
            elif not is_private_descendant(par.get("path"), par.get("md5")):
                findings.append(f"SCHEMA {p}: parent {par.get('path')} has no provenance sidecar "
                                f"(lineage cannot be checked)")
        return cases

    trans = {}
    for p in recs:
        cs = closure(p, set())
        trans[p] = sorted(cs)
        hit = sorted(cs & test)
        if hit and not any(f.startswith(f"LEAK {recs[p]['artifact']['path']}") for f in findings):
            findings.append(f"LEAK {p}: held-out {hit} reached through parents")
    return {"records": recs, "findings": sorted(set(findings)), "transitive": trans}


def assert_one_seat(result_jsons) -> str:
    """Every JSON feeding one table must carry the same provenance.seat_mode."""
    modes = {}
    for f in result_jsons:
        with open(f) as fh:
            d = json.load(fh)
        m = (d.get("provenance") or {}).get("seat_mode")
        modes.setdefault(m, []).append(f)
    if len(modes) != 1 or None in modes:
        raise AssertionError(f"mixed or missing seat modes: { {k: len(v) for k, v in modes.items()} }")
    return next(iter(modes))


# ------------------------------------------------------------------------------ legacy manifests
def legacy_manifest_findings(manifest_path: str, test_patients=None) -> list[str]:
    """data/manifest.json streams: any training-role stream listing a held-out case, or a
    real_pool stream under a private directory."""
    if test_patients is None:
        from .benchmark import TEST_PATIENTS as test_patients
    test = set(test_patients)
    with open(manifest_path) as fh:
        m = json.load(fh)
    out = []
    for s in m.get("streams", []):
        if s.get("retired"):                       # kept for the record, never read by code
            continue
        if s.get("role") in ("sim_pairs", "real_pool"):
            hit = sorted(set(s.get("cases") or []) & test)
            if hit:
                out.append(f"LEAK manifest stream {s.get('name')} ({s.get('file')}): held-out {hit}")
            f = s.get("file") or s.get("path") or ""
            if s.get("role") == "real_pool" and is_private_path(f):
                out.append(f"PRIVATE manifest stream {s.get('name')}: {f}")
    return out


def legacy_nnunet_findings(dataset_dir: str, history_json: str | None = None,
                           test_patients=None) -> list[str]:
    """Segmenter repo: map the nnU-Net frame ids in a training history (train_cases/val_cases) back
    to CT case ids through the dataset's domain.json; flag held-out patients. Without a history,
    report the dataset-level case set (what any split of it could leak)."""
    if test_patients is None:
        from .benchmark import TEST_PATIENTS as test_patients
    test = set(test_patients)
    with open(os.path.join(dataset_dir, "domain.json")) as fh:
        dom = json.load(fh)

    def cid(frame):
        c = str(dom.get(frame, {}).get("case", frame))
        return os.path.splitext(os.path.basename(c))[0]
    name = os.path.basename(dataset_dir.rstrip(os.sep))
    if history_json:
        with open(history_json) as fh:
            h = json.load(fh)
        tr = {cid(f) for f in h.get("train_cases", []) if dom.get(f, {}).get("domain") == "sim"}
        hit = sorted(tr & test)
        return [f"LEAK {history_json} (dataset {name}): held-out {hit} in segmenter TRAIN split"] if hit else []
    allc = {cid(f) for f, v in dom.items() if v.get("split") == "train" and v.get("domain") == "sim"}
    hit = sorted(allc & test)
    return [f"LEAK dataset {name}: held-out {hit} rendered into the training pool"] if hit else []


# --------------------------------------------------------------------------- result-JSON block
RESULT_SCHEMA_VERSION = "usbg-result-prov-1.0"


def _ref(path):
    if not path:
        return None
    return {"path": path, "md5": md5_file(path) if os.path.exists(path) else None}


def result_block(*, texture=None, segmenter=None, policy=None, mask_source=None, dr_mode="off",
                 train_cases=(), eval_cases=(), target_set=(), repo=None, extra=None) -> dict:
    """The ``provenance`` block every script that produces a reported number embeds in its output JSON
    (``out["provenance"] = result_block(...)``). assert_one_seat() and the provenance audit read it."""
    from .benchmark import TEST_PATIENTS
    seg = segmenter or os.environ.get("US_SEG_CKPT")
    blk = {"schema_version": RESULT_SCHEMA_VERSION, "seat_mode": current_seat_mode(), "windows": windows_ref(),
           "texture": _ref(texture), "segmenter": _ref(seg), "policy": _ref(policy),
           "mask_source": mask_source,     # "learned" | "ct" | "pose_oracle" | None (n/a)
           "dr": {"mode": dr_mode}, "cohort": {"train": sorted(set(train_cases)), "eval": sorted(set(eval_cases))},
           "target_set": list(target_set), "git": git_info(repo),
           "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "command": " ".join([os.path.basename(sys.executable)] + sys.argv), "extra": extra or {}}
    if mask_source == "learned" and not os.environ.get("US_SEG_CKPT") and not segmenter:
        raise ValueError("learned-mask result without an explicit US_SEG_CKPT (D2: the silent default "
                         "us_seg_sim_real.pt was trained on held-out patients)")
    leak = sorted(set(train_cases) & set(TEST_PATIENTS))
    if leak:
        raise ValueError(f"result cohort trains on held-out {leak}")
    for k in ("texture", "segmenter", "policy"):
        r = blk[k]
        if r and (is_private_path(r["path"]) or (r["md5"] and r["md5"] in PRIVATE_DESCENDANT_MD5)):
            raise ValueError(f"{k} {r['path']} is private or a private-pool descendant")
    return blk


# ------------------------------------------------------------------- run-level block (trainer/eval)
def windows_ref():
    """The screened-windows file every anatomical seat reads (path + md5 + version), or None."""
    from .seatmode import seat_mode
    if seat_mode() != "anatomical":
        return None
    from . import windows as W
    return {"path": W.WINDOWS_PATH, "md5": md5(W.WINDOWS_PATH), "version": W.WINDOWS_VERSION}


# segmenter checkpoints that must never produce a new reported number (D2)
WITHDRAWN_SEG_MD5 = {"22a4972c5c881356bb3cf0e34366664a"}   # data/dataset/us_seg_sim_real.pt == sibling outputs/us_seg_6cls.pt


def md5(path):
    """md5_file, or None for a missing / unset path."""
    return md5_file(path) if path and os.path.exists(path) else None


def seat_info():
    from .seatmode import seat_mode, gel_coupling
    return {"USBG_SEAT": os.environ.get("USBG_SEAT"), "resolved": seat_mode(), "gel": gel_coupling()}


def mask_info():
    raw = os.environ.get("USBG_LEARNED_SEG")
    info = {"USBG_LEARNED_SEG": raw, "US_SEG_CKPT_env": os.environ.get("US_SEG_CKPT")}
    try:
        from . import view_match as VM
        learned = VM._US is not None
        info["learned"] = learned
        if learned:
            src = getattr(VM._US, "_DEFAULT_CKPT", None)          # module path (env read at import)
            if src is None and hasattr(VM._US, "ckpt_path"):      # object set via set_mask_source
                src = VM._US.ckpt_path
            info["seg_ckpt"] = os.path.abspath(src) if src else None
            info["seg_md5"] = md5(src) if src else None
    except Exception as e:                                         # never fail a run on provenance
        info["error"] = repr(e)
    return info


def texture_info(stage, ckpt):
    d = {"stage": stage, "ckpt": os.path.abspath(ckpt) if ckpt else None, "md5": md5(ckpt)}
    if ckpt:
        pj = sidecar_path(ckpt)                                    # <ckpt>.provenance.json (write_provenance)
        d["provenance_json"] = pj if os.path.exists(pj) else None
        d["provenance_json_md5"] = md5(pj)
    return d


def collect(**kw):
    import numpy, torch
    try:
        import mujoco
        mj = mujoco.__version__
    except Exception:
        mj = None
    prov = {
        "schema": "usbg-prov-1",
        "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv), "cwd": os.getcwd(), "host": platform.node(),
        "python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__,
        "mujoco": mj,
        "env": {k: os.environ.get(k) for k in ("USBG_SEAT", "USBG_LEARNED_SEG", "US_SEG_CKPT",
                                               "USBG_CONTACT", "PYTORCH_ENABLE_MPS_FALLBACK",
                                               "OMP_NUM_THREADS")},
        "git": git_info(), "seat": seat_info(), "mask": mask_info(),
    }
    if "stage" in kw or "texture_ckpt" in kw:
        prov["texture"] = texture_info(kw.pop("stage", None), kw.pop("texture_ckpt", None))
    prov["windows"] = windows_ref()
    prov["seat_mode"] = prov["seat"]["resolved"]
    prov.update(kw)
    return prov


def assert_not_withdrawn(prov, allow=False):
    m = prov.get("mask", {})
    if m.get("learned") and m.get("seg_md5") in WITHDRAWN_SEG_MD5 and not allow:
        raise SystemExit(f"refusing to run: segmenter {m.get('seg_ckpt')} (md5 {m.get('seg_md5')}) is the "
                         "withdrawn shared checkpoint (D2). Set US_SEG_CKPT to the stage/seed-matched one.")
    if m.get("learned") and not m.get("US_SEG_CKPT_env") and not allow:
        raise SystemExit("refusing to run: USBG_LEARNED_SEG=1 but US_SEG_CKPT is unset, so the default "
                         "data/dataset/us_seg_sim_real.pt would load silently. Export US_SEG_CKPT.")
