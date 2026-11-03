#!/usr/bin/env python3
"""CT-anatomy slot of the v2.1 pluggability study: TotalSegmentator v2 automatic labels as the label source.

EXPLORATORY (provenance analysis_status "exploratory, v2.1 pluggability"); the archived v2 numbers stay
primary. This file is new and opt-in: no existing script, cache, windows file or output changes behavior.

What it tests. The testbed's anatomy (the label volume the slicer, the renderers and the CT masks read) is
rebuilt from TotalSegmentator v2 predictions (task "total") on the SAME 19 raw CTs the v2 caches came from
(data/raw/totalseg_small/<case>/ct.nii.gz), instead of the dataset's reference masks. The class mapping is
exactly the existing one: usbg.volume.load_labeled_ct with exclude_structures=volume.ANAT_EXCLUDE, the call
scripts/build_anat_cache.py makes for data/cache_anat/. Each case job re-runs that call on the reference
masks and requires it to reproduce data/cache_anat/<case>.npz bit for bit before it writes anything.

CAVEAT (carried into every output): all 19 cohort CTs (13 TRAIN, 6 held out) are in TotalSegmentator's
own training split (data/raw/totalseg_small/meta.csv, split column). These are train-split predictions,
so this is a label-noise sensitivity rung, not an independent CT-segmenter test.

Seats and episodes. The v1 screened windows are KEPT, so the 60 crossed held-out episodes are identical.
usbg.windows refuses a volume whose labels differ from the screened ones (labels_md5), so the windows step
writes a NEW windows file: usbg-windows-v1.json with every record verbatim except labels_md5, which names
the TotalSegmentator labels. Every eval verifies that equivalence, and checks per episode that patient,
target, window, start pose error and clamp equal those of a v1-label reference output of the same rung
(for pose_oracle and static also the final pose error, which no label can change).

Subcommands
  case      (plugin venv) one case: TotalSegmentator v2 on the raw CT (CPU by default; settings recorded),
            mask checks, the reference-mapping reproduction check, the new bladder-free cache, label
            agreement with the v1 cache (canonical-label Dice, per-structure Dice) and a re-screen of the v1
            windows on the new labels (diagnostic only). Writes under --out-dir; report at --out.
  windows   after all cases: the derived windows file and the manifest (--out) the evals read.
  eval      run scripts/eval_view_ladder.py IN-PROCESS with every case loaded from the new caches, the
            derived windows file (USBG_WINDOWS), the episode-identity checks above and the provenance tag.
            The output file appears only if every check passes.
  analyze   paired comparison of each eval with its v1-label reference (same episodes): per-patient
            seed-paired differences, patient bootstrap CI, k/6.
  jobs      write the job spec outputs/v2_1/jobs/plug_ctanat.json for scripts/run_table4_v2.py --extra-jobs.

Telemetry and network. TotalSegmentator sends usage statistics unless $TOTALSEG_HOME_DIR/config.json says
send_usage_stats false. `case` refuses to run unless it does, replaces the package's prediction-counter write
and stats call with no-ops (so parallel jobs never write config.json), and blocks every non-local socket
connection in the process (an attempt aborts the job).

  TOTALSEG_HOME_DIR=<plugin weights>/totalseg PYTHONPATH=src USBG_SEAT=anatomical \\
    <plugin venv>/bin/python scripts/plug/totalseg_labels.py case --case s0011 \\
      --out-dir data/dataset/v2_1/plug/ctanat_ts2 --out outputs/v2_1/plug/ctanat/labels/s0011.json
  PYTHONPATH=src USBG_SEAT=anatomical .venv/bin/python scripts/plug/totalseg_labels.py jobs
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import glob
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, "scripts")
THIS = os.path.abspath(__file__)
for _p in (os.path.join(REPO, "src"), SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ANALYSIS_STATUS = "exploratory, v2.1 pluggability"
SLOT = "ct_anatomy"
METHOD = "TotalSegmentator v2 automatic labels (task total) as the label source"
CAVEAT = ("train-split TotalSegmentator predictions: all 19 cohort CTs (13 TRAIN, 6 held out) are in "
          "TotalSegmentator's own training split (data/raw/totalseg_small/meta.csv), so this is a label-noise "
          "sensitivity rung, not an independent CT-segmenter test")
CITATION = ("J. Wasserthal, H.-C. Breit, M.T. Meyer, M. Pradella, D. Hinck, A.W. Sauter, T. Heye, D.T. Boll, "
            "J. Cyriac, S. Yang, M. Bach, M. Segeroth, TotalSegmentator: robust segmentation of 104 anatomic "
            "structures in CT images, Radiol. Artif. Intell. 5 (2023) e230024. "
            "https://doi.org/10.1148/ryai.230024")

PY = os.path.join(REPO, ".venv", "bin", "python")
PLUG_PY = os.environ.get("USBG_PLUGIN_PYTHON", PY)
TS_HOME = os.path.join(os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights")), "totalseg")
WEIGHTS_MANIFEST = os.path.join(os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights")), "MANIFEST.json")
RAW = os.path.join(REPO, "data", "raw", "totalseg_small")
V1_ANAT = os.path.join(REPO, "data", "cache_anat")
V1_WINDOWS = os.path.join(REPO, "data", "cache", "windows", "usbg-windows-v1.json")
V1_WINDOWS_MD5 = "af35f0bddeb57c079981eb0ba8aca5b3"
ROOT_BASE = os.path.join(REPO, "data", "dataset", "v2_1", "plug", "ctanat_ts2")   # <mode>/: masks, caches, windows
OUT_BASE = os.path.join(REPO, "outputs", "v2_1", "plug", "ctanat")                 # <mode>/: reports, evals, summary
WINDOWS_NAME = "usbg-windows-v1-ts2labels.json"
JOBS_OUT = os.path.join(REPO, "outputs", "v2_1", "jobs", "plug_ctanat.json")
V2_TABLE4 = os.path.join(REPO, "outputs", "v2", "table4")
S21 = os.path.join(REPO, "outputs", "v2_1", "s21_rerun")
TEX_MANIFEST = os.path.join(REPO, "outputs", "v2", "textures", "manifest.json")
SEG_MANIFEST = os.path.join(os.environ.get("USBG_SEG_REPO", os.path.join(REPO, "segtrain")), "outputs", "v2",
                            "seg_manifest.json")

MODES = {"default": {"fast": False, "task_ids": [291, 292, 293, 294, 295], "resolution_mm": 1.5},
         "fast": {"fast": True, "task_ids": [297], "resolution_mm": 3.0}}
RESCREEN_ORGANS = ("liver", "kidney_right", "kidney_left")
SPEC_MODE = "fast"      # the job spec's TotalSegmentator setting (see cmd_jobs: measured, CPU, documented)
EVAL_STAGES = ("physics", "sonospade")
SEEDS = (0, 1, 2)
TARGETS = ("liver", "kidney")
MIN_WINDOWS = 2
EVAL_SEED = 999
EVAL_EPISODES = 60
FLOAT_TOL = 1e-9
POSE_TOL = 1e-6
ID_KEYS = ("patient", "target", "target_requested", "window", "n_windows", "start_clamped")
ID_FLOATS = ("clamp_mm", "start_pos", "start_ang")


# ------------------------------------------------------------------------------------------- helpers
def root_for(mode):
    return os.path.join(ROOT_BASE, mode)


def out_for(mode):
    return os.path.join(OUT_BASE, mode)


def now_utc():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def md5(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def labels_md5(labels):
    return hashlib.md5(np.ascontiguousarray(labels).tobytes()).hexdigest()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    """Atomic write of a NEW file (refuses to overwrite)."""
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path} (new files only)")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_json_default)
    os.replace(tmp, path)
    return path


def git_info():
    try:
        from usbg import provenance as P
        return P.git_info(REPO)
    except Exception as e:                       # never fail a job on provenance
        return {"error": repr(e)}


def provenance(kind, **extra):
    return {"analysis_status": ANALYSIS_STATUS, "slot": SLOT, "method": METHOD, "kind": kind, "caveat": CAVEAT,
            "citation": CITATION, "created": now_utc(), "argv": list(sys.argv), "cwd": os.getcwd(),
            "python": sys.executable, "python_version": sys.version.split()[0], "host": platform.node(),
            "seat_mode": os.environ.get("USBG_SEAT", "anatomical"), "git": git_info(),
            "script": {"path": THIS, "md5": md5(THIS)}, "archived_v2_primary": True, **extra}


def require_anatomical():
    seat = os.environ.get("USBG_SEAT", "anatomical").strip().lower()
    if seat != "anatomical":
        raise SystemExit(f"USBG_SEAT={seat!r}: the CT-anatomy slot is defined under the anatomical seat only")
    os.environ["USBG_SEAT"] = "anatomical"


def refuse_private(*paths):
    from usbg.provenance import is_private_path
    for p in paths:
        if p and is_private_path(p):
            raise SystemExit(f"refusing a private-pool path: {p}")


def cohort():
    from usbg.benchmark import TRAIN_PATIENTS, TEST_PATIENTS
    return list(TRAIN_PATIENTS), list(TEST_PATIENTS)


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else None


def load_mask(path, shape):
    import nibabel as nib
    m = nib.as_closest_canonical(nib.load(path))
    a = np.asanyarray(m.dataobj) > 0.5
    if a.shape != tuple(shape):
        raise SystemExit(f"{path}: mask shape {a.shape} != CT {tuple(shape)}")
    return a


def weights_record(mode):
    """Task ids, folders and the sha256 of each weight zip from the weights MANIFEST.json."""
    ids = MODES[mode]["task_ids"]
    man = json.load(open(WEIGHTS_MANIFEST))
    out = []
    for tid in ids:
        rec = next((f for f in man.get("files", []) if f"Dataset{tid}_" in str(f.get("path", ""))), None)
        d = sorted(glob.glob(os.path.join(TS_HOME, "nnunet", "results", f"Dataset{tid}_*")))
        out.append({"task_id": tid, "dir": d[0] if d else None, "dir_exists": bool(d),
                    "zip_sha256": (rec or {}).get("sha256"), "url": (rec or {}).get("url"),
                    "license": (rec or {}).get("license")})
    return {"weights_manifest": {"path": WEIGHTS_MANIFEST, "md5": md5(WEIGHTS_MANIFEST)}, "parts": out}


# --------------------------------------------------------------------------- TotalSegmentator guard
class _NetGuard:
    """Blocks every non-AF_UNIX socket connect in this process (local IPC stays allowed)."""
    attempts: list = []

    @classmethod
    def install(cls):
        base = socket.socket
        if getattr(base, "_usbg_guard", False):
            return

        class NoNet(base):
            _usbg_guard = True

            def connect(self, address, *a, **k):
                if self.family == getattr(socket, "AF_UNIX", None):
                    return super().connect(address, *a, **k)
                cls.attempts.append(str(address))
                raise RuntimeError(f"network access attempted: {address}")

            def connect_ex(self, address, *a, **k):
                if self.family == getattr(socket, "AF_UNIX", None):
                    return super().connect_ex(address, *a, **k)
                cls.attempts.append(str(address))
                raise RuntimeError(f"network access attempted: {address}")
        socket.socket = NoNet


def check_totalseg_env():
    """Refuse early (before any directory is made) unless the weights home and the telemetry-off config are set."""
    home = os.environ.get("TOTALSEG_HOME_DIR")
    if not home or os.path.realpath(home) != os.path.realpath(TS_HOME):
        raise SystemExit(f"TOTALSEG_HOME_DIR={home!r}: set it to {TS_HOME} (weights and telemetry-off config)")
    cfgp = os.path.join(home, "config.json")
    if not os.path.exists(cfgp) or json.load(open(cfgp)).get("send_usage_stats") is not False:
        raise SystemExit(f"{cfgp} missing or send_usage_stats not false")
    if os.environ.get("TOTALSEG_WEIGHTS_PATH"):
        raise SystemExit("TOTALSEG_WEIGHTS_PATH is set; unset it so the weights come from TOTALSEG_HOME_DIR")


def setup_totalseg(threads):
    """Telemetry off and verified, config.json never written, sockets blocked, torch threads capped."""
    check_totalseg_env()
    home = os.environ.get("TOTALSEG_HOME_DIR")
    if not home or os.path.realpath(home) != os.path.realpath(TS_HOME):
        raise SystemExit(f"TOTALSEG_HOME_DIR={home!r}: set it to {TS_HOME} (weights and telemetry-off config)")
    cfgp = os.path.join(home, "config.json")
    if not os.path.exists(cfgp):
        raise SystemExit(f"{cfgp} missing: TotalSegmentator would create one with send_usage_stats true")
    cfg = json.load(open(cfgp))
    if cfg.get("send_usage_stats") is not False:
        raise SystemExit(f"{cfgp}: send_usage_stats is {cfg.get('send_usage_stats')!r}; it must be false")
    if os.environ.get("TOTALSEG_WEIGHTS_PATH"):
        raise SystemExit("TOTALSEG_WEIGHTS_PATH is set; unset it so the weights come from TOTALSEG_HOME_DIR")
    _NetGuard.install()
    import torch
    rec = {"config_json": cfgp, "send_usage_stats": False, "config_md5_before": md5(cfgp),
           "stats_calls_suppressed": 0, "counter_writes_suppressed": 0, "threads_cap": int(threads) or None,
           "cpu_count": os.cpu_count()}
    if threads and threads > 0:
        orig = torch.set_num_threads

        def capped(n):
            orig(max(1, min(int(n), int(threads))))
        torch.set_num_threads = capped
        orig(int(threads))
    import totalsegmentator.python_api as TSA

    def no_counter():
        rec["counter_writes_suppressed"] += 1
        return dict(cfg)

    def no_stats(config, params):
        rec["stats_calls_suppressed"] += 1
    TSA.increase_prediction_counter = no_counter
    TSA.send_usage_stats = no_stats
    return TSA, rec


def run_totalseg(ct_path, seg_dir, mode, device, threads, nr_thr_saving):
    """TotalSegmentator v2 task total on one CT into seg_dir (one binary NIfTI per class)."""
    import importlib.metadata
    TSA, guard = setup_totalseg(threads)
    settings = {"tool": "TotalSegmentator", "version": importlib.metadata.version("TotalSegmentator"),
                "task": "total", "mode": mode, "fast": MODES[mode]["fast"], "fastest": False,
                "resolution_mm": MODES[mode]["resolution_mm"], "device": device, "roi_subset": None,
                "ml": False, "output_type": "nifti", "nr_thr_resamp": 1, "nr_thr_saving": int(nr_thr_saving),
                "torch_threads_cap": int(threads) or None, "tta": False,
                "call": "totalsegmentator.python_api.totalsegmentator(ct, out, task='total', fast=..., "
                        "device=..., quiet=True, nr_thr_resamp=1, nr_thr_saving=...)",
                "weights": weights_record(mode)}
    miss = [p["task_id"] for p in settings["weights"]["parts"] if not p["dir_exists"]]
    if miss:
        raise SystemExit(f"TotalSegmentator weights missing for task ids {miss} under {TS_HOME}/nnunet/results")
    import torch
    settings["torch"] = torch.__version__
    t0 = time.time()
    TSA.totalsegmentator(ct_path, seg_dir, task="total", fast=MODES[mode]["fast"], device=device, quiet=True,
                         nr_thr_resamp=1, nr_thr_saving=int(nr_thr_saving))
    settings["seconds"] = round(time.time() - t0, 1)
    try:
        import resource
        scale = 1 if sys.platform == "darwin" else 1024          # ru_maxrss: bytes on macOS, KiB on Linux
        settings["peak_rss_gb"] = {
            "self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2 ** 30, 2),
            "children_max": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale / 2 ** 30, 2)}
    except Exception:
        pass
    guard["config_md5_after"] = md5(guard["config_json"])
    guard["config_unchanged"] = guard["config_md5_after"] == guard["config_md5_before"]
    guard["network_connect_attempts"] = list(_NetGuard.attempts)
    settings["telemetry"] = guard
    if _NetGuard.attempts:
        raise SystemExit(f"network access was attempted: {_NetGuard.attempts}")
    return settings


def listing_md5(seg_dir):
    h = hashlib.md5()
    for f in sorted(os.listdir(seg_dir)):
        p = os.path.join(seg_dir, f)
        h.update(f"{f}\0{os.path.getsize(p)}\0{md5(p)}\n".encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ case
def check_masks(ct_path, seg_dir, ref_dir):
    """Every predicted mask on the CT grid (shape, affine); class set vs the dataset's."""
    import nibabel as nib
    ct = nib.load(ct_path)
    names = sorted(f[:-7] for f in os.listdir(seg_dir) if f.endswith(".nii.gz"))
    bad = []
    for n in names:
        m = nib.load(os.path.join(seg_dir, f"{n}.nii.gz"))
        if m.shape != ct.shape or not np.allclose(m.affine, ct.affine, atol=1e-4):
            bad.append(n)
    ref = sorted(f[:-7] for f in os.listdir(ref_dir) if f.endswith(".nii.gz"))
    return {"n_masks": len(names), "n_reference_masks": len(ref), "off_grid": bad,
            "missing_vs_reference": sorted(set(ref) - set(names)), "extra_vs_reference": sorted(set(names) - set(ref)),
            "ct_shape": list(ct.shape), "ct_zooms_mm": [float(z) for z in ct.header.get_zooms()],
            "ct_axcodes": list(nib.aff2axcodes(ct.affine))}


def agreement(lv_v1, lv_ts, ref_dir, ts_dir):
    """Label agreement of the new volume with the v1 volume (both built by the same call)."""
    from usbg import volume as V
    a, b = lv_v1.labels, lv_ts.labels
    n = V.N_LABELS
    conf = np.bincount(a.astype(np.int64).ravel() * n + b.astype(np.int64).ravel(), minlength=n * n).reshape(n, n)
    per_label = {}
    for k, name in V.USBG_LABEL_NAMES.items():
        va, vb = int(conf[k, :].sum()), int(conf[:, k].sum())
        per_label[name] = {"dice": (2.0 * conf[k, k] / (va + vb)) if (va + vb) else None,
                           "voxels_v1": va, "voxels_ts2": vb}
    body = (a != V.BACKGROUND) | (b != V.BACKGROUND)
    shape = a.shape
    stems = sorted(V.TOTALSEG_MAP)
    per_struct = {}
    for s in stems:
        pr, pt = os.path.join(ref_dir, f"{s}.nii.gz"), os.path.join(ts_dir, f"{s}.nii.gz")
        if not (os.path.exists(pr) and os.path.exists(pt)):
            per_struct[s] = {"dice": None, "present_ref": os.path.exists(pr), "present_ts2": os.path.exists(pt)}
            continue
        mr, mt = load_mask(pr, shape), load_mask(pt, shape)
        per_struct[s] = {"dice": dice(mr, mt), "voxels_ref": int(mr.sum()), "voxels_ts2": int(mt.sum())}

    def bone_union(d):
        u = np.zeros(shape, bool)
        for f in os.listdir(d):
            st = f[:-7] if f.endswith(".nii.gz") else os.path.splitext(f)[0]
            if f.endswith(".nii.gz") and st.startswith(V.BONE_PREFIXES):
                u |= load_mask(os.path.join(d, f), shape)
        return u
    per_struct["bone_union"] = {"dice": dice(bone_union(ref_dir), bone_union(ts_dir))}
    return {"canonical_label_dice": per_label,
            "voxels_changed": int((a != b).sum()), "voxels_body": int(body.sum()),
            "frac_body_voxels_changed": float((a != b)[body].mean()) if body.any() else None,
            "confusion_v1_rows_ts2_cols": conf.tolist(), "label_order": [V.USBG_LABEL_NAMES[k] for k in range(n)],
            "structure_dice_vs_reference_masks": per_struct}


def rescreen(lv, raw_root, rec, organs=RESCREEN_ORGANS):
    """Re-screen each stored v1 window of (case, organ) on lv's labels (usbg.windows screen, flat-skin and
    landmark rules). Diagnostic only: the evaluation keeps the v1 windows."""
    from usbg import windows as W
    case = W._Case(lv, raw_root=raw_root)
    depth_m = float(W.V_IMG_DEPTH())
    grid, _ = W.scan_grid(depth_m)
    out = {}
    for organ in organs:
        tmask = case.target_mask(organ)
        rows = []
        for m in rec["windows"].get(organ, []):
            T = np.asarray(m["T_vp"], np.float64)
            s = W._screen(case, T, tmask, grid, depth_m)
            nl = sum(1 for k, f in s["organ_frac"].items() if k != W._ORGAN_LABEL[organ] and f >= W.LANDMARK_MIN_FRAC)
            gmax, gmin = case.curvature_gap(T)
            flat = bool(np.isfinite(gmax) and gmax <= W.FLAT_BELOW_MM and gmin >= -W.FLAT_ABOVE_MM)
            ok = bool(W._passes(s, W.FAMILIES[organ]["min_frac"]) and nl >= W.MIN_LANDMARKS.get(organ, 0) and flat)
            rows.append({"window_id": m.get("window_id"), "passes": ok, "frac": s["frac"], "frac_v1": m["frac"],
                         "clean_frac": s["clean_frac"], "clean_frac_v1": m["clean_frac"], "lit20": s["lit20"],
                         "lit20_v1": m["lit20"], "center_ok": s["center_ok"], "face_lab": s["face_lab"],
                         "n_landmarks": nl, "flat": flat})
        out[organ] = rows
    return out


def rescreen_summary(rows_by_organ, top=5):
    out = {}
    for organ, rows in rows_by_organ.items():
        d = [abs(r["frac"] - r["frac_v1"]) for r in rows]
        out[organ] = {"n": len(rows), "n_pass": sum(r["passes"] for r in rows),
                      "n_pass_top5": sum(r["passes"] for r in rows[:top]), "n_top5": len(rows[:top]),
                      "max_abs_dfrac": max(d) if d else None, "mean_abs_dfrac": float(np.mean(d)) if d else None}
    return out


def cmd_case(a):
    require_anatomical()
    from usbg import volume as V
    train, test = cohort()
    case = a.case
    if case not in train + test:
        raise SystemExit(f"{case} is not one of the 19 cohort cases")
    if a.smoke and case not in train:
        raise SystemExit("smoke runs use TRAIN patients only")
    if a.device != "cpu" and not a.smoke:
        print(f"[case] note: device {a.device} (the spec default is cpu)")
    ct = os.path.join(a.raw_root, case, "ct.nii.gz")
    ref_dir = os.path.join(a.raw_root, case, "segmentations")
    v1_cache = os.path.join(a.v1_anat, f"{case}.npz")
    refuse_private(ct, ref_dir, v1_cache, a.out_dir, a.out)
    for p in (ct, ref_dir, v1_cache, a.v1_windows):
        if not os.path.exists(p):
            raise SystemExit(f"missing input {p}")
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    case_dir = os.path.join(a.out_dir, "masks", case)
    seg_dir = os.path.join(case_dir, "segmentations")
    seg_rec = os.path.join(case_dir, "segment.json")
    cache = os.path.join(a.out_dir, "cache_anat", f"{case}.npz")
    t_all = time.time()

    # 1. TotalSegmentator (or reuse a COMPLETE, verified earlier run of this case; never overwrite)
    if os.path.isdir(seg_dir):
        if not os.path.exists(seg_rec):
            raise SystemExit(f"{seg_dir} exists without its segment.json (an interrupted run): remove it by hand")
        prev = json.load(open(seg_rec))
        if prev.get("listing_md5") != listing_md5(seg_dir) or prev.get("settings", {}).get("mode") != a.mode:
            raise SystemExit(f"{seg_dir} does not match its segment.json (or another --mode): remove it by hand")
        seg = dict(prev, reused=True)
        print(f"[case] {case}: reusing verified masks {seg_dir}")
    else:
        check_totalseg_env()
        if os.path.exists(case_dir) and [f for f in os.listdir(case_dir) if f != "segmentations.partial"]:
            raise SystemExit(f"{case_dir} exists and is not empty: remove it by hand")
        tmp = seg_dir + ".partial"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)                      # our own scratch from an interrupted run, never data
        os.makedirs(tmp)
        print(f"[case] {case}: TotalSegmentator task total, mode {a.mode}, device {a.device}, threads {a.threads}",
              flush=True)
        settings = run_totalseg(ct, tmp, a.mode, a.device, a.threads, a.nr_thr_saving)
        checks = check_masks(ct, tmp, ref_dir)
        if checks["off_grid"]:
            raise SystemExit(f"masks off the CT grid: {checks['off_grid'][:5]}")
        lost = [s for s in sorted(V.TOTALSEG_MAP) if not os.path.exists(os.path.join(tmp, f"{s}.nii.gz"))]
        if lost:
            raise SystemExit(f"TotalSegmentator wrote no mask for mapped structures {lost}")
        os.replace(tmp, seg_dir)
        seg = {"case": case, "ct": {"path": ct, "md5": md5(ct)}, "settings": settings, "mask_checks": checks,
               "seg_dir": seg_dir, "listing_md5": listing_md5(seg_dir), "created": now_utc(), "reused": False,
               "provenance": provenance("segment", case=case)}
        write_json(seg_rec, seg)
        print(f"[case] {case}: {checks['n_masks']} masks in {settings['seconds']} s", flush=True)

    # 2. the existing mapping must reproduce the v1 bladder-free cache from the reference masks
    tb = time.time()
    lv_ref = V.load_labeled_ct(ct, ref_dir, exclude_structures=V.ANAT_EXCLUDE)
    old = np.load(v1_cache, allow_pickle=False)
    same = (old["labels"].shape == lv_ref.labels.shape and np.array_equal(old["labels"], lv_ref.labels)
            and np.allclose(old["affine_m"], lv_ref.affine_m))
    if not same:
        raise SystemExit(f"{case}: volume.load_labeled_ct on the reference masks no longer reproduces {v1_cache}; "
                         "the mapping drifted, so the new cache would not be built 'as the existing build does'")
    lv_ref.meta["case"] = case

    # 3. the new cache: the SAME call on the TotalSegmentator masks
    lv = V.load_labeled_ct(ct, seg_dir, exclude_structures=V.ANAT_EXCLUDE)
    if lv.labels.shape != lv_ref.labels.shape or not np.allclose(lv.affine_m, lv_ref.affine_m):
        raise SystemExit(f"{case}: new volume grid differs from the v1 cache")
    lv.meta["case"] = case
    lv.meta["anat_cache"] = {"excluded": list(V.ANAT_EXCLUDE), "builder": "volume.load_labeled_ct (as "
                             "scripts/build_anat_cache.py)", "v1_twin": v1_cache, "v1_twin_md5": md5(v1_cache)}
    lv.meta["label_source"] = {"method": METHOD, "seg_dir": seg_dir, "listing_md5": seg["listing_md5"],
                               "settings": {k: v for k, v in seg["settings"].items() if k != "telemetry"},
                               "caveat": CAVEAT, "analysis_status": ANALYSIS_STATUS}
    build_sec = time.time() - tb

    # 4. agreement and window re-screen (diagnostics; they never gate the cache)
    td = time.time()
    agr = agreement(lv_ref, lv, ref_dir, seg_dir)
    W1 = json.load(open(a.v1_windows))
    rec = W1["cases"][case]
    if rec["labels_md5"] != labels_md5(lv_ref.labels):
        raise SystemExit(f"{case}: the v1 windows were not screened on the v1 cache labels")
    rs_v1 = rescreen(lv_ref, a.raw_root, rec)
    rs_ts = rescreen(lv, os.path.join(a.out_dir, "masks"), rec)
    repro = {o: {"n_pass": sum(r["passes"] for r in rows), "n": len(rows),
                 "max_abs_dfrac_vs_stored": max([abs(r["frac"] - r["frac_v1"]) for r in rows], default=0.0)}
             for o, rows in rs_v1.items()}
    diag_sec = time.time() - td

    # 5. the cache is written last; an existing one (a rerun after a later failure) is reused only if identical
    cache_reused = False
    if os.path.exists(cache):
        old_ts = np.load(cache, allow_pickle=False)
        if not (np.array_equal(old_ts["labels"], lv.labels) and np.allclose(old_ts["affine_m"], lv.affine_m)):
            raise SystemExit(f"refusing to overwrite {cache}: it holds different labels")
        cache_reused = True
    else:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmpc = cache[:-4] + ".partial.npz"
        V.save_cache(lv, tmpc)
        os.replace(tmpc, cache)

    rep = {"case": case, "split": "TRAIN" if case in train else "held_out",
           "cache": {"path": cache, "md5": md5(cache), "reused_identical": cache_reused, "labels_md5": labels_md5(lv.labels),
                     "labels_md5_v1": labels_md5(lv_ref.labels), "shape": list(lv.labels.shape)},
           "v1_cache": {"path": v1_cache, "md5": md5(v1_cache), "reproduced_by_mapping": True},
           "segment": {k: seg[k] for k in ("settings", "mask_checks", "seg_dir", "listing_md5", "reused") if k in seg},
           "label_counts": lv.meta.get("label_counts"), "label_counts_v1": lv_ref.meta.get("label_counts"),
           "agreement": agr,
           "window_rescreen": {"note": "v1 windows re-screened on the new labels with usbg.windows rules; the "
                                       "evaluation keeps the v1 windows (diagnostic only)",
                               "v1_labels_reproduce_stored": repro, "ts2": rs_ts, "ts2_summary": rescreen_summary(rs_ts)},
           "seconds": {"total": round(time.time() - t_all, 1), "build": round(build_sec, 1),
                       "diagnostics": round(diag_sec, 1),
                       "totalsegmentator": seg["settings"].get("seconds")},
           "smoke": bool(a.smoke),
           "provenance": provenance("case", case=case, out_dir=os.path.abspath(a.out_dir))}
    write_json(a.out, rep)
    ld = rep["agreement"]["canonical_label_dice"]
    print(f"[case] {case}: cache {cache}; Dice liver {ld['liver']['dice']}, kidney {ld['kidney']['dice']}; "
          f"rescreen {rep['window_rescreen']['ts2_summary']}; {rep['seconds']['total']} s -> {a.out}")
    return 0


# --------------------------------------------------------------------------------------------- windows
def derive_windows(v1, ts2_md5):
    """usbg-windows-v1 with labels_md5 of the named cases replaced; nothing else changes."""
    d = copy.deepcopy(v1)
    for c, m in ts2_md5.items():
        d["cases"][c]["labels_md5"] = m
    return d


def windows_equivalence(v1_path, derived_path, ts2_md5):
    """ok iff same params and cases, and every record equals the v1 record except labels_md5, which is the
    new labels' md5 for the swapped cases and unchanged for the others."""
    a, b = json.load(open(v1_path)), json.load(open(derived_path))
    why = []
    if a.get("params") != b.get("params"):
        why.append("window params differ")
    if set(a["cases"]) != set(b["cases"]):
        why.append("case sets differ")
    for c in sorted(set(a["cases"]) & set(b["cases"])):
        ra, rb = a["cases"][c], b["cases"][c]
        if {k: v for k, v in ra.items() if k != "labels_md5"} != {k: v for k, v in rb.items() if k != "labels_md5"}:
            why.append(f"{c}: window records differ")
        want = ts2_md5.get(c, ra["labels_md5"])
        if rb.get("labels_md5") != want:
            why.append(f"{c}: labels_md5 {rb.get('labels_md5')} != {want}")
    return {"ok": not why, "why": why or "every record equals v1 except labels_md5 of the swapped cases",
            "v1": {"path": v1_path, "md5": md5(v1_path)}, "derived": {"path": derived_path, "md5": md5(derived_path)},
            "swapped_cases": sorted(ts2_md5)}


def cmd_windows(a):
    require_anatomical()
    train, test = cohort()
    cases = a.cases or (train + test)
    if a.partial and not a.smoke:
        raise SystemExit("--partial is for smoke tests only")
    if not a.partial and sorted(cases) != sorted(train + test):
        raise SystemExit("the windows file must swap all 19 cases (use --partial --smoke for a subset)")
    if md5(a.v1_windows) != V1_WINDOWS_MD5:
        raise SystemExit(f"{a.v1_windows} md5 {md5(a.v1_windows)} != {V1_WINDOWS_MD5}")
    wpath = os.path.join(a.out_dir, WINDOWS_NAME)
    for p in (wpath, a.out):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    v1 = json.load(open(a.v1_windows))
    paths, ts2_md5, reps, settings = {}, {}, {}, {}
    for c in cases:
        cp = os.path.join(a.out_dir, "cache_anat", f"{c}.npz")
        rp = os.path.join(a.reports_dir, f"{c}.json")
        if not (os.path.exists(cp) and os.path.exists(rp)):
            raise SystemExit(f"{c}: cache {cp} or report {rp} missing (run the case job first)")
        r = json.load(open(rp))
        if r["cache"]["md5"] != md5(cp):
            raise SystemExit(f"{c}: {cp} changed since its case report")
        lab = np.load(cp, allow_pickle=False)["labels"]
        m = labels_md5(lab)
        if m != r["cache"]["labels_md5"]:
            raise SystemExit(f"{c}: labels md5 differs from the case report")
        if c not in v1["cases"]:
            raise SystemExit(f"{c} not in {a.v1_windows}")
        ts2_md5[c] = m
        paths[c] = {"path": cp, "md5": md5(cp), "labels_md5": m, "labels_md5_v1": v1["cases"][c]["labels_md5"],
                    "report": {"path": rp, "md5": md5(rp)}}
        reps[c] = r
        s = r["segment"]["settings"]
        settings[c] = {k: s.get(k) for k in ("version", "task", "mode", "fast", "device", "torch_threads_cap",
                                             "nr_thr_saving", "resolution_mm")}
    uniq = {json.dumps(v, sort_keys=True) for v in settings.values()}
    if len(uniq) != 1:
        raise SystemExit(f"cases were segmented with different settings: {settings}")
    d = derive_windows(v1, ts2_md5)
    d["ctanat_ts2"] = {"derived_from": {"path": a.v1_windows, "md5": md5(a.v1_windows)},
                       "rule": "every record verbatim from usbg-windows-v1 except labels_md5, which names the "
                               "TotalSegmentator-label cache; the screened windows (T_vp) are unchanged",
                       "swapped_cases": sorted(ts2_md5), "analysis_status": ANALYSIS_STATUS, "caveat": CAVEAT,
                       "partial": bool(a.partial)}
    os.makedirs(a.out_dir, exist_ok=True)
    tmp = wpath + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, wpath)
    eq = windows_equivalence(a.v1_windows, wpath, ts2_md5)
    if not eq["ok"]:
        os.replace(wpath, wpath + ".failed")
        raise SystemExit(f"derived windows file is not equivalent: {eq['why']}")
    # the windows module itself accepts the new labels with the derived file (labels md5 check, params)
    os.environ["USBG_WINDOWS"] = wpath
    from usbg import windows as W
    from usbg import volume as V
    W.WINDOWS_PATH = wpath
    W.load_file(wpath)
    for c in cases[:1]:
        lv = V.load_cache(paths[c]["path"])
        lv.meta["case"] = c
        n = len(W.windows_meta(lv, V.LIVER))
        if n != len(v1["cases"][c]["windows"].get("liver", [])):
            raise SystemExit(f"{c}: windows_meta returned {n} liver windows")
    agg = summarize_cases(reps)
    man = {"slot": SLOT, "method": METHOD, "caveat": CAVEAT, "cases": cases, "partial": bool(a.partial),
           "label_root": os.path.abspath(a.out_dir), "cache_paths": paths,
           "windows": {"path": wpath, "md5": md5(wpath), "equivalence": eq},
           "label_source_settings": next(iter(settings.values())), "summary": agg,
           "smoke": bool(a.smoke), "provenance": provenance("manifest")}
    write_json(a.out, man)
    print(f"[windows] {wpath} (md5 {md5(wpath)}); manifest -> {a.out}")
    print(json.dumps(agg["canonical_label_dice_mean"], indent=0, default=_json_default)[:800])
    return 0


def summarize_cases(reps):
    names = None
    dl, ds, rs, sec = {}, {}, {}, []
    for c, r in reps.items():
        ag = r["agreement"]
        names = names or list(ag["canonical_label_dice"])
        for n, v in ag["canonical_label_dice"].items():
            if v["dice"] is not None:
                dl.setdefault(n, []).append(v["dice"])
        for s, v in ag["structure_dice_vs_reference_masks"].items():
            if v.get("dice") is not None:
                ds.setdefault(s, []).append(v["dice"])
        for o, v in r["window_rescreen"]["ts2_summary"].items():
            x = rs.setdefault(o, {"n": 0, "n_pass": 0, "n_top5": 0, "n_pass_top5": 0})
            for k in x:
                x[k] += v[k]
        sec.append(r["seconds"].get("totalsegmentator") or 0.0)
    f = lambda xs: {"mean": float(np.mean(xs)), "min": float(np.min(xs)), "n": len(xs)}
    return {"canonical_label_dice_mean": {n: f(v) for n, v in dl.items()},
            "structure_dice_mean": {n: f(v) for n, v in ds.items()},
            "frac_body_voxels_changed": f([r["agreement"]["frac_body_voxels_changed"] for r in reps.values()]),
            "window_rescreen_ts2": rs, "totalsegmentator_seconds": f(sec)}


# ------------------------------------------------------------------------------------------------ eval
def episode_identity(out, ref, rung):
    """Per-episode identity of the setup (and, for rungs whose pose cannot depend on labels, of the pose
    error) between this run and its v1-label reference. -> (problems, report)."""
    probs, rep = [], {"blocks": []}
    rb = {(r.get("weights"), r.get("contact"), r.get("eval_randomize")): r for r in ref.get("results", [])}
    for r in out.get("results", []):
        key = (r.get("weights"), r.get("contact"), r.get("eval_randomize"))
        q = rb.get(key)
        if q is None:
            probs.append(f"no reference block for {key}")
            continue
        ea, eb = r.get("episodes") or [], q.get("episodes") or []
        if len(ea) != len(eb) or not ea:
            probs.append(f"{key}: {len(ea)} episodes vs reference {len(eb)}")
            continue
        n_bad, worst = 0, 0.0
        for x, y in zip(ea, eb):
            if any(x.get(k) != y.get(k) for k in ID_KEYS):
                n_bad += 1
                continue
            d = max(abs(float(x[k]) - float(y[k])) for k in ID_FLOATS)
            worst = max(worst, d)
            if d > FLOAT_TOL:
                n_bad += 1
        blk = {"key": list(key), "n": len(ea), "setup_mismatches": n_bad, "max_abs_setup_diff": worst}
        if rung in ("pose_oracle", "static"):
            dp = max(max(abs(float(x["final_pos"]) - float(y["final_pos"])),
                         abs(float(x["final_ang"]) - float(y["final_ang"]))) for x, y in zip(ea, eb))
            blk["max_abs_final_pose_diff"] = dp
            if dp > POSE_TOL:
                probs.append(f"{key}: final pose differs from the reference by {dp} (a {rung} pose cannot depend on labels)")
        if n_bad:
            probs.append(f"{key}: {n_bad} episodes differ in setup from the reference")
        rep["blocks"].append(blk)
    rep["ok"] = not probs
    return probs, rep


def cmd_eval(a):
    require_anatomical()
    rest = list(a.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    for f in ("--out", "--allow-mismatch", "--analysis-status", "--no-episodes"):
        if f in rest:
            raise SystemExit(f"{f} is set by the CT-anatomy wrapper (or not allowed); remove it")
    if "--rung" not in rest:
        raise SystemExit("pass eval_view_ladder arguments after --, including --rung")
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    rung = rest[rest.index("--rung") + 1]
    man = json.load(open(a.manifest))
    if man.get("partial") and not a.smoke:
        raise SystemExit("a partial (smoke) manifest can only be used with --smoke")
    if man.get("smoke") and not a.smoke:
        raise SystemExit("a smoke manifest can only be used with --smoke")
    wpath = os.environ.get("USBG_WINDOWS")
    if not wpath or os.path.realpath(wpath) != os.path.realpath(man["windows"]["path"]):
        raise SystemExit(f"USBG_WINDOWS={wpath!r}: set it to the derived windows file {man['windows']['path']}")
    if md5(wpath) != man["windows"]["md5"]:
        raise SystemExit(f"{wpath} md5 {md5(wpath)} != manifest {man['windows']['md5']}")
    ts2_md5 = {c: v["labels_md5"] for c, v in man["cache_paths"].items()}
    eq = windows_equivalence(V1_WINDOWS, wpath, ts2_md5)
    if not eq["ok"]:
        raise SystemExit(f"derived windows file is not equivalent to usbg-windows-v1: {eq['why']}")
    ref = None
    if a.ref:
        ref = json.load(open(a.ref))
    elif not a.smoke:
        raise SystemExit("--ref (the v1-label output of the same rung) is required outside --smoke")
    from usbg import provenance as P
    tag = {"analysis_status": ANALYSIS_STATUS, "slot": SLOT, "method": METHOD, "caveat": CAVEAT,
           "manifest": {"path": os.path.abspath(a.manifest), "md5": md5(a.manifest)},
           "label_root": man["label_root"], "label_source_settings": man.get("label_source_settings"),
           "windows_equivalence": eq, "ref": {"path": os.path.abspath(a.ref), "md5": md5(a.ref)} if a.ref else None,
           "wrapper_argv": list(sys.argv), "smoke": bool(a.smoke), "loaded": {}, "fallback_v1_cases": [],
           "adjustments": ["every train and eval case loads from the TotalSegmentator-label cache (manifest md5s)",
                           "USBG_WINDOWS = the derived windows file (v1 records except labels_md5)",
                           "policy rung: the windows md5 mismatch vs the run is accepted only for that verified "
                           "equivalence; seat, mask, segmenter and texture mismatches still abort",
                           "post-run: per-episode setup identity with --ref; pose identity for pose_oracle/static"]}
    oc = P.collect

    def collect(**kw):
        prov = oc(**kw)
        prov["analysis_status"] = ANALYSIS_STATUS
        prov["v21_plug_ctanat"] = tag
        return prov
    P.collect = collect
    import train_view_match_rl as T
    import eval_view_ladder as EVL
    from usbg import volume as V

    def load_volumes(cases):
        lvs = []
        for c in cases:
            rec = man["cache_paths"].get(c)
            if rec is None:
                if not a.smoke:
                    raise SystemExit(f"{c} has no TotalSegmentator-label cache in {a.manifest}")
                path = os.path.join(REPO, "data", "cache", f"{c}.npz")
                lv = V.load_cache(path)                     # smoke only: v1 bladder-free twin
                lv.meta.setdefault("case", c)
                if c not in tag["fallback_v1_cases"]:
                    tag["fallback_v1_cases"].append(c)
                lvs.append(lv)
                continue
            if md5(rec["path"]) != rec["md5"]:
                raise SystemExit(f"{c}: {rec['path']} changed since the manifest")
            lv = V.load_cache(rec["path"])
            if labels_md5(lv.labels) != rec["labels_md5"]:
                raise SystemExit(f"{c}: labels md5 differs from the manifest")
            lv.meta["case"] = c
            lv.meta["cache_variant"] = "anat_totalseg_v2"
            tag["loaded"][c] = {"path": rec["path"], "md5": rec["md5"], "labels_md5": rec["labels_md5"]}
            lvs.append(lv)
        return lvs
    T.load_volumes = load_volumes

    if rung == "policy":
        orig_mm = EVL.provenance_mismatch

        def provenance_mismatch(run_prov, now_prov):
            keep = []
            for key, ra, rb in orig_mm(run_prov, now_prov):
                if key == "windows_md5" and ra == V1_WINDOWS_MD5 and rb == man["windows"]["md5"] and eq["ok"]:
                    acc = {"run": ra, "now": rb, "reason": "derived windows file verified equal to the run's "
                                                          "usbg-windows-v1 except labels_md5"}
                    tag["windows_md5_mismatch_accepted"] = acc
                    continue
                keep.append((key, ra, rb))
            return keep
        EVL.provenance_mismatch = provenance_mismatch

    partial = a.out[:-5] + ".partial.json"
    for p in (partial, partial[:-5] + ".traj.npz"):
        if os.path.exists(p):
            os.remove(p)                               # our own scratch from an interrupted run
    argv = [*rest, "--analysis-status", ANALYSIS_STATUS, "--out", partial]
    sys.argv = [os.path.join(SCRIPTS, "eval_view_ladder.py"), *argv]
    print(f"[ctanat eval] {rung}; argv: {' '.join(argv)}", flush=True)
    rc = EVL.main(argv)
    if rc not in (0, None):
        return rc
    d = json.load(open(partial))
    prov = d.get("provenance") or {}
    train, test = cohort()
    probs = []
    if prov.get("analysis_status") != ANALYSIS_STATUS or d.get("analysis_status") != ANALYSIS_STATUS:
        probs.append("analysis_status tag missing")
    if "v21_plug_ctanat" not in prov:
        probs.append("provenance has no v21_plug_ctanat block")
    if (prov.get("windows") or {}).get("md5") != man["windows"]["md5"]:
        probs.append("windows md5 in provenance != derived windows md5")
    used = set(prov.get("train_cases") or []) | set(prov.get("eval_cases") or [])
    if rung == "policy":
        used = set(prov.get("eval_cases") or [])            # the policy rung never touches TRAIN volumes
    miss = sorted(c for c in used if c not in tag["loaded"] and c not in tag["fallback_v1_cases"])
    if miss:
        probs.append(f"cases not loaded through the label swap: {miss}")
    if not a.smoke:
        if tag["fallback_v1_cases"]:
            probs.append(f"v1 caches were loaded for {tag['fallback_v1_cases']}")
        if sorted(prov.get("eval_cases") or []) != sorted(test):
            probs.append(f"eval cases {prov.get('eval_cases')} != the 6 held-out patients")
        if prov.get("eval_design") != "crossed" or int(prov.get("eval_seed", -1)) != EVAL_SEED:
            probs.append("not the crossed design with eval seed 999")
        for r in d.get("results", []):
            if r.get("n_episodes") != EVAL_EPISODES or len(r.get("episodes") or []) != EVAL_EPISODES:
                probs.append(f"{r.get('weights')}/{r.get('contact')}: {r.get('n_episodes')} episodes != {EVAL_EPISODES}")
    ident = None
    if ref is not None:
        p2, ident = episode_identity(d, ref, rung)
        probs += p2
    if rung == "policy" and "windows_md5_mismatch_accepted" not in tag:
        probs.append("policy rung: the windows equivalence was not exercised (expected a v1 windows md5 in the run)")
    d["provenance"]["v21_plug_ctanat"] = json.loads(json.dumps(tag, default=_json_default))
    d["provenance"]["v21_plug_ctanat"]["episode_identity"] = ident
    d["ctanat"] = {"label_source": "totalseg_v2", "caveat": CAVEAT, "ref": tag["ref"], "episode_identity_ok":
                   (ident or {}).get("ok"), "smoke": bool(a.smoke)}
    if probs:
        with open(a.out[:-5] + ".failed.json", "w") as fh:
            json.dump(d, fh, indent=1, default=_json_default)
        os.remove(partial)
        tj = (d.get("traj_npz") or {}).get("path")
        if tj and os.path.exists(tj):
            os.replace(tj, a.out[:-5] + ".failed.traj.npz")
        raise SystemExit("post-run checks failed (output kept as .failed.json): " + "; ".join(probs))
    d["provenance"]["v21_plug_ctanat"]["post_run_checks"] = (
        "passed: analysis_status, label swap for every case, derived windows md5, "
        + ("" if a.smoke else "6 held-out x 60 crossed episodes (seed 999), ")
        + ("per-episode setup identity with the reference" if ref is not None else "no reference (smoke)"))
    tj = d.get("traj_npz") or {}
    if tj.get("path") and os.path.exists(tj["path"]):
        final_tj = a.out[:-5] + ".traj.npz"
        if os.path.exists(final_tj):
            raise SystemExit(f"refusing to overwrite {final_tj}")
        os.replace(tj["path"], final_tj)
        d["traj_npz"]["path"] = os.path.abspath(final_tj)
    write_json(a.out, d)
    os.remove(partial)
    print(f"[ctanat eval] verified -> {a.out}")
    return 0


# --------------------------------------------------------------------------------------------- analyze
def _boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cmd_analyze(a):
    man = json.load(open(a.manifest))
    files = sorted(f for f in glob.glob(os.path.join(a.eval_dir, "*", "*.json"))
                   if not f.endswith((".partial.json", ".failed.json", ".tmp")))
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    rows, conds = [], {}
    metrics = ("final_pos", "final_ang", "keepbest_pos", "final_seg")
    for f in files:
        d = json.load(open(f))
        tg = (d.get("provenance") or {}).get("v21_plug_ctanat")
        if not tg or d.get("analysis_status") != ANALYSIS_STATUS:
            continue
        if tg.get("smoke") and not a.smoke:
            raise SystemExit(f"{f} is a smoke output")
        ref = json.load(open(tg["ref"]["path"]))
        if md5(tg["ref"]["path"]) != tg["ref"]["md5"]:
            raise SystemExit(f"reference {tg['ref']['path']} changed since {f}")
        rb = {(r.get("weights"), r.get("contact")): r for r in ref["results"]}
        for r in d["results"]:
            q = rb[(r.get("weights"), r.get("contact"))]
            ea, eb = r["episodes"], q["episodes"]
            row = {"file": f, "rung": d["rung"], "translator": d["translator"], "mask_source": d["mask_source"],
                   "seed": d.get("seed"), "weights": r["weights"], "contact": r["contact"], "n": len(ea), "metrics": {}}
            for m in metrics:
                xa = np.array([e[m] for e in ea], float)
                xb = np.array([e[m] for e in eb], float)
                pats = sorted({e["patient"] for e in ea})
                pd = {p: float(np.mean([xa[i] - xb[i] for i, e in enumerate(ea) if e["patient"] == p])) for p in pats}
                row["metrics"][m] = {"ts2": float(xa.mean()), "v1": float(xb.mean()), "diff": float((xa - xb).mean()),
                                     "per_patient_diff": pd}
                ck = (d["rung"], d["translator"], d["mask_source"], r["weights"], r["contact"], m)
                for p, v in pd.items():
                    conds.setdefault(ck, {}).setdefault(p, []).append(v)
            rows.append(row)
    cond_out = []
    for (rung, tr, ms, w, ct, m), per in sorted(conds.items()):
        pm = {p: float(np.mean(v)) for p, v in per.items()}          # seed-paired per-patient differences
        x = np.array(list(pm.values()))
        ci = _boot_ci(x)
        cond_out.append({"rung": rung, "translator": tr, "mask_source": ms, "weights": w, "contact": ct, "metric": m,
                         "n_patients": len(x), "n_seeds": max(len(v) for v in per.values()),
                         "mean_diff_ts2_minus_v1": float(x.mean()), "ci95_patient_bootstrap": ci,
                         "k_positive": int((x > 0).sum()), "k_of": len(x),
                         "ci_excludes_0": bool(np.isfinite(ci).all() and (ci[0] > 0 or ci[1] < 0)),
                         "per_patient": pm})
    out = {"slot": SLOT, "method": METHOD, "caveat": CAVEAT, "n_files": len({r["file"] for r in rows}),
           "rows": rows, "conditions": cond_out, "label_agreement": man.get("summary"),
           "stats": "difference = TotalSegmentator labels minus v1 reference labels on identical episodes; per "
                    "patient, averaged over episodes then seeds; 10,000-resample patient bootstrap (seed 0); k/6 "
                    "patients with a positive difference",
           "smoke": bool(a.smoke), "provenance": provenance("analysis", manifest={"path": a.manifest, "md5": md5(a.manifest)})}
    write_json(a.out, out)
    print(f"[analyze] {len(rows)} blocks from {out['n_files']} files -> {a.out}")
    return 0


# ------------------------------------------------------------------------------------------------ jobs
MODE_WHY = {
    "fast": ("TotalSegmentator's documented CPU setting (README: --fast for CPU; survey: 'CPU --fast'). Measured on "
             "TRAIN case s1373 while the v2.1 queue ran 3 jobs: the 1.5 mm default (5 models) on CPU with 5 threads "
             "had not finished after 26 min wall (about 12 CPU-min, heavy swapping on 24 GB) and was stopped; the "
             "download role measured --fast on CPU at 93 s for s0011 on a quiet machine"),
    "default": "TotalSegmentator's default 1.5 mm model (Dataset291-295); slow and memory-heavy on CPU here",
}


def label_est_h(case, mode, override=None):
    """Hours for one label job, scaled by CT voxel count. --fast smoke on s1373 (16.9 M voxels, 4 sliding-window
    patches, 5 threads, the v2.1 queue running): 60 s TotalSegmentator + 20 s build and diagnostics; patches grow
    about linearly with voxels. The 1.5 mm default needs 5 models x up to 128 patches of about 20 s on CPU."""
    if override:
        return float(override)
    import nibabel as nib
    sh = nib.load(os.path.join(RAW, case, "ct.nii.gz")).shape
    mvox = sh[0] * sh[1] * sh[2] / 1e6
    return round(0.02 + (0.0025 if mode == "fast" else 0.12) * mvox, 3)


def cmd_jobs(a):
    train, test = cohort()
    tex = json.load(open(TEX_MANIFEST))
    seg = json.load(open(SEG_MANIFEST))
    mode = a.mode
    root, outd = root_for(mode), out_for(mode)
    P = f"v21:plug:ctanat:{mode}"
    base_env = {"PYTHONPATH": f"{REPO}/src", "USBG_SEAT": "anatomical", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    note = (f"CT-anatomy slot (survey: must): {METHOD}, mode {mode} ({MODES[mode]['resolution_mm']} mm), CPU. "
            f"CAVEAT: {CAVEAT}. Seats and the 60 crossed held-out episodes are the v1 ones (derived windows file = "
            f"usbg-windows-v1 except labels_md5; every eval checks per-episode setup identity with its v1-label "
            f"reference).")
    wout = os.path.join(root, WINDOWS_NAME)
    man_out = os.path.join(outd, "manifest.json")
    jobs, prev = [], None
    for c in train + test:
        out = os.path.join(outd, "labels", f"{c}.json")
        cmd = [PLUG_PY, THIS, "case", "--case", c, "--mode", mode, "--device", "cpu", "--threads", str(a.threads),
               "--nr-thr-saving", str(a.nr_thr_saving), "--out-dir", root, "--out", out]
        jobs.append({"name": f"{P}:label:{c}", "cmd": cmd, "out": out, "est_h": label_est_h(c, mode, a.label_est_h),
                     "env": {**base_env, "TOTALSEG_HOME_DIR": TS_HOME, "USBG_WINDOWS": ""},
                     "deps": [prev] if prev else [],
                     "files": [os.path.join(RAW, c, "ct.nii.gz"), os.path.join(RAW, c, "segmentations"),
                               os.path.join(V1_ANAT, f"{c}.npz"), V1_WINDOWS, os.path.join(TS_HOME, "config.json")],
                     "expect": {}, "cwd": REPO, "slot": SLOT, "survey_status": "must",
                     "analysis_status": ANALYSIS_STATUS,
                     "note": note + f" Label build: TotalSegmentator v2 task total, mode {mode} ({MODE_WHY[mode]}), "
                                    f"CPU, {a.threads} torch threads, plugin venv, telemetry off and network blocked. "
                                    "The 19 label jobs are chained so only one TotalSegmentator process runs at a "
                                    "time (memory)."})
        prev = jobs[-1]["name"]
    labels = [j["name"] for j in jobs]
    jobs.append({"name": f"{P}:windows",
                 "cmd": [PY, THIS, "windows", "--out-dir", root, "--reports-dir", os.path.join(outd, "labels"),
                         "--out", man_out],
                 "out": man_out, "est_h": 0.03, "env": {**base_env, "USBG_WINDOWS": ""},
                 "deps": labels, "files": [V1_WINDOWS], "expect": {}, "cwd": REPO, "slot": SLOT,
                 "survey_status": "must", "analysis_status": ANALYSIS_STATUS, "note": note})
    cohort_args = ["--train-cases", *train, "--eval-cases", *test, "--targets", *TARGETS, "--min-windows",
                   str(MIN_WINDOWS), "--eval-episodes", str(EVAL_EPISODES), "--eval-seed", str(EVAL_SEED),
                   "--eval-design", "crossed"]

    def tex_ck(t):
        return "none" if t == "physics" else os.path.join(REPO, tex[t])

    def wall_h(ref, default):
        try:
            return max(0.02, round(1.5 * float(json.load(open(ref))["wall_sec"]) / 3600 + 0.01, 3))
        except Exception:
            return default

    def ev(name, out, ref, eargs, env_extra, files, expect, est):
        cmd = [PY, THIS, "eval", "--manifest", man_out, "--ref", ref, "--out", out, "--", *eargs]
        jobs.append({"name": name, "cmd": cmd, "out": out, "est_h": est,
                     "env": {**base_env, "USBG_WINDOWS": wout, **env_extra}, "deps": [f"{P}:windows"],
                     "files": [man_out, wout, ref, *files], "expect": {"seat": "anatomical", **expect}, "cwd": REPO,
                     "slot": SLOT, "survey_status": "must", "analysis_status": ANALYSIS_STATUS, "note": note,
                     "reference_v1_labels": ref})
    ct_env = {"USBG_LEARNED_SEG": "0", "US_SEG_CKPT": ""}
    for t in EVAL_STAGES:
        tfiles = [] if t == "physics" else [tex_ck(t)]
        tmd5 = None if t == "physics" else md5(tex_ck(t))
        for rung in ("pose_oracle", "static"):
            tag = f"{rung}_{t}_ct"
            ref = os.path.join(S21, "ladder", f"{tag}.json")
            ev(f"{P}:ladder:{tag}", os.path.join(outd, "ladder", f"{tag}.json"), ref,
               ["--rung", rung, "--translator", t, "--texture-ckpt", tex_ck(t), *cohort_args, "--contact", "on", "off",
                "--ext-log"], ct_env, tfiles, {"texture_md5": tmd5, "seg_md5": None}, wall_h(ref, 0.03))
        for s in SEEDS:
            for mask in ("ct", "matched"):
                tag = f"ibvs_base_{t}_{mask}_s{s}"
                ref = os.path.join(S21, "ladder", f"{tag}.json")
                sk = seg[f"{t}_s{s}"] if mask == "matched" else None
                env = ct_env if sk is None else {"USBG_LEARNED_SEG": "1", "US_SEG_CKPT": sk}
                ev(f"{P}:ladder:{tag}", os.path.join(outd, "ladder", f"{tag}.json"), ref,
                   ["--rung", "ibvs_base", "--translator", t, "--texture-ckpt", tex_ck(t), *cohort_args,
                    "--calib-seed", str(s), "--contact", "on", "off", "--ext-log"], env,
                   tfiles + ([sk] if sk else []), {"texture_md5": tmd5, "seg_md5": md5(sk) if sk else None},
                   wall_h(ref, 0.05))
            pj = os.path.join(V2_TABLE4, "ctmask", f"{t}_deploy_s{s}.json")
            ref = os.path.join(S21, "contact", f"ctmask_{t}_s{s}.json")
            if not os.path.exists(ref):
                ref = os.path.join(V2_TABLE4, "contact", f"ctmask_{t}_s{s}.json")
            ev(f"{P}:contact:ctmask:{t}:s{s}", os.path.join(outd, "contact", f"ctmask_{t}_s{s}.json"), ref,
               ["--rung", "policy", "--policy-json", pj, "--weights", "final", "--contact", "on", "off", "--ext-log"],
               ct_env, [pj, pj[:-5] + ".pt"], {"texture_md5": tmd5, "seg_md5": None}, wall_h(ref, 0.05))
    evals = [j["name"] for j in jobs if j["name"].startswith((f"{P}:ladder:", f"{P}:contact:"))]
    sout = os.path.join(outd, "summary.json")
    jobs.append({"name": f"{P}:analysis",
                 "cmd": [PY, THIS, "analyze", "--manifest", man_out, "--eval-dir", outd, "--out", sout],
                 "out": sout, "est_h": 0.02, "env": {**base_env}, "deps": evals, "files": [man_out], "expect": {},
                 "cwd": REPO, "slot": SLOT, "survey_status": "must", "analysis_status": ANALYSIS_STATUS, "note": note})
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names))
    outs = [j["out"] for j in jobs]
    assert len(outs) == len(set(outs)) and all(o.endswith(".json") for o in outs)
    for j in jobs:
        for o in [j["out"]] + [v for k, v in zip(j["cmd"][:-1], j["cmd"][1:]) if k in ("--out", "--out-dir")]:
            if os.path.exists(o) and o.endswith(".json"):
                raise SystemExit(f"{j['name']}: {o} exists already")
        for f in j["files"]:
            if not os.path.exists(f) and not f.startswith((root, outd)):
                raise SystemExit(f"{j['name']}: input {f} missing")
    if a.print_only:
        print(json.dumps(jobs[:1] + jobs[19:21] + jobs[-1:], indent=1))
    else:
        write_json(a.jobs_out, jobs)
    print(f"[jobs] {len(jobs)} jobs, {sum(j['est_h'] for j in jobs):.2f} job-h -> {a.jobs_out}")
    return 0


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("case")
    c.add_argument("--case", required=True)
    c.add_argument("--mode", default="default", choices=sorted(MODES),
                   help="default = task total at 1.5 mm (Dataset291-295); fast = 3 mm (Dataset297)")
    c.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    c.add_argument("--threads", type=int, default=0, help="cap on torch CPU threads (0 = TotalSegmentator's own)")
    c.add_argument("--nr-thr-saving", type=int, default=2)
    c.add_argument("--raw-root", default=RAW)
    c.add_argument("--v1-anat", default=V1_ANAT)
    c.add_argument("--v1-windows", default=V1_WINDOWS)
    c.add_argument("--out-dir", default=None, help="default: data/dataset/v2_1/plug/ctanat_ts2/<mode>")
    c.add_argument("--out", required=True)
    c.add_argument("--smoke", action="store_true")
    w = sp.add_parser("windows")
    w.add_argument("--out-dir", required=True)
    w.add_argument("--reports-dir", required=True)
    w.add_argument("--v1-windows", default=V1_WINDOWS)
    w.add_argument("--cases", nargs="*", default=None)
    w.add_argument("--partial", action="store_true")
    w.add_argument("--smoke", action="store_true")
    w.add_argument("--out", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--manifest", required=True)
    e.add_argument("--ref", default=None, help="v1-label output of the same rung (episode identity check)")
    e.add_argument("--out", required=True)
    e.add_argument("--smoke", action="store_true")
    e.add_argument("rest", nargs=argparse.REMAINDER)
    z = sp.add_parser("analyze")
    z.add_argument("--manifest", required=True)
    z.add_argument("--eval-dir", required=True)
    z.add_argument("--out", required=True)
    z.add_argument("--smoke", action="store_true")
    j = sp.add_parser("jobs")
    j.add_argument("--jobs-out", default=JOBS_OUT)
    j.add_argument("--mode", default=SPEC_MODE, choices=sorted(MODES))
    j.add_argument("--threads", type=int, default=5)
    j.add_argument("--nr-thr-saving", type=int, default=2)
    j.add_argument("--label-est-h", type=float, default=None, help="override the per-case estimate")
    j.add_argument("--print-only", action="store_true")
    a = ap.parse_args(argv)
    if getattr(a, "out", None) and not a.out.endswith(".json"):
        ap.error("--out must end in .json")
    if a.cmd == "case" and a.out_dir is None:
        a.out_dir = root_for(a.mode)
    return {"case": cmd_case, "windows": cmd_windows, "eval": cmd_eval, "analyze": cmd_analyze,
            "jobs": cmd_jobs}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
