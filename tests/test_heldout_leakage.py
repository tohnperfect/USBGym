"""D3/D5 leakage gate: no held-out benchmark patient (usbg.benchmark.TEST_PATIENTS) may appear in the
transitive training cases of any texture stage, S_US, segmenter or policy, and no artifact may descend
from a private clinical pool. The unit tests always run; the on-disk checks run when v2 artifacts with
provenance sidecars exist (scripts/check_leakage.py prints the same report for both
repos). Uses the merged usbg.provenance only."""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

from usbg import benchmark as B
from usbg import provenance as P

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELD_OUT = {"s0358", "s0513", "s0519", "s0591", "s0649", "s0762"}
# v2 texture pairs are rendered at the screened liver and kidney windows of these 8 TRAIN cases
V2_TEXTURE_CASES = ("s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0522", "s0703")


def _check_leakage():
    spec = importlib.util.spec_from_file_location("check_leakage", os.path.join(_ROOT, "scripts", "check_leakage.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CL = _check_leakage()


def _rec(d, name, train, parents=(), real_pool=None, kind="translator", seat="anatomical", val=()):
    art = d / name
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(name.encode())
    rec = {"schema_version": P.SCHEMA_VERSION,
           "artifact": {"path": name, "md5": P.md5_file(str(art)), "bytes": art.stat().st_size, "kind": kind},
           "train_cases": list(train), "val_cases": list(val),
           "real_pool": real_pool or {"name": "kaggle_vitale_pool", "path": "data/real/kaggle_vitale/pool",
                                      "n_frames": 404, "license": "Kaggle (Vitale 2020)", "frames_md5": None},
           "iters": 1, "seed": 0, "seat_mode": seat, "dr": {"mode": "off", "notes": ""},
           "parents": list(parents), "git": {"repo": "x", "hash": "0", "dirty": False},
           "created": "2026-10-13T00:00:00+00:00", "command": "test",
           "benchmark": {"version": B.BENCHMARK_VERSION, "test_patients": list(B.TEST_PATIENTS)}}
    (d / (name + P.SIDECAR_SUFFIX)).write_text(json.dumps(rec))
    return {"path": name, "md5": rec["artifact"]["md5"], "role": "parent"}


# ----------------------------------------------------------------------------------- the split
def test_held_out_patients_are_the_decided_six():
    assert set(B.TEST_PATIENTS) == HELD_OUT and len(B.TEST_PATIENTS) == 6
    assert len(B.TRAIN_PATIENTS) == 13 and not set(B.TRAIN_PATIENTS) & HELD_OUT
    assert set(V2_TEXTURE_CASES) <= set(B.TRAIN_PATIENTS)


# ------------------------------------------------------------------------------ lineage (unit)
def test_clean_lineage_passes(tmp_path):
    pairs = _rec(tmp_path, "pairs.npz", V2_TEXTURE_CASES, kind="pairs")
    _rec(tmp_path, "sono.pt", [], parents=[pairs])
    res = P.scan([str(tmp_path)], repo=str(tmp_path))
    assert res["findings"] == []
    assert set(res["transitive"][str(tmp_path / "sono.pt")]) == set(V2_TEXTURE_CASES)


def test_direct_and_transitive_leak_detected(tmp_path):
    pairs = _rec(tmp_path, "pairs.npz", ["s0011", "s0358"], kind="pairs")      # s0358 is held out
    seg = _rec(tmp_path, "seg.pt", B.TRAIN_PATIENTS, kind="segmenter", parents=[pairs])
    _rec(tmp_path, "policy.pt", B.TRAIN_PATIENTS, kind="policy", parents=[seg])
    f = P.scan([str(tmp_path)], repo=str(tmp_path))["findings"]
    assert any(x.startswith("LEAK") and "pairs.npz" in x for x in f)
    assert any(x.startswith("LEAK") and "seg.pt" in x for x in f)             # inherited through parent
    assert any(x.startswith("LEAK") and "policy.pt" in x for x in f)          # two hops


def test_val_split_leak_detected(tmp_path):
    """Model selection on a held-out patient is a leak too (the D2 segmenter val split)."""
    _rec(tmp_path, "seg.pt", B.TRAIN_PATIENTS, kind="segmenter", val=["s0591"])
    assert any(x.startswith("LEAK") and "val_cases" in x
               for x in P.scan([str(tmp_path)], repo=str(tmp_path))["findings"])


def test_private_pool_and_descendant_detected(tmp_path):
    _rec(tmp_path, "cut.pt", B.TRAIN_PATIENTS,
         real_pool={"name": "own", "path": "data/real_us_crop", "n_frames": 149, "license": "?",
                    "frames_md5": None})
    _rec(tmp_path, "seg.pt", B.TRAIN_PATIENTS, kind="segmenter",
         parents=[{"path": "data/dataset/texture_cut.pt", "md5": "68abc2fc0e195f61bf46e723c1bf1e8f",
                   "role": "translator"}])
    f = P.scan([str(tmp_path)], repo=str(tmp_path))["findings"]
    assert any(x.startswith("PRIVATE") and "cut.pt" in x for x in f)
    assert any(x.startswith("PRIVATE") and "seg.pt" in x for x in f)


def test_stale_sidecar_detected(tmp_path):
    _rec(tmp_path, "a.pt", B.TRAIN_PATIENTS)
    (tmp_path / "a.pt").write_bytes(b"retrained")
    assert any(x.startswith("STALE") for x in P.scan([str(tmp_path)], repo=str(tmp_path))["findings"])


def test_write_provenance_records_absolute_path_across_drives(tmp_path, monkeypatch):
    """On Windows a temporary folder and the repo can sit on two drives (C: and D:), where os.path.relpath
    raises ValueError; the record then keeps the absolute path and the leak check still runs."""
    art = tmp_path / "x.pt"
    art.write_bytes(b"x")

    def no_common_root(path, start=os.curdir):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    monkeypatch.setattr(P.os.path, "relpath", no_common_root)
    assert P._relpath_or_abs(str(art), _ROOT) == os.path.abspath(str(art))
    with pytest.raises(ValueError, match="LEAK"):
        P.write_provenance(str(art), "translator", train_cases=["s0011", "s0513"], repo=_ROOT,
                           seat_mode="anatomical")


def test_write_provenance_refuses_leaky_or_private_record(tmp_path):
    art = tmp_path / "x.pt"
    art.write_bytes(b"x")
    with pytest.raises(ValueError, match="LEAK"):
        P.write_provenance(str(art), "translator", train_cases=["s0011", "s0513"], repo=_ROOT,
                           seat_mode="anatomical")
    with pytest.raises(ValueError, match="PRIVATE"):
        P.write_provenance(str(art), "translator", train_cases=["s0011"], repo=_ROOT, seat_mode="anatomical",
                           real_pool={"name": "own", "path": "data/real_us_crop", "n_frames": 1,
                                      "license": "?", "frames_md5": None})
    assert not os.path.exists(P.sidecar_path(str(art)))


def test_result_block_refuses_heldout_training_cohort():
    with pytest.raises(ValueError, match="held-out"):
        P.result_block(train_cases=["s0011", "s0762"], eval_cases=list(B.TEST_PATIENTS))
    blk = P.result_block(train_cases=B.TRAIN_PATIENTS, eval_cases=B.TEST_PATIENTS)
    assert blk["seat_mode"] in P.SEAT_MODES and blk["cohort"]["eval"] == sorted(B.TEST_PATIENTS)


# --------------------------------------------------------------------- the gate CLI helpers (unit)
def test_gate_skips_scratch_trees(tmp_path):
    out = tmp_path / "outputs"
    _rec(out / "v2", "clean.pt", B.TRAIN_PATIENTS)
    _rec(out / "v2" / "_smoke" / "texture", "leaky.pt", ["s0358"])
    _rec(out / "v2" / "_recipes", "old.pt", ["s0513"])
    res = CL.gate([str(out)], exclude=CL.SCRATCH_DIRS, repo=str(tmp_path))
    assert res["findings"] == [] and len(res["records"]) == 1
    res_all = CL.gate([str(out)], exclude=(), repo=str(tmp_path))
    assert sum(x.startswith("LEAK") for x in res_all["findings"]) >= 2
    seat = CL.gate([str(out)], repo=str(tmp_path), expect_seat="canonical")
    assert any(x.startswith("SEATMIX") for x in seat["findings"])


def test_npz_case_findings(tmp_path):
    from usbg.dataset import write_pairs_npz
    img = np.zeros((2, 8, 8), np.uint8)
    write_pairs_npz(str(tmp_path / "ok.npz"), img, img, sessions=np.array(["s0011", "s0522"]),
                    extra_meta={"cases": ["s0011", "s0522"]})
    write_pairs_npz(str(tmp_path / "bad.npz"), img, img, sessions=np.array(["s0011", "s0649"]))
    (tmp_path / "broken.npz").write_bytes(b"not a zip")
    f, n, bad = CL.npz_case_findings(str(tmp_path))
    assert n == 2 and len(f) == 1 and "bad.npz" in f[0] and "s0649" in f[0]
    assert len(bad) == 1 and "broken.npz" in bad[0]


# ------------------------------------------------------------------------------------ on disk
def test_on_disk_provenance_is_leak_free():
    """The leakage gate over every non-scratch sidecar in this repo and the segmenter repo (md5 re-hash is
    left to scripts/check_leakage.py, which is the slow, complete version)."""
    res = CL.gate(CL.default_roots(), check_md5=False)
    if not res["records"]:
        pytest.skip("no *.provenance.json yet outside _smoke/_recipes (v2 artifacts not produced)")
    assert not res["findings"], "\n".join(res["findings"])


def test_v2_datasets_hold_no_heldout_patient():
    root = os.path.join(_ROOT, "data", "dataset", "v2")
    if not os.path.isdir(root):
        pytest.skip("data/dataset/v2 not created yet")
    f, n, _bad = CL.npz_case_findings(root)
    if n == 0:
        pytest.skip("no v2 npz with case meta yet")
    assert not f, "\n".join(f)


@pytest.mark.skipif(not os.path.exists(os.path.join(_ROOT, "data", "manifest.json")), reason="no manifest")
def test_committed_manifest_has_no_heldout_training_streams():
    f = P.legacy_manifest_findings(os.path.join(_ROOT, "data", "manifest.json"))
    assert not f, "\n".join(f)
