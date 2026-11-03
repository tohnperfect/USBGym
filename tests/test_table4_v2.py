"""v2 oracle-ladder tooling guards (D1, D2, D4): the canonical seat still reproduces the v1
pose-oracle floor, the crossed design gives every eval patient every target, skipped (case, target)
cells are never substituted, the withdrawn segmenter is refused, mixed seats are never aggregated,
the runner plans the full job set, and the env contact switch works."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import train_view_match_rl as T  # noqa: E402

HELD_OUT = ["s0358", "s0513", "s0519", "s0591", "s0649", "s0762"]
# v1 legacy design, pose oracle (analytic base + zero residual), canonical seat, first 6 of the 60
# held-out episodes (eval seed 999). Full 60: 2.05 mm / 0.84 deg (contact on), 1.85 / 1.47 (off).
POSE_ORACLE_FIRST6 = {"on": [1.3910, 0.9817, 1.1963, 1.8173, 1.1004, 3.1887],
                      "off": [0.4968, 0.1602, 0.8821, 0.8263, 1.3752, 2.1643]}


@pytest.fixture(autouse=True)
def _reset_trainer_globals():
    saved = dict(SEAT_ORGANS=list(T.SEAT_ORGANS), _CELLS=T._CELLS, _CONTACT=T._CONTACT, _RANDOMIZE=T._RANDOMIZE,
                 _MIN_WINDOWS=T._MIN_WINDOWS, _RESIDUAL_MODE=T._RESIDUAL_MODE, _IBVS=T._IBVS)
    T._SKIPS.clear()
    yield
    for k, v in saved.items():
        setattr(T, k, v)
    T._SKIPS.clear()


class _FakeVol:
    def __init__(self, case, organs):
        self.meta = {"case": case}
        self._organs = set(organs)

    def has(self, label):
        return label in self._organs


def _ids(*names):
    return {T._organ_id(n) for n in names}


# ------------------------------------------------------------------------------- canonical regression
@pytest.mark.skipif(not all(os.path.exists(os.path.join(REPO, "data", "cache", f"{c}.npz")) for c in HELD_OUT),
                    reason="needs the held-out CT caches data/cache/<case>.npz (local data, not in git)")
def test_pose_oracle_legacy_floor_reproduces(monkeypatch):
    from usbg import view_match as VM
    from usbg.render_lotus import get_device, RenderConfig
    monkeypatch.setattr(VM, "_US", None)                  # CT mask (the pose numbers are mask-independent)
    lvs = T.load_volumes(HELD_OUT)
    T.SEAT_ORGANS = list(T.LEGACY_TARGETS)
    T._RESIDUAL_MODE = "oracle"
    from eval_view_ladder import ZeroPolicy
    kw = dict(lateral_m=0.03, axial_m=0.01, fan_deg=18.0)
    for contact in ("on", "off"):
        T._CONTACT = contact == "on"
        ev = T.evaluate(lvs, HELD_OUT, None, RenderConfig(seed=0), ZeroPolicy(), None, 4, 128, get_device(),
                        0.6, 0.4, kw, 12, 6, seed=999, residual=True, design="legacy")
        got = [e["final_pos"] for e in ev["episodes"]]
        assert np.allclose(got, POSE_ORACLE_FIRST6[contact], atol=0.01), (contact, got)
        assert [e["target"] for e in ev["episodes"]] == ["liver", "kidney", "spleen"] * 2   # the v1 confound
        assert (ev["peak_force"] > 0) == (contact == "on")
        assert not any(e["start_clamped"] for e in ev["episodes"])


# ------------------------------------------------------------------------------- design and cells
def test_crossed_design_gives_every_patient_every_target():
    T.SEAT_ORGANS = ["liver", "kidney"]
    vols = [_FakeVol(c, _ids("liver", "kidney")) for c in HELD_OUT]
    vols[2] = _FakeVol("s0519", _ids("liver"))                 # no kidney: skipped, never substituted
    cells, skipped = T.eval_cells(vols, HELD_OUT, "crossed")
    assert len(cells) == 11 and skipped == {"s0519:kidney": 0}
    assert cells[:2] == [(0, "liver"), (0, "kidney")]          # patient-major
    for i, c in enumerate(HELD_OUT):
        want = {"liver"} if c == "s0519" else {"liver", "kidney"}
        assert {t for vi, t in cells if vi == i} == want
    assert T.eval_cells(vols, HELD_OUT, "legacy") == (None, {})


def test_draw_cell_rejects_skipped_cells_and_legacy_order_is_unchanged():
    T.SEAT_ORGANS = ["liver", "kidney"]
    vols = [_FakeVol("a", _ids("liver", "kidney")), _FakeVol("b", _ids("liver", "kidney"))]
    T._CELLS = {0: ["liver", "kidney"], 1: ["liver"]}
    rng = np.random.default_rng(0)
    seen = [(v.meta["case"], t) for v, t in (T.draw_cell(vols, rng) for _ in range(600))]
    assert ("b", "kidney") not in seen and T._SKIPS["b:kidney"] > 0
    for cell in (("a", "liver"), ("a", "kidney"), ("b", "liver")):      # uniform over valid cells
        assert 150 < seen.count(cell) < 250
    vols3 = vols + [_FakeVol("c", _ids("liver", "kidney"))]
    T._CELLS = {0: ["liver", "kidney"], 1: ["liver"], 2: ["liver", "kidney"]}
    cyc = [T.cycle_cell(vols3, k) for k in range(4)]                # (k % 3, k % 2), skipped -> next valid
    assert [v.meta["case"] + ":" + t for v, t in cyc] == ["a:liver", "c:liver", "c:liver", "a:kidney"]
    T._CELLS = None                                              # legacy: the v1 draw order exactly
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(20):
        v, t = T.draw_cell(vols, r1)
        assert v is vols[r2.integers(2)] and t == T.SEAT_ORGANS[r2.integers(2)]


@pytest.mark.anatomical
def test_anatomical_cells_skip_cases_without_kidney_windows():
    from usbg import windows as W
    if not (os.path.exists(W.WINDOWS_PATH) and os.path.exists(os.path.join(REPO, "data/cache_anat/s0727.npz"))):
        pytest.skip("anatomical cache / windows file not built")
    T.SEAT_ORGANS = ["liver", "kidney"]
    T._MIN_WINDOWS = 2
    cases = ["s0011", "s0058", "s0727"]
    info = T.init_train_cells(T.load_volumes(cases), cases, log=lambda *_: None)
    assert set(info["skipped"]) == {"s0058:kidney", "s0727:kidney"}
    assert "s0011:kidney" in info["valid"] and "s0727:liver" in info["valid"]


# ------------------------------------------------------------------------------- DR, texture, provenance
def test_episode_cfg_off_is_fixed_and_on_is_one_shared_draw():
    from usbg.render_lotus import RenderConfig
    cfg = RenderConfig(seed=3)
    T._RANDOMIZE = "off"
    assert T.episode_cfg(cfg, np.random.default_rng(0)) is cfg
    T._RANDOMIZE = "episode"
    a = T.episode_cfg(cfg, np.random.default_rng([0, T._DR_SALT]))
    b = T.episode_cfg(cfg, np.random.default_rng([0, T._DR_SALT]))
    assert a == b and a != cfg


def test_anatomical_seat_refuses_the_canonical_texture_ckpts(monkeypatch):
    monkeypatch.setenv("USBG_SEAT", "anatomical")
    with pytest.raises(SystemExit):
        T.resolve_texture("sonospade")
    assert T.resolve_texture("physics") is None
    monkeypatch.setenv("USBG_SEAT", "canonical")
    with pytest.raises(SystemExit):
        T.resolve_texture("physics", texture_ckpt="data/dataset/exp_real_cut_curvi.pt")
    if not os.path.exists(os.path.join(REPO, "data", "dataset", "exp_real_cut_curvi.pt")):
        pytest.skip("the v1 canonical CUT checkpoint data/dataset/exp_real_cut_curvi.pt is local data (not in git)")
    assert T.resolve_texture("cut").endswith("data/dataset/exp_real_cut_curvi.pt")


def test_assert_not_withdrawn_refuses_the_shared_segmenter():
    from usbg import provenance as PROV
    bad = {"mask": {"learned": True, "seg_md5": "22a4972c5c881356bb3cf0e34366664a", "US_SEG_CKPT_env": "x.pt"}}
    with pytest.raises(SystemExit):
        PROV.assert_not_withdrawn(bad)
    PROV.assert_not_withdrawn(bad, allow=True)
    with pytest.raises(SystemExit):
        PROV.assert_not_withdrawn({"mask": {"learned": True, "seg_md5": "ab", "US_SEG_CKPT_env": None}})
    PROV.assert_not_withdrawn({"mask": {"learned": False}})


def test_require_seg_ckpt_guard(monkeypatch):
    from usbg import us_segmenter as USS
    monkeypatch.delenv("US_SEG_CKPT", raising=False)
    monkeypatch.setenv("USBG_REQUIRE_SEG_CKPT", "1")
    with pytest.raises(RuntimeError):
        USS.load_segmenter()


# ------------------------------------------------------------------------------- aggregation and runner
def _trainer_json(path, seat):
    ep = dict(ep=0, patient="s0358", target="liver", window=0, start_pos=10.0, start_ang=5.0, final_pos=4.0,
              final_ang=2.0, keepbest_pos=3.0, static_seg=0.3, final_seg=0.5, final_match=0.6, peak_force=5.0)
    ev = {"episodes": [ep], "contact": "on", "eval_randomize": "off", "design": "crossed"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"translator": "physics", "seed": 0, "after_imitation": ev, "final": ev,
               "provenance": {"seat_mode": seat, "seat": {"resolved": seat}, "mask": {"learned": False}}},
              open(path, "w"))


def test_agg_refuses_mixed_seats(tmp_path):
    import agg_table4_v2 as A
    _trainer_json(str(tmp_path / "ctmask" / "physics_deploy_s0.json"), "anatomical")
    rows, reg, files, seat = A.collect_rows(str(tmp_path))
    assert seat == "anatomical" and len(rows) == 2 and rows[0]["patient"] == "s0358"
    _trainer_json(str(tmp_path / "ctmask" / "physics_deploy_s1.json"), "canonical")
    with pytest.raises(SystemExit):
        A.collect_rows(str(tmp_path))


def test_runner_plans_the_full_table4_job_set(tmp_path):
    import run_table4_v2 as RUN
    tex = tmp_path / "tex.json"
    json.dump({"physics": None, "cut": "data/dataset/v2/cut_curvi.pt", "cyclegan": "data/dataset/v2/cyclegan_curvi.pt",
               "sonospade": "data/dataset/v2/sonospade_curvi.pt"}, open(tex, "w"))
    seg = tmp_path / "seg.json"
    json.dump({f"{t}_s{s}": f"/abs/us_seg_{t}_s{s}.pt" for t in RUN.STAGES for s in (0, 1, 2)}, open(seg, "w"))
    out = tmp_path / "table4"
    argv = ["--out", str(out), "--tex-manifest", str(tex), "--seg-manifest", str(seg), "--dry-run"]
    import argparse
    ns = argparse.Namespace(out=str(out), k=3, seeds=[0, 1, 2],
                            phases=["ctmask", "matched", "dr", "ladder", "contact", "registration"],
                            seat="anatomical", train_cases=["s0011"], eval_cases=HELD_OUT, targets=["liver", "kidney"],
                            min_windows=2, tex_manifest=str(tex), seg_manifest=str(seg), vitale_npz="v.npz",
                            aul_npz="a.npz", aul_n=60, kidneyus_npz="k.npz", kidneyus_n=60, cts_per_target=3,
                            extra_jobs=None, only=None, seg_ready_h=0.0, no_early_ladder=False)
    jobs = RUN.build_jobs(ns)
    kinds = {}
    for j in jobs:
        k = ":".join(j.name.split(":")[:2])
        kinds[k] = kinds.get(k, 0) + 1
    assert kinds == {"train:ctmask": 12, "train:matched": 12, "train:dr_on": 3, "eval:ladder": 36,
                     "eval:contact": 24, "eval:dr": 6, "eval:registration": 12}
    ct = next(j for j in jobs if j.name == "train:ctmask:sonospade:s0")
    assert ct.env["USBG_LEARNED_SEG"] == "0" and ct.env["US_SEG_CKPT"] == "" and ct.expect["seg_md5"] is None
    m = next(j for j in jobs if j.name == "train:matched:cut:s2")
    assert m.env["US_SEG_CKPT"] == "/abs/us_seg_cut_s2.pt" and "/abs/us_seg_cut_s2.pt" in m.files
    assert "--min-windows" in m.cmd and m.cmd[m.cmd.index("--texture-ckpt") + 1].endswith("data/dataset/v2/cut_curvi.pt")
    assert RUN.main(argv) == 0 and not out.exists()                # dry run writes nothing
    mk, order = RUN.simulate(jobs, 3)
    assert len(order) == len(jobs) and mk > 0


# ------------------------------------------------------------------------------- env contact switch
def test_env_contact_switch_and_observe_flag(monkeypatch):
    from usbg import volume as V
    from usbg.mujoco_env import USProbeEnv
    from usbg.placement import seat_volume_under_probe
    from usbg.contracts_bridge import pose_from_rpy
    lv = V.synthetic_phantom()
    Wfv, T0 = seat_volume_under_probe(lv, V.GALLBLADDER, standoff_m=0.006, depth_m=0.04)
    pressed = pose_from_rpy(tz=-0.004, rx=180.0)
    forces = {}
    for contact in (True, False):
        env = USProbeEnv(lv, Wfv, T0, renderer="numpy", contact=contact, observe=False)
        obs, _ = env.reset()
        assert obs is None
        forces[contact] = env.force_at_pose(pressed)
    assert forces[True] > 0 and forces[False] == 0
    monkeypatch.setenv("USBG_CONTACT", "off")
    assert USProbeEnv(lv, Wfv, T0, renderer="numpy").contact is False


def test_runner_picks_up_a_manifest_written_after_start(tmp_path):
    import argparse
    import run_table4_v2 as RUN
    seg = tmp_path / "seg.json"
    ns = argparse.Namespace(out=str(tmp_path / "t4"), k=3, seeds=[0], phases=["matched"], seat="anatomical",
                            train_cases=["s0011"], eval_cases=HELD_OUT, targets=["liver"], min_windows=1,
                            tex_manifest=str(tmp_path / "missing_tex.json"), seg_manifest=str(seg),
                            vitale_npz="v.npz", aul_npz="a.npz", aul_n=1, kidneyus_npz="k.npz", kidneyus_n=1,
                            cts_per_target=1, extra_jobs=None, only="physics", seg_ready_h=0.0, no_early_ladder=False)
    jobs = RUN.build_jobs(ns)
    assert [j.name for j in jobs] == ["train:matched:physics:s0"]
    assert any(f.startswith("<MISSING segmenter") for f in jobs[0].files)
    json.dump({"physics_s0": "/abs/us_seg_physics_s0.pt"}, open(seg, "w"))
    msgs = []
    RUN.refresh_pending(jobs, ns, msgs.append)
    assert jobs[0].env["US_SEG_CKPT"] == "/abs/us_seg_physics_s0.pt" and msgs
    assert "/abs/us_seg_physics_s0.pt" in jobs[0].files
