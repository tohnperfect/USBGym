"""Synthetic-input tests of posthoc_logs.py (post hoc log analyses).

Run (about 2 s; nothing is written into the repository):
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src nice -n 10 python -m pytest -p no:cacheprovider -q \
      <this folder>/test_posthoc_logs.py
"""
import json
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import plan_stats as ps  # noqa: E402
import posthoc_logs as PL  # noqa: E402

P6 = PL.PATIENTS


def test_patient_means_pool_seeds_and_level_mean():
    pats = list(P6) * 2                      # two "seeds", one episode per patient each
    vals = list(range(6)) + list(range(10, 16))
    pm = PL.patient_means(vals, pats)
    assert pm["s0358"] == pytest.approx(5.0) and pm["s0762"] == pytest.approx(10.0)
    assert PL.level_mean(pm) == pytest.approx(np.mean(vals))
    with pytest.raises(ps.IncompleteData):
        PL.patient_means([1.0], ["s0358"])


def test_lopo_sign_kept_and_dissenter():
    d = dict(zip(P6, [1.0, 2.0, 3.0, 4.0, 5.0, -1.0]))
    r = PL.lopo(d)
    assert r["mean"] == pytest.approx(14 / 6)
    assert r["sign_k"] == 5 and r["dissenting_patients"] == ["s0762"]
    assert r["lopo_means"]["s0762"] == pytest.approx(3.0)
    assert r["lopo_means"]["s0649"] == pytest.approx(1.8)
    assert r["lopo_min"] == pytest.approx(1.8) and r["lopo_max"] == pytest.approx(3.0)
    assert r["sign_kept_in_all"] is True


def test_lopo_sign_lost_when_one_patient_carries_the_mean():
    d = dict(zip(P6, [10.0, -1.0, -1.0, -1.0, -1.0, -1.0]))
    r = PL.lopo(d)
    assert r["mean"] > 0 and r["sign_k"] == 1
    assert r["lopo_means"]["s0358"] == pytest.approx(-1.0)
    assert r["sign_kept_in_all"] is False
    assert r["most_influential_patient"] == "s0358"


def test_step_stats_skip_start_column():
    a = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]])
    s = PL.step_stats(a)
    assert s["n_values"] == 4 and s["mean"] == pytest.approx(2.5) and s["max"] == pytest.approx(4.0)
    assert s["p95"] == pytest.approx(np.percentile([1, 2, 3, 4], 95))


def test_twist_fraction_and_step_length():
    bound = np.array([0.03, 0.03, 0.03, 0.5, 0.5, 0.5])
    tw = np.zeros((2, 3, 6))
    tw[..., 2] = 0.03                         # axial at the bound, into tissue
    tw[0, :, 0] = -0.015                      # lateral half bound, one episode
    f = PL.twist_fraction(tw, bound)
    assert f["vz"]["signed_mean_frac"] == pytest.approx(1.0) and f["vz"]["at_bound_frac"] == pytest.approx(1.0)
    assert f["vx"]["signed_mean_frac"] == pytest.approx(-0.25) and f["vx"]["mean_abs_frac"] == pytest.approx(0.25)
    assert f["vx"]["at_bound_frac"] == 0.0
    sl = PL.step_length_mm(tw, 0.1)
    assert sl[1, 0] == pytest.approx(3.0)       # 0.03 m/s x 0.1 s = 3 mm


def test_keepbest_index_first_maximum():
    assert PL.keepbest_index([0.1, 0.5, 0.5, 0.2]) == 1
    assert PL.keepbest_index([0.9, 0.5]) == 0


def test_lift_reconstruct_adds_lift_to_axial_offset():
    pre = PL.lift_reconstruct([-3.0, -1.5], [0.0, 7.0])
    assert pre.tolist() == pytest.approx([-3.0, 5.5])


def test_parse_meta_bom_semicolon_and_summary():
    text = ("﻿image_id;age;gender;institute;study_type;split\n"
            "s0001;51;;E;ct polytrauma;train\ns0002;23;f;E;ct thorax;train\ns0003;70;m;A;ct abdomen;test\n")
    m = PL.parse_meta(text)
    assert set(m) == {"s0001", "s0002", "s0003"} and m["s0001"]["gender"] == ""
    s = PL.sex_age_summary(m, ["s0001", "s0002", "s0003"])
    assert s["female"] == 1 and s["male"] == 1 and s["sex_not_recorded"] == 1
    assert s["age_median"] == pytest.approx(51) and s["age_min"] == 23 and s["age_max"] == 70


def test_code_constants_from_source_text():
    env = ('<option timestep="0.002" gravity="0 0 0"/>\n'
           "def _mjcf(patient_half=(1,), weld_solref=(0.02, 1.0)) -> str:\n"
           "    def __init__(self, settle_steps=10, x=1):\n"
           '        max_press_m = float(os.environ.get("USBG_MAX_PRESS_MM", "5")) / 1000.0\n')
    con = "X = 1\nDT_C = 0.1        # 10 Hz\n"
    goal = "MAX_TWIST = np.array([0.03, 0.03, 0.03, 0.5, 0.5, 0.5], dtype=np.float64)\n"
    place = "PROBE_HALF = (0.008, 0.008, 0.012)\n"
    k = PL.code_constants(env, con, goal, place)
    assert k["physics_time_per_step_s"] == pytest.approx(0.02)
    assert k["weld_solref_timeconst_s"] == pytest.approx(0.02) and k["control_step_s"] == pytest.approx(0.1)
    assert k["physics_share_of_control_step"] == pytest.approx(0.2) and k["press_cap_mm"] == 5.0
    with pytest.raises(ps.PlanStatsError):
        PL.code_constants(env, "no constant here\n", goal, place)


def _write_run(tmp_path, ep_order=(0, 1)):
    """A two-episode result JSON and its npz sidecar in the save_traj_npz layout."""
    eps = [{"ep": i, "patient": P6[i], "target": "liver", "window": 0, "start_pos": 10.0 + i, "start_ang": 5.0,
            "final_pos": 3.0 + i, "keepbest_pos": 2.0, "contact_frac": 1.0} for i in range(2)]
    d = {"results": [{"weights": "zero_residual", "contact": "on", "eval_randomize": "off", "episodes": eps}]}
    jp = tmp_path / "run.json"
    jp.write_text(json.dumps(d))
    pos = np.array([[10.0, 3.0], [11.0, 4.0]], np.float32)
    arrs = {"index": np.asarray(json.dumps([{"block": 0, "weights": "zero_residual", "contact": "on",
                                             "eval_randomize": "off"}])),
            "b0__pos": pos, "b0__ep": np.asarray(ep_order, np.int32), "b0__patient": np.asarray(P6[:2])}
    np.savez_compressed(str(tmp_path / "run.traj.npz"), **arrs)
    return str(jp)


def test_blocks_with_traj_matches_and_checks(tmp_path):
    jp = _write_run(tmp_path)
    d, blocks = PL.blocks_with_traj(jp)
    r, tr = PL.pick(blocks, "on")
    assert tr["pos"].shape == (2, 2) and r["episodes"][1]["final_pos"] == 4.0
    with pytest.raises(ps.PlanStatsError):
        PL.pick(blocks, "off")


def test_blocks_with_traj_refuses_misaligned_npz(tmp_path):
    jp = _write_run(tmp_path, ep_order=(1, 0))
    with pytest.raises(ps.PlanStatsError):
        PL.blocks_with_traj(jp)


def test_blinded_and_forbidden_paths_refused(tmp_path):
    for name in ("x_lpips_s0.json", "ibvs_base_sonospade_samus_unet_s0.json", "a_imnet.json", "ctanat_r1.json"):
        with pytest.raises(ps.ForbiddenPath):
            PL.read_json(str(tmp_path / name))
    for r in PL.plug_rows():
        assert not any(PL._blinded(p) for p in r["paths"].values())


def test_episode_block_layouts():
    eps = [{"ep": 0}]
    d1 = {"final": {"episodes": eps}}
    b, k = PL.episode_block(d1, "final")
    assert b["episodes"] is eps and k == "final.episodes"
    d2 = {"results": [{"contact": "on", "bn_mode": "train", "episodes": eps},
                      {"contact": "on", "bn_mode": "eval", "episodes": eps + eps},
                      {"contact": "off", "bn_mode": "eval", "episodes": []}]}
    b, k = PL.episode_block(d2, None, "eval")
    assert len(b["episodes"]) == 2 and "bn_mode=eval" in k
    b, _ = PL.episode_block(d2, None)
    assert len(b["episodes"]) == 1
    with pytest.raises(ps.PlanStatsError):
        PL.episode_block({"results": []}, None)


def test_fingerprint_rounding():
    e = [{"ep": 0, "patient": "s0358", "target": "liver", "window": 1, "start_pos": 1.00000001, "start_ang": 2.0}]
    f = [{"ep": 0, "patient": "s0358", "target": "liver", "window": 1, "start_pos": 1.0, "start_ang": 2.0}]
    assert PL.fingerprint(e) == PL.fingerprint(f)


def test_contrast_degenerate_zero_and_regular():
    C = ps.load_constants()
    z = PL._contrast(np.zeros(6), C)
    assert z["mean"] == 0.0 and z["sign_k"] == 0 and z["lopo"]["sign_kept_in_all"] is False
    c = PL._contrast(np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0]), C)
    assert c["sign_k"] == 5 and c["ci_t"][0] < c["mean"] < c["ci_t"][1]
    assert set(c["per_unit"]) == set(P6)


def test_check_tex_style_gate():
    ok = "Rows 1--12 \\\\ % src: x\n95\\% CI & a \\\\ % src: y --- in a comment is still refused\n"
    with pytest.raises(ps.PlanStatsError):
        PL._check_tex(ok, "t")                  # three hyphens anywhere in the file
    PL._check_tex("Rows 1--12 \\\\ % src: x\n95\\% CI \\\\\n", "t")
    words = ("significant", "comparable", "deployable")         # refused words, each inside a sentence below
    for bad in ("a " + chr(0x2014) + " b", f"the gap is {words[0]}", f"{words[1]} rows", "a -- b",
                "the perception gap", f"{words[2]} base"):
        with pytest.raises(ps.PlanStatsError):
            PL._check_tex(bad + "\n", "t")
