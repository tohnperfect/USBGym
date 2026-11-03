"""Synthetic tests of best_so_far.py (no study output is read; about 2 s).

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -q \
    <this folder>/test_best_so_far.py
"""
import json
import os
import re
import subprocess
import sys
import textwrap

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

import best_so_far as B  # noqa: E402
import plan_stats as PS  # noqa: E402

P = B.TEST_PATIENTS


def pp(vals):
    return dict(zip(P, [float(v) for v in vals]))


def episodes(values_by_patient, start=16.0):
    """60 synthetic episodes, 10 per patient, with the given final errors (one list of 10 per patient)."""
    eps, k = [], 0
    for p in P:
        for i, v in enumerate(values_by_patient[p]):
            eps.append({"ep": k, "patient": p, "target": "liver" if i % 2 == 0 else "kidney", "window": i % 3,
                        "start_pos": start + 0.01 * k, "start_ang": 10.0 + 0.01 * k, "final_pos": float(v),
                        "final_ang": 5.0})
            k += 1
    return eps


def block(eps, **kw):
    b = {"contact": "on", "eval_randomize": "off", "design": "crossed", "eval_seed": 999, "episodes": eps,
         "pos": float(np.mean([e["final_pos"] for e in eps]))}
    b.update(kw)
    return b


@pytest.fixture
def synthetic_fp(monkeypatch):
    base = episodes({p: [0.0] * 10 for p in P})
    monkeypatch.setitem(B._FP, "fp", B.fingerprint(base))
    return base


def test_diff_summary_matches_scipy_and_sign_count():
    a = np.array([10.0, 12.0, 11.0, 9.0, 13.0, 12.5])
    b = np.array([16.0, 15.0, 17.0, 14.0, 16.5, 12.0])
    d = a - b
    r = B.diff_summary(a, b, P)
    t = stats.ttest_rel(a, b)
    ci = t.confidence_interval(0.95)
    assert r["mean"] == pytest.approx(d.mean())
    assert r["ci95_t"] == pytest.approx([ci.low, ci.high])
    assert r["p_t_two_sided"] == pytest.approx(t.pvalue)
    assert (r["k"], r["n"]) == (5, 6)          # one patient (+0.5) against the negative mean
    assert r["per_unit"]["s0762"] == pytest.approx(0.5)


def test_closed_share_is_ratio_of_patient_means():
    st, orc = np.full(6, 16.0), np.full(6, 2.0)
    c = np.full(6, 9.0)
    r = B.closed_share(c, st, orc)
    assert r["estimate"] == pytest.approx(0.5)
    assert r["ci95_boot_patient"] == pytest.approx([0.5, 0.5])


def test_lopo_selection_picks_by_the_other_five_patients():
    c1 = {"id": "x", "pp": pp([10, 10, 10, 10, 10, 30])}    # mean 13.33; best on 5 of 6 folds
    c2 = {"id": "y", "pp": pp([12, 12, 12, 12, 12, 12])}    # mean 12, the full-sample best
    r = B.lopo_select([c1, c2])
    # leaving out s0762, x wins (10 < 12) and scores 30 there; leaving out any other patient, y wins (14 > 12)
    assert r["picks"]["s0762"] == "x" and all(r["picks"][p] == "y" for p in P[:5])
    assert r["estimate"] == pytest.approx((5 * 12 + 30) / 6)


def test_cell_from_files_selects_the_primary_block_and_averages_seeds(tmp_path, synthetic_fp):
    files = []
    for s in (0, 1, 2):
        good = episodes({p: [s + i + 1.0 for _ in range(10)] for i, p in enumerate(P)})
        bad = episodes({p: [99.0] * 10 for p in P})
        d = {"analysis_status": B.PLUG_STATUS, "condition": {"texture": "physics", "mask": "learned"},
             "results": [block(bad, bn_mode="train"), block(good, bn_mode="eval"),
                         block(bad, bn_mode="eval", contact="off")]}
        f = tmp_path / f"physics_matched_s{s}.json"
        f.write_text(json.dumps(d))
        files.append(str(f))
    cell = B.cell_from_files(files, "results[contact=on,bn_mode=eval].episodes", plug=True,
                             condition={"texture": "physics", "mask": "learned"})
    assert sorted(cell) == [0, 1, 2]
    assert cell[2]["s0358"] == pytest.approx(3.0)
    avg = B.seed_average(cell)
    assert avg["s0762"] == pytest.approx(np.mean([0 + 5 + 1, 1 + 5 + 1, 2 + 5 + 1]))
    c = B.cand("s14:test", "a1", "t", "physics", "U-Net", "BC", "src", B.LABEL_14, sorted(cell), cell=cell)
    assert c["err_mm"] == pytest.approx(np.mean([np.mean([s + i + 1 for i in range(6)]) for s in (0, 1, 2)]))
    assert c["per_seed_err_mm"]["0"] == pytest.approx(3.5)
    # the train block is never read when eval is asked for, and a wrong condition is refused
    with pytest.raises(PS.PlanStatsError):
        B.cell_from_files(files, "results[contact=on,bn_mode=eval].episodes", plug=True,
                          condition={"texture": "sonospade", "mask": "learned"})


def test_design_checks_refuse_other_setups_smoke_and_status(tmp_path, synthetic_fp):
    eps = episodes({p: [5.0] * 10 for p in P})
    shifted = [dict(e, start_pos=e["start_pos"] + 1.0) for e in eps]
    f = tmp_path / "x_s0.json"
    f.write_text(json.dumps({"analysis_status": B.PLUG_STATUS, "results": [block(shifted)]}))
    with pytest.raises(PS.PlanStatsError, match="episode setups differ"):
        B.cell_from_files([str(f)], "results[contact=on].episodes", plug=True)
    f.write_text(json.dumps({"analysis_status": B.PLUG_STATUS, "results": [block(eps[:50])]}))
    with pytest.raises(PS.PlanStatsError, match="crossed held-out design"):
        B.cell_from_files([str(f)], "results[contact=on].episodes", plug=True)
    f.write_text(json.dumps({"analysis_status": B.PLUG_STATUS, "smoke": True, "results": [block(eps)]}))
    with pytest.raises(PS.PlanStatsError, match="smoke"):
        B.cell_from_files([str(f)], "results[contact=on].episodes", plug=True)
    f.write_text(json.dumps({"analysis_status": "exploratory, v2.1", "results": [block(eps)]}))
    with pytest.raises(PS.PlanStatsError, match="analysis_status"):
        B.cell_from_files([str(f)], "results[contact=on].episodes", plug=True)
    g = tmp_path / "trainer_s1.json"
    g.write_text(json.dumps({"final": block(eps)}))
    assert B.cell_from_files([str(g)], "final.episodes")[1]["s0513"] == pytest.approx(5.0)


@pytest.mark.parametrize("path", [
    "/x/outputs/v2_1/amend3/ctrl/bc/physics_learned_s0_l3.json",
    "/x/outputs/v2_1/plug/ctrl/summary.json",
    "/x/outputs/v2_1/plug/seg/ladder/ibvs_base_sonospade_lraspp_mbv3_imnet_s0.json",
    "/x/outputs/v2_1/plug/reward/train/lpips/sonospade_s0.json",
])
def test_guard_refuses_blinded_pending_and_amendment3_paths(path):
    with pytest.raises(PS.PlanStatsError):
        B.guard(path)


@pytest.mark.parametrize("path", [
    "/x/outputs/v2_1/plug/ctrl/td3_res/sonospade_ct_s0.json",
    "/x/outputs/v2_1/plug/ctrl/bc_scratch/physics_ct_s1.json",
    "/x/outputs/v2_1/plug/ctrl/dagger_scratch/sonospade_ct_s2.json",
])
def test_guard_admits_late_rows(path):
    """v3: the TD3 residual and the full-twist BC and DAgger learners finished after the Table 5 snapshot and are
    candidates now."""
    B.guard(path)


def test_guard_and_plug_files_match_windows_path_forms():
    """The guard and plug_file match their '/' patterns on Windows path forms (drive letter, backslashes) too. A child
    process runs them with Windows path rules (ntpath as os.path, a backslash as os.sep) on any platform."""
    win = lambda p: "C:" + p.replace("/", "\\")  # noqa: E731
    refused = [win(p) for p in ("/x/outputs/v2_1/plug/ctrl/summary.json",
                                "/x/outputs/v2_1/plug/seg/ladder/ibvs_base_sonospade_lraspp_mbv3_imnet_s0.json",
                                "/x/outputs/v2_1/plug/reward/train/lpips/sonospade_s0.json")]
    admitted = [win("/x/outputs/v2_1/plug/ctrl/td3_res/sonospade_ct_s0.json")]
    code = textwrap.dedent(f"""
        import ntpath, os, sys
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        import best_so_far as B
        import plan_stats as PS
        os.path, os.sep, os.altsep = ntpath, "\\\\", "/"
        for p in {refused!r}:
            try:
                B.guard(p)
            except PS.PlanStatsError:
                continue
            raise SystemExit("not refused: " + p)
        for p in {admitted!r}:
            B.guard(p)
        assert B.plug_file({admitted[0]!r}) and not B.plug_file({win("/x/outputs/v2/table4/a.json")!r})
        print("windows forms ok")
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0 and "windows forms ok" in p.stdout, p.stdout + p.stderr


def synthetic_world():
    """Anchors and candidates of every class, with known answers."""
    rung = {"static": {"pp": pp([16] * 6)}, "pose_oracle": {"pp": pp([2] * 6)}}
    mk = lambda cid, cls, vals, **kw: B.cand(cid, cls, cid, "f", "m", "c", "src", B.LABEL_V2, [0, 1, 2],  # noqa: E731
                                             pp=pp(vals), **kw)
    cands = [mk(B.REFERENCE_OF_CLASS["a1"], "a1", [17, 17, 18, 16, 17, 17]),
             mk("a1:best", "a1", [14, 15, 13, 14, 15, 14]),
             mk(B.REFERENCE_OF_CLASS["a2"], "a2", [13, 13, 13, 13, 13, 13]),
             mk("a2:seed0", "a2", [11, 11, 11, 11, 11, 11], note="seed 0 only"),
             mk(B.REFERENCE_OF_CLASS["b"], "b", [11.6] * 6),
             mk("b:best", "b", [10.5, 10.6, 10.7, 10.8, 10.9, 11.0]),
             mk("c:look", "c", [7] * 6, eligible=False)]
    return rung, cands


def test_analyse_selects_lowest_eligible_per_class_and_never_class_c():
    rung, cands = synthetic_world()
    K = B.analyse(cands, rung)
    assert K["a1"]["best"] == "a1:best" and K["a2"]["best"] == "a2:seed0" and K["b"]["best"] == "b:best"
    assert "best" not in K["c"] and K["c"]["n_eligible"] == 0
    c = next(x for x in cands if x["id"] == "b:best")
    assert c["minus_static"]["mean"] == pytest.approx(10.75 - 16)
    assert c["minus_oracle"]["mean"] == pytest.approx(10.75 - 2) and c["minus_oracle"]["k"] == 6
    assert c["share_closed"]["estimate"] == pytest.approx((16 - 10.75) / 14)
    assert K["a1"]["best_minus_reference"]["mean"] == pytest.approx(14.1666667 - 17.0, abs=1e-6)
    assert K["b"]["lopo_selection"]["estimate"] == pytest.approx(10.75)


def test_render_table_from_a_synthetic_result(tmp_path):
    rung, cands = synthetic_world()
    K = B.analyse(cands, rung)
    for c in cands:
        c["cohort"] = {"status": "not run on the cohort"}
        c["per_patient_mm"] = c.pop("pp")
    ref = next(c for c in cands if c["id"] == B.REFERENCE_OF_CLASS["a1"])
    ref["cohort"] = {"status": "complete", "mean_mm": 15.75, "xref": "x"}
    st6, or6 = np.full(6, 16.0), np.full(6, 2.0)
    anchors = {"static": {"err_mm": 16.0, "minus_static": None, "minus_oracle": B.diff_summary(st6 + 0.1 * np.arange(6),
                                                                                               or6, P),
                          "cohort": {"mean": 15.97}},
               "scripted": {"err_mm": 15.5, "minus_static": B.diff_summary(st6 - 0.5 - 0.1 * np.arange(6), st6, P),
                            "minus_oracle": B.diff_summary(st6 - 0.5, or6 - 0.1 * np.arange(6), P),
                            "cohort": {"mean": 15.38}},
               "pose_oracle": {"err_mm": 2.0, "minus_static": B.diff_summary(or6, st6 + 0.1 * np.arange(6), P),
                               "minus_oracle": None, "cohort": {"mean": 1.47}}}
    J = {"candidates": cands, "anchors": anchors, "classes": K}
    text = B.render_table(J, str(tmp_path / "best_so_far.json"))
    assert "\\label{tab:best-so-far}" in text and "\\begin{tabularx}{\\columnwidth}" in text
    assert "Post hoc lowest: f; m; c" in text and "15.75" in text and "n/r" in text
    assert "Best so far:" not in text                # amendment 5: never 'best so far' without 'post hoc'
    # the caption prints the printed forms of amendment 5's label and sentence
    assert B.SCI_LABEL_PRINT in text and B.SCI_POSTHOC_PRINT in text
    vis = "\n".join(re.split(r"(?<!\\)%", ln, maxsplit=1)[0] for ln in text.splitlines())
    assert "amendment" not in vis.lower() and "$^{\\S}$" not in vis
    assert "Anchors (reference ladder, Table~\\ref{tab:ladder})" in text
    assert "& 10.75 &" in text                       # the class-b best, an archived-label row: printed plain
    assert "7.00" in text                            # the look-ahead reference row
    assert chr(0x2014) not in text and "---" not in text
    for line in text.splitlines():
        if line.rstrip().endswith("\\\\") and "&" in line and "multicolumn" not in line and "Pipeline" not in line:
            pytest.fail(f"a table row without a src comment: {line}")


def test_best_so_far_check_refuses_the_phrase_without_post_hoc():
    B._best_so_far_check("Post hoc lowest error. The post hoc best so far row.")
    with pytest.raises(PS.PlanStatsError):
        B._best_so_far_check("Best so far: the lowest row.")


def test_run_subcommand_is_render_only():
    with pytest.raises(PS.PlanStatsError, match="render-only"):
        B.main(["run"])
