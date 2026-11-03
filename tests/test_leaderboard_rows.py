"""The leaderboard rows (Table 5 of the article, both parts) recomputed from their shipped per-episode values.

Data-free: reads src/usbg/leaderboard/data only. Every row's error, difference, CI, k/6, seeds k/3, Holm p, TOST
and decision must equal the shipped record (1e-9) and its printed cells, and each slot's post hoc MDD must be
recomputed from its rows.
"""
from __future__ import annotations

import re

import pytest

from usbg.benchmark import TEST_PATIENTS, TRAIN_PATIENTS
from usbg.leaderboard import protocol as PRO
from usbg.leaderboard import recompute as RC
from usbg.leaderboard import render as RD
from usbg.leaderboard.submission import DOI_RE

BOARD, VALUES, METHODS = RC.load_board(), RC.load_values(), RC.load_methods()
ROWS = [r for s in BOARD["slots"] for r in s["rows"]]
GOT = RC.recompute_board(BOARD, VALUES)


def _close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    if isinstance(a, dict):
        return all(_close(a[k], b[k], tol) for k in a if k in b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (bool, str)):
        return a == b
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))


def test_counts_and_slots():
    assert [s["key"] for s in BOARD["slots"]] == ["texture", "mask", "ctrl", "reward", "renderer", "anatomy"]
    assert len(ROWS) == 72
    fams = {(s["key"], f["key"]): f["m"] for s in BOARD["slots"] for f in s["families"]}
    assert fams == {("texture", "family"): 4, ("mask", "family"): 7, ("mask", "MASK-P"): 5, ("mask", "IMN-T"): 4,
                    ("ctrl", "family"): 10, ("reward", "family"): 2, ("reward", "LPIPS-R"): 1,
                    ("renderer", "family"): 2, ("anatomy", "CTA-D"): 5, ("anatomy", "CTA-E"): 5}
    for s in BOARD["slots"]:
        for f in s["families"]:
            assert len(f["members"]) == f["m"]


def test_episode_design_and_split():
    assert list(PRO.TRAIN_PATIENTS) == list(TRAIN_PATIENTS)
    assert list(PRO.HELD_OUT_PATIENTS) == list(TEST_PATIENTS)
    got = [(e["ep"], e["patient"], e["target"]) for e in BOARD["episodes"]]
    assert got == [(e["ep"], e["patient"], e["target"]) for e in PRO.episode_design()]
    for key, v in VALUES.items():
        for s, x in v.items():
            assert len(x["final_pos"]) == PRO.N_EPISODES, (key, s)


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_row_recomputed_equals_record(row):
    g = GOT[row["id"]]
    assert _close(row["err_mm"], g["err_mm"])
    assert _close(row["holm_p"], g["holm_p"])
    assert row["decision"] == g["decision"] and sorted(row["notes"]) == sorted(g["notes"])
    if row["diff"] is None:
        assert g["diff"] is None
    else:
        for k in ("mean", "ci", "p_t", "sd", "sign", "n", "seeds", "per_seed", "k_seed", "n_seed", "tost_ci90",
                  "tost_p", "tost_in", "per_patient"):
            assert _close(row["diff"][k], g["diff"][k]), k
    if row["kind"] == "cta":
        assert _close(row["holm_p_tost"], g["holm_p_tost"]) and row["equivalent"] == g["equivalent"]


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_row_recomputed_equals_printed(row):
    slot = next(s for s in BOARD["slots"] if s["key"] == row["slot"])
    g = dict(row, **{k: GOT[row["id"]][k] for k in ("err_mm", "diff", "holm_p", "decision", "notes")})
    if row["kind"] == "cta":
        g["equivalent"] = GOT[row["id"]]["equivalent"]
    pc = RD.plain_cells(g, slot["decimals"])
    for k in ("error", "difference", "seeds", "holm_p", "tost", "decision", "notes"):
        assert pc[k] == row["printed"][k], k


@pytest.mark.parametrize("slot", [s for s in BOARD["slots"] if s.get("mdd")], ids=lambda s: s["key"])
def test_slot_mdd(slot):
    m = RC.slot_mdd(BOARD, VALUES, slot)
    for k in ("sd_mm", "mdd_mm", "mdd_holm_mm"):
        assert _close(slot["mdd"][k], m[k])


def test_methods_list():
    assert METHODS["counts_table2"] == {"anatomy": 1, "renderer": 2, "texture": 4, "mask": 7, "ctrl": 8, "reward": 5}
    assert METHODS["n_published"] == 27 == len(METHODS["published_methods"])
    cites = {k for r in ROWS for k in r["cites"]}
    assert cites <= set(METHODS["works"])
    for k, w in METHODS["works"].items():
        if w["doi"]:
            assert DOI_RE.match(w["doi"]) and w["verified"]["registry"] in ("Crossref", "DataCite"), k
            assert w["year"] in (w["verified"]["issued_year"], w["verified"]["print_year"]), k
        else:
            assert w["no_doi"] and w["identifier"] and w["why"], k
    rows = {r["id"] for r in ROWS}
    for m in METHODS["published_methods"]:
        assert set(m["rows"]) <= rows, m["name"]
        assert m["adapter"] and (m["license_notes"] or m["license"]), m["name"]


def test_no_number_outside_the_rows():
    """Every printed difference has the paired-t form and every Holm p three decimals or <0.001."""
    for r in ROWS:
        p = r["printed"]
        if p["difference"]:
            assert re.fullmatch(r"[-+]?\d+\.\d{2,3} \[[-+]\d+\.\d{2,3}, [-+]\d+\.\d{2,3}\], \d/6", p["difference"])
        if p["holm_p"]:
            assert re.fullmatch(r"<0\.001|\d\.\d{3}", p["holm_p"])


def test_patient_seed_interval_of_every_three_seed_row():
    """The patient-by-seed bootstrap of usbg.leaderboard.stats reproduces the interval of every row that has one."""
    from usbg.leaderboard import stats as S
    ep = BOARD["episodes"]
    n = 0
    for r in ROWS:
        ps = (r.get("diff") or {}).get("ps_ci")
        if not ps:
            continue
        a = RC.pm_of(VALUES, r["minuend"] if r["kind"] == "imnt" else r["values_of"], ep)
        b = RC.pm_of(VALUES, r["against"], ep)
        assert S.patient_seed_ci(a, b, BOARD["patients"]) == pytest.approx(ps, abs=1e-9), r["id"]
        n += 1
    assert n >= 40


def test_reference_years_are_years():
    """Every work's year appears in its reference text outside a page range (a page number is not a year)."""
    for k, w in METHODS["works"].items():
        text = re.sub(r"\b\d+\s*-+\s*\d+\b", " ", w["reference"])
        assert w["year"] and re.search(rf"\b{w['year']}\b", text), (k, w["year"])
        if w.get("arxiv_doi_verified"):
            assert w["arxiv_doi_verified"]["year"] <= w["year"], k
