"""LEADERBOARD.md lists the published rows in the order of Table C.1 of the article (Appendix C), row by row
(data-free).

The order fixture tests/data/leaderboard_order.json holds the 68 rows of Table C.1 in table order (slot, row id,
role, level and printed error). In each block that shares one reference: the reference, then the rows whose
difference is taken from it by unrounded mean error with the privileged rows last; then the reference's seed-0 line
and the rows run at seed 0 only, in the same way; the CT-anatomy rows keep Table 5's order. The four
pretrained-backbone differences have no error of their own and are outside the order.
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from usbg.leaderboard import recompute as RC
from usbg.leaderboard import render as RD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "data", "leaderboard_order.json")
MD = os.path.join(ROOT, "LEADERBOARD.md")
BOARD = RC.load_board()


def _fixture():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def _order(board):
    out = []
    for slot in board["slots"]:
        for r, level in RD.slot_order(slot):
            out.append({"slot": slot["key"], "id": r["id"], "role": RD.role_of(slot, r), "level": level,
                        "error": f"{r['err_mm']:.2f}"})
    return out


def test_fixture_shape():
    fx = _fixture()
    assert fx["n_rows"] == len(fx["rows"]) == 68
    assert [s["key"] for s in BOARD["slots"]] == list(dict.fromkeys(r["slot"] for r in fx["rows"]))


@pytest.mark.parametrize("i", range(68))
def test_row_matches_table_c1(i):
    """Row i of the leaderboard order is row i of Table C.1: same slot, row, role, block level and printed error."""
    assert _order(BOARD)[i] == _fixture()["rows"][i]


def test_order_has_every_published_row_once():
    ids = [r["id"] for r in _order(BOARD)]
    assert len(ids) == len(set(ids)) == 68
    imnt = [r["id"] for s in BOARD["slots"] for r in RD.pretrained_rows(s)]
    assert len(imnt) == 4 and not set(imnt) & set(ids)
    assert set(ids) | set(imnt) == {r["id"] for s in BOARD["slots"] for r in s["rows"]}


def test_blocks_by_unrounded_error_privileged_last():
    for slot in BOARD["slots"]:
        if slot["key"] == "anatomy":
            assert [r["id"] for r, _ in RD.slot_order(slot)] == [r["id"] for r in slot["rows"]]
            continue
        order = RD.slot_order(slot)
        for level in (0, 1):
            block = [r for r, lv in order if lv == level]
            if not block:
                continue
            assert block[0]["role"] == "ref"
            rest = block[1:]
            priv = [bool(r.get("privileged")) for r in rest]
            assert priv == sorted(priv), slot["key"]                  # privileged rows last
            for flag in (False, True):
                errs = [r["err_mm"] for r in rest if bool(r.get("privileged")) == flag]
                assert errs == sorted(errs), (slot["key"], level, flag)
            assert all((list(r["seeds"]) == [0]) == (level == 1) for r in rest)


def test_unrounded_error_and_ties():
    """Two rows that print the same error are ordered by the unrounded error; equal errors keep Table 5's order."""
    board = copy.deepcopy(BOARD)
    slot = next(s for s in board["slots"] if s["key"] == "renderer")
    a, b = (r for r in slot["rows"] if r["id"] in ("renderer/lotus_orig", "renderer/lotus_reimpl"))
    a["err_mm"], b["err_mm"] = 11.8449, 11.8401                       # both print 11.84
    ids = [r["id"] for r, _ in RD.slot_order(slot)]
    assert ids.index("renderer/lotus_reimpl") < ids.index("renderer/lotus_orig")
    a["err_mm"] = b["err_mm"] = 11.85
    ids = [r["id"] for r, _ in RD.slot_order(slot)]
    first = [r["id"] for r in slot["rows"] if r["id"] in ("renderer/lotus_orig", "renderer/lotus_reimpl")][0]
    assert ids.index(first) < ids.index(({"renderer/lotus_orig", "renderer/lotus_reimpl"} - {first}).pop())


def test_seed0_rows_need_the_seed0_line():
    board = copy.deepcopy(BOARD)
    slot = next(s for s in board["slots"] if s["key"] == "renderer")
    slot["rows"] = [r for r in slot["rows"] if r.get("kind") != "ref_seed0"]
    with pytest.raises(ValueError, match="seed-0 line"):
        RD.slot_order(slot)


def test_markdown_lists_rows_in_order():
    text = RD.markdown(BOARD, RC.load_methods())
    lines = text.splitlines()
    for slot in BOARD["slots"]:
        start = lines.index(f"## {slot['title']}")
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        table = [ln for ln in lines[start:end] if ln.startswith("| ") and not ln.startswith("| :--")][1:]
        order = RD.slot_order(slot)
        assert len(table) >= len(order)
        for ln, (r, level) in zip(table, order):
            cell = ln.split(" | ")[0][2:]
            assert cell.startswith(RD.INDENT * level + r["method"]), (ln, r["id"])
            assert not cell.startswith(RD.INDENT * (level + 1))
            assert ln.split(" | ")[1] == RD.role_of(slot, r)
    assert "is not a ranking" in text and "Table C.1" in text
    assert "Pretrained-backbone differences, outside the order" in text


@pytest.mark.skipif(not os.path.exists(MD), reason="LEADERBOARD.md not in this checkout")
def test_leaderboard_md_in_table_c1_order():
    text = open(MD, encoding="utf-8").read()
    method = {x["id"]: x["method"] for s in BOARD["slots"] for x in s["rows"]}
    last = -1
    for r in _fixture()["rows"]:          # each row's line comes after the line of the row before it
        last = text.index("\n| " + RD.INDENT * r["level"] + method[r["id"]], last + 1)
