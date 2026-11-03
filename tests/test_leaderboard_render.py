"""LEADERBOARD.md is up to date with the leaderboard data, and lists every row with its reference (data-free)."""
from __future__ import annotations

import os

import pytest

from usbg.leaderboard import recompute as RC
from usbg.leaderboard import render as RD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "LEADERBOARD.md")


@pytest.mark.skipif(not os.path.exists(MD), reason="LEADERBOARD.md not in this checkout")
def test_leaderboard_md_up_to_date():
    text = RD.markdown(RC.load_board(), RC.load_methods())
    assert open(MD, encoding="utf-8").read() == text, "run: python -m usbg.leaderboard render --out LEADERBOARD.md"


def test_markdown_lists_every_row_and_doi():
    board, methods = RC.load_board(), RC.load_methods()
    text = RD.markdown(board, methods)
    for s in board["slots"]:
        assert f"## {s['title']}" in text
        for r in s["rows"]:
            assert r["method"] in text
    for w in methods["works"].values():
        assert (w["doi"] or w["identifier"]) in text
    assert chr(0x2014) not in text and "-" * 3 not in text
    assert "not a ranking" in text and "not distinguished from the reference or from each other" in text
    assert "1 anatomy source" in text and "8 controller" in text and " ctrl," not in text


def test_combination_benchmark_baseline_and_assets():
    board = RC.load_board()
    cb = board.get("combination")
    assert "benchmark_baseline" not in board         # the combination is not the benchmark baseline with every frame
    if cb is not None:
        text = RD.markdown(board, RC.load_methods())
        assert "## Combination and benchmark baseline" in text and cb["cohort"]["physics"]["combination"] in text
        for bad in ("best measured pipeline", "replaced", "best so far", "Benchmark baseline minus"):
            assert bad.lower() not in text.lower()
        # the benchmark baseline per frame type, as the article defines it: the combination only where its D
        # contrast holds lower, else the authors' original pipeline
        for comp, mid in (("physics", "D-P"), ("sonospade", "D-S")):
            b = cb["benchmark_baseline"][comp]
            d = next(c for c in cb["contrasts"] if c["id"] == mid)
            assert b["contrast"] == mid and b["decision"] == d["decision"]
            assert b["pipeline"] == ("combination" if d["decision"] == "holds lower" else "authors_original")
            assert b["error_mm"] == cb["cohort"][comp][b["pipeline"]]
        assert RD.benchmark_baseline_text(cb) in text
    from usbg.leaderboard.assets import manifest
    m = manifest()
    assert m["url_pattern"].startswith("https://github.com/tohnperfect/USBGym/releases/download/")
    for a in m["assets"]:
        assert len(a["sha256"]) == 64 and a["size"] > 0 and a["license"] and a["provenance"]["leakage_gate"]
    assert {a["slot"] for a in m["assets"]} == {"renderer"}
    assert m["not_released"]
