"""usbg.leaderboard.recompute: every leaderboard row recomputed from the shipped per-episode values.

The leaderboard data (data/leaderboard.json) carries each row's statistics as the article printed them, and
data/values.json carries the per-episode final translation errors behind them (per row and seed, 60 values in
the episode order of leaderboard.json "episodes"). recompute_board() rebuilds every row from the values alone;
tests/test_leaderboard_rows.py checks that the rebuilt rows equal the shipped ones and their printed cells.
"""
from __future__ import annotations

import json
import os

from . import stats as S

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_json(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def load_board():
    return load_json("leaderboard.json")


def load_values():
    return load_json("values.json")


def load_methods():
    return load_json("methods.json")


def pm_of(values, key, episodes, seeds=None):
    """{seed: {patient: mean final error}} of one value set, restricted to `seeds` when given."""
    v = values[key]
    keep = {str(s) for s in seeds} if seeds is not None else set(v)
    return S.seed_patient_means({s: x for s, x in v.items() if s in keep}, episodes)


def row_pm(row, values, episodes):
    if row.get("values_of") is None:
        return None
    return pm_of(values, row["values_of"], episodes, row["seeds"] if row["kind"] == "ref_seed0" else None)


def _rows(board):
    return {r["id"]: r for s in board["slots"] for r in s["rows"]}


def recompute_row(row, board, values, holm_p=None):
    """{"err_mm", "diff", "holm_p", "decision", "notes"} of one row from the per-episode values."""
    ep = board["episodes"]
    pm = row_pm(row, values, ep)
    out = {"err_mm": S.row_error(pm) if pm is not None else None, "holm_p": holm_p}
    c = None
    if row.get("against"):
        a = pm if row["kind"] != "imnt" else pm_of(values, _rows(board)[row["minuend"]]["values_of"], ep)
        b = pm_of(values, row["against"], ep)
        # a contrast needs a complete row (all designed seeds), and a learned method needs three seeds
        complete = len(a) == len(row["designed_seeds"]) and not (row.get("learned", True) and len(a) < 3
                                                                   and row["kind"] == "row")
        c = S.contrast(a, b, board["patients"]) if complete else None
    out["diff"] = c
    if row["kind"] == "cta":
        return out
    short = bool(row.get("fewer_seeds"))
    dec, note = S.decide(row["kind"], row["role"], c, holm_p, short=short, demote=row.get("demoted", False))
    out["decision"] = dec.replace(S.NOT_POWERED, "")
    out["notes"] = ["a"] if note else []
    return out


def recompute_board(board, values):
    """{row id: recomputed row} for every row, with Holm's adjustment within each family of each slot."""
    out = {}
    for slot in board["slots"]:
        rows = {r["id"]: r for r in slot["rows"]}
        for r in slot["rows"]:
            out[r["id"]] = recompute_row(r, board, values)
        for fam in slot["families"]:
            ps = []
            for m in fam["members"]:
                c = out[m]["diff"] if m in out else None
                complete = c is not None
                if fam["key"] == "CTA-E":
                    ps.append(S.tost(list(c["per_patient"].values()))["p"] if complete else 1.0)
                else:
                    ps.append(c["p_t"] if complete else 1.0)
            adj = S.holm(ps)
            for m, h in zip(fam["members"], adj):
                if m not in out or out[m]["diff"] is None:
                    continue
                if fam["key"] == "CTA-E":
                    out[m]["holm_p_tost"] = h
                elif rows[m].get("family") == fam["key"]:
                    out[m]["holm_p"] = h
        for r in slot["rows"]:
            g = out[r["id"]]
            if r["kind"] == "cta":
                dec, eq, notes = S.decide_anatomy(g["diff"], g["holm_p"], g["holm_p_tost"])
                g.update(decision=dec, equivalent=eq, notes=notes)
            else:
                dec, note = S.decide(r["kind"], r["role"], g["diff"], g["holm_p"], short=r.get("fewer_seeds", False),
                                     demote=r.get("demoted", False))
                g.update(decision=dec.replace(S.NOT_POWERED, ""), notes=["a"] if note else [])
    return out


def slot_mdd(board, values, slot):
    """Post hoc MDD of a slot: median SD of the per-patient differences of the slot's MDD rows from the reference,
    paired t at alpha 0.05 and at 0.05 / m, power 0.8, n = 6."""
    import numpy as np
    ep = board["episodes"]
    ref = pm_of(values, slot["reference"], ep)
    sds = [S.contrast(pm_of(values, r, ep), ref, board["patients"])["sd"] for r in slot["mdd"]["rows"]]
    sd = float(np.median(sds))
    m = slot["families"][0]["m"]
    return {"sd_mm": sd, "mdd_mm": S.mdd(sd), "mdd_holm_mm": S.mdd(sd, alpha=S.ALPHA / m)}


__all__ = ["load_board", "load_values", "load_methods", "recompute_board", "recompute_row", "slot_mdd", "pm_of"]
