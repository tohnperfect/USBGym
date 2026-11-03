"""submission.json: schema, validation and the recomputed summary, on good and malformed input (data-free).

The good submission is built from the shipped per-episode values of a published row (the LOTUS original-code
renderer), so its summary must reproduce that row's difference from the physics renderer exactly.
"""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from usbg.leaderboard import protocol as PRO
from usbg.leaderboard import recompute as RC
from usbg.leaderboard import stats as S
from usbg.leaderboard import submission as SUB

BOARD, VALUES = RC.load_board(), RC.load_values()


def make_submission(row_id="renderer/lotus_orig", slot="renderer", learned=True):
    per_seed = {}
    for s, v in VALUES[row_id].items():
        per_seed[s] = {"final_pos": list(v["final_pos"]), "per_patient": S.patient_means(v["final_pos"],
                                                                                         BOARD["episodes"])}
    method = {"name": "Example method", "slot": slot, "doi": "10.1007/978-3-031-43907-0_42",
              "code_url": "https://github.com/example/method", "license": "MIT", "description": "test",
              "learned": learned, "privileged": False, "label_access": False}
    summ = SUB.summarize(per_seed, slot, method, BOARD, VALUES)
    sha = hashlib.sha256(b"example").hexdigest()
    return {"schema": PRO.SUBMISSION_SCHEMA, "protocol": PRO.PROTOCOL_VERSION, "method": method,
            "plugin": {"entry": "example_pkg.plugin:make", "source_sha256": sha},
            "protocol_settings": SUB.protocol_settings([int(s) for s in per_seed]),
            "results": {"episodes": [{"ep": e["ep"], "patient": e["patient"], "target": e["target"]}
                                     for e in BOARD["episodes"]], "per_seed": per_seed},
            "summary": {k: summ[k] for k in ("err_mm", "diff", "decision", "label", "reference")},
            "provenance": {"usbg_version": "2.2.0", "plugin_source_sha256": sha, "python": "3.12", "torch": "2",
                           "numpy": "2", "device": "cpu", "wall_time_s": 1.0}}


GOOD = json.loads(json.dumps(make_submission()))


def test_good_submission_valid_and_equal_to_the_published_row():
    errs, summ = SUB.validate(copy.deepcopy(GOOD), BOARD, VALUES)
    assert errs == []
    row = next(r for s in BOARD["slots"] for r in s["rows"] if r["id"] == "renderer/lotus_orig")
    assert abs(summ["err_mm"] - row["err_mm"]) < 1e-9
    for k in ("mean", "p_t", "sign", "k_seed", "tost_in"):
        assert summ["diff"][k] == pytest.approx(row["diff"][k], abs=1e-12)
    assert summ["label"] == S.SUBMITTED and summ["decision"] == S.DEC_POSTHOC      # no pre-specified decision word
    # the patient-by-seed interval of Appendix B equals the one shipped with the published row
    assert summ["diff"]["ps_ci"] == pytest.approx(row["diff"]["ps_ci"], abs=1e-9)


def test_privileged_submission_is_secondary():
    sub = make_submission()
    sub["method"]["privileged"] = True
    summ = SUB.summarize(sub["results"]["per_seed"], "renderer", sub["method"], BOARD, VALUES)
    assert summ["decision"] == S.DEC_SEC


def test_seed0_method_pairs_with_reference_seed0():
    sub = make_submission("renderer/simus", learned=False)
    errs, summ = SUB.validate(json.loads(json.dumps(sub)), BOARD, VALUES)
    assert errs == []
    row = next(r for s in BOARD["slots"] for r in s["rows"] if r["id"] == "renderer/simus")
    assert summ["diff"]["seeds"] == [0] and summ["diff"]["mean"] == pytest.approx(row["diff"]["mean"], abs=1e-12)
    assert summ["diff"]["ps_ci"] is None and summ["seeds"] == [0]
    ref0 = S.row_error({0: RC.pm_of(VALUES, "renderer/physics", BOARD["episodes"], [0])[0]})
    assert summ["reference_err_mm"] == pytest.approx(ref0, abs=1e-12)
    cells = SUB.summary_row(sub, summ)
    assert cells["seeds"].startswith("seed 0 only, against the reference's seed 0")


def _broken(fn):
    sub = copy.deepcopy(GOOD)
    fn(sub)
    return SUB.validate(sub, BOARD, VALUES)[0]


@pytest.mark.parametrize("name,fn,needle", [
    ("schema", lambda s: s.update(schema="other"), "schema"),
    ("protocol", lambda s: s.update(protocol="usbg-plug-protocol-0"), "protocol"),
    ("doi url", lambda s: s["method"].update(doi="https://doi.org/10.1234/x"), "method.doi"),
    ("doi junk", lambda s: s["method"].update(doi="not a doi"), "method.doi"),
    ("code url", lambda s: s["method"].update(code_url="ftp://x"), "code_url"),
    ("slot", lambda s: s["method"].update(slot="anatomy"), "method.slot"),
    ("missing name", lambda s: s["method"].pop("name"), "method.name"),
    ("missing seed", lambda s: s["results"]["per_seed"].pop("2"), "seeds 0, 1 and 2"),
    ("short seed", lambda s: s["results"]["per_seed"]["1"]["final_pos"].pop(), "final_pos"),
    ("nan", lambda s: s["results"]["per_seed"]["0"]["final_pos"].__setitem__(3, float("nan")), "final_pos"),
    ("per patient", lambda s: s["results"]["per_seed"]["0"]["per_patient"].__setitem__("s0358", 1.0), "per_patient"),
    ("episodes", lambda s: s["results"]["episodes"].reverse(), "results.episodes"),
    ("steps", lambda s: s["protocol_settings"].update(n_steps=36), "n_steps"),
    ("tuned", lambda s: s["protocol_settings"].update(tuned_on_held_out_patients=True), "tuned"),
    ("summary err", lambda s: s["summary"].update(err_mm=1.0), "summary.err_mm"),
    ("summary diff", lambda s: s["summary"]["diff"].update(mean=-5.0), "summary.diff"),
    ("summary decision", lambda s: s["summary"].update(decision="lower"), "summary.decision"),
    ("plugin entry", lambda s: s["plugin"].update(entry="not an entry"), "plugin.entry"),
    ("provenance", lambda s: s["provenance"].pop("device"), "provenance.device"),
    ("abs path", lambda s: s["provenance"].update(commands=["/root/x/.venv/bin/python -m usbg.leaderboard._exec"]),
     "absolute local path"),
    ("windows path", lambda s: s["provenance"].update(commands=["python --out D:\\data\\a.json"]),
     "absolute local path"),
    ("name newline", lambda s: s["method"].update(name="A\nB"), "method.name"),
    ("name mention", lambda s: s["method"].update(name="A @octocat"), "method.name"),
    ("name link", lambda s: s["method"].update(name="[official](https://evil.example/x)"), "method.name"),
    ("name pipe", lambda s: s["method"].update(name="A | B"), "method.name"),
    ("license html", lambda s: s["method"].update(license="<img src=x>"), "method.license"),
    ("description", lambda s: s["method"].update(description=""), "method.description"),
    ("doi brackets", lambda s: s["method"].update(doi="10.1234/](https://evil.example/doi)["), "method.doi"),
    ("code url paren", lambda s: s["method"].update(code_url="https://github.com/a/b)[click](https://evil.example/"),
     "code_url"),
    ("code url space", lambda s: s["method"].update(code_url="https://github.com/a b"), "code_url"),
])
def test_malformed(name, fn, needle):
    errs = _broken(fn)
    assert errs and any(needle in e for e in errs), errs


def _set(path, value):
    def fn(s):
        d = s
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return fn


@pytest.mark.parametrize("path,value", [
    (("plugin",), ["x"]), (("plugin",), "x"), (("provenance",), "x"), (("protocol_settings",), [1]),
    (("summary",), [1]), (("summary", "diff"), "x"), (("summary", "diff"), [1, 2]), (("summary", "diff", "ci"), 1.0),
    (("summary", "err_mm"), True), (("results", "episodes"), [1, 2, 3]), (("results", "episodes"), {"a": 1}),
    (("results", "per_seed"), ["0"]), (("results", "per_seed", "0"), [1.0]), (("results",), None),
    (("method",), "x"), (("method", "learned"), "yes"), (("plugin", "source_sha256"), 5),
])
def test_wrong_types_give_errors_not_exceptions(path, value):
    """A hand-edited or truncated paste: validate reports errors and never raises."""
    errs = _broken(_set(path, value))
    assert errs, (path, value)


def test_extract_json_with_a_pasted_fence():
    body = "### submission.json\n\n```json\n```json\n" + json.dumps(GOOD) + "\n```\n```\n\n### Checks\n"
    assert SUB.extract_json(body) == GOOD


def test_not_an_object():
    assert SUB.validate([], BOARD, VALUES)[0]


def test_extract_json_from_issue_body():
    body = "### submission.json\n\n```json\n" + json.dumps(GOOD) + "\n```\n\n### Checks\n"
    assert SUB.extract_json(body) == GOOD
