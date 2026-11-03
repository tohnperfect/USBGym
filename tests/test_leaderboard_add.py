"""The maintainers' add: a validated submission enters the leaderboard with its marks, its patient-by-seed interval and
its registry-checked reference in the methods list; recompute covers it (data-free, no network).

The registry records are synthetic (Crossref's test prefix 10.5555), given with --doi-metadata as a saved record.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys

import pytest

from usbg.leaderboard import __main__ as CLI
from usbg.leaderboard import doi as DOI
from usbg.leaderboard import recompute as RC
from usbg.leaderboard import render as RD

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_leaderboard_submission import make_submission  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(RC.__file__)), "data")
TITLE = "A synthetic renderer for testing the leaderboard"


def _crossref(doi, title=TITLE):
    return {"message": {"DOI": doi, "title": [f"<i>{title}</i>"], "type": "journal-article",
                        "author": [{"given": "Ann-Marie", "family": "Tester"}, {"given": "Bo", "family": "Example"}],
                        "issued": {"date-parts": [[2026, 5]]}, "published-print": {"date-parts": [[2026, 6]]},
                        "container-title": ["J. Test. Leaderboards"], "volume": "1", "page": "1-10"}}


@pytest.fixture()
def data(tmp_path):
    d = tmp_path / "data"
    shutil.copytree(DATA, d)
    return d


def _add(tmp_path, data, sub, doi_meta, *extra):
    s, m = tmp_path / f"sub_{len(os.listdir(tmp_path))}.json", tmp_path / f"meta_{len(os.listdir(tmp_path))}.json"
    s.write_text(json.dumps(sub), encoding="utf-8")
    m.write_text(json.dumps(doi_meta), encoding="utf-8")
    return CLI.main(["add", str(s), "--data-dir", str(data), "--leaderboard-md", str(tmp_path / "LB.md"),
                     "--doi-metadata", str(m), *extra])


def _sub(name, doi, row="renderer/lotus_orig", learned=True, privileged=False, label_access=False):
    sub = json.loads(json.dumps(make_submission(row, learned=learned)))
    sub["method"].update(name=name, doi=doi, privileged=privileged, label_access=label_access)
    from usbg.leaderboard import submission as SUB
    board, values = RC.load_board(), RC.load_values()
    summ = SUB.summarize(sub["results"]["per_seed"], "renderer", sub["method"], board, values)
    sub["summary"] = json.loads(json.dumps({k: summ[k] for k in ("err_mm", "diff", "decision", "label", "reference")}))
    return sub


def test_add_resolves_the_doi_and_builds_the_reference(tmp_path, data):
    doi = "10.5555/usbg.test.0001"
    assert _add(tmp_path, data, _sub("Test renderer", doi), _crossref(doi), "--confirm-title", TITLE) == 0
    methods = json.load(open(data / "methods.json", encoding="utf-8"))
    m = methods["submitted_methods"][-1]
    w = m["references"][0]
    assert w["doi"] == doi and w["verified"]["registry"] == "Crossref" and w["title"] == TITLE and w["year"] == 2026
    assert w["reference"] == f"A.-M. Tester, B. Example, {TITLE}, J. Test. Leaderboards 1 (2026) 1-10."
    text = (tmp_path / "LB.md").read_text(encoding="utf-8")
    assert w["reference"] in text and f"[{doi}](https://doi.org/{doi})" in text
    assert "patient-by-seed [" in text and "| post hoc |" in text


def test_add_refuses_a_title_that_does_not_match(tmp_path, data):
    doi = "10.5555/usbg.test.0002"
    with pytest.raises(SystemExit, match="title"):
        _add(tmp_path, data, _sub("Wrong paper", doi), _crossref(doi, title="Deep Residual Learning"),
             "--confirm-title", TITLE)
    with pytest.raises(SystemExit, match="title"):
        _add(tmp_path, data, _sub("No title given", doi), _crossref(doi))


def test_add_refuses_a_doi_already_listed(tmp_path, data):
    used = "10.1109/CVPR.2016.90"                                  # the ResNet work of the published methods list
    assert _add(tmp_path, data, _sub("Reused doi", used), _crossref(used), "--confirm-title", TITLE) == 1
    doi = "10.5555/usbg.test.0003"
    assert _add(tmp_path, data, _sub("First", doi), _crossref(doi), "--confirm-title", TITLE) == 0
    assert _add(tmp_path, data, _sub("Second", doi.upper()), _crossref(doi), "--confirm-title", TITLE) == 1


def test_marks_privileged_group_seed0_and_recompute(tmp_path, data):
    doi_a, doi_b = "10.5555/usbg.test.0004", "10.5555/usbg.test.0005"
    assert _add(tmp_path, data, _sub("Pose renderer", doi_a, privileged=True, label_access=True), _crossref(doi_a),
                "--confirm-title", TITLE) == 0
    sub0 = _sub("Seed zero renderer", doi_b, row="renderer/simus", learned=False)
    assert _add(tmp_path, data, sub0, _crossref(doi_b), "--confirm-title", TITLE) == 0
    board = json.load(open(data / "leaderboard.json", encoding="utf-8"))
    values = json.load(open(data / "values.json", encoding="utf-8"))
    priv = next(r for r in board["submitted"] if r["method"]["name"] == "Pose renderer")
    assert priv["privileged"] and priv["label_access"] and priv["decision"] == "secondary"
    text = (tmp_path / "LB.md").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if "Pose renderer" in ln and "| " in ln and "code" in ln)
    assert "ᴸ" in line and "†" in line and "secondary (privileged)" in line
    assert "*Submitted, privileged, secondary;" in text
    line0 = next(ln for ln in text.splitlines() if "Seed zero renderer" in ln and "code" in ln)
    assert "seed 0 only, against the reference's seed 0" in line0
    assert CLI.recompute_submitted(board, values) == []
    tampered = copy.deepcopy(board)
    tampered["submitted"][0]["err_mm"] += 0.5
    assert CLI.recompute_submitted(tampered, values)


def test_rendering_escapes_submitted_text():
    assert RD.md_text("a | b @c [d](e) <img> `x` *y* _z_") == \
        "a \\| b &#64;c \\[d\\](e) &lt;img&gt; \\`x\\` \\*y\\* \\_z\\_"
    assert RD.doi_link("10.1016/S0031-3203(98)00091-0") == \
        "[10.1016/S0031-3203(98)00091-0](https://doi.org/10.1016/S0031-3203%2898%2900091-0)"
    assert RD.url_link("code", "https://github.com/a/b)[x](https://evil.example/") == "code"


def test_registry_parsers():
    rec = DOI.parse("DataCite", {"data": {"attributes": {
        "doi": "10.48550/ARXIV.1234.56789", "titles": [{"title": "An arXiv work"}], "publicationYear": 2024,
        "creators": [{"givenName": "Jun-Yan", "familyName": "Zhu"}, {"name": "A Consortium"}],
        "publisher": "arXiv", "types": {"resourceTypeGeneral": "Preprint"}}}})
    assert rec["title"] == "An arXiv work" and rec["authors"] == ["J.-Y. Zhu", "A Consortium"]
    assert DOI.reference_text(rec) == "J.-Y. Zhu, A Consortium, An arXiv work, arXiv, 2024."
    rec = DOI.parse("Crossref", {"message": {        # a record with an article number: no page field
        "DOI": "10.5555/usbg.test.0002", "title": ["An article"], "type": "journal-article",
        "author": [{"given": "Jun", "family": "Ma"}], "issued": {"date-parts": [[2024, 1, 22]]},
        "container-title": ["Nature Communications"], "volume": "15", "article-number": "654"}})
    assert DOI.reference_text(rec) == "J. Ma, An article, Nature Communications 15 (2024) 654."
    with pytest.raises(SystemExit, match="not"):
        DOI.resolve("10.5555/other", TITLE, metadata=_crossref("10.5555/usbg.test.0001"))


def test_unprivileged_controller_never_sees_privileged_info():
    """Controller slot: a plug-in not declared privileged gets info without its privileged keys."""
    from usbg.leaderboard._exec import Unprivileged

    class Spy:
        seen = []

        def reset(self, *a, **kw):
            return None

        def act(self, obs, info):
            self.seen.append(set(info))
            return [0.0] * 6

    spy = Spy()
    w = Unprivileged(spy)
    info = {"privileged": {"pose_error_mm": 3.0}, "match": 0.5, "seg_iou": 0.1}
    w.reset(0, None)
    w.act(None, info)
    assert spy.seen == [{"match", "seg_iou"}] and "privileged" in info      # the scoring code keeps the full info
