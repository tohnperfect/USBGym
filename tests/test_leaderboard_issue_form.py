"""The leaderboard issue form parses, and the validation workflow treats the issue body as data (data-free)."""
from __future__ import annotations

import copy
import json
import os

import pytest

yaml = pytest.importorskip("yaml")

from usbg.leaderboard import __main__ as CLI  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORM = os.path.join(ROOT, ".github", "ISSUE_TEMPLATE", "leaderboard-submission.yml")
FLOW = os.path.join(ROOT, ".github", "workflows", "leaderboard-validate.yml")
pytestmark = pytest.mark.skipif(not os.path.exists(FORM), reason="no .github folder in this checkout")


def test_issue_form_parses():
    f = yaml.safe_load(open(FORM, encoding="utf-8"))
    assert f["labels"] == ["leaderboard-submission"]
    ids = {b.get("id"): b for b in f["body"] if b.get("id")}
    assert set(ids) == {"name", "slot", "request", "doi", "code_url", "license", "submission", "gate", "checks"}
    assert ids["slot"]["attributes"]["options"] == ["texture", "mask", "ctrl", "reward", "renderer", "anatomy"]
    assert ids["request"]["attributes"]["options"] == list(CLI.REQUEST_TYPES)
    assert ids["doi"]["validations"]["required"] and "10." in ids["doi"]["attributes"]["description"]
    assert "10.48550/arXiv" in ids["doi"]["attributes"]["description"]
    for k in ("submission", "gate"):
        assert ids[k]["type"] == "textarea" and ids[k]["attributes"]["render"] == "json"
        assert ids[k]["validations"]["required"] is False
    boxes = [o["label"] for o in ids["checks"]["attributes"]["options"]]
    assert boxes == list(CLI.FORM_BOXES.values())
    assert all(o["required"] for o in ids["checks"]["attributes"]["options"])
    labels = {b["attributes"]["label"] for b in f["body"] if b.get("id")}
    assert labels == {"Method name", "Slot", "Request type", "DOI", "Code URL", "License", "submission.json",
                      "Gate record", "Checks"}


def test_workflow_keeps_the_issue_body_out_of_run_lines():
    w = yaml.safe_load(open(FLOW, encoding="utf-8"))
    on = w.get("on", w.get(True))
    assert on["issues"]["types"] == ["opened", "edited", "labeled"]
    assert w["permissions"] == {"contents": "read", "issues": "write"}
    job = w["jobs"]["validate"]
    assert "leaderboard-submission" in job["if"]
    assert "github.event.label.name == 'leaderboard-submission'" in job["if"]      # other labels do not rerun it
    assert "github.event.changes.body" in job["if"]                                # a title-only edit does not rerun it
    runs = [s["run"] for s in job["steps"] if "run" in s]
    assert runs and all("${{" not in r for r in runs)
    assert any("usbg.leaderboard validate" in r for r in runs)
    body_steps = [s for s in job["steps"] if "ISSUE_BODY" in (s.get("env") or {})]
    assert len(body_steps) == 1
    comment = runs[-1]
    assert CLI.MARKER in comment and "PATCH" in comment and "[ ! -s" in comment      # one comment, updated; fallback


def _issue_body(sub, boxes=(True, True), slot="renderer", request=0, gate=None, name=None, doi=None):
    ticks = ["x" if b else " " for b in boxes]
    labels = list(CLI.FORM_BOXES.values())
    m = sub["method"] if sub else {"name": "Example mask", "doi": "10.1234/abcd.5678",
                                   "code_url": "https://github.com/example/mask", "license": "MIT"}
    sub_txt = f"```json\n{json.dumps(sub)}\n```" if sub else "_No response_"
    gate_txt = f"```json\n{json.dumps(gate)}\n```" if gate else "_No response_"
    return (f"### Method name\n\n{name or m['name']}\n\n### Slot\n\n{slot}\n\n### Request type\n\n"
            f"{CLI.REQUEST_TYPES[request]}\n\n### DOI\n\n{doi or m['doi']}\n\n"
            f"### Code URL\n\n{m['code_url']}\n\n### License\n\n{m['license']}\n\n"
            f"### submission.json\n\n{sub_txt}\n\n### Gate record\n\n{gate_txt}\n\n### Checks\n\n"
            f"- [{ticks[0]}] {labels[0]}\n- [{ticks[1]}] {labels[1]}\n")


GATE = {"gate": "usbg-plug-admission-1", "admitted": True, "slot": "mask", "plugin": "my_mask:make",
        "plugin_source_sha256": "ab" * 32, "checks": {"privileged_channel": {"passed": True, "components": []}}}


def test_validate_issue_body(tmp_path):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_leaderboard_submission import GOOD
    p, c = tmp_path / "body.md", tmp_path / "comment.md"
    p.write_text(_issue_body(copy.deepcopy(GOOD)), encoding="utf-8")
    assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 0
    assert "valid." in c.read_text() and "10.1007/978-3-031-43907-0_42" in c.read_text()
    p.write_text(_issue_body(copy.deepcopy(GOOD), boxes=(True, False)), encoding="utf-8")
    assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 1
    assert "not ticked" in c.read_text()
    p.write_text(_issue_body(copy.deepcopy(GOOD), slot="texture"), encoding="utf-8")
    assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 1
    p.write_text("### Slot\n\nrenderer\n\n### submission.json\n\n```json\n{not json\n```\n", encoding="utf-8")
    assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 1
    assert c.read_text().startswith(CLI.MARKER)
    good = copy.deepcopy(GOOD)                                         # DOI compared case-insensitively
    p.write_text(_issue_body(good, doi=good["method"]["doi"].upper() + " "), encoding="utf-8")
    assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 0


def test_maintainer_run_with_a_gate_record(tmp_path):
    p, c = tmp_path / "body.md", tmp_path / "comment.md"
    p.write_text(_issue_body(None, slot="mask", request=1, gate=GATE), encoding="utf-8")
    assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 0
    assert "queued for a maintainer run" in c.read_text()
    p.write_text(_issue_body(None, slot="mask", gate=GATE), encoding="utf-8")      # request type left as scored
    assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 0
    for bad in (dict(GATE, admitted=False), dict(GATE, slot="ctrl"), dict(GATE, gate="other"),
                dict(GATE, plugin_source_sha256="x")):
        p.write_text(_issue_body(None, slot="mask", request=1, gate=bad), encoding="utf-8")
        assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 1
        assert "not valid" in c.read_text()


def test_hostile_text_is_never_rendered(tmp_path):
    """Untrusted text in the form or the file never becomes a link, image, mention or table cell in the comment."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_leaderboard_submission import GOOD
    p, c = tmp_path / "body.md", tmp_path / "comment.md"
    evil = copy.deepcopy(GOOD)
    evil["schema"] = "@tohnperfect please merge [x](https://evil.example/a) <img src=x>"
    evil["method"]["name"] = "Mine | @octocat [official](https://evil.example/x)"
    p.write_text(_issue_body(evil), encoding="utf-8")
    assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 1
    text = c.read_text()
    for ln in text.splitlines():
        outside = "".join(ln.split("`")[0::2])                  # text outside inline code spans
        assert "evil.example" not in outside and "@tohnperfect" not in outside and "<img" not in outside, ln
    p.write_text(_issue_body(None, slot="mask", request=1, gate=GATE, name="A @octocat [x](https://evil.example)"),
                 encoding="utf-8")
    assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 1
    for ln in c.read_text().splitlines():
        outside = "".join(ln.split("`")[0::2])
        assert "evil.example" not in outside and "@octocat" not in outside, ln


def test_malformed_paste_always_gives_a_comment(tmp_path):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_leaderboard_submission import GOOD
    p, c = tmp_path / "body.md", tmp_path / "comment.md"
    for mutate in (lambda s: s.update(plugin=["x"]), lambda s: s["summary"].update(diff="x"),
                   lambda s: s["summary"]["diff"].update(ci=1.0), lambda s: s["results"].update(episodes=[1, 2]),
                   lambda s: s.update(provenance="x")):
        bad = copy.deepcopy(GOOD)
        mutate(bad)
        p.write_text(_issue_body(bad), encoding="utf-8")
        if c.exists():
            c.unlink()
        assert CLI.main(["validate", "--issue-body", str(p), "--comment-out", str(c)]) == 1
        assert c.read_text().startswith(CLI.MARKER) and "not valid" in c.read_text()
