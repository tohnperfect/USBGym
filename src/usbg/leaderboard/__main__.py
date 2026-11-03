"""python -m usbg.leaderboard: the plug-slot leaderboard of USB Gym.

  check      the admission gate for a plug-in given as package.module:factory (training patients only)
  run        run the slot condition with the plug-in on the 60 held-out episodes and write submission.json
  validate   check a submission.json (or the body of a submission issue: a submission.json, or a gate record for a
             maintainer run) and recompute its summary
  render     write LEADERBOARD.md from the leaderboard data (--check: fail if the file is out of date)
  add        (maintainers) merge a validated submission into the leaderboard data, and its DOI, resolved at its
             registry, into the methods list
  recompute  recompute every leaderboard row, submitted rows included, from its shipped per-episode values
  fetch      download the reference artifacts of the release and verify their sha256
  template   copy the template plug-in of a slot
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sys

from . import protocol as PRO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _dump(path, obj, compact=False):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=None if compact else 1, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def cmd_check(a):
    from .gate import check
    body = check(a.slot, a.plugin, privileged=a.privileged, out=a.out)
    ch = body["checks"]
    bad = [c.get("component") for c in ch["privileged_channel"]["components"]
           + ch.get("renderer_checks", {}).get("components", []) + ch["device_parity"]["results"]
           if not c.get("passed")]
    print(f"[check] {a.plugin} ({a.slot}): {'ADMITTED' if body['admitted'] else 'NOT ADMITTED'}"
          + (f"; failed: {bad}" if bad else "") + (f" -> {a.out}" if a.out else ""))
    return 0 if body["admitted"] else 1


def cmd_run(a):
    from .runner import run
    return run(a)


MARKER = "<!-- usbg-leaderboard-validation -->"     # the workflow updates the one comment that carries it


def safe(text, n=300):
    """Untrusted text for a Markdown comment: one line, at most n characters, inside a code span (no link, image,
    HTML or mention renders there)."""
    t = " ".join(str(text).split()).replace("`", "'")
    if len(t) > n:
        t = t[:n] + "..."
    return f"`{t}`"


def _md_cell(s):
    """A validated plain-text value in a table cell (the validator refused | @ [ ] < > ` and line breaks)."""
    from .render import md_text
    return md_text(s)


def _comment(errs, sub, summ):
    from .submission import summary_row
    if errs:
        return (MARKER + "\n**Leaderboard validation: not valid.**\n\n" + "\n".join(f"- {safe(e)}" for e in errs[:30])
                + ("\n- (more errors not shown)" if len(errs) > 30 else "")
                + "\n\nFix the run (or the pasted JSON) and edit the issue; validation runs again.\n")
    r = summary_row(sub, summ)
    from .render import doi_link
    return (MARKER + "\n**Leaderboard validation: valid.** The row this submission adds, recomputed from its "
            "per-episode values. It is post hoc and outside the pre-specified families: its p is unadjusted, and it "
            "gets no pre-specified decision word. The order of the leaderboard is by mean error only and is not a "
            "ranking.\n\n"
            "| Slot | Method | Error (mm) | Difference (mm), 95% t CI, k/6 | Patient-by-seed 95% CI | Seeds | p | TOST "
            "| Decision | DOI |\n"
            "| :-- | :-- | --: | :-- | :-- | :-: | --: | :-: | :-- | :-- |\n"
            f"| {r['slot']} | {_md_cell(r['method'])} | {r['error']} | {r['difference']} | "
            f"{r['patient_seed'].replace('patient-by-seed ', '')} | {r['seeds']} | {r['p']} | {r['tost']} | "
            f"{r['decision']} | {doi_link(r['doi'])} |\n\n"
            "A maintainer resolves the DOI at its registry and reruns the plug-in from its code URL under the "
            "protocol before adding the row.\n")


def _gate_comment(errs, form, gate):
    from .render import doi_link
    if errs:
        return (MARKER + "\n**Leaderboard validation: not valid** (maintainer-run request).\n\n"
                + "\n".join(f"- {safe(e)}" for e in errs[:30])
                + "\n\nFix the gate record or the form and edit the issue; validation runs again.\n")
    from .submission import norm_doi
    return (MARKER + "\n**Leaderboard validation: admitted; queued for a maintainer run.** The gate record admits "
            "the plug-in for this slot. A maintainer resolves the DOI at its registry, checks the plug-in's source "
            "sha256 at the code URL, runs it under the plug protocol and posts the result here.\n\n"
            "| Slot | Method | Plug-in | Source sha256 | DOI |\n| :-- | :-- | :-- | :-- | :-- |\n"
            f"| {form['slot']} | {_md_cell(form['name'])} | {safe(gate['plugin'], 80)} | "
            f"{safe(gate['plugin_source_sha256'][:16] + '...', 40)} | {doi_link(norm_doi(form['doi']))} |\n")


def _form_method(form):
    return {k: (form.get(k) or "").strip() for k in ("name", "slot", "doi", "code_url", "license")} | \
        {"description": "(from the issue form)"}


def _validate_issue(path):
    """-> (comment text, exit code) for an issue body."""
    from .submission import (MAINTAINER_SLOTS, RUNNABLE_SLOTS, check_method_fields, extract_json, norm_doi,
                             validate, validate_gate)
    form = parse_issue_form(open(path, encoding="utf-8").read())
    slot = (form.get("slot") or "").strip()
    method_errs = check_method_fields(_form_method(form))
    box_errs = [f"the checkbox '{FORM_BOXES[b]}' is not ticked" for b in ("protocol", "held_out")
                if form.get(b) is not True]
    if slot == "anatomy":
        errs = method_errs + box_errs
        if errs:
            return _gate_comment(errs, form, None), 1
        return (MARKER + "\n**Leaderboard validation:** the CT-anatomy slot is run by the maintainers "
                "(SUBMITTING.md); no submission.json or gate record is checked here. A maintainer will reply on this "
                "issue.\n"), 0
    wants_gate = form.get("request", "").lower().startswith("maintainer") or \
        (slot in MAINTAINER_SLOTS and not form.get("json") and form.get("gate"))
    if wants_gate:
        try:
            gate = extract_json(form.get("gate") or "")
        except ValueError as e:
            return _gate_comment([f"could not read the gate record JSON: {e}"], form, None), 1
        errs = method_errs + box_errs + validate_gate(gate, slot)
        return _gate_comment(errs, form, gate), (1 if errs else 0)
    if not form.get("json"):
        return _comment(["no submission.json pasted (for a maintainer run, choose that request type and paste the "
                         "gate record)"], None, None), 1
    try:
        sub = extract_json(form["json"])
    except ValueError as e:
        return _comment([f"could not read the submission JSON: {e}"], None, None), 1
    errs, summ = validate(sub)
    if not errs:
        m = sub["method"]
        if (form.get("slot") or "").strip() and (form.get("slot") or "").strip() != str(m.get("slot", "")).strip():
            errs.append("the form's slot differs from the submission's")
        if (form.get("doi") or "").strip() and norm_doi(form["doi"]) != norm_doi(m.get("doi")):
            errs.append("the form's DOI differs from the submission's")
        if slot not in RUNNABLE_SLOTS:
            errs.append("the form's slot is not a runnable slot")
        errs += box_errs
    return _comment(errs, sub if not errs else None, summ), (1 if errs else 0)


def cmd_validate(a):
    """Write the validation comment (always: an internal error still gives a comment) and return 0 if valid."""
    from .submission import validate
    try:
        if a.issue_body:
            text, code = _validate_issue(a.issue_body)
        else:
            try:
                sub = json.load(open(a.submission, encoding="utf-8"))
            except (OSError, ValueError) as e:
                errs, sub, summ = [f"could not read the submission JSON: {e}"], None, None
            else:
                errs, summ = validate(sub)
            text, code = _comment(errs, sub, summ), (1 if errs else 0)
    except Exception as e:                        # noqa: BLE001  (never leave the issue without an answer)
        text = (MARKER + "\n**Leaderboard validation: internal error** while reading this request "
                f"({safe(type(e).__name__, 60)}). A maintainer will look; editing the issue runs validation again.\n")
        code = 2
    if a.comment_out:
        with open(a.comment_out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return code


FORM_BOXES = {"protocol": "The plug-in follows the plug protocol unchanged",
              "held_out": "No setting was tuned on held-out patients"}
REQUEST_TYPES = ("Scored submission (I ran the protocol; submission.json pasted)",
                 "Maintainer run (I passed the gate; gate record pasted)")


def parse_issue_form(text):
    """The fields of a leaderboard-submission issue (GitHub renders an issue form as '### Label' sections)."""
    out, cur, buf = {}, None, []
    fields = {"Method name": "name", "Slot": "slot", "Request type": "request", "DOI": "doi", "Code URL": "code_url",
              "License": "license", "submission.json": "json", "Gate record": "gate", "Checks": "checks"}
    for ln in text.splitlines() + ["### END"]:
        if ln.startswith("### "):
            if cur:
                out[cur] = "\n".join(buf).strip()
            cur, buf = fields.get(ln[4:].strip()), []
        elif cur:
            buf.append(ln)
    for k in ("json", "gate", "request", "doi", "code_url", "license", "name", "slot"):
        if out.get(k, "").strip() == "_No response_":          # GitHub's text for an empty optional field
            out[k] = ""
    checks = out.get("checks", "")
    for k, label in FORM_BOXES.items():
        out[k] = bool(re.search(r"- \[[xX]\] " + re.escape(label), checks))
    return out


def cmd_render(a):
    from . import render as RD
    from .recompute import load_board, load_methods
    text = RD.markdown(load_board(), load_methods())
    if a.check:
        old = open(a.out, encoding="utf-8").read() if os.path.exists(a.out) else ""
        if old != text:
            print(f"{a.out} is out of date: run python -m usbg.leaderboard render --out {a.out}")
            return 1
        print(f"{a.out} is up to date")
        return 0
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[render] wrote {a.out}")
    return 0


def _dois_in_use(methods):
    """{normalized DOI: where} of every DOI the methods list already holds."""
    from .submission import norm_doi
    out = {}
    for k, w in (methods.get("works") or {}).items():
        for d in (w.get("doi"), w.get("arxiv_doi")):
            if d:
                out[norm_doi(d)] = f"work {k}"
    for m in methods.get("submitted_methods", []):
        for w in m.get("references") or []:
            if w.get("doi"):
                out[norm_doi(w["doi"])] = f"submitted method {m['name']!r}"
    return out


def cmd_add(a):
    from . import doi as DOI
    from . import render as RD
    from .submission import norm_doi, validate
    data = a.data_dir
    board = json.load(open(os.path.join(data, "leaderboard.json"), encoding="utf-8"))
    values = json.load(open(os.path.join(data, "values.json"), encoding="utf-8"))
    methods = json.load(open(os.path.join(data, "methods.json"), encoding="utf-8"))
    sub = json.load(open(a.submission, encoding="utf-8"))
    errs, summ = validate(sub, board, values)
    if errs:
        print("not valid:\n" + "\n".join(f"  {e}" for e in errs))
        return 1
    m = dict(sub["method"])
    m["doi"] = norm_doi(m["doi"])
    slug = re.sub(r"[^a-z0-9]+", "_", m["name"].lower()).strip("_")[:60]
    rid = f"{m['slot']}/submitted/{slug}"
    if rid in values or any(r["id"] == rid for r in board["submitted"]):
        print(f"{rid} is already on the leaderboard")
        return 1
    used = _dois_in_use(methods)
    if m["doi"] in used and not a.allow_duplicate_doi:
        print(f"DOI {m['doi']} is already in the methods list ({used[m['doi']]}); a method enters with the DOI of its "
              "own publication (--allow-duplicate-doi for a variant of a listed method, after checking)")
        return 1
    meta = json.load(open(a.doi_metadata, encoding="utf-8")) if a.doi_metadata else None
    work = DOI.resolve(m["doi"], a.confirm_title, metadata=meta)       # SystemExit unless the registry title matches
    values[rid] = {s: {"final_pos": v["final_pos"]} for s, v in sub["results"]["per_seed"].items()}
    seeds = sorted(int(s) for s in values[rid])
    board["submitted"].append({
        "id": rid, "slot": m["slot"], "method": m, "plugin": sub["plugin"], "seeds": seeds,
        "values_of": rid, "against": summ["reference"], "err_mm": summ["err_mm"], "diff": summ["diff"],
        "decision": summ["decision"], "label": summ["label"], "protocol": sub["protocol"],
        "privileged": bool(m.get("privileged")), "label_access": bool(m.get("label_access")),
        "learned": bool(m.get("learned", True)), "seed0_only": seeds == [0],
        "reference_err_mm": summ["reference_err_mm"],
        "provenance": {k: sub["provenance"].get(k) for k in ("usbg_version", "python", "torch", "numpy", "device",
                                                              "wall_time_s")},
        "added_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rerun_by_maintainer": bool(a.rerun_by_maintainer)})
    methods.setdefault("submitted_methods", []).append({
        "slot": m["slot"], "name": m["name"], "code_url": m["code_url"], "license": m["license"],
        "description": m.get("description", ""), "adapter": sub["plugin"]["entry"], "rows": [rid],
        "references": [work]})
    _dump(os.path.join(data, "leaderboard.json"), board)
    _dump(os.path.join(data, "values.json"), values, compact=True)
    _dump(os.path.join(data, "methods.json"), methods)
    with open(a.leaderboard_md, "w", encoding="utf-8") as fh:
        fh.write(RD.markdown(board, methods))
    print(f"[add] {rid}: error {summ['err_mm']:.2f} mm; reference {work['reference']} ({work['verified']['registry']}); "
          f"wrote the data in {data} and {a.leaderboard_md}")
    return 0


def recompute_submitted(board, values):
    """Mismatches of the submitted rows, each recomputed from its shipped per-episode values."""
    from .submission import _close, summarize
    bad = []
    for r in board.get("submitted", []):
        s = summarize(values[r["values_of"]], r["slot"], dict(r["method"], privileged=r.get("privileged", False)),
                      board, values)
        for k in ("err_mm", "decision", "reference_err_mm"):
            if k in r and not _close(r[k], s[k]):
                bad.append(f"{r['id']} {k}: shipped {r[k]!r}, recomputed {s[k]!r}")
        keys = ("mean", "ci", "p_t", "sign", "n", "seeds", "per_seed", "k_seed", "tost_ci90", "tost_p", "tost_in",
                "ps_ci")
        if (r["diff"] is None) != (s["diff"] is None) or \
                (s["diff"] is not None and not all(_close(r["diff"].get(k), s["diff"].get(k)) for k in keys)):
            bad.append(f"{r['id']} diff")
    return bad


def cmd_recompute(a):
    from . import render as RD
    from .recompute import load_board, load_values, recompute_board, slot_mdd
    board, values = load_board(), load_values()
    got = recompute_board(board, values)
    bad, n = [], 0
    for slot in board["slots"]:
        for r in slot["rows"]:
            g = dict(r, err_mm=got[r["id"]]["err_mm"], diff=got[r["id"]]["diff"], holm_p=got[r["id"]]["holm_p"],
                     decision=got[r["id"]]["decision"], notes=got[r["id"]]["notes"])
            if r["kind"] == "cta":
                g["equivalent"] = got[r["id"]]["equivalent"]
            pc = RD.plain_cells(g, slot["decimals"])
            for k in ("error", "difference", "seeds", "holm_p", "tost", "decision", "notes"):
                n += 1
                if pc[k] != r["printed"][k]:
                    bad.append(f"{r['id']} {k}: recomputed {pc[k]!r}, printed {r['printed'][k]!r}")
        if slot.get("mdd"):
            md = slot_mdd(board, values, slot)
            d = slot["decimals"]
            if f"{md['mdd_mm']:.{d}f}" != f"{slot['mdd']['mdd_mm']:.{d}f}":
                bad.append(f"{slot['key']} MDD")
    bad += recompute_submitted(board, values)
    for b in bad:
        print("MISMATCH", b)
    print(f"[recompute] {sum(len(s['rows']) for s in board['slots'])} rows, {n} printed cells and "
          f"{len(board.get('submitted', []))} submitted rows recomputed from the shipped per-episode values: "
          f"{len(bad)} mismatches")
    return 1 if bad else 0


def cmd_fetch(a):
    from .assets import fetch
    fetch(a.dest, a.names, a.from_dir)
    return 0


def cmd_template(a):
    src = os.path.join(HERE, "templates", f"{a.slot}_plugin.py")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    shutil.copyfile(src, a.out)
    print(f"[template] wrote {a.out}; its factory is {os.path.splitext(os.path.basename(a.out))[0]}:make")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m usbg.leaderboard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    slots = ["texture", "mask", "ctrl", "reward", "renderer"]

    p = sp.add_parser("check", help="admission gate (training patients only)")
    p.add_argument("--slot", required=True, choices=slots)
    p.add_argument("--plugin", required=True, help="package.module:factory")
    p.add_argument("--privileged", action="store_true", help="declare that the plug-in reads simulator state")
    p.add_argument("--out", default=None, help="write the gate record here")
    p.set_defaults(fn=cmd_check)

    p = sp.add_parser("run", help="run the slot condition and write submission.json")
    p.add_argument("--slot", required=True, choices=slots)
    p.add_argument("--plugin", required=True, help="package.module:factory ('reference': the slot's own reference "
                                                   "component, for identity runs)")
    p.add_argument("--name", required=True)
    p.add_argument("--doi", required=True, help="the method's DOI, e.g. 10.1234/abcd.5678")
    p.add_argument("--code-url", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--description", required=True, help="one sentence on the method (plain text)")
    p.add_argument("--no-training", dest="learned", action="store_false",
                   help="the method trains nothing (seed 0 only, paired with seed 0 of the reference)")
    p.add_argument("--privileged", action="store_true")
    p.add_argument("--label-access", action="store_true", help="the plug-in reads the CT label slice")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--patients", nargs="+", default=None, help="smoke run on these held-out patients only "
                                                               "(the result is partial, not a valid submission)")
    p.add_argument("--device", default=None, help="cpu, mps or cuda: the device of the plug-in and of the evaluation "
                                                  "(default: MPS when available, else the CPU)")
    p.add_argument("--assets-dir", default="assets", help="the fetched release assets")
    p.add_argument("--windows", default=os.path.join(os.path.dirname(HERE), "data", PRO.WINDOWS_FILE))
    p.add_argument("--vitale-split", default="data/real/vitale_split_v2.npz",
                   help="texture slot: the segmenter trainer's fixed split of the annotated Vitale frames "
                        "(relative to segtrain/)")
    p.add_argument("--sonospade-ckpt", default=None, help="maintainers: the SonoSPADE stage of the reported runs")
    p.add_argument("--reference-dir", default=None, help="maintainers: the reference policies of the reported runs")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--run-name", default="run")
    p.add_argument("--out", default="submission.json")
    p.add_argument("--dry-run", action="store_true", help="print the steps and commands only")
    p.set_defaults(fn=cmd_run)

    p = sp.add_parser("validate", help="validate a submission.json and recompute its summary")
    p.add_argument("submission", nargs="?")
    p.add_argument("--issue-body", default=None, help="a submission issue's body (the JSON in its json block)")
    p.add_argument("--comment-out", default=None, help="write the result as a Markdown comment")
    p.set_defaults(fn=cmd_validate)

    p = sp.add_parser("render", help="write LEADERBOARD.md")
    p.add_argument("--out", default="LEADERBOARD.md")
    p.add_argument("--check", action="store_true")
    p.set_defaults(fn=cmd_render)

    p = sp.add_parser("add", help="maintainers: add a validated submission")
    p.add_argument("submission")
    p.add_argument("--data-dir", default=DATA)
    p.add_argument("--leaderboard-md", default="LEADERBOARD.md")
    p.add_argument("--rerun-by-maintainer", action="store_true",
                   help="the numbers come from the maintainers' own rerun of the plug-in")
    p.add_argument("--confirm-title", default=None,
                   help="the title of the method's publication; it must equal the DOI's registry title")
    p.add_argument("--doi-metadata", default=None,
                   help="a saved registry record of the DOI (Crossref or DataCite JSON) instead of a download")
    p.add_argument("--allow-duplicate-doi", action="store_true",
                   help="accept a DOI the methods list already holds (a variant of a listed method)")
    p.set_defaults(fn=cmd_add)

    p = sp.add_parser("recompute", help="recompute every row from the shipped per-episode values")
    p.set_defaults(fn=cmd_recompute)

    p = sp.add_parser("fetch", help="download the release assets (sha256 verified)")
    p.add_argument("--dest", default="assets")
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--from-dir", default=None, help="copy from a local folder instead of downloading")
    p.set_defaults(fn=cmd_fetch)

    p = sp.add_parser("template", help="copy a template plug-in")
    p.add_argument("--slot", required=True, choices=slots)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_template)

    a = ap.parse_args(argv)
    if a.cmd == "validate" and not (a.submission or a.issue_body):
        ap.error("validate needs a submission.json or --issue-body")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
