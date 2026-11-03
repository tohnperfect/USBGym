"""usbg.leaderboard.submission: the submission.json schema, its validation and the summary it must carry.

A submission (schema "usbg-leaderboard-submission-1") is written by ``python -m usbg.leaderboard run`` and
checked by ``python -m usbg.leaderboard validate``:

    {
      "schema": "usbg-leaderboard-submission-1",
      "protocol": "usbg-plug-protocol-1",
      "method": {"name", "slot", "doi", "code_url", "license", "description",
                 "learned": true, "privileged": false, "label_access": false},
      "plugin": {"entry": "package.module:factory", "source_sha256": "..."},
      "protocol_settings": {...the constants of usbg.leaderboard.protocol that the run used...},
      "results": {"episodes": [{"ep", "patient", "target"} x 60],
                  "per_seed": {"0": {"final_pos": [60 floats], "final_ang": [60 floats],
                                     "per_patient": {"s0358": mean, ...}}, "1": ..., "2": ...}},
      "summary": {"err_mm", "diff": {...}, "decision", "label"},
      "provenance": {"usbg_version", "plugin_source_sha256", "python", "torch", "numpy", "device",
                     "wall_time_s", "platform", "created_utc", "commands"}
    }

validate() never imports or runs the plug-in: it reads the JSON only. It checks the type of every section, the
schema and protocol version, the method's text fields (plain text: no line breaks, no Markdown or HTML markup, no
@-mention), the DOI pattern, the code URL, the protocol settings, that every seed has the 60 per-episode values of
the fixed episodes, that the per-patient means follow from them, that the provenance holds no absolute local path,
and that the summary equals the one recomputed from the per-episode values against the slot's reference values
shipped with the package.

validate_gate() checks a gate record (python -m usbg.leaderboard check --out gate.json) for a maintainer-run
request: the gate version, admitted, the slot, the plug-in entry and its source sha256.
"""
from __future__ import annotations

import json
import math
import re

from . import protocol as PRO
from . import recompute as RC
from . import stats as S

DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
URL_RE = re.compile(r"^https://[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(/[A-Za-z0-9._~%/+=-]*)?"
                    r"(#[A-Za-z0-9._~%/+=-]*)?$")
ENTRY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
RUNNABLE_SLOTS = ("texture", "mask", "ctrl", "reward", "renderer")
# conditions that need an artifact that is not released: the maintainers run the plug-in (a gate record in the form)
MAINTAINER_SLOTS = tuple(k for k, v in PRO.SLOTS.items() if v["who"] == "maintainer")
REQUIRED_PROVENANCE = ("usbg_version", "plugin_source_sha256", "python", "torch", "numpy", "device", "wall_time_s")
TOL = 1e-9
# plain-text method fields: (max length); no line break or control character, and none of | @ [ ] < > ` (they
# would make a Markdown table, a link, an image, HTML or a mention out of the text in the comment and the table)
TEXT_FIELDS = {"name": 120, "license": 120, "description": 300}
UNSAFE_TEXT = re.compile(r"[\x00-\x1f\x7f|@\[\]<>`]")
ABS_PATH = re.compile(r"(^|[\s=:'\"])(/(Users|home|private|root|tmp|var)/|[A-Za-z]:[\\/])")   # macOS, Linux, Windows
GATE_VERSION = "usbg-plug-admission-1"


def norm_doi(doi):
    """A DOI as stored and compared: stripped, without a resolver prefix, lower case (DOIs are case-insensitive)."""
    d = str(doi or "").strip()
    for pre in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if d.lower().startswith(pre):
            d = d[len(pre):]
    return d.lower()


def check_method_fields(m, slots=RUNNABLE_SLOTS + ("anatomy",)):
    """Errors of the method's text fields, DOI, code URL and slot (shared by the submission and the form)."""
    errs = []
    for k, n in TEXT_FIELDS.items():
        v = m.get(k)
        if not isinstance(v, str) or not v.strip():
            errs.append(f"method.{k}: missing or empty")
        elif len(v) > n:
            errs.append(f"method.{k}: longer than {n} characters")
        elif UNSAFE_TEXT.search(v):
            errs.append(f"method.{k}: plain text only (no line break, and none of the characters | @ [ ] < > `)")
    doi = m.get("doi")
    if not isinstance(doi, str) or not doi.strip():
        errs.append("method.doi: missing or empty")
    elif not DOI_RE.match(doi.strip()):
        errs.append("method.doi does not match 10.<registrant>/<suffix> (give the bare DOI, without https://doi.org/; "
                    "a work without a publisher DOI gives its arXiv DOI, 10.48550/arXiv.<id>)")
    url = m.get("code_url")
    if not isinstance(url, str) or not url.strip():
        errs.append("method.code_url: missing or empty")
    elif not URL_RE.match(url.strip()):
        errs.append("method.code_url is not a plain https URL (host and path only; no spaces, quotes, brackets or "
                    "parentheses)")
    if m.get("slot") not in slots:
        errs.append(f"method.slot: expected one of {', '.join(slots)}")
    return errs


def _strings(x, path=""):
    if isinstance(x, str):
        yield path, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _strings(v, f"{path}[{i}]")


def protocol_settings(seeds):
    """The protocol constants a run records (and validate compares)."""
    return {"version": PRO.PROTOCOL_VERSION, "n_steps": PRO.N_STEPS, "n_episodes": PRO.N_EPISODES,
            "eval_seed": PRO.EVAL_SEED, "eval_design": PRO.EVAL_DESIGN, "contact": PRO.CONTACT,
            "domain_randomization": PRO.DOMAIN_RANDOMIZATION, "seat": PRO.SEAT, "windows_md5": PRO.WINDOWS_MD5,
            "batch_norm": PRO.BATCH_NORM, "train_patients": list(PRO.TRAIN_PATIENTS),
            "held_out_patients": list(PRO.HELD_OUT_PATIENTS), "seeds": [int(s) for s in seeds],
            "rl_env_steps": PRO.RL_ENV_STEPS, "imitation_samples": PRO.IMITATION_SAMPLES,
            "tuned_on_held_out_patients": False}


def reference_pm(board, values, slot):
    """{seed: {patient: mean}} of the slot's reference row (the shipped per-episode values)."""
    s = next(x for x in board["slots"] if x["key"] == slot)
    return RC.pm_of(values, s["reference"], board["episodes"])


def summarize(per_seed, slot, method, board=None, values=None):
    """The summary of a submission from its per-episode values: error, the paired difference from the slot's
    reference (seed-paired; a seed-0 submission pairs with seed 0) with its patient-by-seed interval (two or more
    seeds), and the decision cell of a submitted row (no pre-specified decision word)."""
    board = board or RC.load_board()
    values = values or RC.load_values()
    pm = {int(s): S.patient_means(v["final_pos"], board["episodes"]) for s, v in per_seed.items()}
    err = S.row_error(pm)
    ref = reference_pm(board, values, slot)
    complete = all(len(p) == len(PRO.HELD_OUT_PATIENTS) for p in pm.values()) and bool(pm)
    c = S.contrast(pm, ref, board["patients"]) if complete else None
    if c is not None:
        c["ps_ci"] = S.patient_seed_ci(pm, ref, board["patients"])
    dec, label = S.decide_submitted(c, learned=bool(method.get("learned", True)), n_seeds=len(pm),
                                    privileged=bool(method.get("privileged", False)))
    seeds = sorted(pm)
    ref_err = S.row_error({s: ref[s] for s in seeds if s in ref})
    return {"err_mm": err, "diff": c, "decision": dec, "decision_full": dec, "label": label,
            "reference": next(x for x in board["slots"] if x["key"] == slot)["reference"],
            "reference_err_mm": ref_err, "seeds": seeds, "per_patient": {str(s): p for s, p in pm.items()}}


def _finite_list(x, n):
    return (isinstance(x, list) and len(x) == n
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v < 1e4
                    for v in x))


def _close(a, b):
    """Equal within TOL, structure and types included; False (never an exception) on mismatched types."""
    if a is None or b is None:
        return a is b
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return type(a) is type(b) and a == b
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    try:
        return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(a)))
    except (TypeError, ValueError, OverflowError):
        return False


def validate(sub, board=None, values=None):
    """-> (errors, summary or None). An empty error list means the submission is valid."""
    errs = []
    if not isinstance(sub, dict):
        return ["the submission is not a JSON object"], None
    if sub.get("schema") != PRO.SUBMISSION_SCHEMA:
        errs.append(f"schema {sub.get('schema')!r}, expected {PRO.SUBMISSION_SCHEMA!r}")
    if sub.get("protocol") != PRO.PROTOCOL_VERSION:
        errs.append(f"protocol {sub.get('protocol')!r}, expected {PRO.PROTOCOL_VERSION!r}")
    m = sub.get("method")
    if not isinstance(m, dict):
        return errs + ["method: missing or not an object"], None
    for k in ("plugin", "results", "protocol_settings", "provenance"):
        if not isinstance(sub.get(k), dict):
            errs.append(f"{k}: missing or not an object")
    if sub.get("summary") is not None and not isinstance(sub.get("summary"), dict):
        errs.append("summary: not an object")
    if errs:
        return errs, None
    errs += check_method_fields(m, RUNNABLE_SLOTS)
    slot = m.get("slot")
    if slot == "anatomy":
        errs.append("method.slot: the CT-anatomy slot is run by the maintainers")
    for k in ("learned", "privileged", "label_access"):
        if not isinstance(m.get(k), bool):
            errs.append(f"method.{k}: true or false")
    pl = sub["plugin"]
    if not isinstance(pl.get("entry"), str) or not ENTRY_RE.match(pl["entry"]):
        errs.append("plugin.entry: expected package.module:factory")
    if not isinstance(pl.get("source_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", pl["source_sha256"]):
        errs.append("plugin.source_sha256: expected 64 hexadecimal characters")
    res = sub["results"]
    if not isinstance(res.get("per_seed"), dict) or not res["per_seed"]:
        return errs + ["results.per_seed: missing or not an object"], None
    if not isinstance(res.get("episodes"), list) or not all(isinstance(e, dict) for e in res["episodes"]):
        return errs + ["results.episodes: expected a list of the 60 episode records"], None
    board = board or RC.load_board()
    values = values or RC.load_values()
    design = [(e["ep"], e["patient"], e["target"]) for e in board["episodes"]]
    got = [(e.get("ep"), e.get("patient"), e.get("target")) for e in res["episodes"]]
    if got != design:
        errs.append("results.episodes: not the 60 fixed held-out episodes in order (ep, patient, target)")
    seeds = sorted(res["per_seed"])
    if not all(s in ("0", "1", "2") for s in seeds):
        errs.append(f"results.per_seed: seeds {seeds}, expected keys among '0', '1', '2'")
        return errs, None
    learned = m.get("learned", True)
    want = ["0", "1", "2"] if learned else None
    if learned and seeds != want:
        errs.append(f"results.per_seed: a learned method needs seeds 0, 1 and 2 (got {seeds})")
    if not learned and seeds not in (["0"], ["0", "1", "2"]):
        errs.append(f"results.per_seed: a method without training runs seed 0 (or 0, 1 and 2), got {seeds}")
    for s in seeds:
        v = res["per_seed"][s]
        if not isinstance(v, dict) or not _finite_list(v.get("final_pos"), PRO.N_EPISODES):
            errs.append(f"results.per_seed[{s}].final_pos: expected {PRO.N_EPISODES} finite errors in mm")
            continue
        if "final_ang" in v and not _finite_list(v["final_ang"], PRO.N_EPISODES):
            errs.append(f"results.per_seed[{s}].final_ang: expected {PRO.N_EPISODES} finite values")
        pm = S.patient_means(v["final_pos"], board["episodes"])
        if not _close(pm, v.get("per_patient")):
            errs.append(f"results.per_seed[{s}].per_patient: does not follow from final_pos")
    ps = sub["protocol_settings"]
    want_ps = protocol_settings([int(s) for s in seeds])
    for k, v in want_ps.items():
        if ps.get(k) != v:
            errs.append(f"protocol_settings.{k}: differs from the protocol, which says {v!r}")
    pv = sub["provenance"]
    for k in REQUIRED_PROVENANCE:
        if k not in pv:
            errs.append(f"provenance.{k}: missing")
    if pv.get("plugin_source_sha256") != pl.get("source_sha256"):
        errs.append("provenance.plugin_source_sha256 differs from plugin.source_sha256")
    for path, s in _strings(pv, "provenance"):
        if ABS_PATH.search(s):
            errs.append(f"{path}: holds an absolute local path (rerun with this version of the runner, which "
                        "records paths relative to the repository, or replace the path by hand)")
            break
    if errs or slot not in RUNNABLE_SLOTS:
        return errs, None
    summ = summarize(res["per_seed"], slot, m, board, values)
    stated = sub.get("summary") or {}
    for k in ("err_mm", "decision", "label"):
        if not _close(stated.get(k), summ[k]):
            errs.append(f"summary.{k}: differs from the recomputed value {summ[k]!r}")
    keys = ("mean", "ci", "p_t", "sign", "n", "seeds", "per_seed", "k_seed", "n_seed", "tost_ci90", "tost_p",
            "tost_in", "ps_ci")
    sd, rd = stated.get("diff"), summ["diff"]
    if (sd is None) != (rd is None) or (rd is not None and (not isinstance(sd, dict)
                                                            or not all(_close(sd.get(k), rd[k]) for k in keys))):
        errs.append("summary.diff: differs from the difference recomputed from the per-episode values")
    return errs, summ


def validate_gate(gate, slot, entry=None):
    """Errors of a gate record for a maintainer-run request (empty: admitted for the form's slot)."""
    if not isinstance(gate, dict):
        return ["gate record: not a JSON object"]
    errs = []
    if gate.get("gate") != GATE_VERSION:
        errs.append(f"gate record: gate version differs from {GATE_VERSION!r} (write it with python -m "
                    "usbg.leaderboard check --out gate.json)")
    if gate.get("admitted") is not True:
        errs.append("gate record: not admitted")
    if gate.get("slot") != slot:
        errs.append("gate record: its slot differs from the form's slot")
    if not isinstance(gate.get("plugin"), str) or not ENTRY_RE.match(gate["plugin"]):
        errs.append("gate record: plugin is not package.module:factory")
    elif entry is not None and gate["plugin"] != entry:
        errs.append("gate record: plugin differs from the submission's plug-in entry")
    if not isinstance(gate.get("plugin_source_sha256"), str) or \
            not re.fullmatch(r"[0-9a-f]{64}", gate["plugin_source_sha256"]):
        errs.append("gate record: plugin_source_sha256 is not 64 hexadecimal characters")
    if not isinstance(gate.get("checks"), dict):
        errs.append("gate record: checks missing")
    return errs


def _strip_fences(raw):
    """Strip ``` fences (a pasted file that already had them sits inside the form's own json fence)."""
    raw = raw.strip()
    while raw.startswith("```"):
        nl = raw.find("\n")
        raw = raw[nl + 1:] if nl >= 0 else ""
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
        raw = raw.strip()
    return raw


def extract_json(text):
    """The JSON of an issue-form section or a whole text: the content of the first ```json fence (a fence pasted
    inside it is stripped too), or the whole text."""
    m = re.search(r"```(?:json)?[ \t]*\n(.*)\n[ \t]*```", text, re.S)
    raw = _strip_fences(m.group(1)) if m else _strip_fences(text)
    return json.loads(raw)


def ps_text(c, d=2):
    """The patient-by-seed interval of a submitted row, or ''."""
    return "" if c is None or not c.get("ps_ci") else f"patient-by-seed {S.ci_text(c['ps_ci'], d)}"


def summary_row(sub, summ):
    """The leaderboard row a valid submission adds (plain cells)."""
    c = summ["diff"]
    m = sub["method"]
    seed0 = summ.get("seeds") == [0]
    return {"method": m["name"], "slot": m["slot"], "doi": norm_doi(m["doi"]),
            "error": f"{summ['err_mm']:.2f}", "difference": S.diff_text(c), "patient_seed": ps_text(c),
            "seeds": ("seed 0 only, against the reference's seed 0 (" + f"{summ['reference_err_mm']:.2f} mm)")
            if seed0 else S.seeds_text(c),
            "p": "" if c is None else f"{S.p3(c['p_t'])} (unadjusted)",
            "tost": "" if c is None else ("yes" if c["tost_in"] else "no"),
            "decision": summ["decision"] + (" (privileged)" if m.get("privileged") and summ["decision"] == S.DEC_SEC
                                            else "")}


__all__ = ["validate", "validate_gate", "summarize", "protocol_settings", "extract_json", "summary_row", "DOI_RE",
           "URL_RE", "norm_doi", "check_method_fields", "ps_text", "MAINTAINER_SLOTS", "GATE_VERSION"]
