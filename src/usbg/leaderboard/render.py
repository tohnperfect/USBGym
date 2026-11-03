"""usbg.leaderboard.render: the leaderboard data as plain cells and as LEADERBOARD.md.

Submitted rows and methods carry text from outside the repository (a method name, license, code URL, DOI, and the
registry's reference): every such value is escaped for Markdown (md_text) and every link is built from a validated
DOI or URL, so no submitted text can add a link, an image, HTML, a mention or a table cell.
"""
from __future__ import annotations

import re
import urllib.parse

from . import stats as S

FLAG_TEXT = {"L": "ᴸ", "dagger": "†", "ddagger": "‡"}
NOTE_TEXT = {"a": "ᵃ", "e": "ᵉ", "b": "ᵇ"}


def and_join(xs):
    """'a', 'a and b', 'a, b and c'."""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def flags_of(row):
    f = []
    if row.get("label_access"):
        f.append("L")
    if row.get("privileged"):
        f.append("dagger")
    if row.get("fewer_seeds"):
        f.append("ddagger")
    return sorted(f)


_MD_ESC = {"\\": "\\\\", "|": "\\|", "*": "\\*", "_": "\\_", "[": "\\[", "]": "\\]", "`": "\\`", "<": "&lt;",
           ">": "&gt;", "@": "&#64;", "#": "\\#", "!": "\\!"}
SLOT_WORDS = {"anatomy": "anatomy source", "renderer": "renderer", "texture": "texture stage", "mask": "mask source",
              "ctrl": "controller", "reward": "reward"}


def md_text(s, n=400):
    """Untrusted text as one Markdown table-cell line: line breaks folded, Markdown and HTML characters escaped,
    '@' as an entity (no mention), at most n characters."""
    t = " ".join(str(s).split())[:n]
    return "".join(_MD_ESC.get(ch, ch) for ch in t)


def doi_link(doi):
    """[doi](https://doi.org/<percent-encoded doi>) for a DOI that matched the DOI pattern."""
    from .submission import DOI_RE
    d = str(doi).strip()
    if not DOI_RE.match(d):
        return md_text(d)
    return f"[{md_text(d)}](https://doi.org/{urllib.parse.quote(d, safe='/')})"


def url_link(text, url):
    from .submission import URL_RE
    u = str(url).strip()
    return f"[{text}]({u})" if URL_RE.match(u) and not re.search(r"[()\[\]<>\s]", u) else md_text(text)


def plain_cells(row, decimals=2):
    """The cells of one row as Table 5 prints them, in plain text: method (without citations and marks), cites,
    flags, error, difference, seeds, holm_p, tost, decision and notes. TOST is the post hoc, unadjusted test of
    every row (the 90% paired t CI inside +-1 mm); the CT-anatomy rows' Holm-adjusted equivalence family is their
    decision."""
    c = row.get("diff")
    tost = "" if c is None else ("yes" if c["tost_in"] else "no")
    return {"method": row["method"], "cites": list(row.get("cites") or []), "flags": flags_of(row),
            "error": "" if row.get("err_mm") is None else f"{row['err_mm']:.2f}",
            "difference": S.diff_text(c, decimals), "seeds": S.seeds_text(c),
            "holm_p": "" if row.get("holm_p") is None else S.p3(row["holm_p"]), "tost": tost,
            "decision": row.get("decision") or "", "notes": sorted(row.get("notes") or [])}


ROLE_TEXT = {"ref": "reference", "family": "family", "a2": "second family", "secondary": "secondary"}
INDENT = "&emsp;"           # the seed-0 block: rows whose difference is taken from the reference's seed-0 line


def role_of(slot, row):
    """The row's role as Table C.1 of the article prints it: reference, family, second family, secondary or
    privileged (a privileged row is secondary); every CT-anatomy row is a family row."""
    if slot["key"] == "anatomy":
        return "family"
    if row["role"] == "secondary" and row.get("privileged"):
        return "privileged"
    return ROLE_TEXT[row["role"]]


def seed0_row(row):
    """True for a row run at seed 0 only (its difference is taken from the reference's seed-0 line)."""
    return list(row.get("seeds") or []) == [0]


def slot_order(slot):
    """The published rows of one slot in the order of Table C.1 (Appendix C of the article): [(row, level)], level 1
    for the seed-0 block.

    In each block that shares one reference: the reference, then the rows whose difference is taken from it by mean
    error (unrounded; ties keep Table 5's order), lowest first, with the privileged rows last; then the reference's
    seed-0 line, followed in the same way by the rows run at seed 0 only. The CT-anatomy rows keep Table 5's order,
    since each is scored against its own reference level. The pretrained-backbone differences, which have no error of
    their own, are not in the order (pretrained_rows). The order is not a ranking: each row was tested against its
    slot's reference only."""
    rows = slot["rows"]
    if slot["key"] == "anatomy":
        return [(r, 0) for r in rows]
    pos = {id(r): i for i, r in enumerate(rows)}
    shown = [r for r in rows if r.get("kind") != "imnt"]
    refs = [r for r in shown if r["role"] == "ref"]
    main = [r for r in refs if r.get("kind") != "ref_seed0"]
    line0 = [r for r in refs if r.get("kind") == "ref_seed0"]
    if len(main) != 1 or len(line0) > 1:
        raise ValueError(f"slot {slot['key']}: one reference row and at most one seed-0 line are expected")
    rest = [r for r in shown if r["role"] != "ref"]
    if any(r.get("err_mm") is None for r in rest):
        raise ValueError(f"slot {slot['key']}: a row in the order has no error")
    s0 = [r for r in rest if seed0_row(r)]
    if s0 and not line0:
        raise ValueError(f"slot {slot['key']}: rows run at seed 0 only, but no seed-0 line of the reference")

    def block(xs, level):
        key = (lambda r: (r["err_mm"], pos[id(r)]))
        return ([(r, level) for r in sorted((r for r in xs if not r.get("privileged")), key=key)]
                + [(r, level) for r in sorted((r for r in xs if r.get("privileged")), key=key)])

    order = [(main[0], 0)] + block([r for r in rest if r not in s0], 0)
    if line0:
        order += [(line0[0], 1)] + block(s0, 1)
    return order


def pretrained_rows(slot):
    """The pretrained-backbone differences of a slot (each network with an ImageNet backbone minus the same network
    trained from scratch): rows of their own family, with no error of their own, so not in the order."""
    return [r for r in slot["rows"] if r.get("kind") == "imnt"]


def _ref_text(keys, methods):
    out = []
    for k in keys:
        w = methods["works"].get(k)
        if not w:
            continue
        if w.get("doi"):
            out.append(f"[doi:{w['doi']}](https://doi.org/{w['doi']})")
        else:
            out.append(f"[{w['identifier']}]({w['identifier']})" if str(w.get("identifier", "")).startswith("http")
                       else str(w.get("identifier")))
    return "; ".join(out)


def _method_md(row):
    marks = "".join(FLAG_TEXT[f] for f in flags_of(row))
    return f"{row['method']}{marks}"


def _md_escape(s):
    return str(s).replace("|", "\\|")


def submitted_cells(row, decimals=2):
    """Cells of a submitted row: the patient-by-seed interval beside the paired t CI, the unadjusted paired t p in
    the p column, and no pre-specified decision word ("post hoc"; "secondary" for a privileged method)."""
    c = row.get("diff")
    diff = S.diff_text(c, decimals)
    if c is not None and c.get("ps_ci"):
        diff += f"; patient-by-seed {S.ci_text(c['ps_ci'], decimals)}"
    if row.get("seed0_only") and row.get("reference_err_mm") is not None:
        seeds = f"seed 0 only, against the reference's seed 0 ({row['reference_err_mm']:.2f} mm)"
    else:
        seeds = S.seeds_text(c)
    dec = row.get("decision", "")
    if row.get("privileged") and dec == S.DEC_SEC:
        dec = "secondary (privileged)"
    return {"error": "" if row.get("err_mm") is None else f"{row['err_mm']:.2f}",
            "difference": diff, "seeds": seeds,
            "p": "" if c is None else f"{S.p3(c['p_t'])} (unadjusted)",
            "tost": "" if c is None else ("yes" if c["tost_in"] else "no"),
            "decision": dec}


def markdown(board, methods):
    """LEADERBOARD.md from the leaderboard data and the methods list."""
    L = ["# USB Gym plug-slot leaderboard", "",
         "<!-- generated by `python -m usbg.leaderboard render`; do not edit by hand -->", "",
         "Every row runs the plug protocol: one slot swapped at a time, the rest of the loop at that slot's "
         "condition, the same 60 held-out episodes (six held-out patients, evaluation seed 999, contact on) at 12 "
         "steps. Error: final translation error in mm, mean over seeds of patient means (lower is better). "
         "Difference: from the slot's reference component, paired by seed, with its 95% paired t CI and the "
         "patients of 6 sharing its sign. Seeds: the seeds, of 3, whose own difference shares that sign (n/a for a "
         "seed-0 row). Holm p: within the slot's pre-specified family. TOST: yes when the 90% paired t CI lies "
         "inside ±1 mm (post hoc, unadjusted). Decision: the pre-specified outcome words; a decision of no "
         "detectable difference is not powered for equivalence.", "",
         "Order (as in Table C.1, Appendix C of the article): in each slot the reference comes first, then the rows "
         "whose difference is taken from it, by mean error (unrounded), lowest first, with the privileged rows last. "
         "Rows run at seed 0 only follow the reference's seed-0 line in the same order, indented, since their "
         "differences are taken from that line and their error is that of one seed. The CT-anatomy rows keep the "
         "order of Table 5, since each is scored against its own reference level. Rows submitted after the article "
         "come after the published rows of their slot, in a group of their own that is outside this order and not "
         "indented, whatever their seeds. The pretrained-backbone differences, which have no error of their own, "
         "are listed under the mask-source table, outside the order.", "",
         "Role: the row's group in Table 5 (reference, family, second family, secondary or privileged); second "
         "family: the second mask-source family or the LPIPS reward family; privileged: the group of privileged rows, "
         "which are secondary; submitted: a row added after the article.", "",
         "The order is by mean error only and is not a ranking. Each row was tested against its slot's reference, "
         "not against the other rows, and where the decision is no detectable difference, the methods are not "
         "distinguished from the reference or from each other. At n = 6 held-out patients, a result is an estimate "
         "with its interval, read beside coverage and cost.", "",
         "Rows from Table 5 of the USB Gym article are listed as published and are recomputed from their "
         "per-episode values by `python -m usbg.leaderboard recompute` (and by the tests). Rows added after the "
         "article are marked **submitted**: they are post hoc and outside the pre-specified families, so their p "
         "is the unadjusted paired t p, no Holm adjustment applies and they get no pre-specified decision word; "
         "a row with three seeds also shows its patient-by-seed bootstrap interval (Appendix B of the article). "
         "To add a method, see [SUBMITTING.md](SUBMITTING.md).", "",
         "Marks: ᴸ reads the CT label slice; † privileged (simulator state that no rig provides); ‡ fewer seeds "
         "than designed; ᵃ unadjusted CI excludes 0, not declared a difference after Holm (post hoc note); ᵉ "
         "equivalence within ±1 mm not shown. Every contrast is conditional on the trained artifacts; seeds index "
         "artifacts and the patient is the sampling unit. All results are in simulation.", ""]
    subs = {}
    for r in board.get("submitted", []):
        subs.setdefault(r["slot"], []).append(r)
    for slot in board["slots"]:
        dd = slot["decimals"]
        L.append(f"## {slot['title']}")
        L.append("")
        cond = slot.get("printed_condition") or ""
        if slot["key"] != "anatomy":
            fams = slot["families"]
            fam_txt = (f"family m = {fams[0]['m']}" if len(fams) == 1 else
                       "families " + and_join([f"{f['name']} (m = {f['m']})" for f in fams]) + ", never pooled")
            md = slot["mdd"]
            L.append(f"Condition: {cond}. Reference error {slot['reference_err_mm']:.2f} mm; {fam_txt}; post hoc "
                     f"MDD {md['mdd_mm']:.{dd}f} mm ({md['mdd_holm_mm']:.{dd}f} mm at alpha = 0.05/{fams[0]['m']}).")
            if slot.get("printed_note"):
                L.append("")
                L.append(slot["printed_note"])
        else:
            L.append(f"Condition: {cond}. Each row is that level minus the same level on the reference labels. "
                     "Families: CT-anatomy difference (paired t, Holm, m = 5) and CT-anatomy "
                     "equivalence (TOST with a ±1 mm margin, Holm over the TOST p values, m = 5); the Decision column carries both "
                     "families' outcomes, and the TOST column is the unadjusted TOST of every row. The trained "
                     "artifacts never saw the new labels, so a change mixes label noise with the shift between "
                     "training and evaluation labels. " + slot.get("note", ""))
        L.append("")
        L.append("| Method | Role | Error (mm) | Difference (mm), 95% t CI, k/6 | Seeds | Holm p | TOST | Decision | "
                 "Reference |")
        L.append("| :-- | :-- | --: | :-- | :-: | --: | :-: | :-- | :-- |")
        for r, level in slot_order(slot):
            pc = plain_cells(r, dd)
            dec = pc["decision"] + "".join(NOTE_TEXT[n] for n in pc["notes"])
            L.append("| " + " | ".join(_md_escape(x) for x in (
                INDENT * level + _method_md(r), role_of(slot, r), pc["error"], pc["difference"], pc["seeds"],
                pc["holm_p"], pc["tost"], dec, _ref_text(r.get("cites") or r.get("method_cites") or [], methods)))
                     + " |")
        group = None
        for priv in (False, True):
            for r in [x for x in subs.get(slot["key"], []) if bool(x.get("privileged")) == priv]:
                g = "submitted, privileged" if priv else "submitted"
                if group != g:
                    group = g
                    L.append("| *Submitted, " + ("privileged, secondary; " if priv else "")
                             + "post hoc (outside the pre-specified families; p unadjusted; no decision word)* "
                               "| | | | | | | | |")
                sc = submitted_cells(r, dd)
                marks = "".join(FLAG_TEXT[f] for f in flags_of(r))
                name = f"**{md_text(r['method']['name'])}**{marks} ({url_link('code', r['method']['code_url'])})"
                L.append("| " + " | ".join((name, "privileged" if priv else "submitted", sc["error"], sc["difference"],
                                            sc["seeds"], sc["p"], sc["tost"], sc["decision"],
                                            doi_link(r["method"]["doi"]))) + " |")
        pre = pretrained_rows(slot)
        if pre:
            L += ["", "Pretrained-backbone differences, outside the order: each is the network with an ImageNet "
                      "backbone minus the same network trained from scratch, so it has no error of its own. Family: "
                      f"{_md_escape(pre[0]['group'])}.", "",
                  "| Difference | 95% t CI, k/6 | Seeds | Holm p | TOST | Decision | Reference |",
                  "| :-- | :-- | :-: | --: | :-: | :-- | :-- |"]
            for r in pre:
                pc = plain_cells(r, dd)
                dec = pc["decision"] + "".join(NOTE_TEXT[n] for n in pc["notes"])
                L.append("| " + " | ".join(_md_escape(x) for x in (
                    _method_md(r), pc["difference"], pc["seeds"], pc["holm_p"], pc["tost"], dec,
                    _ref_text(r.get("cites") or r.get("method_cites") or [], methods))) + " |")
        L.append("")
    if board.get("combination"):
        L += _combination_md(board["combination"])
    L += _methods_md(methods)
    return "\n".join(L).rstrip() + "\n"


FRAMES = (("physics", "physics"), ("sonospade", "SonoSPADE"))


def benchmark_baseline_text(cb):
    """The benchmark baseline per frame type, as the article defines it: the authors' original pipeline or, where
    the combination tests lower (its D contrast holds lower), the combination."""
    parts = []
    for comp, frames in FRAMES:
        b = cb["benchmark_baseline"][comp]
        comb = b["pipeline"] == "combination"
        who = "the combination" if comb else "the authors' original pipeline"
        dec = b["decision"]
        dtxt = (("it " if comb else "the combination ") + dec) if dec.startswith("holds") else dec
        parts.append(f"with {frames} frames it is {who}, {b['error_mm']} mm ({b['contrast']}: {dtxt})")
    text = "; ".join(parts)
    return f"The benchmark baseline is, {cb['benchmark_baseline_rule']}. " + text[0].upper() + text[1:] + "."


def _combination_md(cb):
    """The combination, as the article's combination table prints it (not a slot row: a whole pipeline), and the
    benchmark baseline that it defines per frame type."""
    n = cb["cohort_n"]
    L = ["## Combination and benchmark baseline", "",
         f"The combination is defined as {cb['definition']}. It is a whole pipeline, not a slot row, so it has no row in the "
         "slot tables above. Final translation error (mm); the six held-out patients took part in the choice, so "
         "their values are selection-biased and for estimation only. On the cohort CTs the static start ends at "
         f"{cb['cohort']['static_start']} mm and the pose oracle (privileged) at {cb['cohort']['pose_oracle']} mm.",
         "", benchmark_baseline_text(cb), "",
         "| Frames | Patients | Combination | Authors' original pipeline | The combination's learned-mask servo base |",
         "| :-- | :-- | --: | --: | --: |"]
    for comp, frames in FRAMES:
        for where, lab in (("cohort", f"{n} cohort CTs"), ("six", "6 held-out patients")):
            x = cb[where][comp]
            L.append(f"| {frames} | {lab} | {x['combination']} | {x['authors_original']} | "
                     f"{x['learned_mask_servo_base']} |")
    L += ["", f"Contrasts on the {n} cohort CTs: the difference family (m = 4; exact sign-flip test, Holm) and, under "
              "each member, its equivalence test within ±1 mm (equivalence family, m = 4; TOST, Holm).", "",
          f"| Contrast on the {n} cohort CTs | Family | Difference (mm), 95% t CI, or 90% CI | k/{n} | p | Holm p | "
          "Decision |", "| :-- | :-- | :-- | :-: | --: | --: | :-- |"]
    for c in cb["contrasts"]:
        L.append(f"| {c['id']}: {c['name']} | {c['family']} | {c['difference_ci95_t']} | {c['k']} | "
                 f"{c['p_signflip']} (sign-flip) | {c['p_holm']} | {c['decision']} |")
        e = c.get("equivalence")
        if e:
            L.append(f"| {c['id']}: equivalence within ±1 mm | {e['family']} | 90% {e['ci90']} | | {e['p_tost']} (TOST) | "
                     f"{e['p_holm']} | {e['decision']} |")
    return L + [""]


def _methods_md(methods):
    L = ["## Methods list", "",
         f"The {methods['n_published']} published methods placed in the slots (Table 2 of the article: "
         + and_join([f"{v} {SLOT_WORDS.get(k, k)}" for k, v in methods["counts_table2"].items()])
         + "), the reference components, and the methods added since. Each DOI was checked against its registry "
           "(title and year); a work without a DOI gives its stable identifier. A method added since enters with "
           "the DOI of its publication, resolved at Crossref or DataCite when it is added, and its reference is "
           "built from that registry record.", "",
         "| Slot | Method | Reference | DOI or identifier | Code and weights | Adapter |",
         "| :-- | :-- | :-- | :-- | :-- | :-- |"]
    for m in methods["published_methods"] + methods["reference_components"]:
        refs = m.get("references") or []
        ref_txt = " ".join(w["reference"] for w in refs) if refs else m.get("reference", "")
        ids = []
        for w in refs:
            if w.get("doi"):
                ids.append(f"[{w['doi']}](https://doi.org/{urllib.parse.quote(w['doi'], safe='/')})")
            else:
                ids.append(f"no DOI: {w['identifier']}" + (f" (arXiv DOI {w['arxiv_doi']})" if w.get("arxiv_doi")
                                                          else ""))
        lic = " ".join(m.get("license_notes") or []) or m.get("license", "")
        if m.get("note"):
            lic = f"{lic} Note: {m['note']}." if lic else m["note"]
        L.append("| " + " | ".join(_md_escape(x) for x in (
            m["slot"], m["name"], ref_txt, "; ".join(ids) or "none", lic, m.get("adapter", ""))) + " |")
    for m in methods.get("submitted_methods", []):          # text from outside the repository: escaped
        refs = m.get("references") or []
        ref_txt = " ".join(md_text(w["reference"], 1000) for w in refs)
        ids = "; ".join(doi_link(w["doi"]) for w in refs if w.get("doi")) or "none"
        code = url_link("code", m["code_url"]) + f"; {md_text(m['license'])}"
        L.append("| " + " | ".join((SLOT_WORDS.get(m["slot"], m["slot"]), md_text(m["name"]), ref_txt, ids, code,
                                    md_text(m.get("adapter", "")))) + " |")
    return L + [""]


__all__ = ["plain_cells", "submitted_cells", "markdown", "and_join", "flags_of", "slot_order", "role_of",
           "pretrained_rows"]
