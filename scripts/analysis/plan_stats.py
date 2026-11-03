#!/usr/bin/env python3
"""Shared statistics, formatting and provenance layer of the v2.1 analysis scripts.

Every analysis script of this folder imports this module. Its parts:

  Constants    load_constants(): the decision constants of the v2.1 plan, read after a sha256 check (development
               repository: the files that plan_private.py names; public mode: plan_inputs.py and the committed
               constants); assert_constants() raises when a script's own constants differ.
  Contrasts    contrast(): the mean difference with its 95% paired t interval, the number of units that share the
               sign of the mean, the paired t and exact sign-flip p values, and the patient-cluster percentile
               bootstrap interval of collect_results_v2.boot_ci.
  Decisions    decide_n6(), decide_cohort(), signflip_p() (exact enumeration up to enumerate_max_n units, Monte
               Carlo draws above that), tost_t(), tost_signflip(), intersection_union().
  Power        power_paired_t(), t_critical(), min_signflip_p(), tost_power(), mdd_paired_t().
  Families     holm_family(): Holm within a named family of the size the constants fix; a member without a complete
               result counts as p = 1.
  Strings      fmt_*(), string_n6(), string_cohort(), label_n6(), fill(), check_text(), sentence_key().
  Bootstraps   patient_bootstrap_ci(), patient_seed_bootstrap_ci(), crossed_bootstrap_ci(), ratio_bootstrap_ci().
  Provenance   write_result(): never overwrites; stamps each output with its status string, the digest of the v2.1
               plan, the repository commit (read from .git; git is never run), the md5 of every input and of the
               script, the argv, the working folder and the UTC time.

Helpers reused from scripts/posthoc_v2.py: boot_ci, holm, crossed_boot and sign_test_p. Not reused, because they
compute a different statistic: posthoc_v2.signflip_p, posthoc_v2.t_stats, posthoc_v2.tost and posthoc_v2.contrast.

The module never runs git, never renders, never uses the network and never reads the inputs of
FORBIDDEN_INPUT_PATTERNS (check_input_path() raises on them).

Amendments 3 to 6 (every function keeps its behaviour for the earlier amendments):
  load_amendment(n), amendment_info(n)   the Appendix block of amendment n (2 to 6) after its sha256 check and the
                       check of its field 'amendment'; n = 1 raises.
  STATUS_A4, STATUS_A5, STATUS_A6        the status strings of amendments 4 to 6, all in ALLOWED_STATUS.
  BLIND_ROOTS          the output roots of amendments 3, 5 and 6 are refused by check_input_path() (except their
                       check folders and the pilots of amendment 6) until lift_blind_set() opens one set for its one
                       reader, after the set's check record passes, cites the amendment's sha256 and is in the
                       records ledger.
  AMOS_PATTERNS, AMEND6_AMOS_PATTERNS    the AMOS outputs, refused until open_amos_read() or open_amos_exception()
                       opens them for their readers; open_amend6_route() lifts the AMOS route folder of amendment 6
                       alone.
  A6_READ_EXCEPTIONS   open_amend6_reads() (amend6.py only) waives the token lpips or ctanat for named folders; never
                       for checkpoints, logs or histories, nor for any path that usbg.provenance.is_private_path
                       refuses.
                       blind_state() gains 'a6_reads' only when one was opened.
  ledger_entries(), ledger_has(), ledger_at_head(), committed_problems()   the records ledger beside this file.
  signflip_method_label(res)             the printed method of a sign-flip p, from its 'method' field only.
  successor_label(), TAG_SUCCESSOR, successor_rule_record()   an output pinned to tag v2.1.2 is accepted at the
                       commit of tag v2.1.3 only, with the recorded content of each declared file; GIT_RO_ARGS and
                       git_ro_env() for the callers' read-only git runner; GitRunnerError when git cannot answer.
"""
from __future__ import annotations

import copy
import datetime as _dt
import functools
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import script_dirs as _SD  # noqa: E402  (code tree, result root, public mode)
import plan_inputs as _PI  # noqa: E402  (the development text and the archive's forms of the status strings)

# the plan documents: the development backend (plan_private.py), None in public mode and in the public mirror
_PP = _SD.private_module()
# the successor-rule label of the results archive's records (public mode and the public mirror)
PUBLIC_SUCCESSOR_DECISION = "tag successor rule"
PLAN_DIR = _PP.PLAN_DIR if _PP else None
PLAN_V21_PATH = _PP.PLAN_V21_PATH if _PP else None
PLAN_V21_SHA_PATH = _PP.PLAN_V21_SHA_PATH if _PP else None
PLAN_V21_COPY = _PP.PLAN_V21_COPY if _PP else None
AMENDMENTS = dict(_PP.AMENDMENTS) if _PP else {}
# the value of the Appendix field 'amendment' of each amendment that has an Appendix block (amendment 1 has none)
AMENDMENT_FIELD = {2: "v2.1 amendment 2", 3: "v2.1 amendment 3", 4: "v2.1 amendment 4", 5: "v2.1 amendment 5",
                   6: "v2.1 amendment 6"}
DATA_ROOT = _SD.results_root()        # the result files: the code tree, or the result root in public mode
RESULTS_ROOT = os.path.join(DATA_ROOT, "outputs", "v2_1", "analysis_plan")
OUTPUTS_ROOT = os.path.join(DATA_ROOT, "outputs")
RECORDS_LEDGER = os.path.join(HERE, "records_ledger.txt")

import posthoc_v2 as _ph  # noqa: E402  (reused helpers: boot_ci, holm, crossed_boot, sign_test_p)

# the provenance status strings, the only values write_result() accepts; amendments 4, 5 and 6 add one
# each
STATUS_A4 = _PI.text("status a4")
STATUS_A5 = _PI.text("status a5")
STATUS_A6 = _PI.text("status a6")
ALLOWED_STATUS = ("exploratory, v2.1", "exploratory, v2.1 pluggability", "post hoc, v2", STATUS_A4, STATUS_A5,
                  STATUS_A6)
# fixed strings of the printed results.
NO_DIFFERENCE = "no difference detected (not powered for equivalence)"
PRE_SPECIFIED = "exploratory, pre-specified rule"
SECONDARY = "secondary"
NOT_RUN = "not run"
NOT_COMPLETED = "not completed"
UNANTICIPATED = _PI.text("unanticipated")
Q2R_WORDS = {"holds": "replicates", "reversed": "reverses", "no_difference": "not replicated"}
# Inputs this layer refuses to read (task scope: amendment 2 groups, real-frame pools, private derivatives).
def _package_private_dirnames():
    """Public mode: usbg.provenance.PRIVATE_DIRNAMES read from the package source (this layer does not import usbg,
    whose __init__ changes warning filters), or () for a package without that list (the closed world of the result
    root then reads the listed files only)."""
    import ast
    src = os.path.join(REPO_ROOT, "src", "usbg", "provenance.py")
    if not os.path.isfile(src):
        return ()
    with open(src, encoding="utf-8") as fh:
        for node in ast.parse(fh.read()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PRIVATE_DIRNAMES" for t in node.targets):
                return tuple(ast.literal_eval(node.value))
    return ()


if _PP is not None:     # development repository: the private folder names and patterns, written out in plan_private
    _PRIVATE_PARTS, _PRIVATE_PATTERNS = tuple(_PP.PRIVATE_DIRNAMES), tuple(_PP.PRIVATE_INPUT_PATTERNS)
else:                   # public mode: every private folder name of the package as a pattern of its own
    _PRIVATE_PARTS = _package_private_dirnames()
    _PRIVATE_PATTERNS = _PRIVATE_PARTS
FORBIDDEN_INPUT_PATTERNS = ("imnet", "samus", "lpips", "ctanat") + _PRIVATE_PATTERNS
# Amendments 3 and 5: an output root (the root, not the token, so the job files and analysis_plan/amend5_*.json stay
# readable), refused except its checks/ folder; one reader lifts one set once the set's check record passes
# (lift_blind_set). records: the check records each set needs, relative to the root; the amend5 record holds one entry
# per set under 'sets'.
BLIND_ROOTS = {
    "amend3": {"root": "outputs/v2_1/amend3/", "allow": ("checks/",), "reader": "amend3.py", "amendment": 3,
               "sets": {"A": "ctrl/", "B": "contact/", "C": "phantoms/"},
               "records": {"A": ("checks/arm_a.json",), "B": ("checks/arm_b.json", "checks/arm_b_post.json"),
                           "C": ("checks/arm_c.json",)}},
    "amend5": {"root": "outputs/v2_1/amend5/", "allow": ("checks/",), "reader": "amend5.py", "amendment": 5,
               "sets": {"six": "train/", "cohort": "cohort/", "armh": "armh/", "amos": "amos/"},
               "records": {s: ("checks/amend5_checks.json",) for s in ("six", "cohort", "armh", "amos")},
               # amendment 5: an identity failure stops only its composition, so one composition of a
               # set may be lifted alone when the record marks it readable (sets[<set>].compositions[<P|S>])
               "compositions": {"P": "physics_", "S": "sonospade_"},
               "set_dirs": {"six": ("train/",), "cohort": ("cohort/",), "armh": ("armh/train/", "armh/cohort/"),
                            "amos": ("amos/",)}},
    # amendment 6: checks/ holds the check records, p9/pilot/ and r13/pilot/ the TRAIN-only records;
    # one record holds one entry per item under 'items', and an item opens once the record records it
    # (record_rule 'items': items[<set>].recorded is true; amend6.py applies the check outcomes)
    "amend6": {"root": "outputs/v2_1/amend6/", "allow": ("checks/", "p9/pilot/", "r13/pilot/"), "reader": "amend6.py",
               "amendment": 6,
               "sets": {"g2": "g2/", "p9": "p9/", "l10": "l10/", "c30": "c30/", "r13": "r13/", "look": "look/",
                        "sp": "sp/", "td": "td/"},
               "records": {s: ("checks/amend6_checks.json",) for s in
                           ("g2", "p9", "l10", "c30", "r13", "look", "sp", "td")},
               "record_rule": "items"},
}
# Amendment 4: refused (each also covers table4_ext) unless an AMOS gate of this process is open.
AMOS_PATTERNS = ("outputs/v2_1/amos/table4", "outputs/v2_1/amos_x/table4")
AMOS_READER = "amos_q2r.py"
AMOS_EXCEPTION_READERS = ("amend5.py", "make_amend5_jobs.py", "amend6.py")
# Amendment 6: the files under look/amos stay closed after 'look' is lifted too; only open_amos_exception(), called
# for amend6.py, opens them in this process.
A6_READER = BLIND_ROOTS["amend6"]["reader"]
AMEND6_AMOS_PATTERNS = ("outputs/v2_1/amend6/look/amos",)
# the folder that amend6.py route-amos compares with the cell, readable ahead of the record (open_amend6_route)
A6_ROUTE_PREFIX = "look/amos/route/"
# Amendment 6: the read exceptions of amend6.py, {kind: (the token waived, the folders it is waived
# for)}; a kind opens only through open_amend6_reads(), and only files under its folders are waived
A6_READ_EXCEPTIONS = {"amend2_lpips": ("lpips", ("outputs/v2_1/plug/reward/train/lpips/",)),
                      "amend2_ctanat": ("ctanat", ("outputs/v2_1/plug/ctanat/fast/",)),
                      "amend6_sp": ("lpips", ("outputs/v2_1/amend6/sp/",)),
                      "td_label_caches": ("ctanat", ("data/dataset/v2_1/plug/ctanat_ts2/default/cache_anat/",
                                                     "data/dataset/v2_1/plug/ctanat_ts2/fast/cache_anat/"))}
# {folder: set}: amendment 6 outputs outside outputs/v2_1/amend6/ (the label caches of the set 'td' in the default mode),
# which the blind root does not cover; such a folder is waived only after its set was lifted for amend6.py
# (lift_blind_set); the caches of the fast mode (from amendment 2) need no lift
A6_WAIVER_AFTER_LIFT = {"data/dataset/v2_1/plug/ctanat_ts2/default/cache_anat/": "td"}
A6_NEVER_WAIVED = ("imnet", "samus") + _PRIVATE_PATTERNS
# the private folder names (usbg.provenance.PRIVATE_DIRNAMES, written out so that this layer need not import usbg): a
# path that has one of them as a part is private and is never read through a waiver
A6_PRIVATE_DIRNAMES = _PRIVATE_PARTS
A6_WAIVER_REFUSED_SUFFIXES = (".pt", ".pth", ".ckpt", ".log")   # weights and logs are never read through a waiver
A6_WAIVER_REFUSED_PART = "history"                               # nor a training history
_LIFTED = {g: set() for g in BLIND_ROOTS}       # process-local: set prefixes lifted per group
_AMOS_OPEN = set()                             # process-local: AMOS patterns opened
_A6_OPEN = set()                               # process-local: amendment 6 read exceptions opened (their kinds)
_BLIND_LOG = []                                # every lift and every opened gate, for the provenance of a result
BANNED_TEXT = (chr(0x2014), "-" * 3)
BANNED_WORDS = re.compile(r"\b(confirmed|significant|significantly|comparable)\b", re.IGNORECASE)
SIGNFLIP_TOL = 1e-12            # a sign vector counts when |mean(s * d)| >= |mean(d)| - 1e-12
MC_CHUNK = 100_000              # Monte Carlo rows per block; the draw stream does not depend on it


class PlanStatsError(Exception):
    """Base class of every error raised by this module."""


class PlanHashMismatch(PlanStatsError):
    pass


class ConstantsMismatch(PlanStatsError):
    pass


class RefuseOverwrite(PlanStatsError, FileExistsError):
    pass


class ForbiddenPath(PlanStatsError):
    pass


class IncompleteData(PlanStatsError):
    pass


# ===================================================================================== hashing and paths
def _hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path):
    return _hash(path, "sha256")


def md5(path):
    return _hash(path, "md5")


def rel(path):
    """Path relative to the main repository when inside it (to the result root first in public mode), else with ~
    for the home directory."""
    p = os.path.abspath(path)
    if _SD.public() and (p == DATA_ROOT or p.startswith(DATA_ROOT + os.sep)):
        return os.path.relpath(p, DATA_ROOT)
    if p == REPO_ROOT or p.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(p, REPO_ROOT)
    home = _PP.HOME if _PP is not None else os.environ.get("HOME", "")
    return "~" + p[len(home):] if home and p.startswith(home + os.sep) else p


def _slashes(form):
    """A path form with every separator written as '/', so the patterns below match on Windows too."""
    return str(form).replace("\\", "/").replace(os.sep, "/")


def _blind_hits(form):
    """Blinded roots a lowercased absolute path form lies under, outside the allow-list and the lifted sets."""
    hits = []
    probe = _slashes(form).rstrip("/") + "/"
    for g, spec in BLIND_ROOTS.items():
        key = "/" + spec["root"].lower()
        i = probe.find(key)
        if i < 0:
            continue
        rest = probe[i + len(key):]
        ok = [p.lower() for p in spec["allow"]] + sorted(_LIFTED[g])
        if rest and any(rest.startswith(p) for p in ok):
            continue
        hits.append(spec["root"])
    return hits


def _a6_folder_open(prefix):
    """A read-exception folder is live unless A6_WAIVER_AFTER_LIFT holds it back until its item's set is lifted."""
    s = A6_WAIVER_AFTER_LIFT.get(prefix)
    return s is None or BLIND_ROOTS["amend6"]["sets"][s].lower() in _LIFTED["amend6"]


def _a6_waived(token, forms):
    """True when an amendment 6 read exception opened in this process waives `token` for every form of a path: the
    token belongs to a kind opened here, each form of the path sits under a live folder of that kind (a folder of
    A6_WAIVER_AFTER_LIFT is live only after its set was lifted), no form has a private part (A6_PRIVATE_DIRNAMES, the
    rule of usbg.provenance.is_private_path) and no form's file name ends in a weight or log suffix or holds
    'history'. The A6_NEVER_WAIVED tokens are never waived."""
    if token in A6_NEVER_WAIVED or not _A6_OPEN:
        return False
    for s in forms:
        parts = s.rstrip("/").split("/")
        if any(d in parts for d in A6_PRIVATE_DIRNAMES):
            return False
        if parts[-1].endswith(A6_WAIVER_REFUSED_SUFFIXES) or A6_WAIVER_REFUSED_PART in parts[-1]:
            return False
    for kind in sorted(_A6_OPEN):
        tok, prefixes = A6_READ_EXCEPTIONS[kind]
        live = [p.lower() for p in prefixes if _a6_folder_open(p)]
        if tok == token and live and all(any("/" + p in s for p in live) for s in forms):
            return True
    return False


def check_input_path(path):
    """Raise ForbiddenPath if a path matches a pattern this analysis must never read. Call before opening.

    Refused: the FORBIDDEN_INPUT_PATTERNS tokens (amendment 6: lpips or ctanat is waived only under the live folders
    of a read exception open_amend6_reads() opened, never for a weight, log or history file or a private path part);
    any path under a BLIND_ROOTS root (outputs/v2_1/amend3/, outputs/v2_1/amend5/, outputs/v2_1/amend6/) except its
    allowed folders and the sets lift_blind_set() opened in this process; the AMOS_PATTERNS
    (outputs/v2_1/amos/table4, outputs/v2_1/amos_x/table4, also table4_ext) unless an AMOS gate is open; and
    AMEND6_AMOS_PATTERNS (outputs/v2_1/amend6/look/amos) unless amendment 4's exception opened it for amend6.py, even
    when the set 'look' is lifted. Both the absolute path and the resolved path are checked, so a symlink out of an
    allowed folder into a blinded one is refused. In public mode both forms are taken below the result root
    (script_dirs.guard_forms), so the location of ROOT itself never matches a pattern."""
    forms = {_slashes(f).lower() for f in _SD.guard_forms(path)}
    tokens = sorted({pat for pat in FORBIDDEN_INPUT_PATTERNS for s in forms if pat in s})
    hits = [pat for pat in tokens if not _a6_waived(pat, forms)]
    hits += sorted({r for s in forms for r in _blind_hits(s)})
    hits += sorted({pat for pat in AMOS_PATTERNS for s in forms if "/" + pat in s and pat not in _AMOS_OPEN})
    hits += sorted({pat for pat in AMEND6_AMOS_PATTERNS for s in forms if "/" + pat in s and pat not in _AMOS_OPEN})
    if hits:
        raise ForbiddenPath(f"refusing to read {path}: matches forbidden pattern(s) {hits}")
    return path


# ===================================================================================== Appendix 2 constants
def _read_sha_file(sha_path):
    with open(sha_path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    if not lines:
        raise PlanHashMismatch(f"{sha_path} is empty")
    parts = lines[0].split()
    if not parts or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
        raise PlanHashMismatch(f"{sha_path}: first token is not a sha256 hex digest")
    return {"sha256": parts[0], "recorded_path": parts[1] if len(parts) > 1 else None,
            "recorded_time": lines[1].strip() if len(lines) > 1 else None}


def _extract_appendix2(text):
    i = text.find("## Appendix 2")
    if i < 0:
        raise PlanStatsError("the v2.1 plan has no '## Appendix 2' section")
    m = re.search(r"```json\s*\n(.*?)\n```", text[i:], re.DOTALL)
    if not m:
        raise PlanStatsError("Appendix 2 has no ```json block")
    return json.loads(m.group(1))


@functools.lru_cache(maxsize=8)
def _load(path, sha_path):
    rec = _read_sha_file(sha_path)
    got = sha256(path)
    if got != rec["sha256"]:
        raise PlanHashMismatch(f"sha256 of {path} is {got}, but {sha_path} records {rec['sha256']}")
    with open(path, "r", encoding="utf-8") as fh:
        consts = _extract_appendix2(fh.read())
    field = _PP.APPENDIX2_FIELD
    if consts.get(field) != "v2.1":
        raise PlanStatsError(f"Appendix 2 field {field!r} is {consts.get(field)!r}, expected 'v2.1'")
    info = {"path": rel(path), "sha256": got, "sha12": got[:12], "sha256_file": rel(sha_path),
            "sha256_file_recorded_path": rec["recorded_path"], "sha256_file_recorded_time": rec["recorded_time"],
            "verified": True}
    return consts, info


def load_constants(path=PLAN_V21_PATH, sha_path=PLAN_V21_SHA_PATH):
    """The decision constants of the v2.1 plan (Appendix 2), after verifying its sha256.

    Raises PlanHashMismatch if the file's digest differs from the one in the .sha256 file. Returns a fresh
    deep copy on every call, so a caller cannot change the constants another caller sees. In public mode (and in the
    public mirror) the constants come from plan_inputs.py: alpha, the decision rules, the families and the bootstrap,
    the ones this layer reads."""
    if _SD.public() or _PP is None:
        import plan_inputs as PI
        return PI.plan_v21_constants()
    consts, _ = _load(os.path.abspath(path), os.path.abspath(sha_path))
    return copy.deepcopy(consts)


def _public_plan():
    """True when the plan documents are not at hand (public mode, or the public mirror): the digests and constants
    then come from plan_inputs.py."""
    return _SD.public() or _PP is None


PUBLIC_DIGEST_NOTE = ("public mode: the digest recorded in the public constants (plan_inputs.py); the plan document "
                      "is not part of the public mirror, so it is not re-hashed here")


def plan_v21_info(path=PLAN_V21_PATH, sha_path=PLAN_V21_SHA_PATH):
    """Verified digest record of the v2.1 plan (path, sha256, sha12, the .sha256 file and its recorded fields), plus
    whether its registered copy under outputs/v2/ exists and matches. In public mode (and in the public mirror) the
    recorded digest and hashing time come from plan_inputs.py, with verified None."""
    if _public_plan():
        import plan_inputs as PI
        d = PI.plan_v21_digest()
        PI.notice("the digest of the v2.1 plan is the recorded one (the plan document is in the development "
                  "repository only)")
        return {"path": "the v2.1 plan", "sha256": d["sha256"], "sha12": d["sha12"],
                "sha256_file_recorded_time": d.get("recorded"), "verified": None, "note": PUBLIC_DIGEST_NOTE}
    _, info = _load(os.path.abspath(path), os.path.abspath(sha_path))
    info = dict(info)
    now = sha256(path)
    if now != info["sha256"]:
        raise PlanHashMismatch(f"{path} changed after its constants were loaded ({now})")
    copy_path = PLAN_V21_COPY
    if os.path.exists(copy_path):
        info["registered_copy_outputs_v2"] = {"path": rel(copy_path), "matches": sha256(copy_path) == info["sha256"]}
    else:
        info["registered_copy_outputs_v2"] = {"path": rel(copy_path), "matches": None, "exists": False}
    return info


def amendment_info(n, plan_dir=PLAN_DIR):
    """sha256 record of amendment n (1 to 6), verified against its own .sha256 file. For n >= 3 the record also
    holds the file name and the hashing time (amendments 3 and 6). In public mode (and in the public mirror): the recorded sha256 from plan_inputs.py, with
    verified None."""
    if _public_plan():
        import plan_inputs as PI
        if n not in (1, 2, 3, 4, 5, 6):
            raise PlanStatsError(f"amendment {n} is not known (known: [1, 2, 3, 4, 5, 6])")
        h = PI.amendment_digest(n)
        PI.notice("amendment digests are the recorded ones (the plan documents are in the development repository "
                  "only)")
        return {"path": f"amendment {n}", "sha256": h, "sha12": h[:12], "verified": None, "note": PUBLIC_DIGEST_NOTE}
    if n not in AMENDMENTS:
        raise PlanStatsError(f"amendment {n} is not known (known: {sorted(AMENDMENTS)})")
    stem = AMENDMENTS[n]
    p, sp = os.path.join(plan_dir, stem + ".md"), os.path.join(plan_dir, stem + ".sha256")
    rec = _read_sha_file(sp)
    got = sha256(p)
    if got != rec["sha256"]:
        raise PlanHashMismatch(f"sha256 of {p} is {got}, but {sp} records {rec['sha256']}")
    info = {"path": rel(p), "sha256": got, "sha12": got[:12], "sha256_file": rel(sp), "verified": True}
    if n >= 3:
        info.update(file=os.path.basename(p), hashed_utc=rec["recorded_time"])
    return info


# the first json block after the Appendix heading: the pattern is the development backend's (plan_private.py
# APPENDIX_RE), because only the development repository holds the plan documents


@functools.lru_cache(maxsize=16)
def _amendment_block(path, digest):
    with open(path, "rb") as fh:
        raw = fh.read()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise PlanHashMismatch(f"{path} changed while it was read")
    m = _PI.private().APPENDIX_RE.search(raw.decode("utf-8"))
    if not m:
        raise PlanStatsError(f"{os.path.basename(path)} has no json block under its Appendix heading")
    return json.loads(m.group(1))


def load_amendment(n, *, plan_dir=PLAN_DIR):
    """The Appendix block of amendment n (2 to 6): the file is located by name beside the other plan documents, its
    sha256 is checked against the first token of its .sha256 file, the first json block after its Appendix
    heading is parsed and its field 'amendment' must read 'v2.1 amendment n'. Returns a fresh deep copy. Amendment 1 has no Appendix
    block and raises."""
    if n == 1:
        raise PlanStatsError("amendment 1 has no Appendix block (its constants are those of the v2.1 plan)")
    if _public_plan():
        raise PlanStatsError(f"the Appendix block of amendment {n} is read in the development repository only; the "
                             "public build reads the constants it needs through plan_inputs.amendment_value")
    if n not in AMENDMENT_FIELD:
        raise PlanStatsError(f"amendment {n} is not known (known: {sorted(AMENDMENT_FIELD)})")
    info = amendment_info(n, plan_dir)
    path = os.path.join(plan_dir, AMENDMENTS[n] + ".md")
    block = _amendment_block(os.path.abspath(path), info["sha256"])
    if block.get("amendment") != AMENDMENT_FIELD[n]:
        raise PlanStatsError(f"{os.path.basename(path)}: field 'amendment' is {block.get('amendment')!r}, "
                             f"expected {AMENDMENT_FIELD[n]!r}")
    return copy.deepcopy(block)


# ===================================================================================== records ledger
def _parse_ledger(text, where):
    out = {}
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("  ", 2)
        if len(parts) != 3 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]) or os.path.isabs(parts[1].strip()) \
                or not parts[1].strip() or not parts[2].strip():
            raise PlanStatsError(f"{where} line {i} is not '<sha256>  <relative path>  <label>'")
        h, p, label = parts[0], parts[1].strip(), parts[2].strip()
        if h in out and out[h][0] != p:
            raise PlanStatsError(f"{where} lists {h[:12]} for two paths ({out[h][0]}, {p})")
        out.setdefault(h, (p, label))
    return out


def ledger_entries(path=RECORDS_LEDGER):
    """{sha256: (path relative to the main repository, label)} of the records ledger, one line per committed
    record: '<sha256>  <relative path>  <label>' ('#' lines are comments), read from the file as it is. An analysis
    run reads the committed version instead (ledger_at_head); scripts/plug/merge_step3_jobs.py checks the file
    against HEAD before any gated job runs."""
    with open(path, "r", encoding="utf-8") as fh:
        return _parse_ledger(fh.read(), rel(path))


class CommittedLedger(dict):
    """{sha256: (relative path, label)} of the records ledger as committed at HEAD, with its source and the md5 of
    the committed text. ledger_has() and the gates accept it in place of a ledger path."""

    def __init__(self, entries, source, text_md5):
        super().__init__(entries)
        self.source, self.md5 = source, text_md5


def ledger_at_head(show, path=RECORDS_LEDGER, root=REPO_ROOT):
    """The records ledger as committed (amendments 3, 4 and 5: a record counts only once its sha256 is in a
    commit). show(relative path) must return the text of that path at
    HEAD; the caller runs the read-only git command, because this module never runs git."""
    rel_ = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    text = show(rel_)
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return CommittedLedger(_parse_ledger(text, f"HEAD:{rel_}"), f"HEAD:{rel_}",
                           hashlib.md5(text.encode("utf-8")).hexdigest())


def committed_problems(show_bytes, paths, root=REPO_ROOT):
    """[(relative path, why)] for each file whose working copy differs from its HEAD blob or is not committed
    (amendments 3 and 5: the analysis code is committed before it first reads).
    show_bytes(relative path) returns the HEAD blob's bytes or raises; the caller runs git."""
    out = []
    for p in paths:
        rel_ = os.path.relpath(os.path.realpath(p), os.path.realpath(root))
        try:
            blob = show_bytes(rel_)
        except Exception as e:                          # not in HEAD, or git failed
            out.append((rel_, f"not committed ({str(e)[:120]})"))
            continue
        if hashlib.md5(blob).hexdigest() != md5(p):
            out.append((rel_, "the working copy differs from HEAD"))
    return out


def ledger_has(record, ledger=RECORDS_LEDGER, root=REPO_ROOT):
    """True if the current content of `record` is listed in the ledger for that path. ledger: a ledger file path, or
    the committed entries (CommittedLedger or any {sha256: (path, label)} mapping)."""
    if not os.path.isfile(record):
        return False
    entries = ledger if isinstance(ledger, Mapping) else ledger_entries(ledger)
    e = entries.get(sha256(record))
    return bool(e) and os.path.realpath(os.path.join(root, e[0])) == os.path.realpath(record)


def ledger_md5(ledger):
    """md5 of the ledger a run used: the committed text's for a CommittedLedger, the file's for a path."""
    if isinstance(ledger, CommittedLedger):
        return ledger.md5
    if isinstance(ledger, str) and os.path.isfile(ledger):
        return md5(ledger)
    return None


# ===================================================================================== tag successor (v2.1.3)
# Outputs pinned to tag v2.1.2 (--tag-commit 6bc6ac6) are also accepted when they name the commit of tag v2.1.3,
# whose changes are limited to the declared files below (recorded by md5). The v2.1 checkout's
# scripts/plug/step3_common.py holds the same rule (successor_check); test_plan_stats.py asserts that both copies name
# the same base, tag, declared and pinned files. This copy also holds the content of each declared file at the tag
# ('blobs': git mode and object id; 'md5'), and accepts only a commit whose declared files have exactly that content,
# so a tag moved or made again on other content is refused.
TAG_SUCCESSOR = {
    "base": "6bc6ac644362d0f92d7e03c1a03418e90f534272",
    "base_tag": "v2.1.2",
    "tag": "v2.1.3",
    # the files that differ between tags v2.1.2 and v2.1.3, sorted
    "files": ("scripts/amos/build_amos_cohort.py",
              "scripts/amos/tests/test_build.py",
              "scripts/amos/tests/test_checks.py",
              "scripts/plug/make_amend3_jobs.py",
              "scripts/plug/make_amend5_jobs.py",
              "scripts/plug/step3_common.py",
              "tests/test_amend3_checks.py",
              "tests/test_amend5_checks.py",
              "tests/test_leakage_lineage.py",
              "tests/test_step3_common.py",
              "tests/test_tag_successor.py"),
    # the files whose md5s amendments 4 and 5 pin (the keys of step3_common.V211_MD5 of the v2.1 checkout): a
    # successor whose declared files name one of them, or anything under src/, is refused
    "never": ("scripts/build_anat_cache.py",
              "scripts/build_cohort_v21.py",
              "scripts/build_ct_volume.py",
              "scripts/check_leakage.py",
              "scripts/eval_view_ladder.py",
              "scripts/plug/admission_gate.py",
              "scripts/plug/cohort_plug_eval.py",
              "scripts/plug/episode_audit.py",
              "scripts/plug/totalseg_labels.py",
              "scripts/plug_reward_train.py",
              "scripts/plug_train_sb3.py",
              "scripts/run_table4_v2.py",
              "scripts/train_view_match_rl.py",
              "src/usbg/plug/rewards.py",
              "src/usbg/volume.py",
              "src/usbg/windows.py"),
    "never_prefixes": ("src/",),
    # the label that successor_rule_record() writes: the development backend's own (plan_private.SUCCESSOR_DECISION);
    # in public mode and in the public mirror PUBLIC_SUCCESSOR_DECISION, the label the results archive's records carry
    "decision": _PP.SUCCESSOR_DECISION if _PP else PUBLIC_SUCCESSOR_DECISION,
    # each declared file at tag v2.1.3: "<git mode> <git object id>" (checked by successor_label, b8) and its md5 (the
    # record of the decision; test_plan_stats.py asserts both against the tag once it exists)
    "blobs": {"scripts/amos/build_amos_cohort.py": "100644 53b21811d76e30303bd3877098189e28301da50b",
              "scripts/amos/tests/test_build.py": "100644 c249f9072d457f941cbd0e5539d7900121dd4af0",
              "scripts/amos/tests/test_checks.py": "100644 ce0471675bff45b231587ed92ef50e8e102a9948",
              "scripts/plug/make_amend3_jobs.py": "100644 06735379142ad6d0e5d533f01fb9b462074ae3b9",
              "scripts/plug/make_amend5_jobs.py": "100644 49203208e2f89e4d125cd102db267bb0f717a4d2",
              "scripts/plug/step3_common.py": "100644 3cf39e6f491fb0bb7cf875d7b8d4a9379f9e7a6b",
              "tests/test_amend3_checks.py": "100644 f4be1a5c77e95481b31502c3ff8d232feae8541e",
              "tests/test_amend5_checks.py": "100644 c0dda38fe28b85d778694e17329b578d7cf0ef60",
              "tests/test_leakage_lineage.py": "100644 b9ba2b401a32a002e89c29123363e3605911c986",
              "tests/test_step3_common.py": "100644 da27e9b4f7ca5bb9a26bda890df59268f74ae943",
              "tests/test_tag_successor.py": "100644 a31915272aeb804f050c6d5080b70bf6643e38de"},
    "md5": {"scripts/amos/build_amos_cohort.py": "0ae39127bd1e34854fa602e8ff846469",
            "scripts/amos/tests/test_build.py": "01957b44420e42da48bf02fd663b1df2",
            "scripts/amos/tests/test_checks.py": "f54db1a7932a849cbfe1e92353aa6598",
            "scripts/plug/make_amend3_jobs.py": "8b763b2097cb51ca1711596b77f6f369",
            "scripts/plug/make_amend5_jobs.py": "ecdad28dd617baf80b5ba14d2934b24e",
            "scripts/plug/step3_common.py": "a02b6c76ddc021f70c011d28b62bae0a",
            "tests/test_amend3_checks.py": "db59461d4dd91d7c8fcd61bb1e0e7033",
            "tests/test_amend5_checks.py": "54e0262095f63aac1aab6f003335e22e",
            "tests/test_leakage_lineage.py": "d2753963bd8eda58d8bee68da266f59f",
            "tests/test_step3_common.py": "ed79f3adfed5cdffee2731d292dd1da6",
            "tests/test_tag_successor.py": "113d546818628497654956dff7d40ac6"},
}
CODE_AT_TAG = "tag commit"                      # the code label of rule (a)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_PIN = re.compile(r"[0-9a-f]{7,40}")
# the read-only git a caller of successor_label runs (amend5.py, amos_q2r.py): no optional locks, no replace refs, no
# file system monitor, paths unquoted, and git_ro_env() drops every GIT_* variable (this module never runs git)
GIT_RO_ARGS = ("git", "--no-optional-locks", "--no-replace-objects", "-c", "core.quotepath=off", "-c",
               "core.fsmonitor=false")


class GitRunnerError(PlanStatsError):
    """v2.1.3: the caller's git runner could not answer a question of the successor rule (it raised, or it exited with
    a code that is neither of the rule's two answers). This is not a refusal: the analysis stops and writes nothing,
    so no result says that an output is not at the tag because of it."""


def git_ro_env(environ=None):
    """The environment for the caller's read-only git: environ (default os.environ) without any GIT_* variable, so
    GIT_DIR, GIT_INDEX_FILE, GIT_REPLACE_REF_BASE or a GIT_CONFIG_* setting cannot change what git reads."""
    return {k: v for k, v in (os.environ if environ is None else environ).items() if not k.startswith("GIT_")}


def successor_rule_record(rule=None):
    """The successor rule as plain JSON values, for the record of a result that applied it."""
    r = TAG_SUCCESSOR if rule is None else rule
    return {"base": r["base"], "base_tag": r.get("base_tag"), "tag": r["tag"], "files": list(r.get("files") or ()),
            "never": list(r.get("never") or ()), "never_prefixes": list(r.get("never_prefixes") or ()),
            "decision": r.get("decision"), "blobs": dict(r.get("blobs") or {}), "md5": dict(r.get("md5") or {}),
            "rule": "the commit starts with the pin; or the pin is a prefix of the base commit and the commit is "
                    "the full commit of the tag, clean, a descendant of the base, its diff against the base lists "
                    "only the declared files, and each declared file has the content recorded for the tag"}


def successor_label(commit, pin, git, *, dirty=None, rule=None):
    """The code label of a commit for a --tag-commit pin, or None when the commit is refused. -> (label, info).

    The pin must be 7 to 40 hex digits (as the --tag-commit of every job writer); any other pin refuses every commit.
    (a) The commit starts with the pin: label "tag commit" (the v2.1.2 rule, unchanged); git is not called.
    (b) Otherwise label "successor <tag>" only when all of these hold, checked in this order:
        b1 the pin is a prefix of the rule's base commit (only a pin to the base has a successor);
        b2 the commit is 40 hex digits (a successor is matched by its full commit, never by a prefix);
        b3 dirty is exactly False (the output's provenance git.dirty: tracked files only);
        b4 the declared files name nothing under the never prefixes (src/) and none of the never files;
        b5 `git rev-parse --verify --quiet refs/tags/<tag>^{commit}` prints the commit (only the tagged commit);
        b6 `git merge-base --is-ancestor <base> <commit>` returns 0 (it descends from the base);
        b7 `git diff --name-only --no-renames --no-ext-diff --ignore-submodules=none <base> <commit>` lists only
           declared files;
        b8 `git ls-tree --full-tree <commit> -- <declared files>` gives each declared file the mode and object id the
           rule records ('blobs'), so the content is the content recorded for the tag.
    (c) Anything else: None, and info['why'] names the first condition that failed.
    git(args) is the caller's read-only git runner and returns (return code, stdout); this module never runs git. A
    runner that raises, or an exit code that is not an answer (rev-parse 0 or 1, merge-base 0 or 1, diff and ls-tree
    0), raises GitRunnerError: an error, never a refusal. rule: a mapping like TAG_SUCCESSOR (None reads
    TAG_SUCCESSOR at call time). info: {ok, how, commit, pin, base, tag, n_changed, why}."""
    r = TAG_SUCCESSOR if rule is None else rule
    c, p = str(commit or ""), str(pin or "")
    base, tag, base_tag = str(r.get("base") or ""), str(r.get("tag") or ""), r.get("base_tag") or "the base"
    info = {"ok": False, "how": None, "commit": c, "pin": p, "base": base, "tag": tag, "n_changed": None, "why": None}

    def refused(why):
        info["why"] = why
        return None, info

    def ask(args, answers=(0,)):
        try:
            rc, out = git(list(args))
        except Exception as e:                          # git missing, a time-out or a runner error: not an answer
            raise GitRunnerError(f"git {args[0]} could not be run: {type(e).__name__}: {str(e)[:160]}") from e
        if rc not in answers:
            raise GitRunnerError(f"git {args[0]} exited with code {rc}, which is not an answer of the successor rule")
        return rc, str(out)
    if not _PIN.fullmatch(p):
        return refused(f"the pin {p!r} is not 7 to 40 hex digits")
    if c.startswith(p):
        info.update(ok=True, how=CODE_AT_TAG, why="the commit starts with the pin")
        return CODE_AT_TAG, info
    if not (_HEX40.fullmatch(base) and base.startswith(p)):
        return refused(f"the commit does not start with the pin {p!r}, and the pin is not a prefix of the "
                       f"{base_tag} commit, the only pin with a successor")
    if not _HEX40.fullmatch(c):
        return refused(f"the commit {c!r} is not 40 hex digits (a successor is matched by its full commit)")
    if dirty is not False:
        return refused(f"the output does not record a clean working tree (dirty is {dirty!r})")
    files = tuple(r.get("files") or ())
    never = set(r.get("never") or ())
    pinned = [f for f in files if f in never or any(f.startswith(x) for x in r.get("never_prefixes") or ())]
    if pinned:
        return refused(f"the declared files name pinned files {pinned}")
    if not tag:
        return refused("the rule names no successor tag")
    blobs = dict(r.get("blobs") or {})
    if not files or set(blobs) != set(files):
        return refused("the rule does not record the content of every declared file")
    rc, out = ask(["rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"], (0, 1))
    if rc == 1:
        return refused(f"tag {tag} does not exist")
    if out.strip() != c:
        return refused(f"{c[:12]} is not the commit of tag {tag}")
    rc, _ = ask(["merge-base", "--is-ancestor", base, c], (0, 1))
    if rc == 1:
        return refused(f"{c[:12]} does not descend from {base_tag} ({base[:12]})")
    _, out = ask(["diff", "--name-only", "--no-renames", "--no-ext-diff", "--ignore-submodules=none", base, c])
    changed = [x for x in out.splitlines() if x]                    # a name of spaces is a file too
    extra = sorted(set(changed) - set(files))
    if extra:
        return refused(f"the diff against {base_tag} lists files that are not declared: {extra[:8]}")
    _, out = ask(["ls-tree", "--full-tree", c, "--", *files])
    got = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and path:
            got[path] = f"{parts[0]} {parts[2]}"
    differ = sorted(f for f in files if got.get(f) != blobs.get(f))
    if differ:
        return refused(f"declared files differ from the content recorded for tag {tag}: {differ[:8]}")
    how = f"successor {tag}"
    info.update(ok=True, how=how, n_changed=len(changed),
                why=f"the commit of tag {tag}, a descendant of {base_tag}, clean, changing only {len(changed)} "
                    f"declared files, each with the content recorded for the tag "
                    f"({r.get('decision') or 'no decision recorded'})")
    return how, info


# ===================================================================================== blinding
def _caller_file(depth=2):
    """File name of the function that called the public gate function (depth frames up)."""
    try:
        return os.path.basename(sys._getframe(depth).f_code.co_filename)
    except ValueError:
        return None


def _record_json(path):
    if not os.path.isfile(path):
        raise PlanStatsError(f"check record {path} does not exist")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _cites(rec, digest):
    pl = rec.get("plan") or {}
    return pl.get("amendment_sha256") == digest and rec.get("cites", digest) == digest


def lift_blind_set(group, set_name, records, *, caller, ledger=RECORDS_LEDGER, outputs_root=OUTPUTS_ROOT,
                   plan_dir=PLAN_DIR, ledger_root=REPO_ROOT, composition=None):
    """Open one set of a blinded root for this process (the read locks of amendments 3, 5 and 6).

    Only the group's reader may call it (amend3.py for 'amend3', amend5.py for 'amend5', amend6.py for 'amend6'):
    `caller` (pass __file__) and the calling frame must both be that file. `records`: the set's check records, which
    must include the paths BLIND_ROOTS names for the set under outputs_root; each must exist, cite the verified
    amendment sha256 (plan.amendment_sha256 and 'cites'), record a pass (amend3: top-level 'passed'; amend5:
    sets[set_name].passed; amend6, record_rule 'items': items[set_name].recorded, the record exists for the item,
    whose check outcomes amend6.py applies) and, unless ledger is None, be listed in the records ledger.
    composition (amend5 only, 'P' or 'S'): lift only that composition's files of the set, which needs
    sets[set_name].compositions[composition].readable instead of the set's pass. Raises PlanStatsError otherwise."""
    if group not in BLIND_ROOTS:
        raise PlanStatsError(f"unknown blinded group {group!r}")
    spec = BLIND_ROOTS[group]
    if set_name not in spec["sets"]:
        raise PlanStatsError(f"{group} has no set {set_name!r} (sets: {sorted(spec['sets'])})")
    reader = spec["reader"]
    if os.path.basename(str(caller)) != reader or _caller_file() != reader:
        raise PlanStatsError(f"only {reader} may lift a set of {spec['root']} (caller {caller!r}, "
                             f"calling file {_caller_file()!r})")
    digest = amendment_info(spec["amendment"], plan_dir)["sha256"]
    want = [os.path.realpath(os.path.join(outputs_root, os.path.relpath(spec["root"], "outputs"), r))
            for r in spec["records"][set_name]]
    given = {os.path.realpath(str(r)) for r in records}
    missing = [w for w in want if w not in given]
    if missing:
        raise PlanStatsError(f"lifting {group}:{set_name} needs the check records {missing}")
    if composition is not None and composition not in (spec.get("compositions") or {}):
        raise PlanStatsError(f"{group} has no composition {composition!r} to lift alone")
    for p in want:
        rec = _record_json(p)
        if not _cites(rec, digest):
            raise PlanStatsError(f"{p} does not cite the verified sha256 of amendment {spec['amendment']}")
        if spec.get("record_rule") == "items":
            items = rec.get("items")
            entry = items.get(set_name) if isinstance(items, Mapping) else None
            if not (isinstance(entry, Mapping) and entry.get("recorded") is True):
                raise PlanStatsError(f"{p} does not record item {set_name} (items.{set_name}.recorded is not true)")
            passed = True
        elif group == "amend3":
            passed = rec.get("passed")
        else:
            entry = (rec.get("sets") or {}).get(set_name) or {}
            passed = entry.get("passed") if composition is None else \
                ((entry.get("compositions") or {}).get(composition) or {}).get("readable")
        if passed is not True:
            raise PlanStatsError(f"{p} does not record a pass for set {set_name}"
                                 + ("" if composition is None else f", composition {composition}"))
        if ledger is not None and not ledger_has(p, ledger, ledger_root):
            raise PlanStatsError(f"{p} is not listed in the records ledger (commit its sha256 first)")
    prefixes = ([spec["sets"][set_name]] if composition is None else
                [d + spec["compositions"][composition] for d in spec["set_dirs"][set_name]])
    for pre in prefixes:
        _LIFTED[group].add(pre.lower())
    _BLIND_LOG.append({"lifted": f"{group}:{set_name}" + ("" if composition is None else f":{composition}"),
                       "prefix": spec["root"] + prefixes[0], "prefixes": [spec["root"] + pre for pre in prefixes],
                       "records": [{"path": rel(p), "sha256": sha256(p)} for p in want],
                       "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})


def _amos_record_ok(path, digest, ledger, what, ledger_root=REPO_ROOT):
    rec = _record_json(path)
    if not _cites(rec, digest):
        raise PlanStatsError(f"{what} {path} does not cite the verified sha256 of amendment 4")
    if rec.get("passed") is not True:
        raise PlanStatsError(f"{what} {path} does not record a pass")
    if ledger is not None and not ledger_has(path, ledger, ledger_root):
        raise PlanStatsError(f"{what} {path} is not listed in the records ledger (commit its sha256 first)")
    return rec


def open_amos_read(caller, *, check_record, go_record, read_order_record, armx_check_record=None,
                   ledger=RECORDS_LEDGER, plan_dir=PLAN_DIR, ledger_root=REPO_ROOT):
    """Open outputs/v2_1/amos/table4 for amos_q2r.py (amendment 4): the AMOS check record
    passes and cites amendment 4's sha256; the arm X go record exists (decision 'run' or 'not run'); the read-order
    record passes (its own checks of the CT-anatomy output and of the listed items); each is listed in the ledger
    unless ledger is None. outputs/v2_1/amos_x/table4 opens too
    only when the go record says 'run' and armx_check_record passes. Returns the list of patterns opened."""
    if os.path.basename(str(caller)) != AMOS_READER or _caller_file() != AMOS_READER:
        raise PlanStatsError(f"only {AMOS_READER} may open the AMOS evaluation outputs (caller {caller!r})")
    digest = amendment_info(4, plan_dir)["sha256"]
    _amos_record_ok(check_record, digest, ledger, "AMOS check record", ledger_root)
    go = _record_json(go_record)
    decision = go.get("decision")
    if decision not in ("run", "not run"):
        raise PlanStatsError(f"go record {go_record}: decision {decision!r} is not 'run' or 'not run'")
    if ledger is not None and not ledger_has(go_record, ledger, ledger_root):
        raise PlanStatsError(f"go record {go_record} is not listed in the records ledger")
    ro = _record_json(read_order_record)
    if ro.get("passed") is not True:
        raise PlanStatsError(f"read-order record {read_order_record} does not record that the read order allows it")
    if ledger is not None and not ledger_has(read_order_record, ledger, ledger_root):
        raise PlanStatsError(f"read-order record {read_order_record} is not listed in the records ledger")
    opened = [AMOS_PATTERNS[0]]
    if decision == "run" and armx_check_record is not None:
        _amos_record_ok(armx_check_record, digest, ledger, "arm X check record", ledger_root)
        opened.append(AMOS_PATTERNS[1])
    _AMOS_OPEN.update(opened)
    _BLIND_LOG.append({"opened": opened, "by": AMOS_READER, "go_decision": decision,
                       "records": [{"path": rel(p), "sha256": sha256(p)} for p in
                                   (check_record, go_record, read_order_record, armx_check_record) if p],
                       "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return opened


def open_amos_exception(caller, *, q2r_result, check_record, ledger=RECORDS_LEDGER, plan_dir=PLAN_DIR,
                        ledger_root=REPO_ROOT):
    """Amendment 5: open outputs/v2_1/amos/table4 for amend5.py or make_amend5_jobs.py (the AMOS identity check),
    once q2r_result (amos_q2r.json) exists and the check record of amendment 4 passed, cites its sha256 and is in the
    ledger (unless ledger is None). amos_x stays closed. Returns the patterns opened. For amend6.py (amendment 6, the
    same conditions) the exception also opens AMEND6_AMOS_PATTERNS (the AMOS outputs of the set 'look' under
    outputs/v2_1/amend6/look/amos, which the blind root of amendment 6 still refuses until the set 'look' is lifted or
    open_amend6_route() opens the route)."""
    who = os.path.basename(str(caller))
    if who not in AMOS_EXCEPTION_READERS or _caller_file() != who:
        raise PlanStatsError(f"only {AMOS_EXCEPTION_READERS} may use the AMOS exception (caller {caller!r})")
    if os.path.basename(str(q2r_result)) != "amos_q2r.json" or not os.path.isfile(q2r_result):
        raise PlanStatsError(f"the AMOS exception needs outputs/v2_1/analysis_plan/amos_q2r.json ({q2r_result} "
                             "does not exist)")
    digest = amendment_info(4, plan_dir)["sha256"]
    _amos_record_ok(check_record, digest, ledger, "AMOS check record", ledger_root)
    opened = [AMOS_PATTERNS[0]] + ([AMEND6_AMOS_PATTERNS[0]] if who == A6_READER else [])
    _AMOS_OPEN.update(opened)
    _BLIND_LOG.append({"opened": list(opened), "by": who,
                       "exception": _PI.text("a6 amos exception" if who == A6_READER else "a5 amos exception"),
                       "records": [{"path": rel(p), "sha256": sha256(p)} for p in (q2r_result, check_record)],
                       "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return opened


def _a6_reader_only(caller, what):
    who = os.path.basename(str(caller))
    if who != A6_READER or _caller_file(3) != A6_READER:
        raise PlanStatsError(f"only {A6_READER} may {what} (caller {caller!r}, calling file {_caller_file(3)!r})")


def open_amend6_route(caller):
    """Amendment 6: lift the folder look/amos/route/ of the amendment 6 root alone, for amend6.py route-amos (it
    compares the route folder with the AMOS cell ahead of the record). Only amend6.py may call it, and only once
    open_amos_exception() has run for amend6.py in this process; the rest of look/ stays refused. Returns the prefix
    lifted."""
    _a6_reader_only(caller, "open the AMOS route folder of amendment 6")
    if AMEND6_AMOS_PATTERNS[0] not in _AMOS_OPEN:
        raise PlanStatsError("the AMOS route folder of amendment 6 opens only after amendment 4's AMOS exception was "
                             "opened for amend6.py in this process (open_amos_exception)")
    root = BLIND_ROOTS["amend6"]["root"]
    _LIFTED["amend6"].add(A6_ROUTE_PREFIX.lower())
    _BLIND_LOG.append({"lifted": "amend6:" + A6_ROUTE_PREFIX.rstrip("/"), "prefix": root + A6_ROUTE_PREFIX,
                       "prefixes": [root + A6_ROUTE_PREFIX], "by": A6_READER,
                       "reading": _PI.text("a6 route reading"),
                       "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return root + A6_ROUTE_PREFIX


def open_amend6_reads(caller, kinds):
    """Amendment 6: open the named read exceptions of A6_READ_EXCEPTIONS for amend6.py (checked on both the
    argument and the frame that calls it) in this process. Under an opened kind, check_input_path() waives that
    kind's token (lpips or ctanat) for files under its folders only, never for a file whose name ends in .pt, .pth,
    .ckpt or .log or holds 'history', nor for a path with a private part (A6_PRIVATE_DIRNAMES); imnet, samus and the
    private real-frame folders are never waived, and the blind roots and AMOS patterns still apply. A folder of
    A6_WAIVER_AFTER_LIFT (the label caches of the set 'td' in the default mode, an amendment 6 output outside the
    blind root) is waived
    only once lift_blind_set('amend6', <its set>, ...) has run in this process, whichever call comes first; the kind's
    other folders are waived at once. kinds: one kind name or a list of them. Returns the kinds open in this
    process."""
    _a6_reader_only(caller, "open the read exceptions of amendment 6")
    names = (kinds,) if isinstance(kinds, str) else tuple(kinds or ())
    unknown = sorted(str(k) for k in set(names) - set(A6_READ_EXCEPTIONS))
    if not names or unknown:
        raise PlanStatsError(f"unknown or no amendment 6 read exception {unknown or list(names)} "
                             f"(known: {sorted(A6_READ_EXCEPTIONS)})")
    _A6_OPEN.update(names)
    exceptions = {}
    for k in sorted(set(names)):
        tok, prefixes = A6_READ_EXCEPTIONS[k]
        exceptions[k] = {"token": tok, "prefixes": list(prefixes)}
        held = {p: A6_WAIVER_AFTER_LIFT[p] for p in prefixes if p in A6_WAIVER_AFTER_LIFT}
        if held:
            exceptions[k]["after_lift"] = held
    _BLIND_LOG.append({"opened_reads": sorted(set(names)), "by": A6_READER, "exceptions": exceptions,
                       "never_waived": list(A6_NEVER_WAIVED), "private_parts": list(A6_PRIVATE_DIRNAMES),
                       "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return sorted(_A6_OPEN)


# an LC bound of amendment 6, the only file read_amend6_lc_bound() opens
A6_LC_BOUND = re.compile(r"/outputs/v2_1/amend6/look/cohort/lookahead_physics_ct_s[12]_n12\.json$")


def read_amend6_lc_bound(caller, path):
    """Read the provenance of one LC bound for amend6.py before any lift, as amend6_checks.py does; only amend6.py
    may call it. A6_LC_BOUND must match the path as given and with its links resolved, and no part of it may be a
    private folder. Returns obs_seed and target_seed
    of provenance.v21.lookahead, the md5 of provenance.v21_cohort.cohort_manifest and a top-level calib_seed; all
    other keys are dropped. The blind log gets one entry with the path, its sha256 and the returned field names.
    It lifts nothing."""
    _a6_reader_only(caller, "read the provenance of an LC bound")
    forms = {_slashes(os.path.abspath(str(path))).lower(), _slashes(os.path.realpath(str(path))).lower()}
    if not all(A6_LC_BOUND.search(s) for s in forms) or \
            any(d in s.split("/") for s in forms for d in A6_PRIVATE_DIRNAMES):
        raise ForbiddenPath(f"refusing to read {path}: not an LC bound of amendment 6 (as written and resolved, "
                            f"{A6_LC_BOUND.pattern})")
    with open(path, "rb") as fh:
        raw = fh.read()
    d = json.loads(raw.decode("utf-8"))
    pv = d.get("provenance") or {}
    lk = (pv.get("v21") or {}).get("lookahead") or {}
    out = {"obs_seed": lk.get("obs_seed"), "target_seed": lk.get("target_seed"),
           "cohort_manifest_md5": ((pv.get("v21_cohort") or {}).get("cohort_manifest") or {}).get("md5")}
    if "calib_seed" in d:
        out["calib_seed"] = d["calib_seed"]
    del d, pv, lk
    _BLIND_LOG.append({"read": rel(path), "sha256": hashlib.sha256(raw).hexdigest(), "by": A6_READER,
                       "kept": sorted(out), "reading": _PI.text("a6 lc reading"),
                       "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return out


def blind_state():
    """What this process lifted or opened (a copy, for the provenance of a result). 'a6_reads' (the amendment 6 read
    exceptions opened) appears only when open_amend6_reads() opened one, so the block of an earlier reader keeps its
    keys."""
    st = {"lifted": {g: sorted(s) for g, s in _LIFTED.items()}, "amos_open": sorted(_AMOS_OPEN),
          "log": copy.deepcopy(_BLIND_LOG)}
    if _A6_OPEN:
        st["a6_reads"] = sorted(_A6_OPEN)
    return st


def _reset_blinding():
    """Close every lifted set, AMOS gate and amendment 6 read exception of this process (tests only)."""
    for s in _LIFTED.values():
        s.clear()
    _AMOS_OPEN.clear()
    _A6_OPEN.clear()
    _BLIND_LOG.clear()


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping) and v:
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _lookup(consts, dotted):
    cur = consts
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        return float(a) == float(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    return a == b


def assert_constants(used, constants=None):
    """Raise ConstantsMismatch unless every constant a script uses equals Appendix 2.

    `used` is a nested dict that mirrors Appendix 2 (for example {"alpha": 0.05, "families": {"Q8-P": 1}}) or a
    flat dict with dotted keys ({"q8.delta_S": 0.01}); both forms may be mixed. A key absent from Appendix 2 is a
    mismatch. Returns a provenance record of what was asserted."""
    C = load_constants() if constants is None else constants
    flat = {}
    for k, v in used.items():
        if isinstance(v, Mapping) and v:
            flat.update(_flatten(v, str(k)))
        else:
            flat[str(k)] = v
    bad = []
    for key, val in sorted(flat.items()):
        try:
            ref = _lookup(C, key)
        except KeyError:
            bad.append(f"{key}: not in Appendix 2 (script uses {val!r})")
            continue
        if not _same(val, ref):
            bad.append(f"{key}: script uses {val!r}, Appendix 2 says {ref!r}")
    if bad:
        raise ConstantsMismatch("constants differ from Appendix 2:\n  " + "\n  ".join(bad))
    return {"n_asserted": len(flat), "asserted": {k: _jsonable(v) for k, v in sorted(flat.items())}}


def family_size(name, constants=None):
    C = load_constants() if constants is None else constants
    if name not in C["families"]:
        raise PlanStatsError(f"family {name!r} is not in Appendix 2 (families: {sorted(C['families'])})")
    return int(C["families"][name])


# ===================================================================================== input checks
def _vector(d, name="d", allow_nan=False):
    x = np.asarray(d, dtype=float).ravel()
    if not allow_nan and not np.all(np.isfinite(x)):
        raise IncompleteData(f"{name} contains non-finite values (a missing patient or cell); the missing-data rule applies")
    return x


def _alt(alternative):
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', not {alternative!r}")
    return alternative


def _direction_of(alternative):
    return {"greater": 1, "less": -1}.get(alternative, 0)


# ===================================================================================== paired t
def paired_t(d, alternative="two-sided", level=0.95):
    """Paired t on per-unit differences d (df n - 1).

    Returns the mean, SD, SE, t, df, dz, the p of `alternative`, the two-sided p, the two-sided `level` t CI
    (the reported interval) and the one-sided `level` bounds (lower = mean - t_{level, df} SE, upper likewise),
    which Q8 uses. If every difference equals the mean (SD 0), t is +/-inf with p 0 in the direction of the mean
    and 1 against it; if every difference is 0, t is NaN and every p is 1."""
    from scipy import stats as st
    alternative = _alt(alternative)
    x = _vector(d)
    n = len(x)
    if n < 2:
        raise IncompleteData(f"paired t needs n >= 2 (n = {n})")
    df = n - 1
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    t2 = float(st.t.ppf(0.5 + level / 2.0, df))
    t1 = float(st.t.ppf(level, df))
    if se > 0:
        t = m / se
        p_two = float(2.0 * st.t.sf(abs(t), df))
        p_gt = float(st.t.sf(t, df))
        p_lt = float(st.t.cdf(t, df))
    elif m != 0:
        t = math.copysign(math.inf, m)
        p_two = 0.0
        p_gt, p_lt = (0.0, 1.0) if m > 0 else (1.0, 0.0)
    else:
        t, p_two, p_gt, p_lt = math.nan, 1.0, 1.0, 1.0
    p = {"two-sided": p_two, "greater": p_gt, "less": p_lt}[alternative]
    return {"n": n, "mean": m, "sd": sd, "se": se, "t": float(t), "df": df,
            "dz": m / sd if sd > 0 else math.nan, "alternative": alternative, "p": p, "p_two_sided": p_two,
            "level": level, "ci_t": [m - t2 * se, m + t2 * se],
            "lower_one_sided": m - t1 * se, "upper_one_sided": m + t1 * se}


def t_critical(df, alpha=0.05, sides=2):
    """Critical |t| at df for level alpha (for example df 5, alpha 0.05 / m)."""
    from scipy import stats as st
    return float(st.t.ppf(1.0 - alpha / (2.0 if sides == 2 else 1.0), df))


# ===================================================================================== sign count
def sign_count(d, direction=None):
    """k/n: the number of units whose difference has the sign of the mean (or of `direction`, +1 or -1, for a
    one-sided test). A zero difference never counts; it stays in n."""
    x = _vector(d)
    if direction is None:
        direction = int(np.sign(x.mean()))
    if direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0, +1 or None")
    k = int((np.sign(x) == direction).sum()) if direction != 0 else 0
    return {"k": k, "n": len(x), "direction": int(direction)}


def sign_rule(d, direction=None, constants=None):
    """The n = 6 sign clause: at least min_sign_agree (5) of 6 units share the sign of the mean (or `direction`).
    Defined only at n = 6; raises at any other n."""
    C = load_constants() if constants is None else constants
    sc = sign_count(d, direction)
    if sc["n"] != 6:
        raise PlanStatsError(f"the sign rule is defined only at n = 6 (n = {sc['n']})")
    return bool(sc["direction"] != 0 and sc["k"] >= int(C["n6_decision"]["min_sign_agree"]))


def sign_binom_p(d):
    """Exact two-sided binomial p of the sign count (descriptive at n other than 6); zeros are
    excluded from the binomial n, as posthoc_v2.sign_test_p (reused)."""
    return float(_ph.sign_test_p(_vector(d)))


# ===================================================================================== exact sign-flip
def _subset_sums(v):
    s = np.zeros(1)
    for y in v:
        s = np.concatenate([s + y, s - y])
    return s


def _count_extreme(means, obs, alternative, tol=SIGNFLIP_TOL):
    if alternative == "two-sided":
        return int(np.count_nonzero(np.abs(means) >= abs(obs) - tol))
    if alternative == "greater":
        return int(np.count_nonzero(means >= obs - tol))
    return int(np.count_nonzero(means <= obs + tol))


def _signflip_count_exact(x, alternative):
    """Count of the 2^n sign vectors s with mean(s * x) at least as extreme as mean(x). Split enumeration: the
    2^a sums of the first half are combined with the 2^b sums of the second half block by block, so memory
    stays small (2^22 vectors use about 35 MB at most)."""
    n = len(x)
    obs = float(np.mean(x))
    a = n // 2
    lo, hi = _subset_sums(x[:a]), _subset_sums(x[a:])
    cnt = 0
    step = max(1, (1 << 20) // max(1, len(hi)))
    for i in range(0, len(lo), step):
        means = (lo[i:i + step, None] + hi[None, :]).ravel() / n
        cnt += _count_extreme(means, obs, alternative)
    return cnt, 1 << n


def _signflip_count_mc(x, alternative, draws, seed, chunk=MC_CHUNK):
    """Monte Carlo count over `draws` random sign vectors of default_rng(seed). A sign is -1 when its draw of
    rng.random() is below 0.5 and +1 otherwise, row after row; every draw uses one 64-bit output, so the count does not
    depend on the block size."""
    n = len(x)
    obs = float(np.mean(x))
    rng = np.random.default_rng(seed)
    cnt, done = 0, 0
    while done < draws:
        b = min(chunk, draws - done)
        S = np.where(rng.random((b, n)) < 0.5, -1.0, 1.0)
        cnt += _count_extreme((S * x).sum(axis=1) / n, obs, alternative)   # no BLAS: order-stable sums
        done += b
    return cnt


def signflip_p(d, alternative="two-sided", constants=None):
    """Exact sign-flip test of the mean of paired differences.

    The statistic is |mean(d)| (two-sided) or mean(d) (one-sided; the cohort TOST uses it). p is the share of sign
    vectors s with |mean(s * d)| >= |mean(d)| - SIGNFLIP_TOL; zero differences count in n. Up to enumerate_max_n
    units all 2^n vectors are enumerated; for more units, mc_draws random vectors of default_rng(mc_seed) give
    p = (1 + count) / (1 + mc_draws)."""
    C = load_constants() if constants is None else constants
    cd = C["cohort_decision"]
    alternative = _alt(alternative)
    x = _vector(d)
    n = len(x)
    if n == 0:
        raise IncompleteData("sign-flip test of an empty vector")
    if n <= int(cd["enumerate_max_n"]):
        cnt, total = _signflip_count_exact(x, alternative)
        return {"p": cnt / total, "method": "exact enumeration", "n": n, "alternative": alternative,
                "count": cnt, "n_vectors": total}
    draws, seed = int(cd["mc_draws"]), int(cd["mc_seed"])
    cnt = _signflip_count_mc(x, alternative, draws, seed)
    return {"p": (1 + cnt) / (1 + draws), "method": "monte carlo", "n": n, "alternative": alternative,
            "count": cnt, "n_vectors": draws, "mc_seed": seed}


def min_signflip_p(n):
    """Smallest attainable two-sided exact sign-flip p at n: 2 / 2^n."""
    return 2.0 / (2 ** int(n))


def signflip_method_label(res, constants=None):
    """The printed method of a signflip_p or tost_signflip result, read from its 'method' field only (amendment 4):
    'exact sign-flip p', or 'Monte Carlo sign-flip p (1,000,000 sign vectors, seed 0)' with the draw
    count and seed of the result (or of Appendix 2 cohort_decision when the result does not carry them)."""
    method = res.get("method")
    if method == "exact enumeration":
        return "exact sign-flip p"
    if method == "monte carlo":
        cd = (load_constants() if constants is None else constants)["cohort_decision"]
        draws = int(res.get("n_vectors") or cd["mc_draws"])
        seed = int(res.get("mc_seed", cd["mc_seed"]))
        return f"Monte Carlo sign-flip p ({draws:,} sign vectors, seed {seed})"
    raise PlanStatsError(f"unknown sign-flip method {method!r}")


# ===================================================================================== Holm
def holm(pvals):
    """Holm step-down adjusted p (same order as the input); posthoc_v2.holm, reused."""
    return _ph.holm([float(p) for p in pvals])


def holm_family(family, pvals, members=None, constants=None):
    """Holm within a named family whose size m comes from Appendix 2.

    pvals: {member: p or None}. members: the full ordered member list (defaults to the keys of pvals). A member
    that is listed but absent from pvals, or whose p is None or NaN (not run, incomplete), enters at p = 1
    (the missing-data rule). The member count must equal the Appendix 2 size m; a family is never resized."""
    m = family_size(family, constants)
    names = list(members) if members is not None else list(pvals)
    extra = [k for k in pvals if k not in names]
    if extra:
        raise PlanStatsError(f"family {family}: p given for members not in the member list: {extra}")
    if len(names) != m or len(set(names)) != m:
        raise PlanStatsError(f"family {family} has m = {m} in Appendix 2, but {len(names)} member(s) were given")
    raw, missing = [], []
    for k in names:
        p = pvals.get(k)
        if p is None or not np.isfinite(p):
            missing.append(k)
            raw.append(1.0)
        else:
            if not 0.0 <= float(p) <= 1.0:
                raise ValueError(f"family {family}, member {k}: p = {p} is not in [0, 1]")
            raw.append(float(p))
    adj = holm(raw)
    return {"family": family, "m": m, "members": names, "p_raw": dict(zip(names, raw)),
            "p_holm": dict(zip(names, adj)), "missing_at_p1": missing, "complete": not missing}


# ===================================================================================== TOST and IUT
def tost_t(d, margin, alpha=None, constants=None):
    """Equivalence at n = 6 with paired t statistics (n - 1 degrees of freedom) against -margin and +margin.

    p_lower is the p value of H0 mean <= -margin, p_upper that of H0 mean >= +margin, and p_tost their maximum.
    Without Holm, equivalence means that the 90% t CI is within +/- margin, that is p_tost < 0.05; with Holm, the
    caller adjusts p_tost within the family."""
    from scipy import stats as st
    C = load_constants() if constants is None else constants
    a = float(C["alpha"]) if alpha is None else float(alpha)
    x = _vector(d)
    n = len(x)
    if n < 2:
        raise IncompleteData(f"TOST needs n >= 2 (n = {n})")
    df = n - 1
    m = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(n)
    if se > 0:
        p_lo = float(st.t.sf((m + margin) / se, df))
        p_hi = float(st.t.cdf((m - margin) / se, df))
    else:
        p_lo = 0.0 if m > -margin else 1.0
        p_hi = 0.0 if m < margin else 1.0
    tq = float(st.t.ppf(1.0 - a, df))
    ci = [m - tq * se, m + tq * se]
    p = max(p_lo, p_hi)
    return {"n": n, "mean": m, "margin": float(margin), "test": "TOST paired t", "p_lower": p_lo, "p_upper": p_hi,
            "p_tost": p, f"ci{int(round(100 * (1 - 2 * a)))}_t": ci, "ci_tost_t": ci,
            "equivalent_unadjusted": bool(p < a), "ci_inside_margin": bool(-margin < ci[0] and ci[1] < margin)}


def tost_signflip(d, margin, constants=None):
    """Equivalence on the cohort with exact sign-flip p values: d + margin against H0 mean <= -margin (alternative
    greater) and d - margin against H0 mean >= +margin (alternative less); p_tost is the maximum of the two."""
    x = _vector(d)
    lo = signflip_p(x + margin, "greater", constants)
    hi = signflip_p(x - margin, "less", constants)
    return {"n": len(x), "mean": float(x.mean()), "margin": float(margin), "test": "TOST exact sign-flip",
            "p_lower": lo["p"], "p_upper": hi["p"], "p_tost": max(lo["p"], hi["p"]), "method": lo["method"]}


def intersection_union(pvals, expected=None):
    """Intersection-union test of a claim that must hold in every arm: p = max of the per-arm p, with no further
    adjustment across arms. A listed arm that is missing (absent, None or NaN) makes p = 1 (the claim cannot be
    shown) and is reported."""
    names = list(expected) if expected is not None else list(pvals)
    missing = [k for k in names if pvals.get(k) is None or not np.isfinite(pvals.get(k))]
    vals = [float(pvals[k]) for k in names if k not in missing]
    p = 1.0 if missing or not vals else max(vals)
    arg = None if missing or not vals else names[int(np.argmax([pvals[k] for k in names]))]
    return {"p": p, "arms": names, "max_arm": arg, "missing": missing, "complete": not missing}


# ===================================================================================== bootstraps
def _q(level):
    return (50.0 - 50.0 * level, 50.0 + 50.0 * level)


def _boot(C):
    return int(C["bootstrap"]["resamples"]), int(C["bootstrap"]["seed"])


def patient_bootstrap_ci(d, level=0.95, constants=None):
    """Patient-cluster percentile bootstrap CI of the mean, with the resamples and seed of the constants: the
    computation of collect_results_v2.boot_ci (posthoc_v2.boot_ci, reused)."""
    C = load_constants() if constants is None else constants
    nboot, seed = _boot(C)
    x = _vector(d)
    _, ci = _ph.boot_ci(x, nboot=nboot, seed=seed, q=_q(level))
    return [float(ci[0]), float(ci[1])]


def patient_seed_bootstrap_ci(D, level=0.95, constants=None):
    """Patient-by-seed bootstrap of the grand mean of D[patient, seed], where D holds the paired
    difference arm A seed s minus arm B seed s. Each resample draws patients with replacement, then, within each
    sampled patient, seeds with replacement (the seed index stays paired across arms because D is already a
    difference). 10,000 resamples, default_rng(0); the patient indices of all resamples are drawn first, then
    the seed indices."""
    C = load_constants() if constants is None else constants
    nboot, seed = _boot(C)
    A = np.asarray(D, dtype=float)
    if A.ndim != 2:
        raise ValueError("D must be a 2-D array [patient, seed]")
    if not np.all(np.isfinite(A)):
        raise IncompleteData("D has missing (patient, seed) cells; the missing-data rule applies")
    P, S = A.shape
    rng = np.random.default_rng(seed)
    pidx = rng.integers(0, P, size=(nboot, P))
    sidx = rng.integers(0, S, size=(nboot, P, S))
    stat = A[pidx[:, :, None], sidx].mean(axis=(1, 2))
    lo, hi = _q(level)
    return [float(np.percentile(stat, lo)), float(np.percentile(stat, hi))]


def crossed_bootstrap_ci(D, level=0.95, constants=None):
    """Two- or three-way pigeonhole (crossed) bootstrap of the grand mean of D (Owen 2007), every axis resampled
    on its own; NaN cells carry no weight (posthoc_v2.crossed_boot, reused). Q12: subjects x CTs (seeds
    averaged within the cell) as primary, subjects x CTs x seeds beside it. Returns {ci, n_dropped}."""
    C = load_constants() if constants is None else constants
    nboot, seed = _boot(C)
    ci, dropped = _ph.crossed_boot(np.asarray(D, dtype=float), nboot=nboot, seed=seed, q=_q(level))
    return {"ci": [float(ci[0]), float(ci[1])], "n_dropped": int(dropped), "resamples": nboot, "seed": seed}


def ratio_bootstrap_ci(num, den, level=0.95, constants=None):
    """Percentile patient bootstrap CI of a ratio of means, mean(num) / mean(den), resampling patients jointly
    (Q10 fraction f, Q13 recovered fraction): 10,000 resamples, default_rng(0), the index draw of boot_ci.
    Resamples with mean(den) = 0 are dropped and counted."""
    C = load_constants() if constants is None else constants
    nboot, seed = _boot(C)
    a, b = _vector(num, "num"), _vector(den, "den")
    if len(a) != len(b):
        raise ValueError("num and den must have the same length")
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    dm = b[idx].mean(1)
    ok = dm != 0
    r = a[idx].mean(1)[ok] / dm[ok]
    lo, hi = _q(level)
    est = float(a.mean() / b.mean()) if b.mean() != 0 else math.nan
    return {"estimate": est, "ci": [float(np.percentile(r, lo)), float(np.percentile(r, hi))],
            "n_dropped": int((~ok).sum()), "resamples": nboot, "seed": seed}


# ===================================================================================== power and MDD
def power_paired_t(dz, n, alpha=0.05, sides=2):
    """Power of the paired t-test (df n - 1) at true standardized effect dz, from the noncentral t."""
    from scipy import stats as st
    df = int(n) - 1
    ncp = float(dz) * math.sqrt(n)
    if sides == 2:
        tc = st.t.ppf(1.0 - alpha / 2.0, df)
        pw = st.nct.sf(tc, df, ncp) + st.nct.cdf(-tc, df, ncp)
    else:
        tc = st.t.ppf(1.0 - alpha, df)
        pw = st.nct.sf(tc, df, ncp)
    pw = float(pw)
    return 1.0 if not np.isfinite(pw) else min(1.0, pw)


def mdd_paired_t(sd, n, alpha=0.05, power=0.8, sides=2):
    """Minimum detectable difference of the paired t-test (amendment 1 uses the defaults): the mean difference whose
    power is `power`, solved on the noncentral t distribution. Returns the MDD in the units of sd and the standardized
    dz."""
    from scipy.optimize import brentq
    if n < 2:
        raise ValueError("MDD needs n >= 2")
    f = lambda z: power_paired_t(z, n, alpha, sides) - power  # noqa: E731
    hi = 1.0
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e4:
            raise PlanStatsError("MDD search did not bracket the target power")
    dz = float(brentq(f, 1e-9, hi, xtol=1e-12))
    return {"mdd": dz * float(sd), "dz": dz, "sd": float(sd), "n": int(n), "alpha": alpha, "power": power,
            "sides": sides}


def tost_power(margin, sd, n, delta=0.0, alpha=0.05):
    """Power of the paired-t TOST at true difference delta, by integration over the sample SD."""
    from scipy import integrate
    from scipy import stats as st
    df = int(n) - 1
    c = st.t.ppf(1.0 - alpha, df) / math.sqrt(n)
    sm = sd / math.sqrt(n)

    def f(s):
        hi, lo = margin - c * s, -margin + c * s
        if hi <= lo:
            return 0.0
        dens = st.chi.pdf(s * math.sqrt(df) / sd, df) * math.sqrt(df) / sd
        return dens * (st.norm.cdf((hi - delta) / sm) - st.norm.cdf((lo - delta) / sm))

    return float(integrate.quad(f, 0.0, margin / c, limit=200)[0])


# ===================================================================================== contrasts and decisions
def contrast(d, alternative="two-sided", design="n6", D_ps=None, constants=None, units=None):
    """Every item of one paired contrast, except the Holm p (which needs the family).

    d: per-unit differences A_p - B_p (patient means, seeds averaged where the cell has seeds).
    alternative: the decision test's alternative ('two-sided', or 'greater' / 'less' where the question states
    the direction). design: 'n6' (decision test paired t) or 'cohort' / 'subject' (decision test exact
    sign-flip). D_ps: optional [unit, seed] difference array for the patient-by-seed bootstrap; its row means
    must equal d. units: optional unit names, kept with the per-unit values."""
    C = load_constants() if constants is None else constants
    if design not in ("n6", "cohort", "subject"):
        raise ValueError("design must be 'n6', 'cohort' or 'subject'")
    alternative = _alt(alternative)
    x = _vector(d)
    t = paired_t(x, alternative)
    sf2 = signflip_p(x, "two-sided", C)
    direction = _direction_of(alternative) or None
    sc = sign_count(x, direction)
    out = {"n": t["n"], "mean": t["mean"], "ci_t": t["ci_t"], "sd": t["sd"], "se": t["se"], "dz": t["dz"],
           "t": t["t"], "df": t["df"], "alternative": alternative, "p_t": t["p"], "p_t_two_sided": t["p_two_sided"],
           "lower_one_sided_95": t["lower_one_sided"], "upper_one_sided_95": t["upper_one_sided"],
           "sign_k": sc["k"], "sign_n": sc["n"], "sign_direction": sc["direction"],
           "p_signflip_two_sided": sf2["p"], "signflip_method": sf2["method"],
           "ci_boot_patient": patient_bootstrap_ci(x, constants=C), "design": design}
    if design == "n6":
        out["decision_test"] = "paired t (df n - 1)" + ("" if alternative == "two-sided" else f", one-sided {alternative}")
        out["p_decision"] = t["p"]
    else:
        sfa = sf2 if alternative == "two-sided" else signflip_p(x, alternative, C)
        out["decision_test"] = "exact sign-flip" + ("" if alternative == "two-sided" else f", one-sided {alternative}")
        out["p_decision"] = sfa["p"]
        out["p_sign_binomial"] = sign_binom_p(x)
    if D_ps is not None:
        A = np.asarray(D_ps, dtype=float)
        if A.shape[0] != len(x) or not np.allclose(A.mean(axis=1), x, rtol=0, atol=1e-9):
            raise ValueError("D_ps row means must equal d (seed-averaged per-unit differences)")
        out["ci_boot_patient_seed"] = patient_seed_bootstrap_ci(A, constants=C)
        out["per_seed_mean"] = [float(v) for v in A.mean(axis=0)]
    out["per_unit"] = ({str(u): float(v) for u, v in zip(units, x)} if units is not None else [float(v) for v in x])
    return out


def decide_n6(c, holm_p, stated_sign=None, constants=None):
    """The decision at n = 6. A contrast holds if Holm p < alpha and min_sign_agree or
    more of the 6 units share the sign of the mean (two-sided), or lie in the stated direction (one-sided). Returns the
    sentence key: 'holds' (holds in the stated direction, or holds when no direction is stated), 'reversed' (holds
    opposite to stated_sign; two-sided only), 'no_difference', or 'incomplete' (n is not 6)."""
    C = load_constants() if constants is None else constants
    alpha = float(C["alpha"])
    need = int(C["n6_decision"]["min_sign_agree"])
    if c["n"] != 6:
        return {"key": "incomplete", "holds": False, "reason": f"n = {c['n']}, the n = 6 rule does not apply"}
    alt = c["alternative"]
    if alt == "two-sided":
        direction = int(np.sign(c["mean"]))
    else:
        direction = _direction_of(alt)
        if stated_sign is not None and stated_sign != direction:
            raise PlanStatsError("a one-sided test's stated direction must equal its alternative")
    k = c["sign_k"] if c["sign_direction"] == direction else None
    if k is None:
        raise PlanStatsError("sign count was taken in a different direction than the decision")
    holds = bool(holm_p < alpha and direction != 0 and k >= need)
    if not holds:
        key = "no_difference"
    elif stated_sign is None or direction == stated_sign:
        key = "holds"
    else:
        key = "reversed"
    return {"key": key, "holds": holds, "direction": direction, "sign_k": k, "holm_p": float(holm_p),
            "rule": f"Holm p < {alpha} and >= {need}/6 sharing the sign"}


def decide_cohort(c, holm_p, original_sign=None, constants=None):
    """The cohort and Q2R decision. Holds when the Holm-adjusted exact sign-flip p < alpha; the sign count is
    never part of it. With n_c < min_n_decision (15) the result is estimation-only (no Holm decision). Key
    'holds' (same sign as the original six), 'reversed', 'no_difference' or 'estimation_only'; Q2R_WORDS maps
    the first three to replicates / reverses / not replicated."""
    C = load_constants() if constants is None else constants
    cd = C["cohort_decision"]
    if c["n"] < int(cd["min_n_decision"]):
        return {"key": "estimation_only", "holds": False,
                "reason": f"n_c = {c['n']} < {cd['min_n_decision']}: estimation only, no Holm decision"}
    holds = bool(holm_p < float(C["alpha"]))
    direction = int(np.sign(c["mean"]))
    if not holds:
        key = "no_difference"
    elif original_sign is None or direction == original_sign:
        key = "holds"
    else:
        key = "reversed"
    return {"key": key, "holds": holds, "direction": direction, "holm_p": float(holm_p),
            "rule": f"Holm-adjusted exact sign-flip p < {C['alpha']}"}


def run_family(family, members, design="n6", alternative="two-sided", stated_signs=None, D_ps=None,
               units=None, constants=None):
    """Compute every member contrast of one Holm family, adjust, and decide.

    members: ordered {member: d, or None if not run, or a dict with a precomputed 'p_decision' (for example the
    Q2R C3 intersection-union TOST p)}; the count must equal the Appendix 2 size m. A member that did not run,
    or whose n differs from 6 under design 'n6', enters Holm at p = 1 and gets key 'not_run' or 'incomplete'.
    alternative / stated_signs / D_ps / units: one value for every member, or {member: value}."""
    C = load_constants() if constants is None else constants

    def per(v, k):
        return v.get(k) if isinstance(v, Mapping) else v

    res, pv = {}, {}
    for name, d in members.items():
        if d is None:
            res[name] = {"status": NOT_RUN, "key": "not_run"}
            pv[name] = None
            continue
        if isinstance(d, Mapping):
            res[name] = dict(d, status="precomputed")
            pv[name] = d["p_decision"]
            continue
        x = _vector(d, allow_nan=True)
        if not np.all(np.isfinite(x)) or (design == "n6" and len(x) != 6):
            res[name] = {"status": "incomplete", "key": "incomplete", "n_finite": int(np.isfinite(x).sum())}
            pv[name] = None
            continue
        c = contrast(x, per(alternative, name) or "two-sided", design, per(D_ps, name), C, per(units, name))
        c["status"] = "computed"
        res[name] = c
        pv[name] = c["p_decision"]
    H = holm_family(family, pv, list(members), C)
    for name, r in res.items():
        r["p_holm"] = H["p_holm"][name]
        if r["status"] == "computed":
            dec = (decide_n6(r, r["p_holm"], per(stated_signs, name), C) if design == "n6"
                   else decide_cohort(r, r["p_holm"], per(stated_signs, name), C))
            r.update({"key": dec["key"], "holds": dec["holds"], "decision": dec})
        elif r["status"] == "precomputed":
            if design != "n6" and r.get("n") is not None and r["n"] < int(C["cohort_decision"]["min_n_decision"]):
                r.update({"key": "estimation_only", "holds": False})
            else:
                r.update({"key": "holds" if r["p_holm"] < float(C["alpha"]) else "no_difference",
                          "holds": bool(r["p_holm"] < float(C["alpha"]))})
        r["family"] = family
    return {"family": family, "m": H["m"], "design": design, "members": res, "holm": H,
            "complete": H["complete"], "missing": H["missing_at_p1"]}


def sentence_key(holds, direction, stated_sign):
    """The sentence key from the test outcome and the sign of the estimate."""
    if not holds:
        return "no_difference"
    return "holds" if stated_sign is None or direction == stated_sign else "reversed"


# ===================================================================================== formatting
def _finite(x):
    return x is not None and isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)


def fmt_num(x, d=2, signed=True):
    """A number with d decimals (2 for mm, as the collector), with an explicit sign for differences."""
    if not _finite(x):
        return "n/a"
    s = f"{float(x):+.{d}f}" if signed else f"{float(x):.{d}f}"
    if float(s) == 0.0:
        s = ("+" if signed else "") + f"{0.0:.{d}f}"
    return s


def fmt_ci(ci, d=2, signed=True):
    return f"[{fmt_num(ci[0], d, signed)}, {fmt_num(ci[1], d, signed)}]"


def fmt_p(p, digits=3):
    """p with `digits` decimals, rounded half up (p >= 10^-digits), else two leading digits in scientific
    form (5.0e-05), so a small p never prints as 0.000."""
    if not _finite(p):
        return "n/a"
    p = float(p)
    if p >= 10.0 ** (-digits):
        return str(Decimal(repr(p)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
    return f"{p:.1e}"


def fmt_k(k, n):
    return f"{int(k)}/{int(n)}"


def fmt_pct(frac, d=0):
    return "n/a" if not _finite(frac) else f"{100.0 * float(frac):.{d}f}%"


def _x_ci(c, unit, d, ci_key="ci_t"):
    u = f" {unit}" if unit else ""
    return f"{fmt_num(c['mean'], d)}{u} {fmt_ci(c[ci_key], d)}"


def label_n6(c, holm_p, family, digits=3):
    """The label of an n = 6 contrast after its CI:
    the sign count over n, the paired t p of the decision test (one-sided where the question states the
    direction) with its Holm p and family, and the two-sided exact sign-flip p, then the status words."""
    return (f"{fmt_k(c['sign_k'], c['n'])}, paired t p = {fmt_p(c['p_t'], digits)} "
            f"(Holm p = {fmt_p(holm_p, digits)}, family {family}); "
            f"exact sign-flip p = {fmt_p(c['p_signflip_two_sided'], digits)}; {PRE_SPECIFIED}")


def string_n6(c, holm_p, family, unit="mm", d=2, digits=3):
    """The string of an n = 6 contrast that holds: the estimate with its 95% t CI, then label_n6()."""
    return f"{_x_ci(c, unit, d)}, {label_n6(c, holm_p, family, digits)}"


def label_cohort(c, holm_p, family, digits=3):
    return (f"{fmt_k(c['sign_k'], c['n'])}, exact sign-flip p = {fmt_p(c['p_decision'], digits)} "
            f"(Holm p = {fmt_p(holm_p, digits)}, family {family})")


def string_cohort(c, holm_p, family, unit="mm", d=2, digits=3):
    """The string of a cohort contrast: the estimate with its CI, then label_cohort()."""
    return f"{_x_ci(c, unit, d)}, {label_cohort(c, holm_p, family, digits)}"


def string_secondary(c, unit="mm", d=2):
    """A secondary contrast: 'X mm [lo, hi] (secondary)', with the 95% t CI; never a 'holds' sentence."""
    return f"{_x_ci(c, unit, d)} ({SECONDARY})"


_PLACEHOLDER = re.compile(r"\[([^\[\]]+)\]")


def fill(template, fields=None, **kw):
    """Fill the bracketed fields of a plan sentence in one pass, e.g. fill(t, {"X": "3.52", "CI": "[...]"}).

    Every bracketed placeholder must have a value and every value must be used, so a misspelt key or a missed
    field raises. Inserted values are not rescanned (a CI's own brackets are safe). The result passes
    check_text()."""
    vals = dict(fields or {}, **kw)
    used = set()

    def rep(m):
        key = m.group(1)
        if key not in vals:
            raise PlanStatsError(f"sentence placeholder [{key}] has no value")
        used.add(key)
        return str(vals[key])

    out = _PLACEHOLDER.sub(rep, template)
    unused = sorted(set(vals) - used)
    if unused:
        raise PlanStatsError(f"values given for placeholders the sentence does not have: {unused}")
    return check_text(out)


def check_text(s):
    """Raise if a printed string contains an em-dash, three hyphens in a row, or a word the results never print
    (the words of BANNED_WORDS)."""
    bad = [b for b in BANNED_TEXT if b in s]
    words = sorted({m.group(0) for m in BANNED_WORDS.finditer(s)})
    if bad or words:
        raise PlanStatsError(f"banned text in printed string: {[repr(b) for b in bad] + words}: {s!r}")
    return s


# ===================================================================================== records of missing elements
def not_run(item, deviation, section_f=None):
    """Record of an element that cannot be computed from existing outputs (no number is invented)."""
    return {"status": NOT_RUN, "item": item, "deviation": deviation, "section_f": section_f}


def not_completed(item, missing, section_f=None):
    if section_f is None:     # the development records name the deviation row; the archive withholds it
        section_f = _PP.NOT_COMPLETED_ROW if _PP else "[item withheld]"
    return {"status": NOT_COMPLETED, "item": item, "missing": list(missing), "section_f": section_f}


# ===================================================================================== provenance and writer
def repo_commit(root=REPO_ROOT):
    """Commit of the main repository read directly from .git (HEAD, then the ref file or packed-refs); git is
    never run, so whether the tree is dirty is not determined (the script and module md5 are recorded). In public
    mode and in the public mirror, a code tree without .git (an archive of the mirror) records commit None."""
    gitdir = os.path.join(root, ".git")
    if _public_plan() and not os.path.exists(gitdir):
        return {"head": None, "ref": None, "commit": None, "read_from": [], "git_run": False,
                "dirty": "not determined (the code tree has no .git)"}
    if os.path.isfile(gitdir):
        with open(gitdir) as fh:
            line = fh.read().strip()
        if not line.startswith("gitdir:"):
            raise PlanStatsError(f"{gitdir}: unexpected content")
        gitdir = os.path.normpath(os.path.join(root, line.split(":", 1)[1].strip()))
    with open(os.path.join(gitdir, "HEAD")) as fh:
        head = fh.read().strip()
    rec = {"head": head, "ref": None, "commit": None, "read_from": [rel(os.path.join(gitdir, "HEAD"))],
           "git_run": False, "dirty": "not determined (git is not run; script and module md5 recorded)"}
    if not head.startswith("ref:"):
        rec["commit"] = head
        return rec
    ref = head.split(":", 1)[1].strip()
    rec["ref"] = ref
    common = gitdir
    cd = os.path.join(gitdir, "commondir")
    if os.path.exists(cd):
        with open(cd) as fh:
            common = os.path.normpath(os.path.join(gitdir, fh.read().strip()))
    for base in dict.fromkeys((gitdir, common)):
        p = os.path.join(base, ref)
        if os.path.isfile(p):
            with open(p) as fh:
                rec["commit"] = fh.read().strip()
            rec["read_from"].append(rel(p))
            return rec
    for base in dict.fromkeys((gitdir, common)):
        pr = os.path.join(base, "packed-refs")
        if os.path.isfile(pr):
            with open(pr) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) == 2 and parts[1] == ref and not line.startswith(("#", "^")):
                        rec["commit"] = parts[0]
                        rec["read_from"].append(rel(pr))
                        return rec
    raise PlanStatsError(f"could not resolve {ref} under {gitdir}")


def _jsonable(o):
    if isinstance(o, Mapping):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    return o


def _versions():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform()}


def write_result(path, payload, *, analysis_status, inputs, script, constants_asserted=None, status_note=None,
                 amendments=(), root=RESULTS_ROOT):
    """Write one result JSON and refuse to overwrite.

    The file must lie under `root` (outputs/v2_1/analysis_plan/) and end in .json. Every output is stamped with
    a provenance block: analysis_status (one of the status strings, passed explicitly), the verified sha256 and
    sha12 of the v2.1 plan (and of any amendments used), the main repository commit read from .git, every input file
    with md5 and size (forbidden inputs are refused), the script path and md5, this module's md5, the constants
    asserted, library versions and the UTC time. NaN is written as null and +/-inf as "inf"/"-inf". Raises
    RefuseOverwrite if the file exists."""
    if analysis_status not in ALLOWED_STATUS:
        raise PlanStatsError(f"analysis_status must be one of {ALLOWED_STATUS}, not {analysis_status!r}")
    if not isinstance(payload, Mapping) or "provenance" in payload:
        raise PlanStatsError("payload must be a dict without a 'provenance' key")
    out = os.path.realpath(os.path.abspath(path))
    if _SD.public():
        # public mode: the result root is the deposit, checked against its MANIFEST.tsv, so nothing is written into
        # it (nor into the code tree); --out names a file elsewhere
        for keep in (_SD.results_root(), REPO_ROOT):
            k = os.path.realpath(keep)
            if out == k or out.startswith(k + os.sep):
                raise PlanStatsError(f"public mode writes outside the result root and the code tree; refusing {out} "
                                     "(pass --out with a path elsewhere)")
    else:
        rroot = os.path.realpath(os.path.abspath(root))
        if not out.startswith(rroot + os.sep):
            raise PlanStatsError(f"results go only under {rroot}; refusing {out}")
    if not out.endswith(".json"):
        raise PlanStatsError("result files are .json")
    if os.path.exists(out):
        raise RefuseOverwrite(f"refusing to overwrite existing result {out}")
    ins = []
    for p in sorted(dict.fromkeys(str(x) for x in inputs)):
        check_input_path(p)
        if not os.path.isfile(p):
            raise PlanStatsError(f"input {p} is not a file")
        ins.append({"path": rel(p), "md5": md5(p), "bytes": os.path.getsize(p)})
    script = os.path.abspath(script)
    prov = {"analysis_status": analysis_status, "status_note": status_note,
            "argv": [str(a) for a in sys.argv], "cwd": os.getcwd(),
            _PI.key("plan_v21"): plan_v21_info(),
            "amendments": {f"amend{n}": amendment_info(n) for n in amendments},
            "repo_commit": repo_commit(),
            "script": {"path": rel(script), "md5": md5(script)},
            "plan_stats": {"path": rel(os.path.abspath(__file__)), "md5": md5(os.path.abspath(__file__))},
            "inputs": ins, "constants_asserted": constants_asserted,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "environment": _versions()}
    if _BLIND_LOG:                                 # a set lifted or an AMOS gate opened in this process
        prov["blinding"] = blind_state()
    doc = _jsonable({"provenance": prov, **payload})
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    bad = [b for b in BANNED_TEXT if b in text]
    if bad:
        raise PlanStatsError(f"refusing to write banned text {[repr(b) for b in bad]} into {out}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out


def read_json(path):
    """Read a JSON input after the forbidden-path check."""
    check_input_path(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _main(argv=None):
    """Self-check: verify the digest of the v2.1 plan, print the sha12, the family sizes and the commit."""
    C = load_constants()
    info = plan_v21_info()
    print(f"v2.1 plan {info['path']} sha256 {info['sha256']} (sha12 {info['sha12']}) "
          f"{'verified' if info['verified'] else 'recorded (public constants)'}")
    if "registered_copy_outputs_v2" in info:
        print(f"registered copy under outputs/v2: {info['registered_copy_outputs_v2']}")
    print("families:", ", ".join(f"{k} m={v}" for k, v in C["families"].items()))
    print("commit:", repo_commit()["commit"])
    return 0


if __name__ == "__main__":
    sys.exit(_main())
