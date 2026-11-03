#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The pre-specified constants and record digests that the table and figure build needs, behind one interface with two
backends.

  public backend (--public, and always in the public mirror, which holds the marker scripts/PUBLIC_MIRROR): reads the committed
      public_inputs/plan_constants.json next to the paper build (found as the single
      scripts/*/public_inputs/plan_constants.json); no plan document, no git and no absolute path;
  development backend (the default): the development repository's plan_private.py, which reads the plan documents of
      that repository, checks their sha256 and reads the records ledger committed at HEAD.

Both backends return the same values for every accessor below (a test of the development repository replays every
call of a full build against both). Checks that need the development repository (its plan documents, the paper's
own source text, the absolute paths that records cite) are skipped in public mode with one notice each on stderr
(skip_dev). Set USBG_RECORD_PLAN_CALLS=<file> to append every accessor call and its result to <file> (JSON lines).
"""
from __future__ import annotations

import functools
import glob
import json
import os
import sys

import script_dirs as SD

JSON_NAME = "plan_constants.json"
# neutral text printed in public mode where the development build prints a source note of its own tree
PUBLIC_TEXT = {
    "lotus defaults src": "the renderer adapters of the testbed (render_lotus_original, default parameters)",
    "identity src": "segmenter/outputs/v2/tablev_v2_{vitale,vitaleho}.json, "
                    "consistency_<stage>_s<seed>_{vitale,vitaleho,kidneyus}.json and outputs/v2/RESULTS.json (table5)",
    "unet score dir": "segmenter/outputs/v2",
    "rule input src": "amendment 5 Appendix selection.rows_read",
    "amend2 file": "amendment 2",
    "amend2 sha file": "the digest of amendment 2",
    "forest figure": "fig:forest (Fig. 4): per-row 95% and 90% paired t CIs against each slot's reference; filled "
                     "marker = post hoc TOST at +/- 1 mm passes",
    "stages twin figure": "fig:stages-twin (supplement)",
    'a6 record file': 'the amendment 6 record',
    'v213 record file': 'the v2.1.3 record',
    'prefix 1': '',
    'prefix 2': '',
    'lotus label src': '',
    "q5 sentence": "\n\nRule (Q5): the keep-best IoU gain holds for an organ when its 95% CI over targets lies above 0.",
    "q3 rule": "\n\nRule (Q3): see the article.\n\n",
    "q6 rule": "Rule (Q6): see the article.",
    "make_tables script": "make_tables.py",
    "section_e script": "section_e.py",
    "best_so_far script": "best_so_far.py",
    "posthoc_logs script": "posthoc_logs.py",
    "plug_rows generated_by": "make_tables.py pluggability()",
    "plug_rows path": "figures/plug_rows.json",
    "horizon source": ("outputs/v2_1/s21_h36/*_n36.json and .traj.npz, contact on, seeds pooled, via posthoc_logs.py "
                       "blocks_with_traj, pick, patient_means and level_mean"),
    "q14 templates": "",
    "best_so_far note": "% ported into best_so_far.py (table)",
}
# >>> moved text of the table build (ids: neutral words)
PUBLIC_TEXT.update({
    'ladder src notes': '',
    'contact src note': '',
    'compute cut note': '',
    'cpu cell tail': '',
    'compute lic a': '',
    'compute lic b': '',
    'posthoc src tail': ', outside the v2 plan',
    'renderers not run': 'Field II~\\cite{jensen1992field}: needs MATLAB. k-Wave~\\cite{treeby2010kwave}: needs CUDA or a long CPU run.',
    'masks not run': 'UltraSam~\\cite{meyer2026ultrasam} and USFM~\\cite{jiao2024usfm}: need an NVIDIA GPU.',
    'tost margin src': 'the +/- 1 mm margin of the pre-specified RL, label-control and CT-anatomy tests',
    'posthoc date': '',
    'ps boot form': '',
    'samus withheld src': 'score withheld; segeval output not opened',
    'cta label volume': '',
    'cta in-sample src': 'amendment 2',
    'cta not run src': 'amendment 2',
    'cta tost test': '',
    'status a4': 'exploratory, v2.1 AMOS cohort',
    'status a5': 'exploratory, v2.1 combination',
    'status a6': 'exploratory, v2.1 estimation only',
    'unanticipated': 'outcome not anticipated by the v2.1 plan',
    'a6 route reading': 'route comparison before the check record exists',
    'a6 lc reading': 'the provenance of an LC bound before any lift',
    'twin arch': '{arch}',
    'painted tost note': 'TOST of this row (Q10)',
    'not run note': '',
    'twin stats src': '',
    'twin caption src': '',
    'dec null': 'no detectable difference, equivalence not tested',
    'not powered': ', equivalence not tested',
    'dec imn null': 'does not detectably change, equivalence not tested',
    'mdd rule': 'post hoc: two-sided paired t, alpha 0.05, power 0.8, n = 6; SD = median over the reference contrasts of the slot; second value at alpha = 0.05/m',
    'dec words src': '',
    't5 src a': '; 60 episodes',
    't5 src b': '; TOST margin of ',
    'ps boot consts': '',
    'ps planned form': 'patient-by-seed and crossed bootstrap',
    'ps caption src': 'amendment 1',
    'q14 band src': 'Q14 row (i)',
    'a2 audit note': '',
    'q14 ii note': '',
    'rerun row': '',
    'rows q8': '',
    'row q13': '',
    'rule src q2r': 'v2.1 plan; Q2R',
    'rule src q2r plain': 'v2.1 plan',
    'rule src q13': 'Q13',
    'rule src q13 plain': 'v2.1 plan',
    'rule src plan plain': 'v2.1 plan',
    'rule src a1 q14 plain': 'Amendment~1; v2.1 plan',
    'rule src rerun': 'v2.1 plan, clean-tag rerun',
    'a2 admitted': '',
    'rl clause key': 'Q2 RL clause at 0.5 mm',
    'bn threshold src': '',
    'h36 src': '36-step minus 12-step servo base',
    'cs caption src': '',
    'bias pilot note': '',
    'earlier label': 'exploratory, no pre-specified rule; earlier run',
    'amos label note': '',
    'q8g1 status': 'exploratory, no pre-specified rule',
    'tag rerun note': '',
    'plane normal note': 'by the reading of the design',
    'plane normal src': '',
    'identity rerun note': '',
    'mm early clause': 'includes runs that began before their decision rules were fixed',
    'mm early log': ' early',
    'a6 printed label': 'exploratory; estimation only; defined after the results',
    'a6 records src': '',
    'est src note': '',
    'registration col src': '',
    'twin real frame': 'no real ultrasound frame in any figure',
    'md t3 head': '| scope | views | stage | Spearman | dz [CI], sign | holds |',
    'md cue head': 'Cue ablation.\n\n',
    'md t2 w1': '\n\nW1 physics below LOTUS: ',
    'md t2 pooled': ' (pooled: ',
    'md t4 head': '| stage | cells | seeds |',
    'md t4 obs': '\nGap to the pose oracle: ',
    'md t4 perc': 'Gap between the masks: ',
    'md t4 pairs': 'Stage pairs: ',
    'md rule yes': 'yes',
    'md rule no': 'no',
    'md t4 learned': 'SonoSPADE against each stage: ',
    'md t4 vs static': '\nExploratory. Against the static start: ',
    'md t4 did': 'Exploratory. Gap between the masks, SonoSPADE against each stage: ',
    'md dr head': '| group | arm | stage | phase | DR | mm | n |',
    'md dr deltas': '\nDR deltas: ',
    'md reg head': '| stage | source | organ | targets | IoU gain [CI] | holds |',
    'md reg intro': 'Registration.\n\n',
    'md drl intro': 'DR landscape.',
    'md compute a': 'MPS bytes of the CPU run: ',
    'md compute b': '.',
    'md compute parts': 'Step components:',
    'md compute run': 'Trainer hours (',
    'a5 label text': 'post hoc: lowest point estimate on the same six held-out patients that measured it, so optimistic; not the pre-specified combination rule',
    'posthoc sentence': 'The pre-specified combination selects by the slot decisions and tests its choice on 20 CTs that played no part in it, so the two can name different pipelines.',
    'bsf status note': 'post hoc: lowest measured error per run-time input class, selected on the six held-out patients that measured it (optimistic); no episode run',
    't5 label': 'exploratory, pre-specified rule; provisional',
    'bsf a3 used for': 'label of the earlier controller rows',
    'bsf run label': 'post hoc; selected on the six held-out patients that measured it, so optimistic',
    'bsf design ci': 'paired t interval with the sign count; patient bootstrap interval as sensitivity',
    'bsf src a': '; label and [posthoc]: ',
    'bsf src b': '; BC residual rows: ',
    'bsf src c': '; random-action anchor: ',
    'bsf src d': '',
    'bsf gen note': '',
    'q2a label': 'Oracle-to-CT-mask gap: CT-mask final minus pose oracle',
    'q2b label': 'CT-to-learned-mask gap: learned-mask final minus CT-mask final',
    'q2c label': 'Change during RL, final minus after imitation',
    'q2d label': 'CT-mask final, stage minus stage',
    'q2e label': 'SonoSPADE minus stage, learned-mask final',
    'ph rule definition': 'the CI-and-sign rule of collect_results_v2.rule (fixed seed-0 resampling matrix)',
    'ph sim head': 'Simulated type-I error at n = 6, normal: ',
    'ph margin source': 'the pre-specified RL threshold (|mean| < 1 mm), chosen before v2 results but applied here as an equivalence margin after the results were known',
    'ph pose metric md': 'Pose metric: final_pos overstates the probe-face displacement by at most ',
    'phl status note': 'post hoc, outside every pre-specified plan: defined and computed from existing logs only (no episode run). Data: the v2.1 extended log of the clean-tag rerun, whose shared episode fields are bit-identical to the archived v2 episodes (archived v2 results stay primary); the v2.1 36-step arms and the completed non-blinded v2.1 pluggability outputs (exploratory). Descriptive; no claim of a difference is made from these numbers.',
    'phl label': 'post hoc, descriptive',
    'phl s0513 note': 'descriptive only; the case study is a separate record',
    'phl head note': '',
    'phl edit note': '',
    'phl edit axes': '',
    'phl edit keepbest': '',
    'phl edit lopo': '',
    'ph mixed model': 'not fitted: statsmodels is not installed and nothing was installed',
    'ph planned key': 'pre_specified',
    'sec bn token': 'BatchNorm in train mode minus the reference',
    'sec reinforce token': 'minus the learned-mask REINFORCE final of the primary results',
    'r1 flag': ': with TotalSegmentator labels the CT-mask base differs between stages',
    'rerun row tail': '',
    'realism date': '',
    'cpu cell tag': 'not printed',
    'cpu cell moved': 'not printed',
    'cuda line tag': '',
    'license moved': 'not printed',
    'wall time tag': 'not printed',
    'rule sim tag': 'not printed',
    'seg sim note': '',
    'cta invariance moved': 'not printed',
    'mdd ref rule': 'reference contrasts (post hoc rule)',
    'drift tag': '',
    'posthoc rule text': 'mdd from reference contrasts, k_seeds, TOST at +/- 1 mm, patient-by-seed and crossed bootstrap CIs (post hoc)',
    'cohort check date': '',
    'cohort note': '',
    'rerun tag': 'not printed',
    'rerun script row': 'committed analysis script',
    'not printed tag': 'not printed',
    'rerun outcome': '{n_same:,} of {n_all:,} episodes identical bit for bit; largest difference per patient {mpp:.2f}~mm',
    'learner sum note': '',
    'phantom tag': 'not printed',
    'rule input caption': "\\caption{Input of amendment~5's rule: each family row of Table~\\ref{tab:pluggability} the rule read, with the decision word it read (from the Table~5 snapshot), beside the word of ",
    'sign check tag': 'sign check',
    'late date': '',
    'phl edit starts': '',
    'phl starts moved': 'not printed',
    'md t4 rungs': '\nReference levels in mm (means over patients): {st:.2f} at the static start, {sc:.2f} for the scripted sweep, {po:.2f} for the pose oracle (privileged).',
    'md compute cost': 'Cost of the runs (`{source}`): {jobs_done} jobs, {jobs_rc0} of them with exit code 0 ({train_jobs} training and {eval_jobs} evaluation jobs), up to {concurrent_jobs} in parallel, from {first_event} to {last_event}: {wall_h:.2f} h of wall time and {job_hours_total:.1f} job-hours. ',
    'md compute median': '{concurrent_jobs} jobs in parallel): {train_h_median_all:.2f} h median, {train_h_min:.2f} to {train_h_max:.2f} h; per arm: ',
    'ph rot head': 'Geodesic rotation error in degrees, means over patients: static start {st:.2f}, pose oracle {po:.2f}, scripted sweep {sc:.2f}; with CT masks, servo base {cb:.2f} and final {c0:.2f} to {c1:.2f}; with learned masks, final {l0:.2f} to {l1:.2f}.',
    'ph succ head': 'Success within {lab} at the final step, patient level, in %: pose oracle {po:.1f}; with CT masks, servo base {cb:.1f} and final {c0:.1f} to {c1:.1f}; with learned masks, servo base {lb0:.1f} to {lb1:.1f} and final {l0:.1f} to {l1:.1f}.',
    'lift tag': 'not printed',
})
# >>> moved text of the table build, second part (ids: neutral words)
PUBLIC_TEXT.update({
    'a6 amos exception': "amendment 6 (amendment 4's exception)",
    'a5 amos exception': 'amendment 5',
    'bsf s14 src': 'outputs/v2_1/plug/ctrl (amendment 3)',
    'bsf s14 label': 'post hoc, descriptive; a selected row against its class reference. A row of the earlier runs is never used to state or deny that two methods separate',
    'a6 leak part': '',
    'src a2 cta': 'amend2',
    'src a2 cta rows': 'amendment 2',
    'dev listed': 'listed among the deviations',
    'src a1 stats': 'amend1',
    'src a2 fam': 'amend2',
    'src a2 lpips': 'amendment 2',
    'src a1 notrun': 'amend1 (not run)',
    'all pairs rule': 'all method pairs (amend1, cohort ordering)',
    'mdd pairs note': 'all method pairs (amend1, cohort ordering)',
    'mdd sizes note': 'family sizes and conditions of amend1 and amend2',
    'src a2 cta parts': 'amend2 (rows R1 to R5; no cohort row)',
    'row a1 gate': 'Admission gate (amendment 1)',
    'row a1 audit': 'Identical-episode audit (amendment 1)',
    'se log name': 'Clean-tag',
    'se job name': 'clean-tag rerun',
    'rule src a1 gate q14': 'Amendment~1; Q14',
    'rule src a1 gate': 'Amendment~1',
    'rule src a1 audit': 'Amendment~1',
    'rule src a1 speckle': 'Amendment~1',
    'rule src a1 cohort': 'Amendment~1',
    'rule src a1 sb3': 'Amendment~1',
    'rule src a1 slots': 'Amendment~1',
    'rule src a3 b': 'Amendment~3',
    'rule src a3 c': 'Amendment~3',
    'rule src a3 checks': 'Amendment~3',
    'rule src a5 checks': 'Amendment~5',
    'rule src a4': 'Amendment~4',
    'src a3 earlier': 'amendment 3',
    'src a3 bands': 'amendment 3',
    'src a4 yield': 'amendment 4',
    'src a4 results': 'amendment 4',
    'src a5 results': 'amendment 5',
    'a5 rule item': 'amendment~5',
    'v213 parts': '',
    'row a3 a': 'Amendment 3, arm A record',
    'row a3 b': 'Amendment 3, arm B records',
    'row a3 c': 'Amendment 3, arm C record',
    'row a4': 'Amendment 4, AMOS records and design checks',
    'row a5': 'Amendment 5, check records',
    'src a6 record': 'amendment 6',
    'src a6 sentences': 'amendment 6',
    'cpu pending cell': 'CPU-only retrain of the CT-mask physics cell not finished: {0} of {1} jobs',
})
# <<< moved text of the table build
_NOTICED = set()


class PlanInputsError(RuntimeError):
    """A constant or digest is missing from the public inputs."""


def json_path() -> str:
    hits = sorted(glob.glob(os.path.join(SD.SCRIPTS, "*", "public_inputs", JSON_NAME)))
    if len(hits) != 1:
        raise PlanInputsError(f"expected exactly one scripts/*/public_inputs/{JSON_NAME}, found {len(hits)}")
    return hits[0]


@functools.lru_cache(maxsize=1)
def _public():
    with open(json_path(), encoding="utf-8") as fh:
        return json.load(fh)


def _pv(key):
    J = _public()
    if key not in J:
        raise PlanInputsError(f"{JSON_NAME} has no entry {key!r}")
    return J[key]["value"]


def _pub() -> bool:
    """The public backend: in public mode, and always in the public mirror (script_dirs.mirror())."""
    return SD.public() or SD.mirror()


def private():
    """The development backend (plan_private.py): the development repository outside public mode only. Its import is
    not guarded, so a missing or broken plan_private.py stops a development run instead of switching backends."""
    if _pub():
        raise PlanInputsError("this step needs the development repository; the public build runs with --public "
                              "--results ROOT --out DIR")
    import plan_private
    return plan_private


def _record(fn, args, result):
    f = os.environ.get("USBG_RECORD_PLAN_CALLS")
    if f:
        name = fn if _pub() else private().CALL_NAMES.get(fn, fn)   # the development build's own accessor names
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"fn": name, "args": list(args), "result": result}, sort_keys=True) + "\n")
    return result


def notice(msg):
    """One line on stderr (public mode), once per message."""
    if msg not in _NOTICED:
        _NOTICED.add(msg)
        print(f"[public] {msg}", file=sys.stderr)


def skip_dev(what) -> bool:
    """True (with one notice) in public mode: the check `what` needs the development repository and is skipped."""
    if SD.public():
        notice(f"skipped, development repository only: {what}")
        return True
    return False


def manifest_instead(what) -> bool:
    """True (with one notice) in public mode: a digest that the development build pins in its code (the records ledger
    lines of result files, a pinned record) names the development file's bytes; the deposit's files differ from those
    bytes (paths and machine names replaced), so the file is checked against its row of the deposit's MANIFEST.tsv
    instead (results_world.listed). The digests that one deposited record holds of another cite the deposited file and
    are checked in both modes."""
    if SD.public():
        notice(f"checked against MANIFEST.tsv instead: {what}")
        return True
    return False


def text(key) -> str:
    """Source text printed into a table comment, a figure data field or a log line, and the archive's form of a
    string the build compares with the result records (not recorded): the public text in public mode and in the
    public mirror, the development build's own text in the development repository."""
    return PUBLIC_TEXT[key] if _pub() else private().text(key)


def key(name) -> str:
    """The name of a result-record key: the results archive's name in public mode and in the public mirror; in the
    development repository the name its own records use (plan_private.KEYS)."""
    return name if _pub() else private().KEYS.get(name, name)


# ================================================================================================= accessors
def amendment_digest(n, plan_dir=None):
    """sha256 of amendment n (1 to 6) as its digest file records it."""
    r = _pv("plan digests")[f"amendment {n}"] if _pub() else private().amendment_digest(n, plan_dir)
    return _record("amendment_digest", (n,), r)


def plan_v21_digest():
    """{sha256, sha12, recorded} of the v2.1 plan."""
    r = dict(_pv("plan digests")["v2.1 plan"]) if _pub() else private().plan_v21_digest()
    return _record("plan_v21_digest", (), r)


def plan_v21_constants():
    """The decision constants of the v2.1 plan (Appendix 2) that the analysis layer reads (alpha, the n = 6 and
    cohort decision rules, the family sizes and the bootstrap); plan_stats.load_constants returns them in public mode."""
    r = json.loads(json.dumps(_pv("v2.1 plan constants"))) if _pub() else private().plan_v21_constants()
    return _record("plan_v21_constants", (), r)


def plan_v21_bootstrap():
    """{resamples, seed} of the bootstrap of the v2.1 plan (Appendix 2)."""
    r = dict(_pv("v2.1 plan constants")["bootstrap"]) if _pub() else private().plan_v21_bootstrap()
    return _record("plan_v21_bootstrap", (), r)


def q14_tolerance(root=None):
    """{band_mm, imit: [mean, seeds], final: [mean, seeds]} of the Q14 tolerance row (i) of the v2.1 plan."""
    r = json.loads(json.dumps(_pv("v2.1 plan Q14 tolerance"))) if _pub() else private().q14_tolerance(root)
    return _record("q14_tolerance", (), r)


def amendment2_appendix(root=None, expect=None):
    """Amendment 2's Appendix block ('schedule' reduced to its sorted keys)."""
    if _pub():
        r = json.loads(json.dumps(_pv("amendment 2 appendix")))
    else:
        r = private().amendment2_appendix(root, expect)
    return _record("amendment2_appendix", (), r)


def amendment_value(n, kp):
    """A constant of amendment n's Appendix block, by key path."""
    if _pub():
        vals = _pv(f"amendment {n}")
        if kp not in vals:
            raise PlanInputsError(f"{JSON_NAME}: amendment {n} has no constant {kp!r}")
        r = vals[kp]
    else:
        r = private().amendment_value(n, kp)
    return _record("amendment_value", (n, kp), r)


def record_digest(rid, root=None):
    """sha256 of a plan-tree record named by its neutral id (None when the development file is missing)."""
    r = _pv("record digests").get(rid) if _pub() else private().record_digest(rid, root)
    return _record("record_digest", (rid,), r)


def ledger_has(rid, sha, head=None):
    """True when the records ledger committed with the analyses lists sha for rid (a result path or a neutral id).
    head: the development build's ledger reader (make_tables._ledger_head), ignored in public mode."""
    r = sha in _pv("ledger").get(rid, []) if _pub() else private().ledger_has(rid, sha, head)
    return _record("ledger_has", (rid, sha), bool(r))


def withheld():
    """Result paths (public layout) that the build checks for existence only and that the deposit withholds."""
    r = list(_pv("withheld")) if _pub() else private().withheld()
    return _record("withheld", (), r)


def printed_versions():
    """{gymnasium, stable-baselines3}: the versions compute.tex prints. Public mode prints the recorded versions and
    gives a notice when the installed ones differ."""
    if _pub():
        r = dict(_pv("printed software versions"))
        from importlib import metadata as _md
        for k, v in r.items():
            try:
                got = _md.version(k)
            except _md.PackageNotFoundError:
                got = None
            if got != v:
                notice(f"installed {k} {got} differs from the recorded {v}; compute.tex prints the recorded version")
    else:
        r = private().printed_versions()
    return _record("printed_versions", (), r)


ACCESSORS = ("amendment_digest", "plan_v21_digest", "plan_v21_constants", "plan_v21_bootstrap", "q14_tolerance",
             "amendment2_appendix", "amendment_value", "record_digest", "ledger_has", "withheld", "printed_versions")


def public_call(fn, args):
    """Replay one recorded call against the public backend (the parity test of the development repository)."""
    old = os.environ.get(SD.PUBLIC_ENV)
    os.environ[SD.PUBLIC_ENV] = "1"
    try:
        f = globals()[fn]
        if fn == "ledger_has":
            return f(args[0], args[1])
        return f(*args)
    finally:
        if old is None:
            os.environ.pop(SD.PUBLIC_ENV, None)
        else:
            os.environ[SD.PUBLIC_ENV] = old

