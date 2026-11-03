"""usbg.leaderboard.stats: the row statistics of the plug-slot leaderboard.

These are the statistics of Table 5 of the USB Gym article, written out so that anyone can recompute a
leaderboard row from its per-episode values:

* Error: the final translation error, mean over seeds of patient means (each patient mean over that
  patient's 10 held-out episodes).
* Difference: from the slot's reference component, paired by seed. For each held-out patient the method's
  patient means are averaged over the common seeds, and so are the reference's; the difference is taken
  per patient (n = 6). Its 95% paired t interval (df 5), the two-sided paired t p, and k/6, the patients
  whose difference shares the sign of the mean.
* Seeds: the seeds whose own difference (mean over patients) shares that sign, k/3 (post hoc; n/a for a
  seed-0 row).
* Holm p: Holm's step-down adjustment of the paired t p within a pre-specified family; a member without a
  complete result enters at p = 1.
* TOST: post hoc and unadjusted; yes when the 90% paired t interval lies inside +-1 mm.
* Decision: the pre-specified outcome words. A family member holds when its Holm p < 0.05 and at least 5
  of 6 patients share the sign; it is then "lower" or "higher", else "no detectable difference".
* Patient-by-seed interval (rows with two or more seeds): the bootstrap of Appendix B of the article, which
  resamples patients, then seeds within each sampled patient, with the seed index paired across arms; 10,000
  resamples, numpy default_rng(0), percentile 95% interval.

A new method is not a member of any pre-specified family, so it gets no Holm p and no pre-specified decision
word: its paired t p is shown unadjusted, beside the paired t CI and the patient-by-seed interval, and the row
is marked "post hoc" ("secondary" for a privileged method, as in Table 5).
"""
from __future__ import annotations

import math

import numpy as np

PATIENTS = ("s0358", "s0513", "s0519", "s0591", "s0649", "s0762")   # usbg.benchmark.TEST_PATIENTS, sorted
ALPHA = 0.05
MIN_SIGN = 5                    # patients of 6 that must share the sign for a family member to hold
TOST_MARGIN_MM = 1.0
DEC_LOWER, DEC_HIGHER, DEC_SEC, DEC_NC = "lower", "higher", "secondary", "not completed"
DEC_NULL = "no detectable difference; not powered for equivalence"
DEC_IMN_NULL = "does not detectably change; not powered for equivalence"
NOT_POWERED = "; not powered for equivalence"
NOTE_A = "Unadjusted CI excludes 0; not declared a difference after Holm (post hoc note)."
SUBMITTED = "submitted, post hoc, outside the pre-specified families"
DEC_POSTHOC = "post hoc"
BOOT_RESAMPLES, BOOT_SEED = 10_000, 0


# ============================================================================================ errors
def patient_means(final_pos, episodes):
    """{patient: mean final error} over the episodes of one seed, in the order the patients first appear.
    final_pos: the 60 per-episode values in episode order; episodes: the 60 episode records (each with a
    'patient'). A None value (an episode that was not run) is skipped."""
    acc = {}
    for e, v in zip(episodes, final_pos):
        if v is None:
            continue
        acc.setdefault(e["patient"], []).append(float(v))
    return {p: float(np.mean(v)) for p, v in acc.items()}


def seed_patient_means(values, episodes):
    """{seed (int): {patient: mean}} from a row's per-episode values {"<seed>": {"final_pos": [...]}}."""
    return {int(s): patient_means(v["final_pos"], episodes) for s, v in sorted(values.items(), key=lambda kv: int(kv[0]))}


def row_error(pm):
    """Mean over seeds of patient means (pm: {seed: {patient: mean}})."""
    return float(np.mean([np.mean(list(c.values())) for c in pm.values()])) if pm else None


def success_rate(final_pos, final_ang, mm=10.0, deg=10.0):
    ok = [float(p) <= mm and float(a) <= deg for p, a in zip(final_pos, final_ang) if p is not None]
    return float(np.mean(ok)) if ok else None


# ============================================================================================ tests
def t_stats(d, level=0.95):
    """Paired t statistics of per-patient differences d (df n - 1): mean, sd, CI and the two-sided p."""
    from scipy import stats as st
    d = np.asarray(d, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tq = float(st.t.ppf(0.5 + level / 2, n - 1))
    if se > 0:
        t = m / se
        p = float(2 * st.t.sf(abs(t), n - 1))
    else:                       # equal per-patient differences: p = 1 at a zero mean, else 0 (as the anatomy rows)
        t = 0.0 if m == 0 else float("inf") * float(np.sign(m))
        p = 1.0 if m == 0 else 0.0
    return {"mean": m, "sd": sd, "se": se, "t": float(t), "df": n - 1, "p_t": p, "ci": [m - tq * se, m + tq * se]}


def tost(d, margin=TOST_MARGIN_MM):
    """Two one-sided paired t-tests at +-margin (df n - 1): the 90% t interval, the TOST p (the larger one-sided
    p) and whether the interval lies inside (-margin, +margin)."""
    from scipy import stats as st
    x = np.asarray(d, float)
    n = len(x)
    mu, se = float(x.mean()), float(x.std(ddof=1)) / np.sqrt(n)
    if se > 0:
        p_lo, p_hi = float(st.t.sf((mu + margin) / se, n - 1)), float(st.t.cdf((mu - margin) / se, n - 1))
    else:
        p_lo, p_hi = (0.0 if mu > -margin else 1.0), (0.0 if mu < margin else 1.0)
    tq = float(st.t.ppf(0.95, n - 1))
    ci = [mu - tq * se, mu + tq * se]
    return {"ci90": [float(v) for v in ci], "p": float(max(p_lo, p_hi)),
            "within": bool(-margin < ci[0] and ci[1] < margin)}


def holm(pvals):
    """Holm step-down adjusted p-values, in the order of the input."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return [float(x) for x in adj]


def contrast(a, b, patients=PATIENTS):
    """Seed-paired, patient-level difference a - b over the seeds both have. a, b: {seed: {patient: mean}}.
    Returns None without a common seed."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    P = sorted(patients)
    d = (np.array([[a[s][p] for p in P] for s in seeds]).mean(0)
         - np.array([[b[s][p] for p in P] for s in seeds]).mean(0))
    ts = t_stats(d)
    m = ts["mean"]
    D = np.array([[a[s][p] - b[s][p] for s in seeds] for p in P])        # [patient, seed]
    per_seed = [float(x) for x in D.mean(0)]
    tt = tost(d)
    return {"mean": m, "ci": [float(x) for x in ts["ci"]], "p_t": ts["p_t"], "sd": ts["sd"],
            "sign": int((np.sign(d) == np.sign(m)).sum()), "n": len(P), "seeds": seeds,
            "per_patient": {p: float(x) for p, x in zip(P, d)}, "per_seed": per_seed,
            "k_seed": int(sum(np.sign(x) == np.sign(m) for x in per_seed)), "n_seed": len(seeds),
            "tost_ci90": tt["ci90"], "tost_p": tt["p"], "tost_in": tt["within"]}


def patient_seed_ci(a, b, patients=PATIENTS, nboot=BOOT_RESAMPLES, seed=BOOT_SEED, level=0.95):
    """Patient-by-seed bootstrap 95% interval of the grand mean of D[patient, seed] = a - b (seed index paired
    across arms), over the seeds both have. Each resample draws patients with replacement, then, within each
    sampled patient, seeds with replacement; the patient indices of all resamples are drawn first, then the seed
    indices. None with fewer than two common seeds."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    P = sorted(patients)
    D = np.array([[a[s][p] - b[s][p] for s in seeds] for p in P], float)       # [patient, seed]
    n_p, n_s = D.shape
    rng = np.random.default_rng(seed)
    pidx = rng.integers(0, n_p, size=(nboot, n_p))
    sidx = rng.integers(0, n_s, size=(nboot, n_p, n_s))
    stat = D[pidx[:, :, None], sidx].mean(axis=(1, 2))
    lo, hi = 50.0 - 50.0 * level, 50.0 + 50.0 * level
    return [float(np.percentile(stat, lo)), float(np.percentile(stat, hi))]


def holds(holm_p, c):
    """A family member holds: Holm p < 0.05 and at least 5 of 6 patients share the sign."""
    return holm_p is not None and c is not None and holm_p < ALPHA and c["sign"] >= MIN_SIGN


def decide(kind, role, c, holm_p, short=False, demote=False):
    """(decision words, post hoc note or None) as Table 5 prints them, before the caption shortening."""
    if kind == "ref_seed0" or role == "ref":
        return "", None
    if kind in ("bn_train", "samus3") or role == "secondary" or demote:
        return DEC_SEC, None
    if c is None or short:
        return DEC_NC, None
    excl = c["ci"][0] > 0 or c["ci"][1] < 0
    hold = holds(holm_p, c)
    if kind == "imnt":
        if hold:
            return ("lowers" if c["mean"] < 0 else "raises"), None
        return DEC_IMN_NULL, (NOTE_A if excl else None)
    if hold:
        return (DEC_LOWER if c["mean"] < 0 else DEC_HIGHER), None
    return DEC_NULL, (NOTE_A if excl else None)


def decide_anatomy(c, holm_d, holm_e):
    """The CT-anatomy decision (difference family on the paired t p with the sign rule; equivalence family on the
    TOST p): 'falls'/'rises', 'equivalent within +-1 mm', both, or no detectable change (note e: equivalence within
    +-1 mm not shown). -> (words, equivalent, notes)."""
    dD = holm_d < ALPHA and c["sign"] >= MIN_SIGN
    dE = holm_e < ALPHA
    verb = "falls" if c["mean"] < 0 else "rises"
    if dD and dE:
        return f"{verb}; within ±1 mm", dE, []
    if dD:
        return verb, dE, []
    if dE:
        return "equivalent within ±1 mm", dE, []
    return "no detectable change", dE, ["e"]


def decide_submitted(c, learned=True, n_seeds=3, privileged=False):
    """The decision cell of a new method: no pre-specified decision word (it lies outside the pre-specified
    families). "post hoc", or "secondary" for a privileged method (as Table 5's privileged rows), or "not
    completed" without a complete result. -> (word, label)."""
    if c is None or (learned and n_seeds < 3):
        return DEC_NC, SUBMITTED
    if privileged:
        return DEC_SEC, SUBMITTED
    return DEC_POSTHOC, SUBMITTED


# ============================================================================================ MDD
def mdd(sd, n=6, alpha=ALPHA, power=0.8):
    """Minimum detectable difference of a two-sided paired t-test: (t_{1-alpha/2, n-1} + t_{power, n-1}) sd/sqrt(n)."""
    from scipy.stats import t as T
    if n < 2 or sd is None or not np.isfinite(sd):
        return None
    return float((T.ppf(1 - alpha / 2, n - 1) + T.ppf(power, n - 1)) * sd / np.sqrt(n))


# ============================================================================================ text
def signed(x, d=2):
    """Signed number rounded to d decimals; a value that rounds to zero prints unsigned."""
    s = f"{x:+.{d}f}"
    return f"{0:.{d}f}" if float(s) == 0 else s


def ci_text(ci, d=2):
    return f"[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"


def p3(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def diff_text(c, d=2):
    return "" if c is None else f"{signed(c['mean'], d)} {ci_text(c['ci'], d)}, {c['sign']}/{c['n']}"


def seeds_text(c):
    if c is None:
        return ""
    return f"{c['k_seed']}/{c['n_seed']}" if c.get("n_seed", 0) >= 2 else "n/a"


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


__all__ = ["PATIENTS", "patient_means", "seed_patient_means", "row_error", "success_rate", "t_stats", "tost", "holm",
           "contrast", "patient_seed_ci", "holds", "decide", "decide_anatomy", "decide_submitted", "mdd", "signed",
           "ci_text", "p3", "diff_text", "seeds_text", "SUBMITTED", "NOTE_A", "NOT_POWERED", "DEC_POSTHOC"]
