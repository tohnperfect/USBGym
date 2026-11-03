#!/usr/bin/env python3
"""make_figures.py: the generated figures of the article and the supplement (the paper's figures folder).

No number is typed in this file. Every plotted or annotated value is read from the v2 and v2.1 outputs; the only
constant is the REINFORCE pass line at one half (Q13 of the v2.1 plan), drawn in checks_sb3.

  ladder   figures/ladder.pdf (fig:ladder) and figures/ladder_data.json
           The oracle ladder: final translation error (mm) per rung and texture stage, one point per held-out
           patient, from outputs/v2/table4/episodes_long.csv (contact on, evaluation DR off). Per-patient
           means use collect_results_v2.per_patient, the function that produced RESULTS.json, and every
           cell mean and gap is asserted equal to outputs/v2/RESULTS.json (table4.cell_means_mm,
           table4.rungs_mm, table4.observability_gap, table4.perception_gap) before anything is drawn.
           The gap labels print RESULTS.json means (one decimal) and sign consistency (k/6 patients).

  stages   figures/stages.png and figures/stages.json (fig:stages)
           The SAME simulated view under the four texture stages. Inference only (no training, no
           evaluation). The probe is seated at screened acoustic window 0 of the liver and of the kidney,
           which is that window's target pose in view matching. The patient is chosen by an anatomy-only
           rule: the held-out patient whose two window-0 CT-label sectors contain the most reward organs
           (view_match.MATCH_ORGANS, >= 20 px each, summed over the two views), ties broken by
           usbg.benchmark.TEST_PATIENTS order; every patient's count is logged in stages.json. (A first
           rule, TEST_PATIENTS[0], gave a gallbladder-dominated liver view that is atypical of the six
           patients; it was replaced by this content rule, which does not look at any rendered image.)
           The physics sector frame is rendered once (DR off, RenderConfig(seed=0), CPU) and passed to each
           stage exactly as view_match.curvi_render does; SonoSPADE also receives the CT label map of the
           same view. Checkpoints come from outputs/v2/textures/manifest.json and their md5 must match the
           v2 oracle-ladder provenance.

  landscape  figures/landscape.pdf and figures/landscape.json (fig:landscape)
           Re-plot of outputs/v2/view_match/landscape_heldout.json (scripts/view_match_landscape.py, anatomical
           seat, DR arm off, CT-label segmentation term). Left: mean appearance term per translation-distance
           bin; right: mean +- SD over target views of the per-view Spearman correlation between the appearance
           term and closeness. The JSON's legacy summary is checked against its own results block and the
           means against outputs/v2/RESULTS.json cue_ablation.none before anything is drawn.

  layout   figures/layout.pdf and figures/layout.json (fig:layout)
           Re-plot of the matched-layout figure from outputs/v2/fig1/matched_layout.json (rows, IoUs) and
           outputs/v2/fig1/matched_layout_rows.npz, the CT label and stage renders of the figure rows, written by
           (one command) PYTHONPATH=src USBG_SEAT=anatomical USBG_LEARNED_SEG=0 .venv/bin/python
           scripts/fig_matched_layout.py --rerender-from outputs/v2/fig1/matched_layout.json
           --arrays-out outputs/v2/fig1/matched_layout_rows.npz
           (stage 3 only, at the saved poses; every row's pose and IoU is checked against the JSON). Real frames and
           annotations come from the public Vitale test set named in that JSON (md5 checked, private paths refused).
           Panels are cropped to the union of non-black content of all panels (crop logged in layout.json).

  scene    figures/scene.pdf and figures/scene.json (fig:scene)
           The human body as the scanned object. No result file is read: everything is built from one TRAIN
           patient's anatomical-seat label cache (data/cache_anat), the screened-windows file
           (data/cache/windows/usbg-windows-v1.json, usbg.windows) and the seat and render code. The patient is
           the first of usbg.benchmark.TRAIN_PATIENTS with screened windows for the liver and for both kidney
           flanks (never a held-out patient). (a) The CT skin surface (usbg.body_surface.body_mask, labels other
           than background, plus an axial hole fill so the airway lumen draws no inner tube; display only) with
           the liver and kidneys inside, and every screened liver and kidney window as a dot on its face and a
           stroke along its beam (hollow and faint when the face points away from the camera). (b) The probe at
           liver window 0 (usbg.placement.seat_volume_under_probe, the view-matching target pose of that
           window), drawn in the MuJoCo world: part of the box top (the contact plane, z = 0, perpendicular to
           the beam), the probe housing (placement.PROBE_HALF), the CT skin near the window, the skin normal and
           beam axis, and the 60-degree slicing fan painted with the CT label slice. (c) The physics render at
           that pose and its label slice in the measured sector geometry (view_match.curvi_render's path, DR off,
           RenderConfig(seed=0), CPU, gel coupling of the anatomical seat). Meshes are decimated and rasterized;
           text stays vector.

Figures of the article ("draft" runs the seven below, which read result files only and render nothing):
  ladder         Fig. 3 (a), as above; the arrow labels print the value and the sign count only.
  ladder_steps   Fig. 3 (b), figures/ladder_steps.pdf and .json, post hoc: mean translation error against control step
                 0 to 12 from outputs/v2/posthoc/posthoc_v2.json step_curves (certified by RESULTS.json; every curve's
                 step 12 equals its Table 4 cell). Printed at 0.44 textwidth.
  forest         Fig. 4, figures/forest.pdf and .json, from figures/plug_rows.json (written by make_tables.py with
                 Table 5): each completed family row's difference from its slot reference
                 with its 95% (thin) and 90% (thick) paired t CI on one mm axis; filled marker = post hoc TOST at
                 +/- 1 mm passes; no MDD band. Printed at 1.0 textwidth, 7 pt gate.
  checks_sb3     Fig. 5 (a), figures/checks_sb3.pdf and .json: the SB3 PPO positive control against environment steps
                 and the REINFORCE planted-deficit control against iteration. Printed at 0.325 textwidth, 7 pt gate.
  checks_cohort  Fig. 5 (b), figures/checks_cohort.pdf and .json: per-patient cohort contrasts, original six against the
                 20 cohort CTs, from outputs/v2_1/cohort/analysis_q2r.json. Printed at 0.325 textwidth, 7 pt gate.
  checks_horizon Fig. 5 (c), figures/checks_horizon.pdf and .json (exploratory v2.1): mean error
                 against control step 0 to 36 from the 36-step arms (outputs/v2_1/s21_h36), checked against
                 posthoc_logs.json and the archived step curves. Printed at 0.325 textwidth, 7 pt gate.
  stages_twin    Supplementary twin of Fig. 2, figures/stages_twin.png and .json: stored realism
                 frames of the Fig. 2 patient under physics, the LOTUS reimplementation, the original LOTUS code and
                 SIMUS. Printed at 0.80 textwidth, 7 pt gate.
Every figure's text passes a style gate (no em-dash, no triple hyphen, no banned word, no "open") before it is saved;
the figures marked "7 pt gate" also refuse any text below 7 pt (they are included 1:1, so that is the printed size).

The re-plots and the scene are authored at their printed width (TEXTWIDTH_PT times the fraction set in the
wrapper), so the type is printed at the size given here (7 pt and up), in Times with STIX math.

Usage (from the repo root):
  PYTHONPATH=src python <this folder>/make_figures.py            # all five
  PYTHONPATH=src python <this folder>/make_figures.py ladder
  PYTHONPATH=src python <this folder>/make_figures.py draft      # Figs. 3 to 5 and the Fig. 2 twin
  PYTHONPATH=src python <this folder>/make_figures.py forest checks_sb3 checks_cohort checks_horizon stages_twin

Public mode (the result files of the public results archive under ROOT, laid out as ROOT/outputs/..., ROOT/data/... and
ROOT/segmenter/outputs/...; DIR/figures/plug_rows.json written first by make_tables.py --public):
  python <this folder>/make_figures.py --public --results ROOT --out DIR [which ...]
writes the figures and their data files to DIR/figures; every path in a data file is relative (script_dirs.
public_json_view), and nothing outside the code tree, ROOT and DIR is read (results_world.py checks every result file
against ROOT/MANIFEST.tsv). A figure whose input the archive leaves out (ROOT/EXCLUDED.tsv: the texture-stage
checkpoints for stages and stages_twin, the annotated real test frames for layout) is built only when that file is
placed at its path under ROOT, where it is checked against its EXCLUDED.tsv row; otherwise one notice names the input
and DIR/NEEDS.tsv records the figure's files (reproduce.py reports them as NEEDS).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
import script_dirs as SD  # noqa: E402  (code tree, result root, public mode)
if __name__ == "__main__":
    import results_world  # noqa: E402
    results_world.start(sys.argv[1:])     # --public --results ROOT --out DIR, before the data paths below
import plan_inputs as PI  # noqa: E402

ROOT = SD.results_root()     # the result files (outputs/, data/): the code tree, or ROOT of --results in public mode
V2 = os.path.join(ROOT, "outputs", "v2")
_PP = SD.private_module()    # the development backend: None in public mode and in the public mirror
FIG = (os.path.join(SD.public_out(), "figures") if SD.public()
       else os.path.join(_PP.paper_tree(CODE_ROOT), "figures") if _PP
       else os.path.join("DIR", "figures"))    # the public mirror outside public mode: --help only (apply_cli)
if SD.public():              # matplotlib's cache inside DIR, not in the home folder
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(SD.public_out(), ".mplconfig"))
    # a fixed creation date in the PDF files (matplotlib reads SOURCE_DATE_EPOCH), so two runs write the same bytes
    os.environ["SOURCE_DATE_EPOCH"] = "0"
STAGES = ("physics", "cut", "cyclegan", "sonospade")
STAGE_NAME = {"physics": "physics", "cut": "CUT", "cyclegan": "CycleGAN", "sonospade": "SonoSPADE"}
TOL = 1e-9

# Okabe-Ito hues for the two mask arms (validated colorblind-safe pair) and neutrals for the references.
C_STATIC, C_ORACLE, C_CT, C_LEARNED, C_PATIENT = "#7F7F7F", "#000000", "#0072B2", "#D55E00", "#C8C8C8"

# Texture stages in the re-plotted figures (landscape, layout): capitalized names, and one Okabe-Ito hue and
# marker per stage. Black for the untextured stage; the three learned stages avoid blue #0072B2 and vermillion
# #D55E00, which the ladder uses for the CT- and learned-mask arms.
STAGE_LABEL = {"physics": "Physics", "cut": "CUT", "cyclegan": "CycleGAN", "sonospade": "SonoSPADE"}
STAGE_COLOR = {"physics": "#000000", "cut": "#E69F00", "cyclegan": "#56B4E9", "sonospade": "#009E73"}
STAGE_MARKER = {"physics": "o", "cut": "s", "cyclegan": "^", "sonospade": "D"}
TEXTWIDTH_PT = 468.0                                     # elsarticle 3p \textwidth (TeX pt, 72.27 per inch)
GRID = "#E6E6E6"


def _tw_in(frac):
    """Printed width in inches of frac * \\textwidth."""
    return frac * TEXTWIDTH_PT / 72.27


def _paper_rc(plt, size=7.0):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": size, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "xtick.labelsize": size, "ytick.labelsize": size,
                         "axes.labelsize": size + 0.5, "legend.fontsize": size, "pdf.fonttype": 42,
                         "axes.unicode_minus": True})


def _rel(p):
    if SD.public():
        a = os.path.abspath(p)
        return ("figures/" + os.path.relpath(a, FIG)) if a.startswith(FIG + os.sep) else SD.display(a)
    return os.path.relpath(p, ROOT)


def _pub(payload):
    """A figure data payload as written: in public mode every path made relative (script_dirs.public_json_view)."""
    return SD.public_json_view(payload) if SD.public() else payload


def _needs(inputs, outputs, what):
    """Public mode: True when an input of this figure (a path relative to the result root) is left out of the archive
    (a row of ROOT/EXCLUDED.tsv) and no file was placed at its path; the figure is then not built, one notice names the
    input and EXCLUDED.tsv, and DIR/NEEDS.tsv records the outputs (results_world.needs). A placed file is checked
    against its EXCLUDED.tsv row before it is read. Always False in development mode."""
    if not SD.public():
        return False
    import results_world
    return results_world.needs(inputs, [_rel(o) for o in outputs], what)


def _texture_ckpts():
    """The texture-stage checkpoints that outputs/v2/textures/manifest.json names (relative to the result root)."""
    with open(os.path.join(V2, "textures", "manifest.json")) as fh:
        manifest = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    return [manifest[st] for st in STAGES if manifest.get(st)]


def _md5(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


# Words that no figure prints (style rules; the gate covers figure text). Transfer and sim-to-real belong to scope
# statements, which no figure carries.
BANNED = ("confirmed", "significant", "comparable", "deployable", "deploy", "observability", "perception gap",
          "superset", "differentiable", "transfer", "sim-to-real", "realistic", "obs.", "perc.")


def _text_gate(fig, path, min_pt=None):
    """Refuse to save a figure whose text holds an em-dash, a triple hyphen, a banned word or the word 'open'.
    With min_pt, also refuse any visible text set smaller than min_pt points. The figures that pass min_pt are
    authored at their printed width (the wrapper includes them at width 1:1), so the authored size is the printed
    size (every Fig. 4 and Fig. 5 font at 7 pt or larger)."""
    import re
    import matplotlib.text as mtext
    if min_pt is not None:
        fig.canvas.draw()                                   # tick labels get their text at draw time
    objs = [t for t in fig.findobj(mtext.Text) if t.get_text() and t.get_visible()]
    for t in objs:
        s = t.get_text()
        low = s.lower()
        if "\u2014" in s or "-" * 3 in s or any(b in low for b in BANNED) or re.search(r"\bopen\b", low):
            raise SystemExit(f"{_rel(path)}: figure text {s!r} breaks the style gate")
        if min_pt is not None and t.get_fontsize() < min_pt - 1e-9:
            raise SystemExit(f"{_rel(path)}: figure text {s!r} is set at {t.get_fontsize():.2f} pt, below {min_pt} pt")
    return len(objs)


# ============================================================================================= ladder
def ladder(out_pdf=os.path.join(FIG, "ladder.pdf"), out_json=os.path.join(FIG, "ladder_data.json")):
    sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
    import collect_results_v2 as C                      # the exact aggregation behind RESULTS.json

    res_path = os.path.join(V2, "RESULTS.json")
    with open(res_path) as fh:
        T4 = json.load(fh)["table4"]
    rows = C.load_eps()
    if not rows:
        raise SystemExit("episodes_long.csv missing")
    base = dict(contact="on", eval_randomize="off")
    # rung key, tick label, color, filled marker, filter, RESULTS.json key it must reproduce
    rungs = [
        ("static", "static", C_STATIC, True, lambda st: dict(group="ladder", rung="static", stage=st),
         lambda st: ("rungs_mm", "static")),
        ("pose_oracle", "pose\noracle\u2020", C_ORACLE, True, lambda st: dict(group="ladder", rung="pose_oracle", stage=st),
         lambda st: ("rungs_mm", "pose_oracle")),
        ("ct_base", "base", C_CT, False, lambda st: dict(group="ladder", rung="ibvs_base", stage=st, mask="ct"),
         lambda st: ("cell_means_mm", f"ctmask/{st}/base")),
        ("ct_final", "final", C_CT, True, lambda st: dict(group="ctmask", arm="ctmask", stage=st, phase="final"),
         lambda st: ("cell_means_mm", f"ctmask/{st}/final")),
        ("learned_base", "base", C_LEARNED, False,
         lambda st: dict(group="ladder", rung="ibvs_base", stage=st, mask="learned"),
         lambda st: ("cell_means_mm", f"matched/{st}/base")),
        ("learned_final", "final", C_LEARNED, True,
         lambda st: dict(group="matched", arm="matched", stage=st, phase="final"),
         lambda st: ("cell_means_mm", f"matched/{st}/final")),
    ]
    data, checks = {}, []
    for st in STAGES:
        data[st] = {}
        for key, _lab, _c, _f, flt, rkey in rungs:
            pp, n, seeds = C.per_patient(rows, **flt(st), **base)
            if len(pp) != 6:
                raise SystemExit(f"{st}/{key}: {len(pp)} patients, expected 6")
            m = float(np.mean(list(pp.values())))
            sect, k = rkey(st)
            ref = T4[sect][k]
            if abs(m - ref) > TOL:
                raise SystemExit(f"{st}/{key}: CSV mean {m!r} != RESULTS.json table4.{sect}[{k!r}] {ref!r}")
            checks.append(f"table4.{sect}.{k}")
            data[st][key] = {"per_patient": pp, "mean": m, "n_episodes": n, "seeds": seeds,
                             "results_json": f"table4.{sect}.{k}"}
        # gaps: recompute the per-patient paired mean and check it against RESULTS.json
        for gname, hi, lo in (("observability_gap", "ct_final", "pose_oracle"),
                              ("perception_gap", "learned_final", "ct_final")):
            d = C.diff_pp(data[st][hi]["per_patient"], data[st][lo]["per_patient"])
            ref = T4[gname][st]
            if abs(float(np.mean(d)) - ref["mean"]) > TOL:
                raise SystemExit(f"{st}/{gname}: recomputed {np.mean(d)!r} != RESULTS.json {ref['mean']!r}")
            checks.append(f"table4.{gname}.{st}")
            data[st][gname] = {"mean": ref["mean"], "ci": ref["ci"], "sign": ref["sign"], "n": ref["n"],
                               "results_json": f"table4.{gname}.{st}"}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    patients = sorted(data[STAGES[0]]["static"]["per_patient"])
    ymax = max(v for st in STAGES for r in rungs for v in data[st][r[0]]["per_patient"].values())
    ytop = 5.0 * np.ceil((ymax + 1.0) / 5.0)
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.55), sharex=True, sharey=True)
    x = np.arange(len(rungs))
    jit = {p: j for p, j in zip(patients, np.linspace(-0.13, 0.13, len(patients)))}   # fixed, same in every panel
    for ax, st in zip(axes.ravel(), STAGES):
        D = data[st]
        for p in patients:                               # paired structure: one thin line per patient
            ax.plot(x + jit[p], [D[r[0]]["per_patient"][p] for r in rungs], color=C_PATIENT, lw=0.45, zorder=1)
        for i, (key, _lab, col, filled, _f, _r) in enumerate(rungs):
            ys = [D[key]["per_patient"][p] for p in patients]
            ax.scatter(i + np.array([jit[p] for p in patients]), ys, s=7, zorder=3, linewidths=0.6,
                       facecolors=col if filled else "white", edgecolors=col)
            ax.plot([i - 0.28, i + 0.28], [D[key]["mean"]] * 2, color=col, lw=1.8, zorder=4,
                    solid_capstyle="round")
        # oracle-to-CT-mask gap: pose oracle -> CT-mask final; CT-to-learned-mask gap: CT-mask final -> learned final
        # (RESULTS.json keys observability_gap and perception_gap). The arrow labels print the value and the sign count
        # only; the caption names each arrow by its end points.
        for gname, lo_key, hi_key, xa in (("observability_gap", "pose_oracle", "ct_final", 3.42),
                                          ("perception_gap", "ct_final", "learned_final", 5.42)):
            lo, hi = D[lo_key]["mean"], D[hi_key]["mean"]
            x0 = rungs.index(next(r for r in rungs if r[0] == lo_key))
            ax.plot([x0 + 0.3, xa], [lo, lo], color="#555555", lw=0.5, ls=(0, (2, 1.5)), zorder=2)
            ax.annotate("", xy=(xa, hi), xytext=(xa, lo), zorder=5,
                        arrowprops=dict(arrowstyle="<->", lw=0.7, color="#222222", shrinkA=0, shrinkB=0,
                                        mutation_scale=5))
            g = D[gname]
            ax.text(xa + 0.12, 0.5 * (lo + hi), f"{g['mean']:+.1f} ({g['sign']}/{g['n']})",
                    rotation=90, ha="left", va="center", fontsize=6, color="#222222")
        ax.set_title(STAGE_LABEL[st], fontsize=7.5, pad=2.5)
        ax.set_xlim(-0.5, 6.05)
        ax.set_ylim(0, ytop)
        ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
    import matplotlib.transforms as mtransforms
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels([r[1] for r in rungs], fontsize=6.2, linespacing=0.95)
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for x0, x1, lab in ((1.75, 3.25, "CT\nmask\u2020"), (3.75, 5.25, "learned\nmask")):   # the text's terms;
            # the dagger marks privileged rows in every table and figure
            ax.plot([x0, x1], [-0.125, -0.125], transform=tr, color="#444444", lw=0.5, clip_on=False)
            ax.text(0.5 * (x0 + x1), -0.145, lab, transform=tr, ha="center", va="top", fontsize=6.2, linespacing=0.95)
    for ax in axes[:, 0]:
        ax.set_ylabel("Final translation error (mm)", fontsize=7)
    fig.tight_layout(pad=0.25, h_pad=0.6, w_pad=0.4)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    _text_gate(fig, out_pdf)
    fig.savefig(out_pdf)
    plt.close(fig)

    def _jsonable(d):
        return {st: {k: ({kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in D.items()}
                for st, D in d.items()}
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_pub({"source_csv": _rel(os.path.join(V2, "table4", "episodes_long.csv")),
                        "source_json": _rel(res_path), "filter": base,
                        "aggregation": "scripts/collect_results_v2.py per_patient (mean of final_pos per patient)",
                        "checked_against_results_json": sorted(set(checks)), "tolerance": TOL,
                        "patients": patients, "data": _jsonable(data)}), fh, indent=1)
    print(f"ladder: wrote {_rel(out_pdf)} and {_rel(out_json)}; {len(set(checks))} values match RESULTS.json")


# ============================================================================================= stages
def stages(out_png=os.path.join(FIG, "stages.png"), out_json=os.path.join(FIG, "stages.json")):
    os.environ["USBG_SEAT"] = "anatomical"               # the v2 seat
    os.environ["USBG_LEARNED_SEG"] = "0"                 # CT label returned; no segmenter is loaded
    from usbg import volume as V, view_match as VM, render_lotus as R, provenance as P
    from usbg.benchmark import TEST_PATIENTS
    from usbg.placement import seat_volume_under_probe, n_windows
    from usbg.texture_gan import load_translator
    from usbg.segmenter import align_label_to_image
    from usbg import slicer as S

    with open(os.path.join(V2, "textures", "manifest.json")) as fh:
        manifest = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    if _needs([manifest[st] for st in STAGES if manifest.get(st)], (out_png, out_json), "stages"):
        return
    # the texture checkpoints must be the ones every v2 oracle-ladder run recorded
    expect = {}
    for st in STAGES:
        with open(os.path.join(V2, "table4", "ctmask", f"{st}_deploy_s0.json")) as fh:
            expect[st] = (json.load(fh)["provenance"].get("texture") or {}).get("md5")
    tex = {}
    for st in STAGES:
        p = manifest.get(st)
        md5 = P.md5(os.path.join(ROOT, p)) if p else None
        if md5 != expect[st]:
            raise SystemExit(f"{st}: manifest checkpoint md5 {md5} != Table IV provenance {expect[st]}")
        tex[st] = {"ckpt": p, "md5": md5}
    translators = {st: (load_translator(os.path.join(ROOT, tex[st]["ckpt"]), "cpu") if tex[st]["ckpt"] else None)
                   for st in STAGES}

    organs = (("liver", V.LIVER), ("kidney", V.KIDNEY))

    def label_sector(lv, label):
        Wfv, T0 = seat_volume_under_probe(lv, label, window=0)
        lab_lin = S.slice_at_pose(lv, T0, Wfv)
        _, ls = VM.warp_to_sector(np.zeros((128, 128), np.float32), align_label_to_image(lab_lin, (128, 128)))
        return ls

    counts = {}
    for c in TEST_PATIENTS:                               # anatomy-only selection rule (module doc)
        lv_c = V.load_cache(os.path.join(ROOT, "data", "cache", f"{c}.npz"))
        lv_c.meta.setdefault("case", c)
        counts[c] = {o: [V.USBG_LABEL_NAMES[k] for k in VM.MATCH_ORGANS if int((label_sector(lv_c, lab) == k).sum()) >= 20]
                     for o, lab in organs}
    case = max(TEST_PATIENTS, key=lambda c: (sum(len(v) for v in counts[c].values()), -TEST_PATIENTS.index(c)))
    lv = V.load_cache(os.path.join(ROOT, "data", "cache", f"{case}.npz"))
    lv.meta.setdefault("case", case)
    cfg = R.RenderConfig(seed=0)
    views = []
    for organ, label in organs:
        n = n_windows(lv, label)
        if n == 0:
            raise SystemExit(f"{case}: no screened {organ} window")
        Wfv, T0 = seat_volume_under_probe(lv, label, window=0)
        lab_lin = S.slice_at_pose(lv, T0, Wfv)
        img_lin = R.render(lab_lin, "physics", device="cpu", cfg=cfg)
        img_sec, lab_sec = VM.warp_to_sector(VM._resize01(img_lin, 128), align_label_to_image(lab_lin, (128, 128)))
        ims = {}
        for st, tr in translators.items():                # exactly view_match.curvi_render's dispatch
            if tr is None:
                x = img_sec
            elif getattr(tr, "wants_label", False):
                x = tr.translate_aligned(img_sec, lab_sec)
            else:
                x = tr(img_sec)
            ims[st] = np.clip(np.asarray(x, np.float32), 0, 1)
        views.append(dict(organ=organ, window=0, n_windows=int(n), label=lab_sec, ims=ims))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7})
    # label colors: the five reward organs as in the matched-layout figure, other tissue in grays
    lut = np.zeros((V.N_LABELS, 3))
    gray = {V.GI: 0.55, V.PANCREAS: 0.62, V.MUSCLE: 0.45, V.FAT: 0.80, V.BONE: 0.97, V.LUNG: 0.30,
            V.SOFT_TISSUE: 0.70}
    for k, g in gray.items():
        lut[k] = (g, g, g)
    organ_rgb = {V.LIVER: (0.85, 0.37, 0.0), V.GALLBLADDER: (0.0, 0.62, 0.45), V.VESSEL: (0.0, 0.45, 0.70),
                 V.KIDNEY: (0.80, 0.47, 0.65), V.SPLEEN: (0.94, 0.89, 0.26)}
    for k, c in organ_rgb.items():
        lut[k] = c
    cmap = ListedColormap(lut)
    cols = ["CT label (input)"] + [STAGE_NAME[s] for s in STAGES]
    fig, axes = plt.subplots(len(views), len(cols), figsize=(6.3, 1.32 * len(views) + 0.2), squeeze=False)
    for i, v in enumerate(views):
        panels = [("label", v["label"])] + [(s, v["ims"][s]) for s in STAGES]
        for j, (kind, im) in enumerate(panels):
            ax = axes[i][j]
            if kind == "label":
                ax.imshow(im, cmap=cmap, vmin=0, vmax=V.N_LABELS - 1, interpolation="nearest")
            else:
                ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(cols[j], fontsize=7.5, pad=3)
        axes[i][0].set_ylabel(f"{v['organ']} window", fontsize=7.5)
    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.3)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_pub({"rule": "held-out patient whose window-0 liver and kidney CT-label sectors contain the most "
                           "reward organs (>= 20 px each, summed), ties by TEST_PATIENTS order; screened window 0 of "
                           "each target organ; probe at the window's seated target pose; physics DR off, "
                           "RenderConfig(seed=0), CPU; every stage applied to the same physics sector frame",
                   "reward_organs_per_patient": counts,
                   "case": case, "seat_mode": os.environ["USBG_SEAT"],
                   "windows": P.windows_ref(), "texture": tex,
                   "views": [{"organ": v["organ"], "window": v["window"], "n_windows": v["n_windows"],
                              "organ_px": {V.USBG_LABEL_NAMES[k]: int((v["label"] == k).sum())
                                           for k in organ_rgb if (v["label"] == k).sum() > 0}}
                             for v in views]}), fh, indent=1)
    print(f"stages: wrote {_rel(out_png)} and {_rel(out_json)} ({case}, windows 0; counts "
          f"{ {c: sum(len(v) for v in d.values()) for c, d in counts.items()} })")


# ========================================================================================== landscape
def landscape(out_pdf=os.path.join(FIG, "landscape.pdf"), out_json=os.path.join(FIG, "landscape.json"), frac=0.8):
    src = os.path.join(V2, "view_match", "landscape_heldout.json")
    res_path = os.path.join(V2, "RESULTS.json")
    with open(src) as fh:
        d = json.load(fh)
    with open(res_path) as fh:
        ref = json.load(fh)["cue_ablation"]["none"]
    arm = d["dr_arms"][0]
    if d["seat_mode"] != "anatomical" or d["dr_arms"] != ["off"] or d["mask_source"]["seg_learned"] is not None:
        raise SystemExit(f"{_rel(src)}: not the anatomical-seat, DR-off, CT-label run")
    n_views = len(d["per_view"])
    if n_views != d["n_target_views"] or n_views != len(d["cases"]) * len(d["organs"]) * d["targets_per_case"]:
        raise SystemExit(f"{_rel(src)}: {n_views} per-view records vs n_target_views {d['n_target_views']}")
    data = {}
    for st in STAGES:
        tr, r = d["translators"][st], d["results"][arm][st]
        m, s = tr["app_closeness_per_view_mean"], tr["app_closeness_per_view_std"]
        if tr["app_by_bin"] != r["app_by_bin"] or m != r["app"]["per_view_mean"] or s != r["app"]["per_view_std"]:
            raise SystemExit(f"{st}: legacy summary differs from results.{arm}")
        if r["app"]["n_views"] != n_views or abs(m - ref[st]) > TOL:
            raise SystemExit(f"{st}: n_views {r['app']['n_views']} or mean {m!r} != RESULTS.json {ref[st]!r}")
        data[st] = {"app_by_bin": tr["app_by_bin"], "corr_mean": m, "corr_sd": s, "n_views": n_views}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _paper_rc(plt)
    W = _tw_in(frac)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(W, 2.15), layout="constrained",
                                 gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)
    bins = d["bins_mm"]
    x = np.arange(len(bins))
    for st in STAGES:
        a0.plot(x, data[st]["app_by_bin"], color=STAGE_COLOR[st], marker=STAGE_MARKER[st], ms=3.6, lw=1.0,
                mec="white", mew=0.4, label=STAGE_LABEL[st], zorder=3)
    a0.set_xticks(x, [f"{a}–{b}" for a, b in bins])
    a0.set_xlim(-0.3, len(bins) - 0.7)
    a0.set_xlabel("Translation distance to target pose (mm)")
    a0.set_ylabel("Mean appearance term")
    a0.legend(loc="upper right", frameon=False, handlelength=2.0, borderaxespad=0.2, labelspacing=0.3)
    a1.set_xticks(range(len(STAGES)), [STAGE_LABEL[s] for s in STAGES])
    for i, st in enumerate(STAGES):
        m, s = data[st]["corr_mean"], data[st]["corr_sd"]
        a1.errorbar(i, m, yerr=s, fmt=STAGE_MARKER[st], color=STAGE_COLOR[st], ms=4.2, mec="white", mew=0.4,
                    elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=3)
        a1.text(i, m + s + 0.025, f"{m:.3f}", ha="center", va="bottom", fontsize=7)
    a1.set_xlim(-0.5, len(STAGES) - 0.5)
    a1.set_ylim(0, 0.9)
    a1.set_yticks(np.arange(0, 0.81, 0.2))
    a1.set_xlabel("Texture stage")
    a1.set_ylabel("Spearman correlation,\nappearance term vs closeness")
    for ax in (a0, a1):
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(pad=1.5)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_pub({"source_json": _rel(src), "source_md5": _md5(src), "checked_against": f"{_rel(res_path)} "
                   "cue_ablation.none", "dr_arm": arm, "seat_mode": d["seat_mode"], "n_target_views": n_views,
                   "targets_per_case_per_organ": d["targets_per_case"], "samples_per_target": d["samples_per_target"],
                   "cases": d["cases"], "organs": d["organs"], "bins_mm": bins,
                   "sd": "population SD over target views (numpy std), as in the source JSON",
                   "printed_width_in": W, "data": data}), fh, indent=1)
    print(f"landscape: wrote {_rel(out_pdf)} ({W:.3f} in wide) and {_rel(out_json)}")


# ============================================================================================= layout
def layout(out_pdf=os.path.join(FIG, "layout.pdf"), out_json=os.path.join(FIG, "layout.json"), frac=0.83):
    sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
    import fig_matched_layout as FML                     # ORGANS, organ colors, per_organ_iou of the original
    try:     # the testbed's private-path guard (a package without it: public mode relies on the closed world)
        from usbg.provenance import is_private_path
    except ImportError:
        if not SD.public():
            raise
        is_private_path = (lambda p: False)  # noqa: E731

    src = os.path.join(V2, "fig1", "matched_layout.json")
    arr = os.path.join(V2, "fig1", "matched_layout_rows.npz")
    side = os.path.splitext(arr)[0] + ".json"
    if not os.path.exists(arr):
        raise SystemExit(f"{_rel(arr)} missing; make it with the command in this module's doc (layout)")
    with open(src) as fh:
        d = json.load(fh)
    with open(side) as fh:
        sc = json.load(fh)
    if sc["source_json_md5"] != _md5(src):
        raise SystemExit(f"{_rel(arr)} was re-rendered from another version of {_rel(src)}")
    real_path = os.path.join(ROOT, d["real"]["path"])
    if is_private_path(real_path) or is_private_path(d["real"]["path"]):
        raise SystemExit(f"{d['real']['path']}: private pool, refused")
    if _needs([d["real"]["path"]], (out_pdf, out_json), "layout"):
        return
    if _md5(real_path) != d["real"]["md5"]:
        raise SystemExit(f"{real_path}: md5 differs from the matched-layout run")
    z = np.load(arr)
    rows = d["figure_rows"]
    frames = {f["frame"]: f for f in d["frames"]}
    if [int(i) for i in z["frame"]] != [r["frame"] for r in rows]:
        raise SystemExit("re-rendered rows differ from figure_rows")
    real = np.load(real_path)
    RI = real["images"].astype(np.float32) / 255.0
    RL = np.where(np.isin(real["labels"], FML.ORGANS), real["labels"], 0).astype(np.uint8)
    for i, r in enumerate(rows):                          # the arrays are the saved rows
        f = frames[r["frame"]]
        if str(real["sessions"][r["frame"]]) != r["session"] or \
                FML.per_organ_iou(z["ct_label"][i], RL[r["frame"]], d["min_px"]) != f["per_organ_iou"]:
            raise SystemExit(f"row {i} (frame {r['frame']}) does not match {_rel(src)}")
    fr = [r["frame"] for r in rows]
    panels_all = [RI[fr], RL[fr] > 0, z["ct_label"] > 0] + [z[f"stage_{s}"] for s in STAGES]
    content = np.zeros(RI.shape[1:], bool)
    for p in panels_all:
        content |= (np.asarray(p, np.float32) > 4.0 / 255.0).any(0)
    ys, xs = np.where(content)
    pad = 2
    y0, y1 = max(0, ys.min() - pad), min(content.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(content.shape[1], xs.max() + pad + 1)
    crop = (slice(y0, y1), slice(x0, x1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _paper_rc(plt)

    def label_rgb(lab):
        rgb = np.zeros(lab.shape + (3,), np.float32)
        for o, c in FML._CMAP.items():
            rgb[lab == o] = c
        return rgb

    def outline(ax, lab):
        for o, c in FML._CMAP.items():                    # as fig_matched_layout._contours
            m = lab == o
            if m.sum() >= 5:
                ax.contour(m.astype(float), levels=[0.5], colors=[c], linewidths=0.6)

    cols = ["Real frame", "Annotation", "Matched CT label"] + [STAGE_LABEL[s] for s in STAGES]
    W = _tw_in(frac)
    lab_w, gap, head, foot = 0.30, 0.035, 0.18, 0.17      # inches
    pw = (W - lab_w - (len(cols) - 1) * gap) / len(cols)
    ph = pw * (y1 - y0) / (x1 - x0)
    H = head + len(rows) * ph + (len(rows) - 1) * gap + foot
    fig = plt.figure(figsize=(W, H))
    for i, r in enumerate(rows):
        k = r["frame"]
        ct = z["ct_label"][i]
        panels = [(RI[k], RL[k], True), (label_rgb(RL[k]), None, False), (label_rgb(ct), None, False)] + \
                 [(z[f"stage_{s}"][i], ct, True) for s in STAGES]
        for j, (im, cl, gray) in enumerate(panels):
            ax = fig.add_axes([(lab_w + j * (pw + gap)) / W, 1 - (head + i * (ph + gap) + ph) / H, pw / W, ph / H])
            if gray:
                ax.imshow(np.clip(im[crop], 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(im[crop], interpolation="nearest")
            if cl is not None:
                outline(ax, cl[crop])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(cols[j], fontsize=7.5, pad=3)
            if j == 0:
                ax.text(-0.05, 0.5, f"{r['session']}, {r['case']}\nIoU {r['best_iou']:.2f}", transform=ax.transAxes,
                        rotation=90, ha="right", va="center", fontsize=7, linespacing=1.15)
    s = d["best_iou_summary"]
    fig.text(0.5, 0.5 * foot / H, f"Best IoU over all {len(d['frames'])} frames: median {s['median']:.2f} "
             f"(IQR {s['p25']:.2f}–{s['p75']:.2f})", ha="center", va="center", fontsize=7)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, dpi=600)
    plt.close(fig)
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_pub({"source_json": _rel(src), "source_md5": _md5(src), "arrays": _rel(arr), "arrays_md5": _md5(arr),
                   "arrays_sidecar": _rel(side), "real": d["real"], "stage_order": list(STAGES),
                   "columns": cols, "crop_rows_cols": [[int(y0), int(y1)], [int(x0), int(x1)]],
                   "crop_rule": "union over all panels of pixels > 4/255, padded by 2 px",
                   "rows": rows, "best_iou_summary": s, "n_frames": len(d["frames"]),
                   "printed_size_in": [W, H], "image_dpi": 600}), fh, indent=1)
    print(f"layout: wrote {_rel(out_pdf)} ({W:.3f} x {H:.3f} in, crop rows {y0}:{y1} cols {x0}:{x1}) and {_rel(out_json)}")


# ============================================================================================== scene
SKIN_RGB = (0.90, 0.74, 0.62)
C_WIN_LIVER, C_WIN_KIDNEY = "#A84300", "#6E2A5C"         # darker shades of the liver and kidney label colors
C_PLANE = (0.16, 0.38, 0.72)


def _cam(elev, azim):
    """Unit vector from the scene toward a matplotlib 3D camera at (elev, azim) degrees."""
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _shade(tris, rgb, alpha, light):
    """Per-triangle RGBA with two-sided Lambert shading (tris (T,3,3))."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    k = 0.40 + 0.60 * np.abs(n @ light)
    c = np.clip(np.asarray(rgb, float)[None, :] * k[:, None], 0, 1)
    return np.concatenate([c, np.full((len(c), 1), alpha)], 1)


def _tris(mesh, M=None, scale=1000.0):
    """Triangles (T,3,3) of a (verts_m, faces) mesh, optionally moved by the 4x4 M, in mm."""
    v, f = mesh
    v = np.asarray(v, float)
    if M is not None:
        v = v @ M[:3, :3].T + M[:3, 3]
    return v[np.asarray(f)] * scale


def _clip(tris, planes):
    """Clip triangles to the half-spaces n.x >= d (each plane (n, d)) with clean planar cuts."""
    t = tris
    for n, d in planes:
        n = np.asarray(n, float)
        s = t @ n - d
        inside = (s >= 0).all(1)
        cross = ~inside & (s >= 0).any(1)
        out = [t[inside]]
        for tri, sd in zip(t[cross], s[cross]):
            poly = []
            for i in range(3):
                p, q, sp, sq = tri[i], tri[(i + 1) % 3], sd[i], sd[(i + 1) % 3]
                if sp >= 0:
                    poly.append(p)
                if (sp >= 0) != (sq >= 0):
                    poly.append(p + (q - p) * (sp / (sp - sq)))
            out += [np.array([[poly[0], poly[k], poly[k + 1]]]) for k in range(1, len(poly) - 1)]
        t = np.concatenate(out) if sum(len(o) for o in out) else np.zeros((0, 3, 3))
    return t


def _disk(r, n=32):
    """Half-spaces whose intersection is the vertical prism over a regular n-gon of radius r about z."""
    return [((-np.cos(a), -np.sin(a), 0.0), -r) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]


def _decimate(v, f, n_faces, taubin=0):
    import trimesh
    m = trimesh.Trimesh(vertices=np.asarray(v, float), faces=np.asarray(f), process=True)
    if taubin:
        trimesh.smoothing.filter_taubin(m, iterations=int(taubin))
    if len(m.faces) > n_faces:
        m = m.simplify_quadric_decimation(face_count=int(n_faces))
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def _setup3d(ax, lo, hi, elev, azim, zoom=1.0):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.asarray(hi) - np.asarray(lo), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def _p2(ax, p):
    from mpl_toolkits.mplot3d import proj3d
    x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
    return x, y


def _note(ax, text, p, xytext, ha="left", va="center", fs=7.0):
    """Label a 3D point with a leader line from a position given in axes fraction."""
    ax.annotate(text, xy=_p2(ax, p), xytext=xytext, textcoords="axes fraction", ha=ha, va=va, fontsize=fs,
                color="#222222", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444", shrinkA=1, shrinkB=1.5))


def scene(out_pdf=os.path.join(FIG, "scene.pdf"), out_json=os.path.join(FIG, "scene.json")):
    os.environ["USBG_SEAT"] = "anatomical"               # the v2 seat (bladder-free cache, gel coupling)
    os.environ["USBG_LEARNED_SEG"] = "0"                 # no segmenter is loaded
    from scipy import ndimage as ndi
    from skimage import measure
    from usbg import volume as V, body_surface as BS, windows as W, view_match as VM, render_lotus as R
    from usbg import provenance as P, slicer as S
    from usbg.benchmark import TRAIN_PATIENTS, TEST_PATIENTS
    from usbg.placement import seat_from_window, seat_volume_under_probe, PROBE_HALF
    from usbg.seatmode import gel_coupling
    from usbg.segmenter import align_label_to_image

    # ==== patient: the first TRAIN patient with liver windows and windows on both kidney flanks
    wfile = W.load_file()
    case = next(c for c in TRAIN_PATIENTS
                if all(len(wfile["cases"][c]["windows"].get(o, [])) > 0
                       for o in ("liver", "kidney_right", "kidney_left")))
    if case in TEST_PATIENTS:
        raise SystemExit(f"{case} is a held-out patient")
    cache = os.path.join(ROOT, "data", "cache", f"{case}.npz")
    lv = V.load_cache(cache)
    lv.meta.setdefault("case", case)
    if lv.meta.get("cache_variant") != "anat":
        raise SystemExit(f"{case}: expected the bladder-free anatomical cache, got {lv.meta.get('cache_variant')}")
    A = np.asarray(lv.affine_m, float)
    win_l, win_k = W.windows_meta(lv, V.LIVER), W.windows_meta(lv, V.KIDNEY)
    n_side = {o: len(wfile["cases"][case]["windows"].get(o, [])) for o in ("liver", "kidney_right", "kidney_left")}
    if len(win_l) != n_side["liver"] or len(win_k) != n_side["kidney_right"] + n_side["kidney_left"]:
        raise SystemExit("windows_meta and the windows file disagree")

    # ==== meshes (metres, CT volume frame)
    body = BS.body_mask(lv)                              # not air, largest component, 3D holes filled
    for k in range(body.shape[2]):                       # axial fill: the airway lumen opens to the CT top
        body[:, :, k] = ndi.binary_fill_holes(body[:, :, k])
    vv, ff, _n, _v = measure.marching_cubes(body.astype(np.float32), 0.5, step_size=3)
    vv, ff = BS._postprocess(vv @ A[:3, :3].T + A[:3, 3], ff, smooth_iters=12)
    meshes = {"skin": _decimate(vv, ff, 9000)}
    for name, lab, n_f in (("liver", V.LIVER, 3000), ("kidney", V.KIDNEY, 1600)):
        m = ndi.binary_closing(lv.labels == lab, iterations=1)
        vv, ff, _n, _v = measure.marching_cubes(m.astype(np.float32), 0.5, step_size=2)
        meshes[name] = _decimate(vv @ A[:3, :3].T + A[:3, 3], ff, n_f, taubin=10)

    # ==== the close-up: liver window 0 at its seated (view-matching target) pose
    T_vp = np.asarray(win_l[0]["T_vp"], float)
    Wfv, T0 = seat_volume_under_probe(lv, V.LIVER, window=0)
    Wfv_chk, T0_chk = seat_from_window(T_vp, 0.0)
    if not (np.allclose(Wfv, Wfv_chk) and np.allclose(T0, T0_chk)):
        raise SystemExit("seat_volume_under_probe differs from seat_from_window(window 0)")
    lab_lin = S.slice_at_pose(lv, T0, Wfv)
    cfg = R.RenderConfig(seed=0)
    img_lin = R.render(lab_lin, "physics", device="cpu", cfg=cfg)
    img_sec, lab_sec = VM.warp_to_sector(VM._resize01(img_lin, 128), align_label_to_image(lab_lin, (128, 128)))
    fan_pts, _depth = S.scan_grid_probe()

    # ==== colors: the label palette of the stages figure
    lut = np.zeros((V.N_LABELS, 3))
    for k, g in {V.GI: 0.55, V.PANCREAS: 0.62, V.MUSCLE: 0.45, V.FAT: 0.80, V.BONE: 0.97, V.LUNG: 0.30,
                 V.SOFT_TISSUE: 0.70}.items():
        lut[k] = (g, g, g)
    organ_rgb = {V.LIVER: (0.85, 0.37, 0.0), V.GALLBLADDER: (0.0, 0.62, 0.45), V.VESSEL: (0.0, 0.45, 0.70),
                 V.KIDNEY: (0.80, 0.47, 0.65), V.SPLEEN: (0.94, 0.89, 0.26)}
    for k, c in organ_rgb.items():
        lut[k] = c

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    _paper_rc(plt)
    FS = 7.0
    Wd, Hd = _tw_in(1.0), 2.75
    fig = plt.figure(figsize=(Wd, Hd))
    axa = fig.add_axes([0.0, 0.0, 0.335, 0.94], projection="3d", computed_zorder=False)
    axb = fig.add_axes([0.335, 0.0, 0.43, 0.94], projection="3d", computed_zorder=False)
    axc = [fig.add_axes([0.79, 0.515, 0.205, 0.40]), fig.add_axes([0.79, 0.045, 0.205, 0.40])]

    # (a) the body, the organs and the screened windows, in the CT frame (RAS, mm)
    view_a = dict(elev=10.0, azim=58.0)
    cam, light = _cam(**view_a), _cam(view_a["elev"] + 25.0, view_a["azim"] - 30.0)
    zc = (60.0, 500.0)
    zcut = [((0, 0, 1), zc[0]), ((0, 0, -1), -zc[1])]
    P_, C_ = [], []
    for name, rgb, alpha in (("liver", organ_rgb[V.LIVER], 0.92), ("kidney", organ_rgb[V.KIDNEY], 0.95),
                             ("skin", SKIN_RGB, 0.26)):
        t = _clip(_tris(meshes[name]), zcut)
        P_.append(t); C_.append(_shade(t, rgb, alpha, light))
    axa.add_collection3d(Poly3DCollection(np.concatenate(P_), facecolors=np.concatenate(C_), edgecolors="none",
                                          linewidths=0, rasterized=True, zorder=1))
    sv = P_[-1].reshape(-1, 3)
    lo_a, hi_a = sv.min(0), sv.max(0)
    _setup3d(axa, lo_a, hi_a, **view_a, zoom=1.12)
    for wins, col in ((win_l, C_WIN_LIVER), (win_k, C_WIN_KIDNEY)):
        F = np.array([w["face"] for w in wins]) * 1000.0
        N = np.array([w["normal"] for w in wins])
        Bm = np.array([np.asarray(w["T_vp"])[:3, 2] for w in wins])
        vis = N @ cam > 0.0
        for sel, alpha, mfc in ((~vis, 0.45, "white"), (vis, 1.0, col)):
            if sel.any():
                axa.add_collection3d(Line3DCollection(np.stack([F[sel], F[sel] + 20.0 * Bm[sel]], 1), colors=col,
                                                      linewidths=0.6, alpha=alpha, zorder=3))
                axa.scatter(F[sel, 0], F[sel, 1], F[sel, 2], s=2.5, c=mfc, edgecolors=col, linewidths=0.45,
                            alpha=alpha, depthshade=False, zorder=4)
    f0 = np.asarray(win_l[0]["face"]) * 1000.0
    axa.scatter([f0[0]], [f0[1]], [f0[2]], s=40, facecolors="none", edgecolors="k", linewidths=0.8,
                depthshade=False, zorder=5)
    # (b) the close-up in the MuJoCo world of the seat: contact plane z = 0, beam on -z, mm
    view_b = dict(elev=22.0, azim=-60.0)
    light_b = _cam(view_b["elev"] + 25.0, view_b["azim"] - 30.0)
    w = (fan_pts[::4, ::8] @ T0[:3, :3].T + T0[:3, 3]) * 1000.0
    lab_f = lab_lin[::4, ::8]
    quads = np.stack([w[:-1, :-1], w[1:, :-1], w[1:, 1:], w[:-1, 1:]], 2).reshape(-1, 4, 3)
    lab_q = lab_f[:-1, :-1].reshape(-1)
    keep = lab_q != V.BACKGROUND                         # the gel-coupled gap before the skin is left blank
    fc = lut[lab_q[keep]]
    axb.add_collection3d(Poly3DCollection(quads[keep], facecolors=fc, edgecolors=fc, linewidths=0.25,
                                          rasterized=True, zorder=1))
    sk = _clip(_tris(meshes["skin"], Wfv), _disk(55.0) + [((0, 0, 1), -40.0), ((0, 0, -1), -30.0)])
    axb.add_collection3d(Poly3DCollection(sk, facecolors=_shade(sk, SKIN_RGB, 0.55, light_b), edgecolors="none",
                                          linewidths=0, rasterized=True, zorder=1.5))
    s_pl = 46.0
    sq = np.array([[-s_pl, -s_pl, 0], [s_pl, -s_pl, 0], [s_pl, s_pl, 0], [-s_pl, s_pl, 0]])
    axb.add_collection3d(Poly3DCollection([sq], facecolors=[C_PLANE + (0.13,)], edgecolors=[C_PLANE + (1.0,)],
                                          linewidths=0.7, zorder=2))
    bx, by, bz = (1000.0 * np.asarray(PROBE_HALF)) * np.array([1, 1, 2])   # housing behind the face (+z)
    V8 = np.array([[x, y, z] for z in (0, bz) for y in (-by, by) for x in (-bx, bx)])
    box_f = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    axb.add_collection3d(Poly3DCollection([V8[f] for f in box_f],
                                          facecolors=[(0.30 * k, 0.36 * k, 0.46 * k, 1.0)
                                                      for k in (0.55, 0.95, 0.75, 0.75, 0.6, 0.85)],
                                          edgecolors=(0.1, 0.1, 0.1, 1.0), linewidths=0.4, zorder=5))
    z_top, z_bot = 52.0, -122.0
    axb.plot([0, 0], [0, 0], [z_top, z_bot], color="k", lw=0.6, ls=(0, (3, 2)), zorder=6)
    n_w = Wfv[:3, :3] @ np.asarray(win_l[0]["normal"], float)
    axb.plot([0, z_top * n_w[0]], [0, z_top * n_w[1]], [0, z_top * n_w[2]], color="#B00000", lw=0.8, zorder=6)
    R_b = np.array([82.0, 50.0])
    lo_b, hi_b = np.array([-R_b[0], -R_b[1], -128.0]), np.array([R_b[0], R_b[1], z_top])
    _setup3d(axb, lo_b, hi_b, **view_b, zoom=1.15)
    for ax in (axa, axb):
        ax.set_facecolor((1, 1, 1, 0))                   # no 3D-axes background over neighboring labels

    # (c) the observation at that pose: physics render and CT label slice in the sector geometry
    content = (lab_sec > 0) | (img_sec > 4.0 / 255.0)
    ys, xs = np.where(content)
    crop = (slice(max(0, ys.min() - 1), ys.max() + 2), slice(max(0, xs.min() - 1), xs.max() + 2))
    axc[0].imshow(img_sec[crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axc[1].imshow(lut[lab_sec[crop]], interpolation="nearest")
    for ax, title in zip(axc, ("Physics B-mode", "CT label slice")):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=7.5, pad=2)

    fig.canvas.draw()                                    # final projections before the leader lines
    _note(axa, "liver", np.array([60.0, 230.0, 330.0]), (0.97, 0.80), ha="right")
    kid = _tris(meshes["kidney"]).reshape(-1, 3)
    for i, side in enumerate((kid[:, 0] < 0, kid[:, 0] > 0)):   # left, right kidney (-x / +x in RAS): lower pole
        ks = kid[side]
        _note(axa, "kidneys" if i == 0 else "", ks[ks[:, 2] <= np.percentile(ks[:, 2], 10)].mean(0),
              (0.97, 0.36) if i == 0 else (0.825, 0.36), ha="right")
    _note(axa, "window in (b)", f0, (0.97, 0.11), ha="right")
    # direction triad: the projected patient right, anterior and superior axes, drawn in 2D
    o3 = 0.5 * (lo_a + hi_a)
    ox, oy = _p2(axa, o3)
    tri_ax = fig.add_axes([0.255, 0.80, 0.07, 0.13])
    tri_ax.set_xlim(-1.5, 1.5); tri_ax.set_ylim(-1.5, 1.5); tri_ax.set_aspect("equal"); tri_ax.axis("off")
    for d, lab_ in (((1, 0, 0), "R"), ((0, 1, 0), "A"), ((0, 0, 1), "S")):
        ex, ey = _p2(axa, o3 + 50.0 * np.asarray(d, float))
        v2 = np.array([ex - ox, ey - oy]); v2 = v2 / np.linalg.norm(v2)
        tri_ax.annotate("", xy=v2, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#333333",
                                                                    mutation_scale=5, shrinkA=0, shrinkB=0))
        tri_ax.text(*(1.35 * v2), lab_, fontsize=FS, ha="center", va="center")
    # (b) organs on the fan, at the median fan point of each label
    wf = (fan_pts @ T0[:3, :3].T + T0[:3, 3]) * 1000.0
    def at(lab_id):
        return np.median(wf[lab_lin == lab_id], axis=0)
    p_l = at(V.LIVER)
    axb.text(*(p_l + np.array([-20.0, 0.0, 10.0])), "liver", color="white", fontsize=FS, ha="center",
             va="center", zorder=7)
    _note(axb, "gallbladder", at(V.GALLBLADDER), (0.03, 0.30))
    _note(axb, "vessel", at(V.VESSEL), (0.83, 0.25))
    _note(axb, "probe", np.array([bx, 0.0, 0.7 * bz]), (0.80, 0.97))
    _note(axb, "contact plane\n(box top)", np.array([0.8 * s_pl, 0.8 * s_pl, 0.0]), (0.84, 0.80))
    _note(axb, "skin (CT)", np.array([-48.0, -25.0, -6.0]), (0.02, 0.80))
    _note(axb, f"skin normal, {win_l[0]['tilt_deg']:.0f}\u00b0 from the beam", 0.92 * z_top * n_w, (0.03, 0.97))
    _note(axb, "beam axis", np.array([0.0, 0.0, -112.0]), (0.62, 0.05))
    for x, lab_ in ((0.005, "(a)"), (0.335, "(b)"), (0.77, "(c)")):
        fig.text(x, 0.99, lab_, fontsize=8, fontweight="bold", va="top")
    fig.legend(handles=[Line2D([], [], marker="o", ls="-", color=C_WIN_LIVER, ms=2.5, lw=0.6,
                               label=f"liver window ({len(win_l)})"),
                        Line2D([], [], marker="o", ls="-", color=C_WIN_KIDNEY, ms=2.5, lw=0.6,
                               label=f"kidney window ({len(win_k)})"),
                        Line2D([], [], marker="o", ls="-", color="#999999", mfc="white", ms=2.5, lw=0.6,
                               label="far side")],
               loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=FS, handlelength=1.6,
               labelspacing=0.25, borderaxespad=0.3)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, dpi=600)
    plt.close(fig)

    m0 = win_l[0]
    wref = P.windows_ref()
    wref["path"] = _rel(wref["path"])
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_pub({"rule": "first TRAIN patient (usbg.benchmark.TRAIN_PATIENTS order) with screened windows for the "
                           "liver and for both kidney flanks; close-up = liver window 0 at its seated pose "
                           "(seat_volume_under_probe, the view-matching target pose of that window)",
                   "case": case, "split": "TRAIN", "seat_mode": os.environ["USBG_SEAT"], "gel_coupling": gel_coupling(),
                   "cache": {"path": _rel(lv.meta["cache_path"]), "md5": _md5(lv.meta["cache_path"])},
                   "windows": wref, "n_windows": {"liver": len(win_l), "kidney": len(win_k), **n_side},
                   "closeup_window": {"organ": "liver", "index": 0, "window_id": m0["window_id"], "beam": m0["beam"],
                                      "tilt_deg": m0["tilt_deg"], "phi_deg": m0["phi"], "azimuth_deg": m0["th"],
                                      "target_depth_mm": m0["t_depth_mm"], "organ_frac": m0["organ_frac"],
                                      "face_m": m0["face"], "normal": m0["normal"], "T_vp": m0["T_vp"]},
                   "world_from_volume": Wfv.tolist(), "T0_world_from_probe": T0.tolist(),
                   "render": {"renderer": "physics", "device": "cpu", "RenderConfig_seed": 0, "domain_randomization": False,
                              "texture_stage": "physics (none)", "sector": VM.SECTOR, "size_px": 128},
                   "label_slice_px": {V.USBG_LABEL_NAMES[int(k)]: int(c) for k, c in
                                      zip(*np.unique(lab_sec[lab_sec > 0], return_counts=True))},
                   "meshes": {k: {"n_verts": int(len(v)), "n_faces": int(len(f))} for k, (v, f) in meshes.items()},
                   "mesh_recipe": "skin: usbg.body_surface.body_mask + axial binary_fill_holes, marching cubes step 3, "
                                  "body_surface._postprocess Taubin 12, quadric decimation; organs: binary closing, "
                                  "marching cubes step 2, Taubin 10, quadric decimation; panel (a) clipped to "
                                  f"z in {list(zc)} mm",
                   "contact_plane": "MuJoCo box top z = 0 of src/usbg/mujoco_env.py _mjcf, drawn as a 92 mm square",
                   "probe_housing_mm": [2 * bx, 2 * by, bz], "views": {"a": view_a, "b": view_b},
                   "printed_size_in": [Wd, Hd], "raster_dpi": 600}), fh, indent=1)
    print(f"scene: wrote {_rel(out_pdf)} ({os.path.getsize(out_pdf) / 1e6:.2f} MB, {Wd:.3f} x {Hd:.2f} in) and "
          f"{_rel(out_json)} ({case}, liver window 0, {len(win_l)} liver and {len(win_k)} kidney windows)")


# ============================================================================================= figures of the article
# Each figure is authored
# at its printed width (the fraction of \textwidth given in its docstring), so the type prints at 6 to 7 pt. Colors
# are Okabe-Ito (colorblind-safe): blue #0072B2 for the CT-mask arms, vermillion #D55E00 for the learned-mask arms,
# black for the pose oracle and gray for the static start. Every value is read from a result file and checked against
# its certifying file before anything is drawn; each figure writes a JSON sidecar with the plotted values and sources.
C_FAMILY, C_BAND, C_COHORT, C_SIX = "#0072B2", "#E69F00", "#0072B2", "#000000"
SEED_MARK = ("o", "s", "^")
V21 = os.path.join(ROOT, "outputs", "v2_1")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _paper_rc(plt, 7.0)
    return plt


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=1.5)


def _save(fig, out_pdf, out_json, payload, min_pt=None, **savekw):
    n_txt = _text_gate(fig, out_pdf, min_pt=min_pt)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, **savekw)
    if min_pt is not None:
        payload = {**payload, "min_font_pt": min_pt, "printed_size_in": [float(v) for v in fig.get_size_inches()]}
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_pub(payload), fh, indent=1, default=float)
    return n_txt


def _results():
    with open(os.path.join(V2, "RESULTS.json")) as fh:
        return json.load(fh)


# ============================================================================================= Fig. 3 (b)
def ladder_steps(out_pdf=os.path.join(FIG, "ladder_steps.pdf"), out_json=os.path.join(FIG, "ladder_steps.json")):
    """Fig. 3 (b), post hoc, printed at 0.44 \\textwidth: mean translation error against control step 0 to 12, one
    curve each for the static start and the pose oracle and, per stage, for the CT-mask servo base and the final
    policies under both masks, from outputs/v2/posthoc/posthoc_v2.json step_curves (the values of
    tables/step_error.tex). The file must be the one RESULTS.json certifies (sha256), and every curve's step-12 value
    must equal its RESULTS.json Table 4 cell."""
    sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
    import posthoc_v2 as PH
    R = _results()
    J, status = PH.load_checked()
    if J is None or status != "ok" or PH.sha256(PH.OUT_JSON) != (R.get("posthoc") or {}).get("sha256"):
        raise SystemExit(f"ladder_steps: posthoc_v2.json is not the file RESULTS.json certifies ({status})")
    sc, T4 = J["step_curves"], R["table4"]
    curves = {"static": ("static", T4["rungs_mm"]["static"]), "pose_oracle": ("pose_oracle", T4["rungs_mm"]["pose_oracle"])}
    for arm in ("ctmask", "matched"):
        for st in STAGES:
            for ph in ("base", "final"):
                k = f"{arm}/{st}/{ph}"
                curves[k] = (k, T4["cell_means_mm"][k])
    end_tol, end_dev = 1e-4, 0.0    # the per-step trajectories are logged at lower precision than final_pos
    for k, (key, ref) in curves.items():
        y = sc[key]["pos_mean_by_step"]
        end_dev = max(end_dev, abs(y[-1] - ref))
        if len(y) != 13 or abs(y[-1] - ref) > end_tol:
            raise SystemExit(f"ladder_steps: step curve {key} ends at {y[-1]!r}, RESULTS.json Table 4 has {ref!r}")
    base_ct = [sc[f"ctmask/{st}/base"]["pos_mean_by_step"] for st in STAGES]
    if any(np.max(np.abs(np.asarray(b) - np.asarray(base_ct[0]))) > 1e-9 for b in base_ct):
        raise SystemExit("ladder_steps: the CT-mask servo base differs across stages; one curve is not valid")
    plt = _plt()
    x = np.arange(13)
    fig, ax = plt.subplots(figsize=(_tw_in(0.44), 3.35))
    ax.plot(x, sc["static"]["pos_mean_by_step"], color=C_STATIC, lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.plot(x, sc["pose_oracle"]["pos_mean_by_step"], color=C_ORACLE, lw=1.2, zorder=3)
    ax.plot(x, base_ct[0], color=C_CT, lw=1.0, ls=(0, (1.5, 1.2)), zorder=3)
    for st in STAGES:
        for arm, col in (("ctmask", C_CT), ("matched", C_LEARNED)):
            ax.plot(x, sc[f"{arm}/{st}/final"]["pos_mean_by_step"], color=col, lw=0.9, marker=STAGE_MARKER[st],
                    ms=2.6, markevery=2, mfc="white" if st == "physics" else col, mew=0.6, zorder=4)
    # direct labels at the right end, so no legend box hides the curves
    ax.text(0.15, sc["static"]["pos_mean_by_step"][0] + 0.35, "static start", color=C_STATIC, fontsize=6,
            va="bottom", ha="left")
    lab = [[float(np.mean([sc[f"matched/{s}/final"]["pos_mean_by_step"][-1] for s in STAGES[:3]])),
            "learned mask, final,\nthree stages", C_LEARNED],
           [sc["matched/sonospade/final"]["pos_mean_by_step"][-1], "learned mask,\nfinal, SonoSPADE", C_LEARNED],
           [float(np.mean([sc[f"ctmask/{s}/final"]["pos_mean_by_step"][-1] for s in STAGES])),
            "CT mask\u2020, servo\nbase and final", C_CT],
           [sc["pose_oracle"]["pos_mean_by_step"][-1], "pose oracle\u2020", C_ORACLE]]
    lab.sort(key=lambda q: q[0])
    gap = 1.9                                            # mm between two-line labels at this axis height
    for _ in range(50):                                  # spread the labels; the arrows of text are not needed
        for a_, b_ in zip(lab, lab[1:]):
            if b_[0] - a_[0] < gap:
                d_ = 0.5 * (gap - (b_[0] - a_[0]))
                a_[0] -= d_
                b_[0] += d_
    for yv, t, col in lab:
        ax.text(12.45, yv, t, color=col, fontsize=6, va="center", ha="left", linespacing=0.9)
    ax.set_xlim(-0.3, 12.3)
    ax.set_ylim(0, 5.0 * np.ceil((max(max(v["pos_mean_by_step"]) for v in sc.values()) + 0.5) / 5.0))
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("Control step")
    ax.set_ylabel("Mean translation error (mm)")
    _clean(ax)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color="#555555", marker=STAGE_MARKER[s], ms=3, lw=0.8,
                 mfc="white" if s == "physics" else "#555555", mew=0.6, label=STAGE_LABEL[s]) for s in STAGES]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=6,
              handlelength=1.6, columnspacing=0.8, borderaxespad=0.2)
    fig.subplots_adjust(left=0.15, right=0.70, bottom=0.11, top=0.88)
    mins = {k: (sc[k]["pos_min"], sc[k]["pos_min_step"]) for k in curves if k.endswith("/final")}
    n_txt = _save(fig, out_pdf, out_json, {
        "figure": "fig:ladder panel (b), post hoc", "source": _rel(PH.OUT_JSON) + " step_curves",
        "certified_by": "outputs/v2/RESULTS.json posthoc.sha256", "aggregation": "posthoc_v2 pos_mean_by_step: the mean "
        "over the cell's episodes at each step; the crossed design is balanced over patients, so it equals the mean of "
        "patient means; the step-12 value of every curve equals its RESULTS.json table4 cell within 1e-4 mm (checked; the "
        "per-step trajectories are logged at lower precision than final_pos)",
        "curves": {k: sc[key]["pos_mean_by_step"] for k, (key, _) in curves.items()},
        "minima": {k: {"pos_min": v[0], "step": v[1]} for k, v in mins.items()}})
    plt.close(fig)
    print(f"ladder_steps: wrote {_rel(out_pdf)} and {_rel(out_json)}; {len(curves)} curves end at their Table 4 "
          f"cells (largest deviation {end_dev:.2g} mm); minima of the learned-mask finals: " + ", ".join(
              f"{s} {sc[f'matched/{s}/final']['pos_min']:.2f} at step {sc[f'matched/{s}/final']['pos_min_step']}"
              for s in STAGES) + f"; {n_txt} text items passed the style gate")


# ============================================================================================= Fig. 4
FOREST_COND = {"ctrl": ", CT masks†", "reward": ", CT masks†", "renderer": ", CT masks†"}
HOLDS_WORDS = ("higher", "lower")                        # the plans' outcome words of a verdict that holds


def _text_w_in(s, size, weight="normal"):
    """Advance width in inches of a one-line string in the paper font (glyph outlines, no screen hinting)."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    return TextPath((0, 0), s, prop=FontProperties(family="serif", weight=weight, size=size)).get_extents().x1 / 72.0


def _pm(x, nd):
    """Signed number with a typographic minus, as Table 5 prints it."""
    return f"{x:+.{nd}f}".replace("-", "−")


# the drawn name of one renderer row (with the typographic apostrophe, as in Fig. S3.1); plug_rows.json and
# forest.json keep the record label
FOREST_LABEL_PRINT = {"LOTUS, in-repository reimplementation": "LOTUS, authors’ reimplementation"}


def forest(out_pdf=os.path.join(FIG, "forest.pdf"), out_json=os.path.join(FIG, "forest.json"),
           rows_json=os.path.join(FIG, "plug_rows.json"), frac=1.0, pitch=0.115):
    """Fig. 4, printed at 1.0 \\textwidth. One line per
    completed family row of Table 5: the row's own difference from its slot's reference row (paired by seed, mean
    over the six held-out patients) with its 95% paired t CI (thin bar) and 90% paired t CI (thick bar), on one
    millimeter axis whose 0 is each slot's reference. A filled marker means that the post hoc, unadjusted TOST at
    +/- 1 mm passes (the 90% CI lies inside the pre-specified equivalence margin); an open marker
    means that it does not. Dotted lines mark the margin. No band around the reference is drawn any more: the slot's
    post hoc MDD is a statement about the design and stays in the Table 5 slot headers. The left column names the
    slot (with its mask condition where it is the privileged CT mask) and the row; seed-0 rows say so. The right
    column prints the difference and 95% CI as Table 5 does (reward slot to three decimals) and the plans' outcome
    word where it is "higher" or "lower"; slot lines give the reference row's error and the number of completed
    family rows. Every value comes from figures/plug_rows.json (make_tables.py pluggability(), written
    in the run that writes Table 5); each row's 95% and 90% CI and its TOST flag are recomputed from its difference
    and SD (paired t, df = patients - 1) and must match before anything is drawn. Secondary, privileged and pending
    rows stay in Table 5. The figure draws the amendment 1 family rows only: the amendment 2 families (MASK-P, IMN-T,
    LPIPS-R, CTA-D, CTA-E) are never pooled with them and stay in Table 5, so no CT-anatomy line is drawn (the
    earlier "CT anatomy: rows not read in this build" line was drawn whatever plug_rows.json said,
    and the blinding has been lifted)."""
    from scipy import stats
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D
    from usbg.benchmark import TEST_PATIENTS
    if not os.path.isfile(rows_json):
        raise SystemExit(f"forest: {_rel(rows_json)} not found: make_tables.py writes it (run make_tables.py first, "
                         "with the same --out)")
    with open(rows_json) as fh:
        J = json.load(fh)
    R = _results()
    rg = R["table4"]["rungs_mm"]
    if abs(J["oracle_mm"] - rg["pose_oracle"]) > TOL or abs(J["static_mm"] - rg["static"]) > TOL:
        raise SystemExit("forest: plug_rows.json oracle or static differs from RESULTS.json")
    n = len(TEST_PATIENTS)
    t95, t90 = float(stats.t.ppf(0.975, n - 1)), float(stats.t.ppf(0.95, n - 1))
    margin = float(J["tost_margin_mm"])
    lines, data, max_ci_dev = [], [], 0.0
    for SR in J["slots"]:
        fam = [r for r in SR["rows"] if r["kind"] == "row" and r["role"] == "family"]
        done = [r for r in fam if r["err_mm"] is not None and r.get("ci95_t") is not None]
        nd = 3 if SR["key"] == "reward" else 2
        lines.append(("head", SR, done))
        for r in done:
            se = float(r["sd_mm"]) / np.sqrt(n)
            for ci, t in ((r["ci95_t"], t95), (r["ci90_t"], t90)):
                dev = max(abs(ci[0] - (r["diff_mm"] - t * se)), abs(ci[1] - (r["diff_mm"] + t * se)))
                max_ci_dev = max(max_ci_dev, dev)
                if dev > 1e-6:
                    raise SystemExit(f"forest: {SR['key']}/{r['id']} CI does not follow from diff and SD ({dev:.2g})")
            tost = bool(-margin < r["ci90_t"][0] and r["ci90_t"][1] < margin)
            if tost != bool(r["tost_within_1mm"]):
                raise SystemExit(f"forest: {SR['key']}/{r['id']} TOST flag differs from its 90% CI")
            lines.append(("row", SR, r, nd, tost))
            data.append({"slot": SR["key"], "id": r["id"], "label": r["label"], "seeds": r.get("seeds"),
                         "err_mm": r["err_mm"], "diff_mm": r["diff_mm"], "ci95_t": r["ci95_t"], "ci90_t": r["ci90_t"],
                         "k": r["k"], "p_holm": r["p_holm"], "decision": r["decision"], "tost_within_1mm": tost,
                         "p_tost_1mm": r.get("p_tost_1mm"), "provisional": r.get("provisional")})
    cta = J.get("cta") or {}
    lo = min(d["ci95_t"][0] for d in data)
    hi = max(d["ci95_t"][1] for d in data)
    xlo, xhi = float(np.floor(lo - 0.3)), float(np.ceil(hi + 0.3))
    if xlo % 2:
        xlo -= 1.0

    plt = _plt()
    W = _tw_in(frac)
    L_txt, R_txt, gap, g_gap = 2.22, 1.42, 0.08, 0.045       # inches: label column, number column, gaps
    # the two-row legend is dropped and the bottom margin shrinks from 0.66 to 0.36 in;
    # the caption of figures/fig_forest.tex states every marker and line (filled or open marker, thin 95% bar, thick
    # 90% bar, solid reference line, dotted margin lines)
    top, bottom = 0.20, 0.36
    n_head = sum(1 for q in lines if q[0] != "row")
    body = len(lines) * pitch + (n_head - 1) * g_gap
    H = top + body + bottom
    x0, pw = L_txt, W - L_txt - R_txt - gap
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([x0 / W, bottom / H, pw / W, body / H])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(body, 0)
    tl = mtransforms.blended_transform_factory(fig.dpi_scale_trans, ax.transData)    # x in inches, y in body inches
    xr = x0 + pw + gap
    ax.axvline(0.0, color="#333333", lw=0.7, zorder=1)
    for s in (-margin, margin):
        ax.axvline(s, color="#7F7F7F", lw=0.7, ls=(0, (1, 1.6)), zorder=1)
    y, first = 0.0, True
    for q in lines:
        if q[0] != "row" and not first:
            y += g_gap
            ax.plot([0.03, W - 0.03], [y - 0.5 * g_gap] * 2, transform=tl, color="#D9D9D9",
                    lw=0.5, clip_on=False, zorder=0)
        yc = y + 0.5 * pitch
        if q[0] == "head":
            _, SR, done = q
            tb = ax.text(0.03, yc, SR["title"] + FOREST_COND.get(SR["key"], ""), transform=tl, ha="left",
                         va="center", fontsize=7, fontweight="bold")
            wb = _text_w_in(tb.get_text(), 7, "bold")
            # the family size, not a row count
            ax.text(0.03 + wb, yc, f" ($m = {SR['m']}$)" + ("" if len(done) == SR["m"] else
                                                           f"; {len(done)} with results"),
                    transform=tl, ha="left", va="center", fontsize=7, color="#333333")
            ax.text(xr, yc, f"reference {SR['reference']['err_mm']:.2f} mm", transform=tl, ha="left", va="center",
                    fontsize=7, color="#333333")
        elif q[0] == "cta":
            tb = ax.text(0.03, yc, "CT anatomy", transform=tl, ha="left", va="center", fontsize=7, fontweight="bold")
            wb = _text_w_in(tb.get_text(), 7, "bold")
            ax.text(0.03 + wb, yc, ": rows not read in this build" + (" (blinded)" if q[1].get("blinded") else ""),
                    transform=tl, ha="left", va="center", fontsize=7, color="#333333")
        else:
            _, SR, r, nd, tost = q
            # the drawn label goes through FOREST_LABEL_PRINT
            lab = FOREST_LABEL_PRINT.get(r["label"], r["label"]) + (" (seed 0)" if r.get("n_seeds") == 1 else "")
            ax.text(0.15, yc, lab, transform=tl, ha="left", va="center", fontsize=7)
            ax.plot(r["ci95_t"], [yc, yc], color=C_FAMILY, lw=0.8, zorder=2, solid_capstyle="butt")
            ax.plot(r["ci90_t"], [yc, yc], color=C_FAMILY, lw=2.6, zorder=3, solid_capstyle="butt")
            ax.scatter([r["diff_mm"]], [yc], s=15, marker="o", facecolors=C_FAMILY if tost else "white",
                       edgecolors=C_FAMILY, linewidths=0.9, zorder=4)
            num = f"{_pm(r['diff_mm'], nd)} [{_pm(r['ci95_t'][0], nd)}, {_pm(r['ci95_t'][1], nd)}]"
            word = (r["decision"] or "").split(";")[0].strip()
            ax.text(xr, yc, num + (f"  {word}" if word in HOLDS_WORDS else ""), transform=tl, ha="left",
                    va="center", fontsize=7)
        y += pitch
        first = False
    ax.text(xr, -0.06, "Difference [95% CI] (mm)", transform=tl, ha="left", va="bottom", fontsize=7,
            fontstyle="italic", clip_on=False)
    ax.set_xticks(np.arange(xlo, xhi + 0.5, 2.0))
    ax.set_yticks([])
    # the Box 1 term "reference component", as the Fig. B.1 caption prints it
    ax.set_xlabel("Difference from the slot’s reference component (mm)", fontsize=7, labelpad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#EDEDED", lw=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=2, pad=1.5, labelsize=7)
    # legend removed: the caption describes the markers and lines
    n_txt = _save(fig, out_pdf, out_json, {
        "figure": PI.text("forest figure"),
        "source": _rel(rows_json), "status": J.get("status"), "provisional": J.get("provisional"),
        "blinded_amendment2": J.get("blinded_amendment2"), "n_patients": n, "t_df": n - 1,
        "t95": t95, "t90": t90, "tost_margin_mm": margin, "tost_margin_src": J.get("tost_margin_src"),
        "ci_check": f"every row's ci95_t and ci90_t equal diff_mm -/+ t * sd_mm / sqrt({n}) (max deviation "
                    f"{max_ci_dev:.2g} mm); tost_within_1mm equals (90% CI inside +/- margin)",
        PI.key("posthoc_rule"): J.get(PI.key("posthoc_rule")), "x_range_mm": [xlo, xhi],
        "mdd_not_drawn": "the post hoc slot MDDs (plug_rows.json slots[*].mdd) are printed in the Table 5 slot headers",
        "slots": [{"slot": SR["key"], "title": SR["title"], "m": SR["m"],
                   "reference": SR["reference"], "pending": SR.get("pending"),
                   "groups_pending": SR.get("groups_pending")} for SR in J["slots"]],
        "cta": {"blinded": cta.get("blinded")}, "rows": data}, min_pt=7.0)
    plt.close(fig)
    print(f"forest: wrote {_rel(out_pdf)} ({W:.3f} x {H:.3f} in) and {_rel(out_json)} from {_rel(rows_json)}; "
          f"{len(data)} family rows, {sum(d['tost_within_1mm'] for d in data)} pass the post hoc TOST; x "
          f"{xlo:g} to {xhi:g} mm; CI check max deviation {max_ci_dev:.2g} mm; {n_txt} text items passed the "
          "style and 7 pt gates")


# ============================================================================================= Fig. 5
# Each Fig. 5 panel is authored at the width of its minipage in figures/fig_checks.tex (0.325 \textwidth) and included
# at width 1:1, so the 7 pt gate of _save is the printed size.
FIG5_FRAC, FIG5_H = 0.325, 3.0


def checks_sb3(out_pdf=os.path.join(FIG, "checks_sb3.pdf"), out_json=os.path.join(FIG, "checks_sb3.json"),
               frac=FIG5_FRAC, height=FIG5_H):
    """Fig. 5 (a), printed at 0.325 \\textwidth, every font 7 pt or larger. Top: SB3 PPO on the privileged easy
    variant (TRAIN patients), final translation error on the TRAIN-internal evaluation episodes against environment
    steps, three seeds, with lines at the pass bound (pose oracle plus the margin), the pose oracle and the static
    start, from outputs/v2_1/checks/sb3pos/ppo_poseobs_s{0,1,2}.json curves.heldout_train_internal (the 57,600-step
    point equals verdict.json learner_mm). Bottom: the REINFORCE control, the fraction of the planted lateral deficit
    that remains on the held-out patients against RL iteration, for the three settings, with the pass line at
    one half, from outputs/v2_1/rlpos/rlpos_summary.json cells[deficit=bias05].metrics.final_dlat_mm.curve (the
    endpoint equals one minus recovered.fraction). The two controls use different patients and metrics, so they
    share no axis. The top axis is padded below 0 so that the pose-oracle label sits under its line; its left spine
    starts at 0."""
    vd = json.load(open(os.path.join(V21, "checks", "sb3pos", "verdict.json")))
    seeds = []
    for x in vd["seeds"]:
        p = os.path.join(V21, "checks", "sb3pos", os.path.basename(x["path"]))
        d = json.load(open(p))
        cur = d["curves"]["heldout_train_internal"]
        if abs(cur[-1]["final_pos_mm"] - x["learner_mm"]) > 1e-9 or cur[-1]["env_steps"] != d["budget"]["env_steps"]:
            raise SystemExit(f"checks_sb3: {_rel(p)} curve endpoint differs from verdict.json")
        seeds.append({"seed": x["seed"], "steps": [c["env_steps"] for c in cur], "mm": [c["final_pos_mm"] for c in cur],
                      "oracle": x["oracle_mm"], "static": x["static_mm"], "margin": d["pass_rule"]["margin_mm"],
                      "file": _rel(p)})
    om, sm, mg = seeds[0]["oracle"], seeds[0]["static"], seeds[0]["margin"]
    if any(abs(s["oracle"] - om) > 1e-9 or abs(s["static"] - sm) > 1e-9 or s["margin"] != mg for s in seeds):
        raise SystemExit("checks_sb3: the seeds do not share oracle, static and margin")
    rl = json.load(open(os.path.join(V21, "rlpos", "rlpos_summary.json")))
    cells = [c for c in rl["cells"] if c["deficit"] == "bias05"]
    # the drawn setting names ("shipped" prints as "default")
    var_lab = {"shipped": "default settings", "explore": "learned exploration", "reward": "pose shaping†"}
    var_col = {"shipped": "#000000", "explore": "#009E73", "reward": "#CC79A7"}
    rlc = []
    for c in cells:
        m = c["metrics"]["final_dlat_mm"]
        it = [q["iter"] for q in m["curve"]]
        rem = [q["gap_frac_remaining"] for q in m["curve"]]
        if it[-1] != c["final_iter"] or abs(rem[-1] - (1.0 - m["recovered"]["fraction"])) > 1e-6:
            raise SystemExit(f"checks_sb3: REINFORCE {c['variant']} curve endpoint differs from its recovered fraction")
        rlc.append({"variant": c["variant"], "iters": it, "remaining": rem, "verdict": c["verdict"]})
    plt = _plt()
    FS = 7.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(_tw_in(frac), height), gridspec_kw={"height_ratios": [1.2, 1.0]})
    for s, mk in zip(seeds, SEED_MARK):
        a1.plot(np.asarray(s["steps"]) / 1000.0, s["mm"], color=C_FAMILY, lw=0.9, marker=mk, ms=3.0, mfc="white",
                mew=0.7, zorder=3, label=f"seed {s['seed']}")
    xr = seeds[0]["steps"][-1] / 1000.0
    ytop = 5.0 * np.ceil((sm + 1.5) / 5.0)
    a1.axhline(sm, color=C_STATIC, lw=0.8, ls=(0, (4, 2)), zorder=2)
    a1.text(1.0, sm + 0.4, f"static start {sm:.2f} mm", color=C_STATIC, fontsize=FS, va="bottom", ha="left")
    a1.axhline(om + mg, color=C_LEARNED, lw=0.8, ls=(0, (2, 1.5)), zorder=2)
    a1.text(xr, om + mg + 0.4, f"pass bound {om + mg:.2f} mm", color=C_LEARNED, fontsize=FS, va="bottom", ha="right")
    a1.axhline(om, color=C_ORACLE, lw=0.8, zorder=2)
    a1.text(xr, om - 0.45, f"pose oracle† {om:.2f} mm", color=C_ORACLE, fontsize=FS, va="top", ha="right")
    a1.set_xlim(0, xr + 2)
    a1.set_ylim(-0.17 * ytop, ytop)
    a1.set_yticks(np.arange(0, ytop + 0.1, 5.0))
    a1.spines["left"].set_bounds(0, ytop)
    a1.set_title("SB3 PPO, privileged easy variant†", fontsize=FS, pad=2)
    a1.set_xlabel("Environment steps (thousands)", fontsize=FS, labelpad=1)
    a1.set_ylabel("Final translation error (mm)", fontsize=FS, labelpad=2)
    a1.legend(loc="center right", bbox_to_anchor=(1.0, 0.55), frameon=False, fontsize=FS, handlelength=1.5,
              borderaxespad=0.2, labelspacing=0.25)
    _clean(a1)
    for r in rlc:
        a2.plot(r["iters"], r["remaining"], color=var_col[r["variant"]], lw=0.9, marker="o", ms=2.4, zorder=3,
                label=var_lab[r["variant"]])
    a2.axhline(0.5, color=C_LEARNED, lw=0.8, ls=(0, (2, 1.5)), zorder=2)
    a2.text(4, 0.53, "pass: half removed", color=C_LEARNED, fontsize=FS, va="bottom", ha="left")
    a2.set_ylim(0, 1.05)
    a2.set_yticks([0.0, 0.5, 1.0])
    a2.set_xlim(-8, cells[0]["final_iter"] + 8)
    a2.set_title("REINFORCE, planted lateral deficit", fontsize=FS, pad=2)
    a2.set_xlabel("Iteration", fontsize=FS, labelpad=1)
    a2.set_ylabel("Deficit left (fraction)", fontsize=FS, labelpad=2)
    a2.legend(loc="lower left", frameon=False, fontsize=FS, handlelength=1.5, borderaxespad=0.15, labelspacing=0.25)
    _clean(a2)
    for ax in (a1, a2):
        ax.tick_params(labelsize=FS)
    fig.subplots_adjust(left=0.205, right=0.975, bottom=0.085, top=0.955, hspace=0.50)
    n_txt = _save(fig, out_pdf, out_json, {
        "figure": "fig:checks panel (a)", "sb3": seeds, "sb3_verdict": vd["verdict"],
        "sb3_source": "outputs/v2_1/checks/sb3pos/verdict.json and ppo_poseobs_s{0,1,2}.json curves.heldout_train_internal",
        "reinforce": rlc, "reinforce_source": "outputs/v2_1/rlpos/rlpos_summary.json cells[deficit=bias05].metrics."
        "final_dlat_mm.curve.gap_frac_remaining", "reinforce_overall": rl["overall"]["key"]}, min_pt=FS)
    plt.close(fig)
    print(f"checks_sb3: wrote {_rel(out_pdf)} and {_rel(out_json)}; SB3 finals " + ", ".join(
        f"{s['mm'][-1]:.2f}" for s in seeds) + f" mm against bound {om + mg:.2f}; REINFORCE remaining " + ", ".join(
        f"{r['variant']} {r['remaining'][-1]:.2f}" for r in rlc) + f"; {n_txt} text items passed the style and 7 pt gates")


def checks_cohort(out_pdf=os.path.join(FIG, "checks_cohort.pdf"), out_json=os.path.join(FIG, "checks_cohort.json"),
                  frac=FIG5_FRAC, height=FIG5_H):
    """Fig. 5 (b), printed at 0.325 \\textwidth, every font 7 pt or larger: per-patient differences of the two cohort
    contrasts, the physics CT-to-learned-mask gap and SonoSPADE minus physics at the learned-mask servo base, for the
    original six and the 20 cohort CTs, with the mean (vertical tick) and its 95% t interval (bar above the dots),
    from outputs/v2_1/cohort/analysis_q2r.json (primary_cohort and reference_original_six). The original six's
    physics gap is checked against RESULTS.json table4.perception_gap. Each contrast has its own title line, and the
    zero line is drawn through the strips only, so no title crosses a line."""
    sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
    import posthoc_v2 as PH
    import matplotlib.transforms as mtransforms
    co = json.load(open(os.path.join(V21, "cohort", "analysis_q2r.json")))
    R = _results()
    keys = (("C1_physics_perception_gap", "CT-to-learned-mask gap,\nphysics stage"),
            ("C2_sonospade_minus_physics_learned_base", "SonoSPADE minus physics,\nlearned-mask servo base"))
    if abs(co["reference_original_six"][keys[0][0]]["mean"] - R["table4"]["perception_gap"]["physics"]["mean"]) > 1e-3:
        raise SystemExit("checks_cohort: the original six's physics gap differs from RESULTS.json")
    plt = _plt()
    FS = 7.0
    W = _tw_in(frac)
    fig, ax = plt.subplots(figsize=(W, height))
    rng = np.random.default_rng(0)                     # fixed vertical jitter, display only
    data, yt, yl, heads = [], [], [], []
    y = 0.0
    for k, lab in keys:
        heads.append((y, lab))
        y += 1.05
        for grp, blk, col, nm in (("six", co["reference_original_six"][k], C_SIX, "original six"),
                                  ("cohort", co["primary_cohort"][k], C_COHORT, "20 cohort CTs")):
            v = np.array(list(blk["per_patient"].values()), float)
            ts = PH.t_stats(v)
            if abs(float(v.mean()) - blk["mean"]) > 1e-3 or abs(ts["ci_t"][0] - blk["ci95_t"][0]) > 1e-3:
                raise SystemExit(f"checks_cohort: {k} {grp} per-patient values do not reproduce the stored mean and CI")
            jit = rng.uniform(-0.14, 0.14, len(v))
            ax.scatter(v, y + 0.08 + jit, s=9, color=col, alpha=0.55 if grp == "cohort" else 0.85, linewidths=0,
                       zorder=3)
            ax.plot(blk["ci95_t"], [y - 0.30] * 2, color=col, lw=1.2, zorder=4, solid_capstyle="butt")
            ax.plot([blk["mean"]] * 2, [y - 0.42, y - 0.18], color=col, lw=1.6, zorder=5)
            ax.plot([0.0, 0.0], [y - 0.5, y + 0.35], color="#555555", lw=0.6, zorder=1)
            yt.append(y)
            yl.append(f"{nm}\n{blk['sign_consistency']} same sign")
            data.append({"contrast": k, "group": grp, "n": blk["n"], "per_patient": blk["per_patient"],
                         "mean": blk["mean"], "ci95_t": blk["ci95_t"], "sign_consistency": blk["sign_consistency"]})
            y += 1.0
        y += 0.1
    lo = min(min(min(d["per_patient"].values()), d["ci95_t"][0]) for d in data)
    hi = max(max(max(d["per_patient"].values()), d["ci95_t"][1]) for d in data)
    ax.set_xlim(np.floor(lo) - 0.5, np.ceil(hi) + 0.5)
    ax.set_ylim(y - 0.45, -0.55)
    tr = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    for yh, lab in heads:
        ax.text(0.02, yh, lab, transform=tr, fontsize=FS, va="center", ha="left", linespacing=1.0,
                fontweight="bold")
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=FS, linespacing=1.0)
    ax.set_xlabel("Per-patient difference (mm)", fontsize=FS, labelpad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2, pad=1.5, labelsize=FS)
    fig.subplots_adjust(left=0.37, right=0.975, bottom=0.085, top=0.985)
    n_txt = _save(fig, out_pdf, out_json, {"figure": "fig:checks panel (b)", "source":
                                           "outputs/v2_1/cohort/analysis_q2r.json primary_cohort and "
                                           "reference_original_six", "analysis_status": co["analysis_status"],
                                           "groups": data}, min_pt=FS)
    plt.close(fig)
    print(f"checks_cohort: wrote {_rel(out_pdf)} and {_rel(out_json)}; " + "; ".join(
        f"{d['contrast'].split('_')[0]} {d['group']} {d['mean']:+.2f} [{d['ci95_t'][0]:+.2f}, {d['ci95_t'][1]:+.2f}], "
        f"{d['sign_consistency']}" for d in data) + f"; {n_txt} text items passed the style and 7 pt gates")


def checks_horizon(out_pdf=os.path.join(FIG, "checks_horizon.pdf"), out_json=os.path.join(FIG, "checks_horizon.json"),
                   frac=FIG5_FRAC, height=FIG5_H):
    """Fig. 5 (c), printed at 0.325 \\textwidth, every font 7 pt or larger (exploratory, v2.1): mean translation
    error against control step 0 to 36, one curve each for the static start, the pose oracle, the servo base under CT
    masks and the four servo bases under learned masks, contact on, with step 12 (the evaluation horizon) marked.
    Source: the v2.1 36-step arms outputs/v2_1/s21_h36/*_n36.json and their .traj.npz sidecars, read and matched episode
    for episode through posthoc_logs.py of the analysis scripts (blocks_with_traj, pick; the reader refuses the blinded
    amendment-2 paths), seeds pooled; the value at each step is the patient-level mean (mean of per-patient means,
    posthoc_logs patient_means and level_mean). Checks before anything is drawn: steps 0, 12 and 36 equal
    outputs/v2_1/analysis_plan/posthoc_logs.json m8_horizon.arms36.<arm>.steps.t{0,12,36}.pos_mm, and steps 0 to 12
    equal the archived step curves of outputs/v2/posthoc/posthoc_v2.json (the file RESULTS.json certifies) within 1e-3
    mm. No episode is run and nothing renders."""
    sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
    sys.path.insert(0, SD.analysis_dir())
    import posthoc_v2 as PH
    import posthoc_logs as PL
    from matplotlib.lines import Line2D
    R = _results()
    J, status = PH.load_checked()
    if J is None or status != "ok" or PH.sha256(PH.OUT_JSON) != (R.get("posthoc") or {}).get("sha256"):
        raise SystemExit(f"checks_horizon: posthoc_v2.json is not the file RESULTS.json certifies ({status})")
    sc = J["step_curves"]
    plj = os.path.join(V21, "analysis_plan", "posthoc_logs.json")
    with open(plj) as fh:
        HZ = json.load(fh)["m8_horizon"]["arms36"]
    arch = {"static": "static", "pose_oracle": "pose_oracle", "ct_base": "ctmask/physics/base"}
    arch.update({f"learned_base_{st}": f"matched/{st}/base" for st in STAGES})
    files = PL.h36_files()
    curves, checks, dev12, dev_snap = {}, {}, 0.0, 0.0
    for key in arch:
        sel = [PL.pick(PL.blocks_with_traj(p)[1], "on") for p in files[key]]
        pats = np.concatenate([np.asarray(tr["patient"]) for _, tr in sel])
        pos = np.concatenate([tr["pos"] for _, tr in sel]).astype(np.float64)
        if pos.shape[1] != 37:
            raise SystemExit(f"checks_horizon: {key} has {pos.shape[1]} logged steps, expected 37")
        cv = [PL.level_mean(PL.patient_means(pos[:, t], pats)) for t in range(pos.shape[1])]
        for t in (0, 12, 36):
            ref = HZ[key]["steps"][f"t{t}"]["pos_mm"]
            dev_snap = max(dev_snap, abs(cv[t] - ref))
            if abs(cv[t] - ref) > 1e-9:
                raise SystemExit(f"checks_horizon: {key} step {t} {cv[t]!r} != posthoc_logs.json {ref!r}")
        a12 = sc[arch[key]]["pos_mean_by_step"]
        d12 = float(np.max(np.abs(np.asarray(cv[:13]) - np.asarray(a12))))
        dev12 = max(dev12, d12)
        if len(a12) != 13 or d12 > 1e-3:
            raise SystemExit(f"checks_horizon: {key} steps 0 to 12 differ from posthoc_v2.json {arch[key]} by {d12:.3g} mm")
        curves[key] = cv
        checks[key] = {"files": [_rel(p) for p in files[key]], "n_episodes": int(len(pats)),
                       "archived_curve": arch[key], "max_abs_diff_steps_0_12_mm": d12}
    plt = _plt()
    FS = 7.0
    W = _tw_in(frac)
    fig, ax = plt.subplots(figsize=(W, height))
    x = np.arange(37)
    ax.plot(x, curves["static"], color=C_STATIC, lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.plot(x, curves["pose_oracle"], color=C_ORACLE, lw=1.2, zorder=3)
    ax.plot(x, curves["ct_base"], color=C_CT, lw=1.2, zorder=3)
    for st in STAGES:
        ax.plot(x, curves[f"learned_base_{st}"], color=C_LEARNED, lw=0.9, marker=STAGE_MARKER[st], ms=3.0,
                markevery=6, mfc="white" if st == "physics" else C_LEARNED, mew=0.6, zorder=4)
    ytop = 5.0 * np.ceil((max(max(v) for v in curves.values()) + 0.5) / 5.0)
    ax.axvline(12, color="#555555", lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    ax.text(12.6, ytop * 0.985, "step 12", fontsize=FS, color="#333333", ha="left", va="top")
    ax.set_xlim(-0.5, 36.5)
    ax.set_ylim(0, ytop)
    ax.set_xticks(range(0, 37, 6))
    ax.set_xlabel("Control step", fontsize=FS, labelpad=1)
    ax.set_ylabel("Mean translation error (mm)", fontsize=FS, labelpad=2)
    _clean(ax)
    ax.tick_params(labelsize=FS)
    hs = [Line2D([], [], color=C_ORACLE, lw=1.2, label="pose oracle†"),
          Line2D([], [], color=C_CT, lw=1.2, label="CT-mask base†"),
          Line2D([], [], color=C_LEARNED, lw=0.9, label="learned-mask base"),
          Line2D([], [], color=C_STATIC, lw=1.0, ls=(0, (4, 2)), label="static start")]
    hs += [Line2D([], [], color=C_LEARNED, marker=STAGE_MARKER[s], ms=3.0, lw=0,
                  mfc="white" if s == "physics" else C_LEARNED, mew=0.6, label=STAGE_LABEL[s]) for s in STAGES]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=FS,
               handlelength=1.5, columnspacing=0.6, labelspacing=0.2, borderaxespad=0.1, handletextpad=0.4)
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.085, top=0.80)
    n_txt = _save(fig, out_pdf, out_json, {
        "figure": "fig:checks panel (c), exploratory v2.1 (36-step arms)",
        "source": PI.text("horizon source"),
        "aggregation": "patient-level mean (mean of per-patient means) of the translation error at each step",
        "checked_against": {"posthoc_logs_json": _rel(plj) + " m8_horizon.arms36.<arm>.steps.t{0,12,36}.pos_mm "
                            f"(max abs diff {dev_snap:.2g} mm)",
                            "posthoc_v2_json": "outputs/v2/posthoc/posthoc_v2.json step_curves steps 0 to 12 "
                            f"(max abs diff {dev12:.2g} mm; certified by RESULTS.json posthoc.sha256)"},
        "curves": curves, "arms": checks}, min_pt=FS)
    plt.close(fig)
    print(f"checks_horizon: wrote {_rel(out_pdf)} and {_rel(out_json)}; step 12 -> 36: " + ", ".join(
        f"{k} {curves[k][12]:.2f} -> {curves[k][36]:.2f}" for k in arch) + f"; steps 0-12 match posthoc_v2.json within "
        f"{dev12:.2g} mm; {n_txt} text items passed the style and 7 pt gates")


# ============================================================================================= Fig. 2 twin (supplement)
TWIN_RENDERERS = (("physics", "Physics\n(authors\u2019)"), ("lotus_reimpl", "LOTUS, authors\u2019\nreimplementation"),
                  ("lotus_orig", "LOTUS, original code,\ndefault parameters"), ("simus", "SIMUS\n(PyMUST)"))
# (the SIMUS label names no device; the reimplementation is the authors', as in the Fig. S3.1 caption)


def stages_twin(out_png=os.path.join(FIG, "stages_twin.png"), out_json=os.path.join(FIG, "stages_twin.json"),
                frac=0.80, stages_json=os.path.join(FIG, "stages.json")):
    """Supplementary twin of Fig. 2, printed at 0.80 \\textwidth, every font
    7 pt or larger: the same simulated views under the renderer slot's simulators, from frames already on disk; no
    renderer, simulator or network runs. Frames: outputs/v2_1/plug/renderer/realism/torch_frames.npz (physics, the
    in-repository LOTUS reimplementation, the original LOTUS code with its published default parameters) and
    simus_frames.npz (physics, SIMUS through PyMUST on the CPU), the uint8 sector frames that the v2.1 realism runs
    torch.json and simus.json scored (Table A.4). Both files must list the same views and hold bit-identical physics
    frames. Views: the held-out patient of Fig. 2 (figures/stages.json case) and its first liver view and first
    kidney view in the files' view order (a jittered view near screened window 0 under the realism protocol; the rule
    reads no image). No CT label is stored with these frames, so the twin has no label column. Panels are cropped to
    the union of non-black content of all panels (> 4/255, padded by 2 px). The label-painted stage and a learned-mask
    overlay are not drawn: no frame or mask of either is stored on disk. In public mode it needs what the stages
    figure needs (its case comes from figures/stages.json)."""
    if SD.public() and _needs(_texture_ckpts(), (out_png, out_json),
                              "stages_twin (its case comes from figures/stages.json)"):
        return
    rdir = os.path.join(V21, "plug", "renderer", "realism")
    paths = {"torch": os.path.join(rdir, "torch_frames.npz"), "simus": os.path.join(rdir, "simus_frames.npz")}
    meta = {}
    for k in paths:
        with open(os.path.join(rdir, f"{k}.json")) as fh:
            meta[k] = json.load(fh)
        if os.path.basename(meta[k]["frames_npz"]) != os.path.basename(paths[k]):
            raise SystemExit(f"stages_twin: {k}.json names another frames file ({meta[k]['frames_npz']})")
    zt, zs = np.load(paths["torch"]), np.load(paths["simus"])
    if not np.array_equal(zt["groups"], zs["groups"]) or not np.array_equal(zt["physics"], zs["physics"]):
        raise SystemExit("stages_twin: the two frame files do not hold the same views")
    with open(stages_json) as fh:
        case = json.load(fh)["case"]
    if case not in meta["torch"]["cases"] or case not in meta["simus"]["cases"]:
        raise SystemExit(f"stages_twin: {case} is not a realism case")
    groups = [str(g) for g in zt["groups"]]
    views = []
    for organ in ("liver", "kidney"):
        i = next(j for j, g in enumerate(groups) if g.startswith(f"{case}:{organ}"))
        views.append({"organ": organ, "index": i, "group": groups[i]})
    frames = {"physics": zt["physics"], "lotus_reimpl": zt["lotus_reimpl"], "lotus_orig": zt["lotus_orig"],
              "simus": zs["simus"]}
    content = np.zeros(frames["physics"].shape[1:], bool)
    for v in views:
        for k in frames:
            content |= frames[k][v["index"]] > 4
    ys, xs = np.where(content)
    pad = 2
    y0, y1 = max(0, ys.min() - pad), min(content.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(content.shape[1], xs.max() + pad + 1)
    plt = _plt()
    FS = 7.0
    W = _tw_in(frac)
    lab_w, gap, head = 0.20, 0.04, 0.30
    nc = len(TWIN_RENDERERS)
    pw = (W - lab_w - (nc - 1) * gap) / nc
    ph = pw * (y1 - y0) / (x1 - x0)
    H = head + len(views) * ph + (len(views) - 1) * gap + 0.02
    fig = plt.figure(figsize=(W, H))
    for i, v in enumerate(views):
        for j, (k, title) in enumerate(TWIN_RENDERERS):
            ax = fig.add_axes([(lab_w + j * (pw + gap)) / W, 1 - (head + i * (ph + gap) + ph) / H, pw / W, ph / H])
            ax.imshow(frames[k][v["index"]][y0:y1, x0:x1] / 255.0, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=FS, pad=2, linespacing=1.0)
            if j == 0:
                ax.text(-0.04, 0.5, f"{v['organ']} view", transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=FS)
    n_txt = _text_gate(fig, out_png, min_pt=FS)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_pub({"figure": PI.text("stages twin figure"), "status": "exploratory, v2.1 pluggability",
                   "frames": {k: {"path": _rel(p), "md5": _md5(p)} for k, p in paths.items()},
                   "realism_json": {k: {"path": _rel(os.path.join(rdir, f"{k}.json")),
                                        "frames_npz": meta[k]["frames_npz"], "protocol": meta[k]["protocol"],
                                        "jitter": meta[k]["jitter"], "seat_mode": meta[k]["seat_mode"],
                                        "render_device": meta[k]["render_device"],
                                        "citations": meta[k]["citations"]} for k in paths},
                   "case": case, "case_rule": "the held-out patient of Fig. 2 (figures/stages.json case)",
                   "view_rule": "first liver view and first kidney view of that patient in the files' view order",
                   "views": views, "columns": [k for k, _ in TWIN_RENDERERS],
                   "physics_identical_across_files": True,
                   "crop_rows_cols": [[int(y0), int(y1)], [int(x0), int(x1)]],
                   "crop_rule": "union over the shown panels of pixels > 4/255, padded by 2 px",
                   "not_drawn": {"label_painted_stage": "no painted frame is stored on disk (only the checkpoint)",
                                 "learned_mask_overlay": "no stored learned-mask prediction of a simulated frame",
                                 "real_frame": PI.text("twin real frame")},
                   "printed_size_in": [W, H], "min_font_pt": FS, "image_dpi": 600}), fh, indent=1)
    print(f"stages_twin: wrote {_rel(out_png)} ({W:.3f} x {H:.3f} in) and {_rel(out_json)}; {case}, views "
          + ", ".join(f"{v['group']} (#{v['index']})" for v in views) + f"; {n_txt} text items passed the gates")


FIGS = {"ladder": ladder, "stages": stages, "landscape": landscape, "layout": layout, "scene": scene,
        "ladder_steps": ladder_steps, "forest": forest, "checks_sb3": checks_sb3, "checks_cohort": checks_cohort,
        "checks_horizon": checks_horizon, "stages_twin": stages_twin}
# the figures of the article that read result files only (no rendering); "draft" runs these
DRAFT = ("ladder", "ladder_steps", "forest", "checks_sb3", "checks_cohort", "checks_horizon", "stages_twin")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", choices=list(FIGS) + ["draft"],
                    help="default: all; draft: the article's figures that read result files only")
    ap.add_argument("--public", action="store_true", help="public mode: result files under --results ROOT, figures "
                    "to --out DIR/figures (see the module docstring)")
    ap.add_argument("--results", help="public mode: the result root (outputs/, data/, segmenter/outputs/)")
    ap.add_argument("--out", help="public mode: the output root DIR")
    a = ap.parse_args(argv)
    if a.out and not a.public:
        raise SystemExit("--out DIR is an option of public mode: add --public --results ROOT")
    if a.public and not SD.public():
        raise SystemExit("--public: start this script from its command line (results_world.start reads --public, "
                         "--results and --out before any data path is computed)")
    which = [x for w in (a.which or list(FIGS)) for x in (DRAFT if w == "draft" else (w,))]
    for w in dict.fromkeys(which):
        if SD.public():     # the NEEDS.tsv lines of an earlier run of this figure (outputs figures/<name>.<ext>)
            import results_world
            results_world.clear_needs(f"figures/{w}.")
        FIGS[w]()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PI.PlanInputsError as e:      # a missing or incomplete public_inputs/plan_constants.json: one line
        raise SystemExit(f"plan inputs: {e}")
