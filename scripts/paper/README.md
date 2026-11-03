# Rebuilding the article's tables and figures

The scripts in this folder build every table and figure data file that the article and its supplement print, from
the result files of the public results archive. The analysis scripts they import are in `scripts/analysis/`, the
shared helpers in `scripts/`, the testbed package in `src/usbg/`.

## The results archive (ROOT)

The result files are archived on Zenodo as a separate results archive, doi:10.5281/zenodo.23913142
(https://doi.org/10.5281/zenodo.23913142, the file `usbgym_results_v1.zip`). Unzip it; ROOT is the folder
`usbgym_results` it holds. Verify it first, from inside that folder:

    shasum -a 256 -c SHA256SUMS        (or: sha256sum -c SHA256SUMS)

`reproduce.py` checks every file that `MANIFEST.tsv` lists (bytes and sha256) before it builds, and each step checks
every result file it opens or probes against the same file, so `MANIFEST.tsv` itself must be the archive's: its sha256
and number of rows are recorded in `public_inputs/results_archive.json`, and `reproduce.py` prints whether ROOT holds
that file before it builds. The layout:

    ROOT/outputs/...             result files of the testbed runs
    ROOT/data/...                the screened acoustic windows and the CT label caches the build reads
    ROOT/segmenter/outputs/...   result files of the segmenter runs
    ROOT/MANIFEST.tsv            one line per file: path relative to ROOT, bytes, sha256, role and the printed
                                 tables and figures it feeds
    ROOT/EXCLUDED.tsv            inputs of the build that the archive leaves out: path, bytes, sha256, reason and
                                 how to obtain or regenerate each

The records inside the archive give paths relative to the repository roots (`outputs/...`, `data/...`,
`segmenter/...`); the build resolves them against ROOT. The digests that one record holds of another file cite the
archived file, and the build checks them as the development build does. Strings in which a record quoted plan wording
read `[plan wording withheld]` in the archive (its `SANITIZATION.md` lists every rewrite); no printed table or figure
data file depends on them.

Two seed-0 evaluation files of SAMUS are withheld (their real-frame Dice is not reported). The build only checks that
they exist: in public mode it treats them as present without reading them and prints "withheld" in their place
(Table S3.1), so nothing has to be placed for them.

## Items that need inputs left out of the archive

Four inputs are not in the archive because of third-party terms (EXCLUDED.tsv). Without them the build makes every
other table and figure, prints one notice per item that needs them, and `reproduce.py` reports the item's expected
file as `NEEDS <input> (EXCLUDED.tsv)` instead of comparing it:

| Item | Files | Needs |
| :-- | :-- | :-- |
| Fig. A.1 | `figures/stages.png`, `figures/stages.json` | the texture-stage checkpoints `data/dataset/v2/cut_curvi.pt`, `cyclegan_curvi.pt` and `sonospade_curvi.pt` |
| Fig. S3.1 | `figures/stages_twin.png`, `figures/stages_twin.json` | the same three checkpoints (it takes its patient from `figures/stages.json`) |
| Fig. A.3 | `figures/layout.pdf`, `figures/layout.json` | the annotated real test frames `data/eval/kaggle_vitale_test.npz` |

To build these items, obtain or regenerate the files as EXCLUDED.tsv describes and place each at its path under ROOT
(for example `ROOT/data/eval/kaggle_vitale_test.npz`). The build checks a placed file against its EXCLUDED.tsv row
(bytes and sha256) before it reads it and stops on a mismatch; a retrained checkpoint has other bytes than the
authors' file, so it is refused. In the figure data files the md5 of such an input reads `see EXCLUDED.tsv`, and the
md5 of a result file of the archive reads `see MANIFEST.tsv`.

## Environment

`requirements-reproduce.txt` pins the versions that wrote the expected files. From the repository root:

    pip install -e .
    pip install -r scripts/paper/requirements-reproduce.txt

The tests below that rebuild from the archive need this environment as well (they are skipped without it).

## Rebuild and compare

From the repository root:

    PYTHONPATH=src python scripts/paper/reproduce.py --results ROOT --out DIR

DIR is a new or earlier output folder outside ROOT (the build writes nothing under ROOT). `reproduce.py` runs four steps
in public mode, one process each (`make_tables.py`, `best_so_far.py table`, `posthoc_logs.py tables`,
`make_figures.py`), then compares every file of `public_inputs/expected/` with DIR and prints one PASS, FAIL or NEEDS
line per file. Exit codes:

| Code | Meaning |
| --: | :-- |
| 0 | no file fails (NEEDS lines included) and ROOT holds the archive's `MANIFEST.tsv` |
| 1 | a listed file differs from `MANIFEST.tsv`, a step stops, or a file fails |
| 2 | a usage error: an unknown option, a missing `--results` or `--out`, or an `--out` inside ROOT |
| 3 | no file fails, but ROOT's `MANIFEST.tsv` is not the archive's (`--own-results` makes this 0, for a result root of your own) |

Each step also runs alone, for example:

    PYTHONPATH=src python scripts/paper/make_tables.py --public --results ROOT --out DIR
    PYTHONPATH=src python scripts/analysis/best_so_far.py table --public --results ROOT --out DIR
    PYTHONPATH=src python scripts/analysis/posthoc_logs.py tables --public --results ROOT --out DIR
    PYTHONPATH=src python scripts/paper/make_figures.py --public --results ROOT --out DIR

(`make_figures.py` reads `DIR/figures/plug_rows.json`, which `make_tables.py` writes.) Then compare without building:

    PYTHONPATH=src python scripts/paper/reproduce.py --results ROOT --out DIR --compare-only

A figure that a step lists in `DIR/NEEDS.tsv` reads NEEDS only while its file is not in DIR: a later run that builds
it (with the input placed) removes its NEEDS lines, and a file in DIR is always compared.

In public mode a step reads the code tree, ROOT and DIR only. Every result file it opens or probes is checked against
`ROOT/MANIFEST.tsv` (a file the manifest does not list does not exist for the build). Where the development build
checks a file against a digest held in its own code (the records ledger, pinned records), public mode checks it
against its MANIFEST.tsv row instead. The pre-specified constants come from `public_inputs/plan_constants.json`; each
check that needs the development repository (its pre-specified plan documents and the article's sources) is skipped with
one notice on stderr; the tables are written with every LaTeX comment removed. For a result root of your own, write
its manifest first, then its `EXCLUDED.tsv`. Copy the archive's `EXCLUDED.tsv` when your root lacks the inputs it lists
(the three texture-stage checkpoints and the real test frames): their figures then read NEEDS. Write the header line
only (the last command) when your root holds every input; an input that is neither in ROOT nor listed stops the build:

    python scripts/results_world.py manifest ROOT > ROOT/MANIFEST.tsv
    cp <archive>/usbgym_results/EXCLUDED.tsv ROOT/EXCLUDED.tsv      # your root lacks the left-out inputs
    python scripts/results_world.py excluded > ROOT/EXCLUDED.tsv    # your root holds every input

## Tests

Without the archive:

    PYTHONPATH=src python -m pytest scripts/paper/test_public_build.py scripts/paper/test_tex_comments.py \
        scripts/analysis/test_best_so_far.py scripts/analysis/test_posthoc_logs.py

The tests of `test_public_build.py` that rebuild from the archive run when the environment variable
`USBG_TEST_RESULTS_ROOT` names ROOT (and, for the full rebuild, the environment above is installed) and are skipped
otherwise:

    USBG_TEST_RESULTS_ROOT=ROOT PYTHONPATH=src python -m pytest scripts/paper/test_public_build.py
