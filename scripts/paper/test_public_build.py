#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests of the public build: the public backend of plan_inputs.py, the committed constants, the path mapping of
script_dirs.py (relative recorded paths in public mode only), the closed world of results_world.py (MANIFEST.tsv rows
in place of the digests pinned in code, the inputs that EXCLUDED.tsv lists), the compare step of reproduce.py (PASS,
FAIL and NEEDS, the report on ROOT's MANIFEST.tsv) and one public table step (best_so_far.py table) on tiny
synthetic result roots. They run without the results archive.

The tests at the end rebuild from the results archive: they run when USBG_TEST_RESULTS_ROOT names its folder
usbgym_results (ROOT) and are skipped otherwise.

  python -m pytest <this folder>/test_public_build.py
  USBG_TEST_RESULTS_ROOT=ROOT python -m pytest <this folder>/test_public_build.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
CODE_ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
import plan_inputs as PI  # noqa: E402
import reproduce as R  # noqa: E402
import script_dirs as SD  # noqa: E402
import tex_comments  # noqa: E402

KEYS = {"about", "plan digests", "v2.1 plan constants", "v2.1 plan Q14 tolerance", "amendment 2 appendix",
        "amendment 3", "amendment 4", "amendment 5", "record digests", "ledger", "withheld", "printed software versions"}


@pytest.fixture
def public(monkeypatch, tmp_path):
    monkeypatch.setenv(SD.PUBLIC_ENV, "1")
    monkeypatch.setenv(SD.RESULTS_ENV, str(tmp_path))
    monkeypatch.delenv("USBG_RECORD_PLAN_CALLS", raising=False)
    return tmp_path


def _strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


# =============================================================================================== constants and backend
def test_constants_file_shape():
    J = json.load(open(PI.json_path(), encoding="utf-8"))
    assert set(J) == KEYS
    for k in KEYS - {"about"}:
        assert set(J[k]) == {"description", "value"} and J[k]["description"].strip(), k
    raw = open(PI.json_path(), encoding="utf-8").read()
    assert raw == json.dumps(J, sort_keys=True, indent=1, ensure_ascii=False) + "\n"     # the generator's form


def test_constants_carry_no_path_mail_or_document_name():
    J = json.load(open(PI.json_path(), encoding="utf-8"))
    for s in _strings(J):
        assert not s.startswith(("/", "~", "\\")) and not re.match(r"[A-Za-z]:[\\/]", s), s
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", s), s
        assert not re.search(r"\.(md|txt|sha256|tex|pdf|docx)\b", s), s
    for p in J["withheld"]["value"]:
        assert p.startswith("segmenter/outputs/") and not os.path.isabs(p)


def test_public_backend_returns_every_accessor(public):
    assert PI.amendment_digest(2) == PI._public()["plan digests"]["value"]["amendment 2"]
    for n in (3, 4, 5, 6):
        assert re.fullmatch(r"[0-9a-f]{64}", PI.amendment_digest(n))
    a = PI.plan_v21_digest()
    assert a["sha12"] == a["sha256"][:12]
    assert PI.plan_v21_bootstrap() == {"resamples": 10000, "seed": 0}
    C = PI.plan_v21_constants()
    assert set(C) == {"alpha", "bootstrap", "cohort_decision", "families", "n6_decision"} and C["alpha"] == 0.05
    q = PI.q14_tolerance()
    assert q["band_mm"] == 1.0 and len(q["imit"][1]) == 3 and len(q["final"][1]) == 3
    A = PI.amendment2_appendix()
    assert A["amendment"] == "v2.1 amendment 2" and A["samus"]["real_frame_dice"] == "withheld"
    assert "schedule" not in A and "complete_member" not in A
    assert PI.amendment_value(3, "alpha") == 0.05 and PI.amendment_value(5, "alpha") == 0.05
    assert PI.amendment_value(4, "amos.gate.liver_dice_min") == 0.85
    with pytest.raises(PI.PlanInputsError):
        PI.amendment_value(3, "no.such.constant")
    led = PI._public()["ledger"]["value"]
    rid, shas = next(iter(led.items()))
    assert PI.ledger_has(rid, shas[0]) is True and PI.ledger_has(rid, "0" * 64) is False
    assert set(led) == set(PI._public()["record digests"]["value"])      # the dated records only, no result path
    for rid in PI._public()["record digests"]["value"]:
        sha = PI.record_digest(rid)
        assert PI.ledger_has(rid, sha) is True
    assert len(PI.withheld()) == 2
    assert set(PI.printed_versions()) == {"gymnasium", "stable-baselines3"}
    assert set(PI.ACCESSORS) <= set(dir(PI))


def test_public_text_and_development_skips(public, capsys):
    assert PI.text("amend2 file") == "amendment 2"
    assert PI.skip_dev("a test check") is True
    assert PI.skip_dev("a test check") is True
    err = capsys.readouterr().err
    assert err.count("a test check") == 1                # one notice per check


# =============================================================================================== paths
def test_map_recorded_and_display(public, monkeypatch):
    root = str(public)
    assert SD.segmenter_repo_name() is None              # public mode: no folder name of the development machine
    rec = "/somewhere/else/a_checkout/outputs/v2_1/plug/x.json"
    assert SD.map_recorded(rec) == os.path.join(root, "outputs", "v2_1", "plug", "x.json")   # platform separators
    monkeypatch.setattr(SD, "segmenter_repo_name", lambda: "seg_repo")     # a development file's absolute path
    seg = "/a/b/seg_repo/outputs/v2/eval_cut_s0.json"
    assert SD.map_recorded(seg) == os.path.join(root, "segmenter", "outputs", "v2", "eval_cut_s0.json")
    assert SD.public_json_view({"p": seg}) == {"p": "segmenter/outputs/v2/eval_cut_s0.json"}
    assert SD.map_recorded("/x/data/cache/w.json") == os.path.join(root, "data", "cache", "w.json")
    assert SD.map_recorded("outputs/v2/a.json") == os.path.join(root, "outputs", "v2", "a.json")
    assert SD.map_recorded("segmenter/outputs/v2/eval_cut_s0.json") == os.path.join(
        root, "segmenter", "outputs", "v2", "eval_cut_s0.json")             # the deposit's relative form
    assert SD.map_recorded("plan/record.txt") == "plan/record.txt"           # no result-root folder: unchanged
    inside = os.path.join(root, "outputs", "a.json")
    assert SD.map_recorded(inside) == inside
    assert SD.display(os.path.join(root, "segmenter", "outputs", "v2", "a.json")) == "segmenter/outputs/v2/a.json"


def test_refusal_words_match_below_the_result_root_only(tmp_path):
    """The refusal checks of the build (make_tables._hard_refusal, plan_stats.check_input_path, the posthoc_logs
    blinding) match their words in the path below ROOT: a ROOT whose own location holds such words is accepted, and
    the same words below ROOT are still refused."""
    root = tmp_path / "archive_samus_imnet_imagenet_lpips_ctanat" / "res"
    _deposit(str(root), {"outputs/a.json": b"{}"})
    code = textwrap.dedent(f"""
        import os, sys
        sys.path[:0] = [{HERE!r}, {SCRIPTS!r}, {SD.analysis_dir()!r}]
        import make_tables as MT, plan_stats as ps, posthoc_logs as PL, script_dirs as SD
        r = {str(root)!r}
        seg = os.path.join(r, "segmenter", "outputs", "v2_1", "plug", "seg")
        assert SD.guard_forms(os.path.join(r, "outputs", "a.json"))[0] == os.sep + os.path.join("outputs", "a.json")
        assert MT._hard_refusal(os.path.join(seg, "lraspp_mbv3", "eval_sonospade_s0.json")) is None
        assert MT._hard_refusal(os.path.join(seg, "samus_unet", "eval_sonospade_s0.json")) == \
            "SAMUS real-frame Dice (withheld)"
        assert "ImageNet" in MT._hard_refusal(os.path.join(seg, "train", "fcn_imnet", "history_s0.json"))
        ps.check_input_path(os.path.join(r, "outputs", "v2_1", "plug", "x.json"))
        assert not PL._blinded(os.path.join(r, "outputs", "v2_1", "plug", "x.json"))
        assert PL._blinded(os.path.join(r, "outputs", "v2_1", "plug", "seg", "samus", "x.json"))
        try:
            ps.check_input_path(os.path.join(r, "outputs", "v2_1", "plug", "samus", "x.json"))
            raise SystemExit("a samus path below the root was read")
        except ps.ForbiddenPath:
            pass
        print("ok")
    """)
    p = _cli(code, {SD.PUBLIC_ENV: "1", SD.RESULTS_ENV: str(root), "PYTHONPATH": os.path.join(CODE_ROOT, "src")})
    assert p.returncode == 0 and "ok" in p.stdout, p.stdout + p.stderr


def test_the_output_folder_lies_outside_the_result_root(tmp_path):
    root = tmp_path / "res"
    _deposit(str(root), {"outputs/a.json": b"{}"})
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SCRIPTS!r})
        import script_dirs as SD
        try:
            SD.apply_cli(["--public", "--results", {str(root)!r}, "--out", {str(root / "build")!r}])
            raise SystemExit("DIR inside ROOT was accepted")
        except SystemExit as e:
            assert "lies inside --results" in str(e), e
        assert SD.apply_cli(["--public", "--results", {str(root)!r}, "--out", {str(tmp_path / "build")!r}])
        print("ok")
    """)
    p = _cli(code)
    assert p.returncode == 0 and "ok" in p.stdout, p.stdout + p.stderr
    with pytest.raises(SystemExit):
        R.main(["--results", str(root), "--out", str(root / "build")])


def test_recorded_paths_are_mapped_in_public_mode_only(monkeypatch):
    monkeypatch.delenv(SD.PUBLIC_ENV, raising=False)
    for p in ("outputs/v2/a.json", "segmenter/outputs/v2/a.json", "/x/outputs/v2/a.json"):
        assert SD.map_recorded(p) == p


def _close_probe(env_extra, cases):
    """Run make_tables._close on (recorded, computed) pairs in a fresh process; 'ok' or 'refused' per pair."""
    code = textwrap.dedent(f"""
        import os, sys
        sys.path[:0] = [{HERE!r}, {SCRIPTS!r}]
        import make_tables as MT
        r = MT.ROOT
        for a, b in {cases!r}:
            try:
                MT._close(a, b.replace("ROOT", r))
                print("ok")
            except SystemExit:
                print("refused")
    """)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("USBG_", "US_SEG_"))}
    env.update(env_extra, PYTHONPATH=os.path.join(CODE_ROOT, "src"), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=CODE_ROOT)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout.split()


def test_relative_recorded_paths_are_accepted_in_public_mode_only(tmp_path):
    """The deposit records paths relative to the repository roots; public mode resolves them against ROOT, while the
    development build keeps comparing the absolute paths its own files record."""
    cases = [("outputs/v2/a.json", "ROOT/outputs/v2/a.json"), ("segmenter/outputs/v2/a.json",
                                                                "ROOT/segmenter/outputs/v2/a.json"),
             ("outputs/v2/a.json", "ROOT/outputs/v2/b.json"), ("segmenter/outputs/a.json", "ROOT/outputs/a.json"),
             ("notes", "/x/notes")]
    got = _close_probe({SD.PUBLIC_ENV: "1", SD.RESULTS_ENV: str(tmp_path)}, cases)
    assert got == ["ok", "ok", "refused", "refused", "refused"]
    if SD.mirror():
        pytest.skip("the public mirror has no development mode")
    assert _close_probe({}, cases[:2]) == ["refused", "refused"]


def test_public_json_view(public):
    v = SD.public_json_view({"a": ["/u/x/data/cache/w.json", "dev/figures/plug_rows.json", "text"],
                             "b": {"c": os.path.join(str(public), "outputs", "v2", "R.json"), "n": 1.5},
                             "t": "private words"}, {"private words": "public words"}, figures_prefix="dev/figures/")
    assert v == {"a": ["data/cache/w.json", "figures/plug_rows.json", "text"], "b": {"c": "outputs/v2/R.json", "n": 1.5},
                 "t": "public words"}
    with pytest.raises(SD.DirsError):           # an absolute path with no result-root folder never reaches a file
        SD.public_json_view({"p": "/somewhere/figures/x.pdf"})
    # the md5 beside an input that the archive leaves out points to EXCLUDED.tsv, any other md5 to MANIFEST.tsv
    v = SD.public_json_view({"x": {"ckpt": "/u/x/data/dataset/v2/a.pt", "md5": "0" * 32},
                             "y": {"path": "data/cache/w.json", "md5": "1" * 32}, "z_md5": "2" * 32,
                             "n": [{"md5": "3" * 32}]}, excluded=frozenset({"data/dataset/v2/a.pt"}))
    assert v == {"x": {"ckpt": "data/dataset/v2/a.pt", "md5": SD.DIGEST_NOTE_EXCLUDED},
                 "y": {"path": "data/cache/w.json", "md5": SD.DIGEST_NOTE}, "z_md5": SD.DIGEST_NOTE,
                 "n": [{"md5": SD.DIGEST_NOTE}]}
    assert SD.public_json_view({"ckpt": "data/dataset/v2/a.pt", "md5": "0" * 32})["md5"] == SD.DIGEST_NOTE  # no world


# =============================================================================================== command line
def _cli(code, env_extra=None, cwd=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("USBG_", "US_SEG_"))}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=cwd or CODE_ROOT)


def test_only_the_command_line_switches_to_public_mode(tmp_path):
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {SCRIPTS!r})
        import script_dirs as SD
        if SD.mirror():          # the public mirror runs public mode only
            try:
                SD.apply_cli(["x"]); raise AssertionError("ran without --public in the mirror")
            except SystemExit as e:
                assert "public mirror" in str(e), e
        else:
            assert SD.apply_cli(["x"]) is False
        assert not SD.public() and "USBG_RESULTS_ROOT" not in os.environ
        try:
            SD.apply_cli(["--results", {str(tmp_path)!r}]); raise SystemExit("--results without --public ran")
        except SystemExit as e:
            assert "--public" in str(e), e
        print("ok")
    """)
    p = _cli(code, {SD.PUBLIC_ENV: "1", SD.RESULTS_ENV: str(tmp_path)})
    assert p.returncode == 0 and "ok" in p.stdout, p.stdout + p.stderr


def test_public_mode_drops_inherited_variables_and_sets_the_windows_file(tmp_path):
    (tmp_path / "res").mkdir()
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {SCRIPTS!r})
        import script_dirs as SD
        assert SD.apply_cli(["--public", "--results", {str(tmp_path / "res")!r}, "--out", {str(tmp_path / "o")!r}])
        assert os.environ["USBG_WINDOWS"] == os.path.join({str(tmp_path / "res")!r}, "data", "cache", "windows",
                                                          "usbg-windows-v1.json")
        assert "USBG_SEAT" not in os.environ and "US_SEG_CKPT" not in os.environ
        print("ok")
    """)
    p = _cli(code, {"USBG_WINDOWS": "/elsewhere/w.json", "USBG_SEAT": "canonical", "US_SEG_CKPT": "/x.pt"})
    assert p.returncode == 0 and "ok" in p.stdout, p.stdout + p.stderr
    assert "ignored inherited environment variables: USBG_SEAT, USBG_WINDOWS, US_SEG_CKPT" in p.stderr


def test_folders_are_found_by_content():
    assert os.path.isfile(os.path.join(SD.paper_dir(), "make_tables.py"))
    assert os.path.isfile(os.path.join(SD.analysis_dir(), "plan_stats.py"))
    assert os.path.dirname(PI.json_path()) == os.path.join(SD.paper_dir(), "public_inputs")


# =============================================================================================== closed world
def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _deposit(root, files, withheld=(), bad=(), excluded=True):
    """Write files {relative path: bytes}, MANIFEST.tsv (a wrong digest for the paths in bad) and, unless excluded is
    False, an EXCLUDED.tsv that leaves nothing out (its header line)."""
    lines = ["# test deposit", "path\tbytes\tsha256"]
    for rel, b in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b)
        lines.append(f"{rel}\t{len(b)}\t{'0' * 64 if rel in bad else _sha(b)}")
    open(os.path.join(root, "MANIFEST.tsv"), "w").write("\n".join(lines) + "\n")
    if excluded:
        with open(os.path.join(root, "EXCLUDED.tsv"), "w") as fh:
            fh.write("path\tbytes\tsha256\treason\thow_to_obtain_or_regenerate\n")


# On Windows, Python 3.12 and later implement several os.path predicates (exists, isfile, isdir and islink among
# them) without calling os.stat. This prelude gives a POSIX run the same predicates, bound to the os.stat and os.lstat
# of the time before the closed world is installed, so that the closed-world tests below also run as on Windows.
NATIVE_PREDICATES = """
import os as _os, stat as _stat
def _native(fn, test):
    def f(path):
        try:
            st = fn(path)
        except (OSError, ValueError):
            return False
        return test(st.st_mode)
    return f
_os.path.exists = _native(_os.stat, lambda m: True)
_os.path.lexists = _native(_os.lstat, lambda m: True)
_os.path.isfile = _native(_os.stat, _stat.S_ISREG)
_os.path.isdir = _native(_os.stat, _stat.S_ISDIR)
_os.path.islink = _native(_os.lstat, _stat.S_ISLNK)
"""


@pytest.mark.parametrize("native", [False, True], ids=["os_path", "native_predicates"])
def test_closed_world_lists_checks_hides_and_withholds(tmp_path, native):
    root = tmp_path / "res"
    _deposit(str(root), {"outputs/a.json": b'{"a": 1}', "outputs/b.json": b"{}"}, bad=("outputs/b.json",))
    (root / "outputs" / "c.json").write_text("{}")           # present but not listed
    code = (NATIVE_PREDICATES if native else "") + textwrap.dedent(f"""
        import json, os, sys
        sys.path.insert(0, {SCRIPTS!r})
        import results_world as RW
        W = RW.install({str(root)!r}, ["outputs/w.json"])
        r = {str(root)!r}
        assert json.load(open(os.path.join(r, "outputs/a.json"))) == {{"a": 1}}
        try:
            open(os.path.join(r, "outputs/b.json")).read(); raise SystemExit("b.json was not refused")
        except RW.ResultMismatch:
            pass
        assert not os.path.exists(os.path.join(r, "outputs/c.json"))
        try:
            open(os.path.join(r, "outputs/c.json")); raise SystemExit("c.json was not hidden")
        except FileNotFoundError:
            pass
        assert os.path.exists(os.path.join(r, "outputs/w.json")) and not os.path.exists(os.path.join(r, "outputs/x"))
        assert sorted(os.listdir(os.path.join(r, "outputs"))) == ["a.json", "b.json", "w.json"], os.listdir(r + "/outputs")
        try:
            open(os.path.join(r, "outputs/new.json"), "w"); raise SystemExit("a write under the root was allowed")
        except PermissionError:
            pass
        print("closed world ok", RW.summary())
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "closed world ok" in p.stdout


def test_closed_world_needs_a_manifest_and_every_listed_file(tmp_path):
    root = tmp_path / "res"
    root.mkdir()
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SCRIPTS!r})
        import results_world as RW
        RW.install({str(root)!r}, ["outputs/w.json"])
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode != 0 and "MANIFEST.tsv not found" in p.stderr, p.stderr
    _deposit(str(root), {"outputs/a.json": b"{}", "outputs/b.json": b"{}"})
    lines = open(root / "MANIFEST.tsv").read() + f"outputs/w.json\t2\t{_sha(b'{}')}\n"
    open(root / "MANIFEST.tsv", "w").write(lines)                      # a withheld file may be absent
    (root / "outputs" / "b.json").unlink()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode != 0 and "1 listed file(s) missing from the result root: outputs/b.json" in p.stderr, p.stderr


def test_closed_world_needs_an_excluded_list(tmp_path):
    """Without ROOT/EXCLUDED.tsv the build stops with one message, instead of reading a left-out input as a missing
    file; the header line alone (results_world.py excluded) means that nothing is left out."""
    root = tmp_path / "res"
    _deposit(str(root), {"outputs/a.json": b"{}"}, excluded=False)
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SCRIPTS!r})
        import results_world as RW
        RW.install({str(root)!r})
        print("installed")
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode != 0 and "EXCLUDED.tsv not found" in p.stderr and "results_world.py excluded" in p.stderr, (
        p.stdout + p.stderr)
    exc = subprocess.run([sys.executable, os.path.join(SCRIPTS, "results_world.py"), "excluded"], capture_output=True,
                         text=True)
    assert exc.returncode == 0 and exc.stdout.startswith("path\tbytes\tsha256\t"), exc.stdout + exc.stderr
    (root / "EXCLUDED.tsv").write_text(exc.stdout)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0 and "installed" in p.stdout, p.stdout + p.stderr


def _excluded_tsv(root, rows):
    lines = ["path\tbytes\tsha256\treason\thow_to_obtain_or_regenerate"]
    lines += [f"{p}\t{n}\t{sha}\tthird-party terms\tretrain" for p, n, sha in rows]
    open(os.path.join(root, "EXCLUDED.tsv"), "w").write("\n".join(lines) + "\n")


def _world(root, out, body, withheld=(), native=False):
    code = (NATIVE_PREDICATES if native else "") + textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {SCRIPTS!r})
        import results_world as RW
        RW.install({str(root)!r}, {list(withheld)!r})
        r, out = {str(root)!r}, {str(out)!r}
    """) + textwrap.dedent(body)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("USBG_", "US_SEG_"))}
    env.update({SD.PUBLIC_ENV: "1", SD.RESULTS_ENV: str(root), SD.OUT_ENV: str(out)})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


@pytest.mark.parametrize("native", [False, True], ids=["os_path", "native_predicates"])
def test_excluded_inputs_absent_placed_and_refused(tmp_path, native):
    root, out = tmp_path / "res", tmp_path / "out"
    ck = b"weights"
    _deposit(str(root), {"outputs/a.json": b"{}"})
    _excluded_tsv(str(root), [("data/x/ckpt.pt", len(ck), _sha(ck)),
                              ("segmenter/outputs/w.json", 12, "not hashed (never opened)")])
    absent = """
        assert RW.left_out("data/x/ckpt.pt") and not RW.left_out("outputs/a.json")
        assert not os.path.exists(os.path.join(r, "data/x/ckpt.pt"))
        try:
            open(os.path.join(r, "data/x/ckpt.pt"), "rb"); raise SystemExit("an absent left-out input opened")
        except FileNotFoundError as e:
            assert "EXCLUDED.tsv" in str(e), e
        assert RW.needs(["outputs/a.json"], ["figures/g.json"], "g") is False
        assert RW.needs(["data/x/ckpt.pt"], ["figures/f.png", "figures/f.json"], "f") is True
        assert RW.read_needs(out) == {"figures/f.png": ["data/x/ckpt.pt"], "figures/f.json": ["data/x/ckpt.pt"]}
        assert os.path.exists(os.path.join(r, "segmenter/outputs/w.json"))      # withheld: existence only
        print("absent ok")
    """
    p = _world(root, out, absent, withheld=["segmenter/outputs/w.json"], native=native)
    assert p.returncode == 0 and "absent ok" in p.stdout, p.stdout + p.stderr
    view = """
        import script_dirs as SD
        v = SD.public_json_view({"a": {"ckpt": "data/x/ckpt.pt", "md5": "0" * 32},
                                 "b": {"path": "outputs/a.json", "md5": "1" * 32},
                                 "c": {"path": "segmenter/outputs/w.json", "md5": "2" * 32}})
        assert v["a"]["md5"] == "see EXCLUDED.tsv" and v["b"]["md5"] == v["c"]["md5"] == "see MANIFEST.tsv", v
        print("view ok")
    """
    q = _world(root, tmp_path / "out_view", view, withheld=["segmenter/outputs/w.json"], native=native)
    assert q.returncode == 0 and "view ok" in q.stdout, q.stdout + q.stderr
    assert "[public] NEEDS data/x/ckpt.pt (EXCLUDED.tsv): f not built (figures/f.png, figures/f.json)" in p.stderr
    assert "1 inputs left out (EXCLUDED.tsv): 0 placed under the root" in p.stderr
    (root / "data" / "x").mkdir(parents=True)
    (root / "data" / "x" / "ckpt.pt").write_bytes(ck)                          # placed with its EXCLUDED.tsv bytes
    placed = """
        assert not RW.left_out("data/x/ckpt.pt")
        assert open(os.path.join(r, "data/x/ckpt.pt"), "rb").read() == b"weights"
        assert RW.listed("data/x/ckpt.pt") is None                              # not a MANIFEST.tsv row
        assert RW.needs(["data/x/ckpt.pt"], ["figures/f.json"], "f") is False
        print("placed ok", RW.summary()["placed"])
    """
    p = _world(root, tmp_path / "out2", placed, native=native)
    assert p.returncode == 0 and "placed ok 1" in p.stdout, p.stdout + p.stderr
    (root / "data" / "x" / "ckpt.pt").write_bytes(b"weighs!")                  # same size, other bytes
    wrong = """
        try:
            open(os.path.join(r, "data/x/ckpt.pt"), "rb"); raise SystemExit("a wrong placed input opened")
        except RW.ResultMismatch as e:
            assert "data/x/ckpt.pt" in str(e) and "EXCLUDED.tsv says" in str(e), e
        print("wrong ok")
    """
    p = _world(root, tmp_path / "out3", wrong, native=native)
    assert p.returncode == 0 and "wrong ok" in p.stdout, p.stdout + p.stderr
    (root / "data" / "x" / "ckpt.pt").write_bytes(b"other size")
    p = _world(root, tmp_path / "out4", "print('installed')", native=native)
    assert p.returncode != 0 and "data/x/ckpt.pt (its EXCLUDED.tsv row)" in p.stderr, p.stdout + p.stderr


def test_manifest_command_writes_a_manifest_the_world_accepts(tmp_path):
    root = tmp_path / "res"
    (root / "outputs").mkdir(parents=True)
    (root / "outputs" / "a.json").write_text('{"a": 1}')
    man = subprocess.run([sys.executable, os.path.join(SCRIPTS, "results_world.py"), "manifest", str(root)],
                         capture_output=True, text=True)
    assert man.returncode == 0, man.stderr
    (root / "MANIFEST.tsv").write_text(man.stdout)
    exc = subprocess.run([sys.executable, os.path.join(SCRIPTS, "results_world.py"), "excluded"], capture_output=True,
                         text=True)
    (root / "EXCLUDED.tsv").write_text(exc.stdout)
    code = textwrap.dedent(f"""
        import json, os, sys
        sys.path.insert(0, {SCRIPTS!r})
        import results_world as RW
        RW.install({str(root)!r})
        assert json.load(open(os.path.join({str(root)!r}, "outputs", "a.json"))) == {{"a": 1}}
        assert RW.listed("outputs/a.json") == {_sha(b'{"a": 1}')!r} and RW.listed("outputs/none.json") is None
        print("ok")
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0 and "ok" in p.stdout, p.stdout + p.stderr


# =============================================================================================== reproduce compare
def test_reproduce_compare_pass_and_fail(tmp_path, monkeypatch, capsys):
    exp, out = tmp_path / "expected", tmp_path / "out"
    for d in (exp / "tables", exp / "figures", out / "tables", out / "figures"):
        d.mkdir(parents=True)
    (exp / "tables" / "t1.tex").write_text("A\n")
    (exp / "tables" / "t2.tex").write_text("B\n")
    (exp / "figures" / "f.json").write_text("{}\n")
    (out / "tables" / "t1.tex").write_text("A\n")
    (out / "tables" / "t2.tex").write_text("C\n")
    (out / "figures" / "f.json").write_text("{}\n")
    (out / "figures" / "f.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(R, "EXPECTED", str(exp))
    assert R.compare(str(out)) is False
    text = capsys.readouterr().out
    assert "PASS tables/t1.tex" in text and "FAIL tables/t2.tex" in text and "2 PASS, 1 FAIL" in text
    (out / "tables" / "t2.tex").write_text("B\n")
    assert R.compare(str(out)) is True


def test_reproduce_compare_reports_needs(tmp_path, monkeypatch, capsys):
    exp, out = tmp_path / "expected", tmp_path / "out"
    for d in (exp / "tables", exp / "figures", out / "tables", out / "figures"):
        d.mkdir(parents=True)
    (exp / "tables" / "t1.tex").write_text("A\n")
    (exp / "figures" / "f.json").write_text("{}\n")
    (exp / "figures" / "g.json").write_text("{}\n")
    (out / "tables" / "t1.tex").write_text("A\n")
    (out / "figures" / "g.json").write_text("{}\n")
    (out / "NEEDS.tsv").write_text("figures/f.png\tdata/x.pt\nfigures/f.json\tdata/x.pt\n")
    monkeypatch.setattr(R, "EXPECTED", str(exp))
    assert R.compare(str(out)) is True
    text = capsys.readouterr().out
    assert "NEEDS data/x.pt (EXCLUDED.tsv): figures/f.json" in text
    assert "2 PASS, 0 FAIL, 1 NEEDS (inputs left out of the results archive, EXCLUDED.tsv)" in text
    assert R.compare(str(out), archive_manifest=False) is True
    assert "1 NEEDS (inputs left out of the results archive, EXCLUDED.tsv) (MANIFEST.tsv is not the results " \
           "archive's)" in capsys.readouterr().out
    (out / "tables" / "t1.tex").write_text("B\n")
    assert R.compare(str(out)) is False
    capsys.readouterr()
    (out / "tables" / "t1.tex").write_text("A\n")
    (out / "figures" / "f.json").write_text("{\"wrong\": 1}\n")     # a NEEDS line of an earlier run, a new file
    assert R.compare(str(out)) is False
    text = capsys.readouterr().out
    assert "FAIL figures/f.json" in text and "NEEDS data/x.pt" not in text and "2 PASS, 1 FAIL" in text
    (out / "figures" / "f.json").write_text("{}\n")
    assert R.compare(str(out)) is True and "PASS figures/f.json" in capsys.readouterr().out


def test_a_figure_built_again_drops_its_earlier_needs_lines(tmp_path):
    root, out = tmp_path / "res", tmp_path / "out"
    _deposit(str(root), {"outputs/a.json": b"{}"})
    out.mkdir()
    (out / "NEEDS.tsv").write_text("figures/stages.png\tdata/c.pt\nfigures/stages.json\tdata/c.pt\n"
                                   "figures/stages_twin.json\tdata/c.pt\nfigures/layout.json\tdata/e.npz\n")
    body = """
        RW.clear_needs("figures/stages.")
        assert RW.read_needs(out) == {"figures/stages_twin.json": ["data/c.pt"], "figures/layout.json": ["data/e.npz"]}
        RW.clear_needs("figures/none.")
        assert len(RW.read_needs(out)) == 2
        print("cleared")
    """
    p = _world(root, out, body)
    assert p.returncode == 0 and "cleared" in p.stdout, p.stdout + p.stderr


def test_reproduce_reports_whether_root_holds_the_archive_manifest(tmp_path, monkeypatch):
    root = tmp_path / "res"
    _deposit(str(root), {"outputs/a.json": b"{}"})
    man = (root / "MANIFEST.tsv").read_bytes()
    pin = tmp_path / "results_archive.json"
    pin.write_text(json.dumps({"manifest_sha256": _sha(man), "manifest_rows": 1}))
    monkeypatch.setattr(R, "ARCHIVE", str(pin))
    ok, text = R.manifest_status(str(root))
    assert ok and "is the results archive's (1 rows" in text
    (root / "MANIFEST.tsv").write_bytes(man + b"outputs/b.json\t2\t" + _sha(b"{}").encode() + b"\n")
    ok, text = R.manifest_status(str(root))
    assert not ok and "is not the results archive's (2 rows" in text
    assert "checked against this MANIFEST.tsv only" in text
    ok, text = R.manifest_status(str(tmp_path / "none"))
    assert not ok and "not found" in text


def test_reproduce_checks_every_listed_file_and_exits_3_on_another_manifest(tmp_path, monkeypatch, capsys):
    root, out, exp = tmp_path / "res", tmp_path / "out", tmp_path / "expected"
    _deposit(str(root), {"outputs/a.json": b"{}", "outputs/b.json": b"[]"})
    for d in (exp / "tables", out / "tables"):
        d.mkdir(parents=True)
    (exp / "tables" / "t.tex").write_text("A\n")
    (out / "tables" / "t.tex").write_text("A\n")
    pin = tmp_path / "results_archive.json"
    pin.write_text(json.dumps({"manifest_sha256": _sha((root / "MANIFEST.tsv").read_bytes()), "manifest_rows": 2}))
    monkeypatch.setattr(R, "ARCHIVE", str(pin))
    monkeypatch.setattr(R, "EXPECTED", str(exp))
    monkeypatch.setattr(R, "build", lambda r, o: True)          # the compare step only
    args = ["--results", str(root), "--out", str(out)]
    assert R.main(args) == 0
    assert "listed files: all 2 match MANIFEST.tsv" in capsys.readouterr().out
    pin.write_text(json.dumps({"manifest_sha256": "0" * 64, "manifest_rows": 2}))   # a result root of one's own
    assert R.main(args) == 3
    assert "exit 3: ROOT/MANIFEST.tsv is not the results archive's" in capsys.readouterr().out
    assert R.main(args + ["--own-results"]) == 0
    (root / "outputs" / "b.json").write_bytes(b"{}")                 # same size, other bytes, never opened by a step
    assert R.main(args + ["--own-results"]) == 1
    assert "listed files: 1 of 2 differ from MANIFEST.tsv: outputs/b.json" in capsys.readouterr().out


def test_the_archive_manifest_pin_is_well_formed():
    J = json.load(open(R.ARCHIVE, encoding="utf-8"))
    assert set(J) == {"about", "doi", "manifest_sha256", "manifest_rows"}
    assert re.fullmatch(r"[0-9a-f]{64}", J["manifest_sha256"]) and J["manifest_rows"] > 0
    assert re.fullmatch(r"10\.5281/zenodo\.[0-9]+", J["doi"])
    # the build's README (README.md of the published folder, public_inputs/README_public.md in development) names it
    readme = [p for p in (os.path.join(HERE, "README.md"), os.path.join(HERE, "public_inputs", "README_public.md"))
              if os.path.isfile(p)][0]
    assert "doi:" + J["doi"] in open(readme, encoding="utf-8").read()


def test_expected_tables_hold_no_comment():
    d = os.path.join(R.EXPECTED, "tables")
    names = sorted(os.listdir(d))
    assert len(names) >= 50
    for n in names:
        text = open(os.path.join(d, n), encoding="utf-8").read()
        assert [c for _, c in tex_comments.remaining_comments(text) if c != "%"] == [], n


# =============================================================================================== one public step
def _a5_public():
    """The amendment 5 record of a synthetic deposit: its status, label and [posthoc] sentence as the results archive
    holds them (the public texts of plan_inputs.py, which public mode compares with)."""
    return {"provenance": {"analysis_status": PI.PUBLIC_TEXT["status a5"]},
            "posthoc_table_label": PI.PUBLIC_TEXT["a5 label text"],
            "sentences": [{"id": "posthoc", "text": PI.PUBLIC_TEXT["posthoc sentence"]}]}


def test_best_so_far_table_in_public_mode_on_a_synthetic_deposit(tmp_path):
    sys.path.insert(0, SD.analysis_dir())
    import best_so_far as B
    P = B.TEST_PATIENTS

    def pp(vals):
        return dict(zip(P, [float(v) for v in vals]))
    mk = lambda cid, cls, vals, **kw: B.cand(cid, cls, cid, "f", "m", "c", "src", B.LABEL_V2, [0, 1, 2],  # noqa: E731
                                             pp=pp(vals), **kw)
    cands = [mk(B.REFERENCE_OF_CLASS["a1"], "a1", [17, 17, 18, 16, 17, 17]), mk("a1:best", "a1", [14] * 6),
             mk(B.REFERENCE_OF_CLASS["a2"], "a2", [13] * 6), mk("a2:best", "a2", [11] * 6),
             mk(B.REFERENCE_OF_CLASS["b"], "b", [11.6] * 6), mk("b:best", "b", [10.5, 10.6, 10.7, 10.8, 10.9, 11.0]),
             mk("c:look", "c", [7] * 6, eligible=False)]
    rung = {"static": {"pp": pp([16] * 6)}, "pose_oracle": {"pp": pp([2] * 6)}}
    K = B.analyse(cands, rung)
    for c in cands:
        c["cohort"] = {"status": "not run on the cohort"}
        c["per_patient_mm"] = c.pop("pp")
    import numpy as np
    st6, or6 = np.full(6, 16.0), np.full(6, 2.0)
    anchors = {"static": {"err_mm": 16.0, "minus_static": None, "minus_oracle": B.diff_summary(st6, or6 - 0.1, P),
                          "cohort": {"mean": 15.97}},
               "scripted": {"err_mm": 15.5, "minus_static": B.diff_summary(st6 - 0.5, st6, P),
                            "minus_oracle": B.diff_summary(st6 - 0.5, or6, P), "cohort": {"mean": 15.38}},
               "pose_oracle": {"err_mm": 2.0, "minus_static": B.diff_summary(or6, st6, P), "minus_oracle": None,
                               "cohort": {"mean": 1.47}}}
    J = json.loads(json.dumps({"candidates": cands, "anchors": anchors, "classes": K}, default=float))
    a5 = _a5_public()
    a3 = {"result": {"rows": [{}] * 6 + [{"row": "random", "kind": "reference", "n_seeds": 3, "mean_mm": 17.25}]}}
    ap = "outputs/v2_1/analysis_plan/"
    root, out = tmp_path / "res", tmp_path / "out"
    _deposit(str(root), {ap + "synthetic_best_so_far.json": json.dumps(J).encode(),
                         ap + "amend5_combination.json": json.dumps(a5).encode(),
                         ap + "amend3_earlier.json": json.dumps(a3).encode()})
    env = dict(os.environ, PYTHONPATH=os.path.join(CODE_ROOT, "src"))
    env.pop(SD.PUBLIC_ENV, None)
    p = subprocess.run([sys.executable, os.path.join(SD.analysis_dir(), "best_so_far.py"), "table", "--public",
                        "--results", str(root), "--out", str(out), "--json",
                        str(root / ap / "synthetic_best_so_far.json")], capture_output=True, text=True, env=env,
                       cwd=CODE_ROOT)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "closed world" in p.stderr
    got = (out / "tables" / "best_so_far.tex").read_text()
    want = tex_comments.strip_comments(B.render_table(J, "x.json", {"label": a5["posthoc_table_label"],
                                                                     "posthoc": a5["sentences"][0]["text"], "src": "x"},
                                                      {"mean_mm": 17.25, "n_seeds": 3, "src": "x"}))[0]
    assert got == want
    assert "%" not in got.replace("\\%", "")
    p = subprocess.run([sys.executable, os.path.join(SD.analysis_dir(), "best_so_far.py"), "table", "--out",
                        str(tmp_path / "o2")], capture_output=True, text=True, env=env, cwd=CODE_ROOT)
    msg = p.stdout + p.stderr                       # --out needs --public (and the public mirror needs --public)
    assert p.returncode != 0 and ("option of public mode" in msg or "public mirror" in msg), msg


def test_best_so_far_checks_the_version_4_result_against_the_manifest(tmp_path):
    """The md5 that best_so_far.py pins for the Version 4 result names the development file; public mode checks the
    file against its MANIFEST.tsv row instead, and a row that does not match stops the table."""
    sys.path.insert(0, SD.analysis_dir())
    import best_so_far as B
    P = B.TEST_PATIENTS

    def pp(vals):
        return dict(zip(P, [float(v) for v in vals]))
    mk = lambda cid, cls, vals: B.cand(cid, cls, cid, "f", "m", "c", "src", B.LABEL_V2, [0, 1, 2],  # noqa: E731
                                       pp=pp(vals))
    cands = [mk(B.REFERENCE_OF_CLASS["a1"], "a1", [17, 17, 18, 16, 17, 17]), mk(B.RENDER_BEST["a1"], "a1", [14] * 6),
             mk(B.REFERENCE_OF_CLASS["a2"], "a2", [13] * 6), mk(B.RENDER_BEST["a2"], "a2", [11] * 6),
             mk(B.REFERENCE_OF_CLASS["b"], "b", [11.6] * 6),
             mk(B.RENDER_BEST["b"], "b", [10.5, 10.6, 10.7, 10.8, 10.9, 11.0])]
    K = B.analyse(cands, {"static": {"pp": pp([16] * 6)}, "pose_oracle": {"pp": pp([2] * 6)}})
    for c in cands:
        c["cohort"] = {"status": "not run on the cohort"}
        c["per_patient_mm"] = c.pop("pp")
    import numpy as np
    st6, or6 = np.full(6, 16.0), np.full(6, 2.0)
    anchors = {"static": {"err_mm": 16.0, "minus_static": None, "minus_oracle": B.diff_summary(st6, or6 - 0.1, P),
                          "cohort": {"mean": 15.97}},
               "scripted": {"err_mm": 15.5, "minus_static": B.diff_summary(st6 - 0.5, st6, P),
                            "minus_oracle": B.diff_summary(st6 - 0.5, or6, P), "cohort": {"mean": 15.38}},
               "pose_oracle": {"err_mm": 2.0, "minus_static": B.diff_summary(or6, st6, P), "minus_oracle": None,
                               "cohort": {"mean": 1.47}}}
    J = json.dumps({"candidates": cands, "anchors": anchors, "classes": K}, default=float).encode()
    a5 = _a5_public()
    a3 = {"result": {"rows": [{}] * 6 + [{"row": "random", "kind": "reference", "n_seeds": 3, "mean_mm": 17.25}]}}
    ap = "outputs/v2_1/analysis_plan/"
    v4 = ap + os.path.basename(B.RENDER_JSON)
    files = {v4: J, ap + "amend5_combination.json": json.dumps(a5).encode(),
             ap + "amend3_earlier.json": json.dumps(a3).encode()}
    env = dict(os.environ, PYTHONPATH=os.path.join(CODE_ROOT, "src"))
    env.pop(SD.PUBLIC_ENV, None)
    for bad, want_rc in (((), 0), ((v4,), 1)):
        root = tmp_path / f"res{len(bad)}"
        _deposit(str(root), files, bad=bad)
        p = subprocess.run([sys.executable, os.path.join(SD.analysis_dir(), "best_so_far.py"), "table", "--public",
                            "--results", str(root), "--out", str(tmp_path / f"out{len(bad)}")], capture_output=True,
                           text=True, env=env, cwd=CODE_ROOT)
        assert (p.returncode == 0) == (want_rc == 0), p.stdout + p.stderr
        assert "checked against MANIFEST.tsv instead: the pinned md5 of the Version 4 result" in p.stderr, p.stderr
        if bad:
            assert f"{v4}: " in p.stderr and "MANIFEST.tsv says" in p.stderr, p.stderr
        else:
            assert (tmp_path / "out0" / "tables" / "best_so_far.tex").is_file()


# =============================================================================================== the results deposit
DEPOSIT = os.environ.get("USBG_TEST_RESULTS_ROOT")
needs_deposit = pytest.mark.skipif(not DEPOSIT, reason="USBG_TEST_RESULTS_ROOT not set (the results archive's folder "
                                                       "usbgym_results)")
# the expected files whose step reads an input that the deposit leaves out (EXCLUDED.tsv)
NEEDS_FILES = {"figures/stages.json", "figures/stages_twin.json", "figures/layout.json"}


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("USBG_", "US_SEG_"))}
    env.update(PYTHONPATH=os.path.join(CODE_ROOT, "src"), PYTHONDONTWRITEBYTECODE="1")
    return env


def _shadow(dst, changed=None, manifest_rows=None):
    """A copy of the deposit made of symbolic links (ROOT's files are never written), with the files of changed
    ({path: bytes}) written instead and MANIFEST.tsv rows replaced from manifest_rows ({path: bytes})."""
    import shutil
    changed, manifest_rows = changed or {}, manifest_rows or {}
    head, rows = [], []
    for ln in open(os.path.join(DEPOSIT, "MANIFEST.tsv"), encoding="utf-8"):
        if ln.startswith("#") or ln.startswith("path\t"):
            head.append(ln)
            continue
        rel = ln.split("\t")[0]
        p = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if rel in changed:
            open(p, "wb").write(changed[rel])
        else:
            os.symlink(os.path.join(DEPOSIT, rel), p)
        if rel in manifest_rows:
            b = manifest_rows[rel]
            f = ln.rstrip("\n").split("\t")
            ln = "\t".join([rel, str(len(b)), _sha(b)] + f[3:]) + "\n"
        rows.append(ln)
    open(os.path.join(dst, "MANIFEST.tsv"), "w", encoding="utf-8").write("".join(head + rows))
    shutil.copyfile(os.path.join(DEPOSIT, "EXCLUDED.tsv"), os.path.join(dst, "EXCLUDED.tsv"))


@needs_deposit
def test_the_deposit_is_the_pinned_archive_and_withholds_only_what_the_build_never_opens(public):
    """ROOT/MANIFEST.tsv is the one public_inputs/results_archive.json pins; the rows of ROOT/EXCLUDED.tsv without a
    sha256 are exactly the withheld files (existence only: nothing is to be placed for them), and every other row is
    an input that the build checks when it is placed."""
    sys.path.insert(0, SCRIPTS)
    import results_world as RW
    ok, text = R.manifest_status(DEPOSIT)
    assert ok, text
    rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(DEPOSIT, "EXCLUDED.tsv"), encoding="utf-8")][1:]
    unhashed = {r[0] for r in rows if not re.fullmatch(r"[0-9a-f]{64}", r[2])}
    assert unhashed == set(PI.withheld())
    assert set(RW.load_excluded(os.path.join(DEPOSIT, "EXCLUDED.tsv"))) == {r[0] for r in rows} - unhashed


@needs_deposit
def test_reproduce_on_the_results_deposit(tmp_path):
    """Every expected file passes, except the ones whose step needs an input that the deposit leaves out and that
    was not placed under ROOT: those are NEEDS lines naming the input and EXCLUDED.tsv."""
    pytest.importorskip("trimesh", reason="the rebuild environment of README.md (requirements-reproduce.txt) is not "
                                          "installed")
    sys.path.insert(0, SCRIPTS)
    import results_world as RW
    left = {r for r in RW.load_excluded(os.path.join(DEPOSIT, "EXCLUDED.tsv"))
            if not os.path.isfile(os.path.join(DEPOSIT, r))}
    p = subprocess.run([sys.executable, os.path.join(HERE, "reproduce.py"), "--results", DEPOSIT, "--out",
                        str(tmp_path / "out")], capture_output=True, text=True, env=_env(), cwd=CODE_ROOT, timeout=3600)
    assert p.returncode == 0 and " 0 FAIL" in p.stdout, p.stdout[-3000:] + p.stderr[-2000:]
    needs = {}
    for ln in p.stdout.splitlines():
        if ln.startswith("NEEDS "):
            inputs, _, f = ln[len("NEEDS "):].partition(" (EXCLUDED.tsv): ")
            needs[f] = set(inputs.split(", "))
    assert set(needs) <= NEEDS_FILES and all(v <= left for v in needs.values()), needs
    assert bool(needs) == bool(left)
    assert "manifest: ROOT/MANIFEST.tsv is the results archive's" in p.stdout, p.stdout[:2000]
    assert "(MANIFEST.tsv is not the results archive's)" not in p.stdout


@needs_deposit
def test_a_changed_deposited_file_stops_the_build(tmp_path):
    rel = "outputs/v2/RESULTS.json"
    b = bytearray(open(os.path.join(DEPOSIT, rel), "rb").read())
    i = next(k for k, c in enumerate(b) if chr(c).isdigit())
    b[i] = ord("1") if b[i] != ord("1") else ord("2")                         # one byte, same size
    root = tmp_path / "res"
    _shadow(str(root), changed={rel: bytes(b)})
    p = subprocess.run([sys.executable, os.path.join(HERE, "make_tables.py"), "--public", "--results", str(root),
                        "--out", str(tmp_path / "out")], capture_output=True, text=True, env=_env(), cwd=CODE_ROOT,
                       timeout=1800)
    assert p.returncode != 0 and f"{rel}: " in p.stderr and "MANIFEST.tsv says" in p.stderr, p.stderr[-2000:]


@needs_deposit
def test_a_digest_that_a_record_holds_is_checked_with_the_deposited_bytes(tmp_path):
    """A deposited file changed together with its MANIFEST.tsv row still stops the build where another record cites
    its digest (here amos_q2r.json, which cites the frozen AMOS manifest)."""
    rel = "outputs/v2_1/amos/amos_frozen.json"
    b = open(os.path.join(DEPOSIT, rel), "rb").read() + b"\n"
    root = tmp_path / "res"
    _shadow(str(root), changed={rel: b}, manifest_rows={rel: b})
    p = subprocess.run([sys.executable, os.path.join(HERE, "make_tables.py"), "--public", "--results", str(root),
                        "--out", str(tmp_path / "out")], capture_output=True, text=True, env=_env(), cwd=CODE_ROOT,
                       timeout=1800)
    assert p.returncode != 0 and "amos_frozen.json" in p.stderr, p.stdout[-1000:] + p.stderr[-2000:]


@needs_deposit
def test_the_combined_digest_of_the_segmenter_files_is_recomputed(tmp_path):
    """A segmenter evaluation file changed together with its MANIFEST.tsv row (one trailing newline, the same values)
    makes posthoc_v2.json stale: its seg_eval_jsons fingerprint is recomputed in public mode as well."""
    sys.path.insert(0, SCRIPTS)
    root = tmp_path / "res"
    rel = "segmenter/outputs/v2/eval_physics_s0.json"
    b = open(os.path.join(DEPOSIT, rel), "rb").read() + b"\n"
    _shadow(str(root), changed={rel: b}, manifest_rows={rel: b})
    code = textwrap.dedent(f"""
        import sys
        sys.argv = ["x", "--public", "--results", {str(root)!r}, "--out", {str(tmp_path / "out")!r}]
        sys.path.insert(0, {SCRIPTS!r})
        import results_world as RW
        assert RW.start(sys.argv)
        import posthoc_v2 as PH
        J, status = PH.load_checked()
        print("status:", status)
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_env(), cwd=CODE_ROOT,
                       timeout=600)
    assert p.returncode == 0 and "status: stale: seg_eval_jsons" in p.stdout, p.stdout + p.stderr[-2000:]
    _shadow(str(tmp_path / "res2"))
    p = subprocess.run([sys.executable, "-c", code.replace(str(root), str(tmp_path / "res2"))], capture_output=True,
                       text=True, env=_env(), cwd=CODE_ROOT, timeout=600)
    assert p.returncode == 0 and "status: ok" in p.stdout, p.stdout + p.stderr[-2000:]


def _successor_records(o):
    """Every successor-rule record (plan_stats.successor_rule_record) inside a JSON value."""
    if isinstance(o, dict):
        if {"base", "tag", "never_prefixes", "decision"} <= set(o):
            yield o
        for v in o.values():
            yield from _successor_records(v)
    elif isinstance(o, list):
        for v in o:
            yield from _successor_records(v)


def test_the_public_successor_label_is_the_archive_label():
    """In public mode (and in the public mirror) the successor-rule records carry the label of the results archive."""
    import glob
    hits = sorted(glob.glob(os.path.join(SCRIPTS, "*", "plan_stats.py")))
    assert len(hits) == 1, hits
    sys.path.insert(0, os.path.dirname(hits[0]))
    import plan_stats as ps
    assert ps.PUBLIC_SUCCESSOR_DECISION == "tag successor rule"
    rule = dict(ps.TAG_SUCCESSOR, decision=ps.PUBLIC_SUCCESSOR_DECISION)
    assert ps.successor_rule_record(rule)["decision"] == "tag successor rule"
    if DEPOSIT:
        labels = set()
        for dp, _, fns in os.walk(DEPOSIT):
            for fn in fns:
                if fn.endswith(".json"):
                    with open(os.path.join(dp, fn), encoding="utf-8") as fh:
                        labels |= {r["decision"] for r in _successor_records(json.load(fh))}
        assert labels == {ps.PUBLIC_SUCCESSOR_DECISION}, labels
