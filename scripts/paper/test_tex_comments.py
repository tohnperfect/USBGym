#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of tex_comments.py (the comment stripper of the public tables) on small synthetic .tex strings.

Run from the repository root (stdlib unittest; pytest also collects it):
  python -m pytest <this folder>/test_tex_comments.py
The token tests compare what TeX reads before and after comment removal, through tex_tokens below: a small model of
TeX's input processor (trailing blanks removed from each line, an end-of-line character appended, states N, M and S,
control words and control symbols, % as comment character, @ as letter or as other). It does not model ^^ notation
or catcode changes.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tex_comments as ms  # noqa: E402


def tex_tokens(text: str, at_letter: bool = False) -> list[tuple]:
    """The token list TeX's input processor makes from text, with plain catcodes (see the module docstring)."""
    def letter(c: str) -> bool:
        return c.isascii() and (c.isalpha() or (at_letter and c == "@"))

    toks: list[tuple] = []
    lines = text.split("\n")
    if text.endswith("\n") or text == "":    # an empty file has no line at all
        lines.pop()
    for raw in lines:
        line = raw.rstrip(" ") + "\r"
        state, i = "N", 0
        while i < len(line):
            c = line[i]
            if c == "\\":
                j = i + 1
                if letter(line[j]):
                    k = j
                    while letter(line[k]):
                        k += 1
                    toks.append(("cs", line[j:k]))
                    state, i = "S", k
                else:
                    toks.append(("cs", line[j]))
                    state, i = ("S" if line[j] in " \t" else "M"), j + 1   # tex.web 354: cat spacer -> S
                    if line[j] == "\r":
                        break
                continue
            if c == "%":
                break
            if c == "\r":
                if state == "N":
                    toks.append(("cs", "par"))   # a blank line is the control sequence \par itself
                elif state == "M":
                    toks.append(("space",))
                break
            if c in " \t":
                if state == "M":
                    toks.append(("space",))
                    state = "S"
                i += 1
                continue
            toks.append(("char", c))
            state, i = "M", i + 1
    return toks


def strip(text: str) -> str:
    return ms.strip_comments(text)[0]


class TokenModel(unittest.TestCase):
    """The model itself, on cases from The TeXbook, chapter 8."""

    def test_model(self):
        self.assertEqual(tex_tokens("a b\n"), [("char", "a"), ("space",), ("char", "b"), ("space",)])
        self.assertEqual(tex_tokens("a\n\nb\n"), [("char", "a"), ("space",), ("cs", "par"), ("char", "b"), ("space",)])
        self.assertEqual(tex_tokens("\\foo  \nb\n"), [("cs", "foo"), ("char", "b"), ("space",)])
        self.assertEqual(tex_tokens("a%x\nb\n"), [("char", "a"), ("char", "b"), ("space",)])
        self.assertEqual(tex_tokens("a\\ \n"), [("char", "a"), ("cs", "\r")])
        self.assertEqual(tex_tokens("\\c@x\n"), [("cs", "c"), ("char", "@"), ("char", "x"), ("space",)])
        self.assertEqual(tex_tokens("\\c@x\n", at_letter=True), [("cs", "c@x")])


class StripCases(unittest.TestCase):
    def same_tokens(self, src: str) -> str:
        out = strip(src)
        for at in (False, True):
            self.assertEqual(tex_tokens(out, at), tex_tokens(src, at), "at_letter=%s\n%r\n->\n%r" % (at, src, out))
        return out

    def test_full_line_comment_removed_without_blank_line(self):
        self.assertEqual(self.same_tokens("a\n% note\nb\n"), "a\nb\n")
        self.assertEqual(self.same_tokens("a\n   %% indented note\nb\n"), "a\nb\n")

    def test_blank_lines_kept(self):
        self.assertEqual(self.same_tokens("a\n\n% note\n\nb\n"), "a\n\n\nb\n")
        self.assertEqual(self.same_tokens("a\n% note\n\nb\n"), "a\n\nb\n")
        self.assertEqual(self.same_tokens("% header\n\n\\section{X}\n"), "\n\\section{X}\n")

    def test_trailing_comment_after_blank(self):
        self.assertEqual(self.same_tokens("text 12 mm. % src: RESULTS.json\nnext\n"), "text 12 mm.\nnext\n")
        self.assertEqual(self.same_tokens("a & b \\\\ % src: x\n\\midrule\n"), "a & b \\\\\n\\midrule\n")

    def test_escaped_percent_kept(self):
        self.assertEqual(self.same_tokens("95\\% CI % src: x\n"), "95\\% CI\n")
        self.assertEqual(self.same_tokens("50\\%\\% of it\n"), "50\\%\\% of it\n")
        self.assertEqual(self.same_tokens("% 50\\% in a comment\nb\n"), "b\n")

    def test_double_backslash_then_comment(self):
        self.assertEqual(self.same_tokens("a\\\\% row end\n\\hline\n"), "a\\\\\\hline\n")

    def test_line_join_without_percent(self):
        self.assertEqual(self.same_tokens("\\def\\x{%\n  \\relax}%\n\\x\n"), "\\def\\x{\\relax}\\x\n")
        self.assertEqual(self.same_tokens("{%\n% inner note\n   b}\n"), "{b}\n")
        self.assertEqual(self.same_tokens("\\label{sec:a}%\nOf the six\n"), "\\label{sec:a}Of the six\n")

    def test_control_word_drops_percent(self):
        self.assertEqual(self.same_tokens("\\scriptsize% single spacing\n\\begin{x}\n"), "\\scriptsize\n\\begin{x}\n")
        self.assertEqual(self.same_tokens("\\relax   % c\nb\n"), "\\relax\nb\n")

    def test_blank_line_after_join_becomes_par(self):
        self.assertEqual(self.same_tokens("\\def\\a{1}% note\n\n\\b\n"), "\\def\\a{1}\\par\n\\b\n")
        self.assertEqual(self.same_tokens("x}%\n% c\n   \n\n y\n"), "x}\\par\n\n y\n")

    def test_end_of_file_keeps_bare_marker(self):
        out, st = ms.strip_comments("a}% note\n")
        self.assertEqual(out, "a}%\n")
        self.assertEqual(st.markers, [(1, "end of file")])
        self.assertEqual(tex_tokens(out), tex_tokens("a}% note\n"))

    def test_control_space(self):
        self.assertEqual(self.same_tokens("a\\ % c\nb\n"), "a\\ b\n")
        self.assertEqual(self.same_tokens("a\\  % c\n\nb\n"), "a\\ \\par\nb\n")

    def test_at_sign_ambiguity(self):
        out, st = ms.strip_comments("\\c@foo%\nbar\n")
        self.assertEqual(out, "\\c@foo%\nbar\n")
        self.assertEqual(st.markers[0][1], "@ in a control sequence at the line end")
        self.assertEqual(self.same_tokens("\\c@foo%\n{x}\n"), "\\c@foo{x}\n")
        self.assertEqual(self.same_tokens("\\@ifundefined{c@box}{%\n\\newfloat}{}\n"),
                         "\\@ifundefined{c@box}{\\newfloat}{}\n")

    def test_endinput_not_joined(self):
        src = "\\expandafter\\endinput\\fi%\nafter\n"
        out, st = ms.strip_comments(src)
        self.assertEqual(out, "\\expandafter\\endinput\\fi\nafter\n")   # \fi is a control word: % dropped
        src = "\\endinput{}%\nafter\n"
        out, st = ms.strip_comments(src)
        self.assertEqual(out, "\\endinput{}%\nafter\n")
        self.assertEqual(st.markers[0][1], "\\endinput on the line")

    def test_verbatim_untouched(self):
        src = ("before % c\n\\begin{verbatim}\n  x = 1  % not a comment\n%   neither\n\n\\end{verbatim} % c2\n"
               "after\n")
        out = strip(src)
        self.assertEqual(out, "before\n\\begin{verbatim}\n  x = 1  % not a comment\n%   neither\n\n"
                              "\\end{verbatim}\nafter\n")
        self.assertEqual(ms.remaining_comments(out), [])

    def test_other_verbatim_environments(self):
        for env in ("lstlisting", "Verbatim", "minted", "alltt", "verbatim*"):
            arg = "{python}" if env == "minted" else ""
            src = "\\begin{%s}%s\n%% kept\na %% kept\n\\end{%s}\n" % (env, arg, env)
            self.assertEqual(strip(src), src, env)

    def test_verbatim_on_one_line(self):
        src = "\\begin{verbatim}a % b\\end{verbatim} % c\n"
        self.assertEqual(strip(src), "\\begin{verbatim}a % b\\end{verbatim}\n")

    def test_commented_begin_does_not_open_verbatim(self):
        src = "% \\begin{verbatim}\na % c\n"
        self.assertEqual(strip(src), "a\n")

    def test_listing_float_is_ordinary_latex(self):
        src = ("\\begin{listing}[t]\n\\begin{verbatim}\nprint(1)  # %d\n\\end{verbatim}\n"
               "\\caption{Toy.}% note\n\\label{lst:a}\n\\end{listing}\n")
        out = strip(src)
        self.assertIn("print(1)  # %d\n", out)
        self.assertIn("\\caption{Toy.}\\label{lst:a}\n", out)

    def test_verb_and_url(self):
        self.assertEqual(strip("\\verb|50%| and \\verb*+a%b+ % c\n"), "\\verb|50%| and \\verb*+a%b+\n")
        self.assertEqual(strip("\\url{https://x.org/a%20b} % c\n"), "\\url{https://x.org/a%20b}\n")
        self.assertEqual(strip("\\href{https://x.org/%7Ea}{text} % c\n"), "\\href{https://x.org/%7Ea}{text}\n")

    def test_crlf_and_no_final_newline(self):
        self.assertEqual(strip("a % c\r\nb"), "a\nb")
        self.assertEqual(strip("% only\n% comments\n"), "")

    def test_no_comment_left(self):
        src = ("%% header\n\\documentclass{article}% c\n\\begin{document}\n95\\% of {%\n x} % y\n"
               "\\verb|%| \\url{a%b}\n\\begin{verbatim}\n% v\n\\end{verbatim}\n\\end{document}\n")
        self.assertEqual(ms.remaining_comments(strip(src)), [])

    def test_line_map_names_source_lines(self):
        src = "% c1\n% c2\na{%\n% c\n  b}\nc % x\n\n\\fillin{y}\n"
        out, st = ms.strip_comments(src)
        self.assertEqual(out, "a{b}\nc\n\n\\fillin{y}\n")
        self.assertEqual(st.line_map, [3, 6, 7, 8])

    def test_control_tab_kept(self):
        self.assertEqual(self.same_tokens("a\\\t  % c\nb\n"), "a\\\tb\n")      # \^^I stays \^^I, not "\ "
        self.assertEqual(self.same_tokens("a\t% c\nb\n"), "a\nb\n")

    def test_random_inputs_keep_tokens(self):
        frags = ["a", "b", " ", "  ", "%", "% c d", "\\", "\\\\", "\\%", "\\foo", "\\ ", "{", "}", "@", "\\c@x",
                 "\\@", "^", "x", "\n", "\n", "\n\n", "  %% note\n", "}%\n", "{%\n", "\\relax%\n", "\t", "\t%\n",
                 "   \n", "\\par"]
        rng = random.Random(20261020)
        for trial in range(4000):
            src = "".join(rng.choice(frags) for _ in range(rng.randint(1, 30)))
            if rng.random() < 0.5:
                src += "\n"
            out, st = ms.strip_comments(src)
            for at in (False, True):
                self.assertEqual(tex_tokens(out, at), tex_tokens(src, at),
                                 "trial %d at_letter=%s\n%r\n->\n%r" % (trial, at, src, out))
            for _, c in ms.remaining_comments(out):
                self.assertEqual(c, "%", "comment text left: %r -> %r" % (src, out))


if __name__ == "__main__":
    unittest.main(verbosity=1)
