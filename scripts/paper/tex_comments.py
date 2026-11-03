#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
r"""tex_comments.py: remove every LaTeX comment from a .tex text without changing the tokens TeX reads.

Comment removal (strip_comments):
  - a line that holds only a comment is deleted with its newline, so no blank line (a \par) is created or lost;
  - a trailing comment is deleted; when blanks precede it, they go too (TeX reads the same single space), except a
    blank that a backslash escapes ("\ " or "\<tab>", a control symbol), which stays and takes the next line on;
  - escaped \% stays, and so does a % inside verbatim-like environments (verbatim, Verbatim, BVerbatim, LVerbatim,
    lstlisting, minted, alltt, comment, filecontents), \verb, and the \url, \nolinkurl and \href URL argument;
  - where the comment character joined two lines ("text%" at a line end), the next line is appended to this one
    with its leading blanks removed, which gives TeX the same tokens without any %. After a control word such as
    \small the % is simply dropped, because TeX ignores that line end anyway. When a blank line follows, the line
    becomes "text\par" and that blank line goes, since TeX reads the blank line as the token \par. Where a join is
    not safe (the end of the file follows, a verbatim block or \endinput is involved, or an @ makes the end of a
    control sequence depend on \makeatletter), a bare % is kept and listed as a line-join marker;
  - blank lines are kept exactly as they are.
The token stream that TeX reads is unchanged; test_tex_comments.py checks this against a small model of TeX's input
processor (states N, M and S) on fixed and random inputs. The paper build writes its public tables through
strip_comments, and the package script of the manuscript imports the same functions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Environments whose body LaTeX reads with % as an ordinary character (or skips line by line): copied untouched.
VERBATIM_ENVS = frozenset({
    "verbatim", "verbatim*", "Verbatim", "Verbatim*", "BVerbatim", "BVerbatim*", "LVerbatim", "LVerbatim*",
    "lstlisting", "minted", "alltt", "comment", "filecontents", "filecontents*", "spverbatim",
})
# Commands whose braced first argument is read with % as an ordinary character (url.sty, hyperref).
URL_COMMANDS = frozenset({"url", "nolinkurl", "href"})   # not \path, which TikZ also defines


# ---------------------------------------------------------------------------------------------------------------
# Line scanner
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class ScannedLine:
    raw: str              # the line as read, without its newline
    content: str          # the text before the comment character (the whole line when there is none)
    code: str             # content with verbatim, \verb and URL text replaced by blanks
    comment: str | None   # from the comment character to the end of the line, or None
    verbatim: bool        # some part of the line lies inside a verbatim-like environment


_CS_WORD = re.compile(r"\\([A-Za-z]+)")
_BEGIN_ARG = re.compile(r"[ \t]*\{([^{}]*)\}")


def _match_brace(line: str, j: int) -> int | None:
    """Index of the brace that closes line[j] == '{' on the same line, or None."""
    depth = 0
    for k in range(j, len(line)):
        ch = line[k]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return k
    return None


def scan_line(line: str, verb_end: str | None) -> tuple[ScannedLine, str | None]:
    """Split one source line at its comment character.

    verb_end is the end marker of the verbatim-like environment that is open at the start of the line (None in
    ordinary text). Returns the scanned line and the marker that is open at its end.
    """
    content: list[str] = []
    code: list[str] = []
    verbatim = False

    def keep(s: str) -> None:
        content.append(s)
        code.append(s)

    def protect(s: str) -> None:
        content.append(s)
        code.append(" " * len(s))

    i, n = 0, len(line)
    if verb_end is not None:
        k = line.find(verb_end)
        if k < 0:
            return ScannedLine(line, line, " " * n, None, True), verb_end
        protect(line[:k])
        keep(verb_end)
        i = k + len(verb_end)
        verbatim = True
        verb_end = None
    while i < n:
        ch = line[i]
        if ch == "\\":
            m = _CS_WORD.match(line, i)
            if m is None:                       # control symbol, including \% and \\
                keep(line[i:i + 2])
                i += 2
                continue
            name, j = m.group(1), m.end()
            if name == "begin":
                b = _BEGIN_ARG.match(line, j)
                if b and b.group(1) in VERBATIM_ENVS:
                    end = "\\end{%s}" % b.group(1)
                    keep(line[i:b.end()])
                    k = line.find(end, b.end())
                    verbatim = True
                    if k < 0:
                        protect(line[b.end():])
                        return ScannedLine(line, "".join(content), "".join(code), None, True), end
                    protect(line[b.end():k])
                    keep(end)
                    i = k + len(end)
                    continue
            elif name == "verb":
                j2 = j + 1 if line.startswith("*", j) else j
                if j2 < n:
                    k = line.find(line[j2], j2 + 1)
                    k = n - 1 if k < 0 else k
                    keep(line[i:j2 + 1])
                    protect(line[j2 + 1:k])
                    keep(line[k:k + 1] if k > j2 else "")
                    i = max(k + 1, j2 + 1)
                    continue
            elif name in URL_COMMANDS:
                j2 = j
                while j2 < n and line[j2] in " \t":
                    j2 += 1
                if j2 < n and line[j2] == "{":
                    k = _match_brace(line, j2)
                    if k is not None:
                        keep(line[i:j2 + 1])
                        protect(line[j2 + 1:k])
                        keep("}")
                        i = k + 1
                        continue
            keep(m.group(0))
            i = j
            continue
        if ch == "%":
            return ScannedLine(line, "".join(content), "".join(code), line[i:], verbatim), None
        keep(ch)
        i += 1
    return ScannedLine(line, "".join(content), "".join(code), None, verbatim), verb_end


def scan_text(text: str) -> list[ScannedLine]:
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    out, end = [], None
    for ln in lines:
        rec, end = scan_line(ln, end)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Comment removal
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class StripStats:
    full_lines: int = 0          # comment-only lines deleted
    trailing: int = 0            # trailing comments deleted
    joined: int = 0              # line ends joined to the next line in place of a %
    pars: int = 0                # "text%" + blank line written as "text\par" with that blank line removed
    markers: list = field(default_factory=list)   # (line number, reason) of each bare % kept
    line_map: list = field(default_factory=list)  # output line k (1-based) starts on source line line_map[k - 1]

    def add(self, other: "StripStats") -> None:
        self.full_lines += other.full_lines
        self.trailing += other.trailing
        self.joined += other.joined
        self.pars += other.pars
        self.markers.extend(other.markers)


_CTRL_WORD_END = re.compile(r"(?<!\\)(?:\\\\)*\\[A-Za-z]+\Z")
_AT_WORD_END = re.compile(r"(?<!\\)(?:\\\\)*\\[A-Za-z@]*@[A-Za-z@]*\Z")
_LETTER_OR_AT = re.compile(r"[A-Za-z@]")


def _comment_only(rec: ScannedLine) -> bool:
    return rec.comment is not None and not rec.verbatim and rec.content.strip(" \t") == ""


def _line_end(cur: str) -> tuple[str, str]:
    """What the end of a line needs once its comment is gone: ('drop', text) or ('join', text)."""
    if cur[-1:] in (" ", "\t"):
        s = cur.rstrip(" \t")
        backslashes = len(s) - len(s.rstrip("\\"))
        if backslashes % 2:                    # "\ " or "\<tab>" (a control symbol): keep that blank, which must
            return "join", s + cur[len(s)]     # not reach the line end, where TeX would strip a space
        return "drop", s                       # TeX reads one space, from the blank or from the line end
    if _CTRL_WORD_END.search(cur):
        return "drop", cur                     # after a control word TeX ignores the line end
    return "join", cur


def _join_target(recs: list[ScannedLine], idx: int, cur: str) -> tuple[str, int]:
    """('', j) when the line idx (now cur) may take line j onto it; otherwise (reason, j).

    The reason "blank" means that line j is blank: TeX then reads cur and a \\par, which cur + "\\par" gives too.
    """
    if recs[idx].verbatim:
        return "verbatim on the line", -1
    if "\\endinput" in cur:
        return "\\endinput on the line", -1
    j = idx + 1
    while j < len(recs) and _comment_only(recs[j]):
        j += 1
    if j >= len(recs):
        return "end of file", -1
    nxt = recs[j]
    if nxt.raw.strip(" \t") == "":
        return "blank", j
    if cur.endswith("^"):
        return "^ at the line end", -1
    if nxt.verbatim:
        return "verbatim on the next line", -1
    if "\\endinput" in nxt.code:
        return "\\endinput on the next line", -1
    first = nxt.content.lstrip(" \t")[:1]
    if _AT_WORD_END.search(cur) and _LETTER_OR_AT.match(first):
        return "@ in a control sequence at the line end", -1
    return "", j


def strip_comments(text: str) -> tuple[str, StripStats]:
    """Remove every LaTeX comment from text without changing the tokens TeX reads (see the module docstring)."""
    ends_nl = text.replace("\r\n", "\n").endswith("\n")
    recs = scan_text(text)
    st = StripStats()
    out: list[str] = []
    lmap = st.line_map                         # source line on which each output line starts
    pending: str | None = None
    pend_line = 0                              # source line on which the pending (joined) line starts
    skip = -1                                  # a blank line already given as \par
    for idx, rec in enumerate(recs):
        if idx == skip:
            continue
        start = idx + 1 if pending is None else pend_line
        if rec.comment is None:
            out.append(rec.content if pending is None else pending + rec.content.lstrip(" \t"))
            lmap.append(start)
            pending = None
            continue
        if _comment_only(rec):
            st.full_lines += 1
            continue
        st.trailing += 1
        cur = rec.content if pending is None else pending + rec.content.lstrip(" \t")
        pending = None
        action, cur = _line_end(cur)
        if action == "drop":
            out.append(cur)
            lmap.append(start)
            continue
        why, j = _join_target(recs, idx, cur)
        if why == "blank":                     # "text%" + blank line: TeX reads text, \par
            out.append(cur + "\\par")
            lmap.append(start)
            skip = j
            st.pars += 1
        elif why:
            out.append(cur + "%")
            lmap.append(start)
            st.markers.append((idx + 1, why))
        else:
            pending, pend_line = cur, start
            st.joined += 1
    if pending is not None:                    # unreachable: _join_target refuses at the end of the file
        out.append(pending + "%")
        lmap.append(pend_line)
    result = "".join(ln + "\n" for ln in out)
    # Without a final newline in the source, drop the one after the last line only if that line is still the last
    # one written; when the last source line was a deleted comment, the line before it did end with a newline.
    if result and not ends_nl and recs and not _comment_only(recs[-1]) and len(recs) - 1 != skip:
        result = result[:-1]
    return result, st


def remaining_comments(text: str) -> list[tuple[int, str]]:
    """(line, comment) of every comment character left outside verbatim text; a bare '%' is a line-join marker."""
    return [(i + 1, r.comment) for i, r in enumerate(scan_text(text)) if r.comment is not None]


def code_text(text: str) -> str:
    """The text without comments, with verbatim, \\verb and URL text blanked (for reference and macro scans)."""
    return "\n".join(r.code for r in scan_text(text))
