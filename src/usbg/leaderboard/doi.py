"""usbg.leaderboard.doi: resolve a submitted method's DOI at its registry and build its reference.

``python -m usbg.leaderboard add`` (maintainers) calls :func:`resolve` for the DOI of the submission: the record is
fetched from Crossref (https://api.crossref.org/works/<doi>), or from DataCite (https://api.datacite.org/dois/<doi>,
which registers the arXiv DOIs 10.48550/arXiv.<id>) when Crossref has none. The registry title must equal the
title the maintainer gives (``--confirm-title``, compared after normalization), so a DOI of another work is
refused. The reference printed in the methods list is built from the registry record, in the style of the
published entries (authors with initials, title, venue, volume, year, pages), so every DOI in the methods list was
checked against its registry.

A record saved beforehand (the JSON body of either API) can be given with ``--doi-metadata FILE`` instead of a
download.
"""
from __future__ import annotations

import datetime
import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

CROSSREF = "https://api.crossref.org/works/{doi}"
DATACITE = "https://api.datacite.org/dois/{doi}"
MAX_AUTHORS = 20


def norm_title(t):
    """Titles compared after normalization: ASCII folding, lower case, punctuation and spacing ignored."""
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "usbg-leaderboard (https://github.com/tohnperfect/USBGym)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch(doi):
    """The registry record of a DOI: Crossref first, then DataCite. -> (registry, JSON body)."""
    q = urllib.parse.quote(doi, safe="/")
    errors = []
    for registry, pattern in (("Crossref", CROSSREF), ("DataCite", DATACITE)):
        try:
            return registry, _get(pattern.format(doi=q))
        except urllib.error.HTTPError as e:
            errors.append(f"{registry}: HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            errors.append(f"{registry}: {e}")
    raise SystemExit(f"DOI {doi}: no registry record ({'; '.join(errors)})")


def _clean(t):
    """Registry text as plain text: markup tags (Crossref titles may carry <i>, <sub>) and spacing removed."""
    return " ".join(re.sub(r"<[^>]{0,40}>", "", str(t or "")).split())


def _initials(given):
    parts = []
    for word in re.split(r"\s+", (given or "").strip()):
        if not word:
            continue
        sub = [p for p in word.split("-") if p]
        parts.append("-".join(p[0].upper() + "." for p in sub))
    return "".join(parts)


def parse(registry, body):
    """{registry, title, authors [str], year, container, volume, issue, pages, publisher, type} of a record."""
    if registry == "Crossref" or "message" in body:
        m = body["message"]
        year = None
        for k in ("published-print", "issued", "published-online", "published"):
            dp = (m.get(k) or {}).get("date-parts") or [[None]]
            if dp and dp[0] and dp[0][0]:
                year = int(dp[0][0])
                break
        issued = ((m.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        authors = []
        for a in m.get("author") or []:
            fam = _clean(a.get("family") or a.get("name") or "")
            ini = _initials(a.get("given"))
            authors.append(f"{ini} {fam}".strip())
        return {"registry": "Crossref", "doi": m.get("DOI"), "title": _clean((m.get("title") or [""])[0]),
                "authors": authors, "year": year, "issued_year": int(issued) if issued else year,
                "container": _clean((m.get("container-title") or [""])[0]),
                "volume": m.get("volume"), "issue": m.get("issue"),
                "pages": m.get("page") or m.get("article-number"),     # a record with an article number has no pages
                "publisher": m.get("publisher"), "type": m.get("type")}
    a = body["data"]["attributes"]
    authors = []
    for c in a.get("creators") or []:
        if c.get("familyName"):
            authors.append(f"{_initials(c.get('givenName'))} {c['familyName']}".strip())
        elif c.get("name"):
            authors.append(c["name"])
    titles = a.get("titles") or [{}]
    cont = a.get("container") or {}
    return {"registry": "DataCite", "doi": a.get("doi"), "title": _clean(titles[0].get("title")),
            "authors": authors, "year": a.get("publicationYear"), "issued_year": a.get("publicationYear"),
            "container": cont.get("title") or "", "volume": cont.get("volume"), "issue": cont.get("issue"),
            "pages": None, "publisher": a.get("publisher") if isinstance(a.get("publisher"), str)
            else (a.get("publisher") or {}).get("name"), "type": (a.get("types") or {}).get("resourceTypeGeneral")}


def reference_text(rec):
    """The reference in the style of the published entries of the methods list."""
    au = rec["authors"]
    who = ", ".join(au[:MAX_AUTHORS]) + (", et al." if len(au) > MAX_AUTHORS else "")
    title = rec["title"]
    pages = (rec.get("pages") or "").replace("--", "-")
    year = rec.get("year")
    if rec.get("type") == "journal-article" and rec.get("container"):
        vol = f" {rec['volume']}" if rec.get("volume") else ""
        txt = f"{title}, {rec['container']}{vol} ({year})" + (f" {pages}" if pages else "")
    elif rec.get("container"):
        txt = f"{title}, in: {rec['container']}" + (f", {rec['publisher']}" if rec.get("publisher") else "") + \
              f", {year}" + (f", pp. {pages}" if pages else "")
    else:
        txt = f"{title}, {rec.get('publisher') or rec['registry']}, {year}"
    return (f"{who}, " if who else "") + txt + "."


def resolve(doi, confirm_title, metadata=None):
    """Resolve and verify a DOI. -> the work record stored in the methods list. Refuses a DOI whose registry title
    differs from confirm_title (after normalization)."""
    if metadata is not None:
        body = metadata
        registry = "Crossref" if "message" in body else "DataCite"
    else:
        registry, body = fetch(doi)
    rec = parse(registry, body)
    if (rec.get("doi") or "").lower() != doi.lower():
        raise SystemExit(f"the registry record is for DOI {rec.get('doi')!r}, not {doi!r}")
    if not rec["title"] or not rec["year"]:
        raise SystemExit(f"DOI {doi}: the {registry} record has no title or year")
    if confirm_title is None or norm_title(confirm_title) != norm_title(rec["title"]):
        raise SystemExit(f"DOI {doi}: the {registry} title is {rec['title']!r}; rerun with --confirm-title equal to "
                         "it after checking that it is the submitted method's publication")
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    return {"key": f"doi:{doi}", "reference": reference_text(rec), "title": rec["title"], "year": int(rec["year"]),
            "doi": doi, "no_doi": False, "doi_url": "https://doi.org/" + urllib.parse.quote(doi, safe="/"),
            "verified": {"registry": registry, "title": rec["title"], "issued_year": rec.get("issued_year"),
                         "print_year": rec["year"], "checked": today}}


__all__ = ["resolve", "fetch", "parse", "reference_text", "norm_title"]
