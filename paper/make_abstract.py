#!/usr/bin/env python
"""Derive abstract.txt from main.tex, so the two cannot drift apart.

The abstract exists twice: typeset in the PDF and as plain text pasted into a
submission form. Keeping two copies by hand is how they end up disagreeing, and
a reader comparing the listing to the paper would have no way to tell which was
current.

It also enforces arXiv's hard cap. Abstracts over 1920 characters are rejected
at the metadata step, which is late enough to be annoying and easy to miss,
because nothing about a 2600-character abstract looks wrong until the form
refuses it.

    python make_abstract.py
"""
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ARXIV_LIMIT = 1920

# LaTeX to plain text. Ordered: commands with arguments before bare symbols, so
# that \sqrt{2} is consumed before \s... would be. Each symbol keeps a trailing
# space where one is needed, because \sigma\sqrt{2} has no space in the source
# and "sigmasqrt(2)" in the output would be a typo the form would happily take.
SUBS = [
    (r"\\citet\{wilt2016\}", "Wilt & Ruml (2016)"),
    (r"\\citep\{([^}]*)\}", ""),
    (r"\\citet\{([^}]*)\}", ""),
    (r"\\emph\{([^}]*)\}", r"\1"),
    (r"\\mathrm\{([^}]*)\}", r"\1"),
    (r"\\text\{([^}]*)\}", r"\1"),
    (r"\\sqrt\{2\}", "sqrt(2)"),
    (r"\\Phi", "Phi"),
    (r"\\Delta", "Delta"),
    (r"\\sigma", "sigma "),
    (r"\\tau", "tau"),
    (r"\\times", "x"),
    (r"\\!", ""),
    (r"\\,", " "),
    (r"\^\{([^}]*)\}", r"^\1"),
    (r"\$([^$]*)\$", r"\1"),
]


def plain(tex):
    for pat, rep in SUBS:
        tex = re.sub(pat, rep, tex)
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r"\s+\)", ")", tex)          # "sigma )" from the trailing space
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return "\n".join(line.strip() for line in tex.strip().splitlines())


def main():
    src = io.open(HERE / "main.tex", encoding="utf-8").read()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
    if not m:
        raise SystemExit("no abstract found in main.tex")
    text = plain(m.group(1))
    (HERE / "abstract.txt").write_text(text + "\n")

    n = len(text)
    print(f"  abstract.txt: {n} characters, {len(text.split())} words")
    for bad in ("\\", "{", "}", "$"):
        if bad in text:
            print(f"  WARNING: leftover LaTeX character {bad!r} in the plain text")
    if n > ARXIV_LIMIT:
        print(f"  *** {n - ARXIV_LIMIT} characters OVER the arXiv limit of "
              f"{ARXIV_LIMIT}; the submission form will refuse it")
        return 1
    print(f"  arXiv limit {ARXIV_LIMIT}: OK, {ARXIV_LIMIT - n} to spare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
