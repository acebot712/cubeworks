# Overleaf / local build

Upload `tmlr-submission.zip` to Overleaf (New Project → Upload Project). It
compiles as-is with pdfLaTeX; no configuration needed.

## Contents

```
main.tex              the paper
refs.bib              references
tmlr.sty  tmlr.bst    official TMLR style, from github.com/JmlrOrg/tmlr-style-file
fancyhdr.sty          dependency of tmlr.sty
figures/*.pdf         vector figures, converted from the SVGs the analysis emits
tables/*.tex          generated from eval/results/*.json, not typed by hand
```

## Submission-ready as is

Nothing needs changing before uploading to OpenReview. The build is anonymised
for double-blind review: the title page reads "Anonymous authors / Paper under
double-blind review" and the header "Under review as submission to TMLR". The
`\author{...}` block in `main.tex` is ignored in this mode and the compiled PDF
was checked, including inside compressed streams and metadata, to confirm it
contains no author name, email, affiliation or project identifier.

`\def\openreview{XXXXXX}` is likewise never rendered here; `tmlr.sty` only
prints it under `[accepted]`.

**For the camera-ready version only**, after acceptance: change line 2 to
`\usepackage[accepted]{tmlr}` to reveal the author block, and set `\openreview`
to the assigned ID.

## Regenerating figures and tables

Both are derived, never hand-edited. From the repository root:

```bash
# tables, straight from the result JSON
./.venv-mlx/bin/python paper/build_tables.py

# figures: render at localhost:5183/figures.html, export SVG, then
./.venv-mlx/bin/python paper/svg2pdf.py
```

If a measurement is re-run, the JSON changes and both propagate. See
`../REPRODUCE.md` for how each result file is produced.

## Verified

Compiled with Tectonic 0.15.0: **9 pages**, no undefined references, no
undefined citations, no bibliography warnings. Every `\input`, `\includegraphics`
and `\ref` resolves.
