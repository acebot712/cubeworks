# Submitting to TMLR

Everything is ready. What remains needs your OpenReview account, so you do it —
no assistant should be typing your credentials.

## Files

| what | where |
|---|---|
| the PDF you upload | `paper/main.pdf` — 14 pages, anonymous |
| the abstract, to paste | `paper/abstract.txt` — 353 words |
| supplementary code to attach | `paper/supplementary-anonymous.zip` — 1.6 MB, no identity, no photographs, no git history |
| LaTeX source, if you'd rather build on Overleaf | `paper/tmlr-submission.zip` |

## Metadata

**Title** (one line, no line break):

    Learned Heuristics Decay With Distance to Goal — and So Do Pattern Databases, by the Same Two-Moment Law

**Abstract** — paste the contents of `paper/abstract.txt`.

**Keywords** — heuristic search; learned heuristics; pattern databases; rank
correlation; signal detection; evaluation methodology; Rubik's cube

**Authors** — entered into the OpenReview form, never into the PDF. OpenReview
hides authorship from reviewers and reveals it on acceptance.

## Steps

1. Sign in at <https://openreview.net>. If you have no profile, create one first
   — new profiles take a day or two to become usable for submission, so do this
   before you need it.
2. Find the TMLR venue page and choose **Submit**.
3. Paste the title, abstract and keywords. Add yourself as author.
4. Upload `paper/main.pdf`.
5. Attach `paper/supplementary-anonymous.zip` as supplementary material.
6. Answer the venue questions. The three that need a real decision:
   - *Prior or concurrent submission*: no.
   - *Conflicts of interest*: your own affiliations and recent collaborators.
   - *Code and data availability*: **yes — anonymous supplementary attached; the
     repository will be released publicly on acceptance.** Do not link the
     GitHub repo. It is under your own account and would identify you.
7. Submit, then open the anonymous preview OpenReview renders back and read the
   first page. That preview is what reviewers see.

## Do not

- **Do not link the GitHub repo.** It stays private until the decision. Publishing
  it under your account de-anonymizes the submission — which is the whole reason
  the supplementary archive exists.
- **Do not restore the author block.** `main.tex` has it commented out on
  purpose; the source is not the PDF, and reviewers may receive the source.

## At camera-ready, after acceptance

1. Uncomment the `\author{...}` block in `main.tex` and fill in your details.
2. Change line 2 to `\usepackage[accepted]{tmlr}`.
3. Set `\def\openreview{...}` to the assigned ID (currently `XXXXXX`; it only
   renders under `[accepted]`).
4. Push the repository public. `eval/frames/` stays excluded permanently — see
   `eval/README.md`.

## What to expect

TMLR judges two things: whether the claims are supported by evidence, and whether
some subset of the community would be interested. It does not judge novelty or
impact. That suits this paper, whose strength is careful measurement and whose
weakness is a classical core identity and a single domain.

A realistic first round asks for external validity — does the result hold on the
sliding-tile puzzle or in classical planning. That is a fair request and the
honest answer is that we have not tested it. A second domain is roughly a week of
engineering, mostly a general permutation ranking function to replace the
base-24 indexer in `exact.py`.
