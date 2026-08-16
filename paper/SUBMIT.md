# Submitting to TMLR

Everything below is ready. What remains needs your OpenReview account, so it has
to be done by you — no assistant should be typing your credentials.

## Files

| what | where |
|---|---|
| the PDF you upload | `paper/main.pdf` (14 pages, anonymous) |
| LaTeX source, if you'd rather build on Overleaf | `paper/tmlr-submission.zip` |
| abstract, to paste into the form | `paper/abstract.txt` (353 words) |
| reproducibility appendix, if you attach one | `REPRODUCE.md` |

## Metadata to paste

**Title**

    Learned Heuristics Decay With Distance to Goal — and So Do Pattern
    Databases, by the Same Two-Moment Law

**Abstract** — the contents of `paper/abstract.txt`.

**Keywords** — heuristic search; learned heuristics; pattern databases;
rank correlation; signal detection; evaluation methodology; Rubik's cube.

**Authors** — entered into the OpenReview form, not into the PDF. The PDF stays
anonymous; OpenReview keeps authorship hidden from reviewers and reveals it on
acceptance. You will need an OpenReview profile with a verified email.

## Steps

1. Sign in at <https://openreview.net> (create a profile first if you don't have
   one; profiles need a day or two to become usable for submission).
2. Go to the TMLR venue page and choose **Submit**.
3. Paste title, abstract, keywords. Add yourself as author.
4. Upload `paper/main.pdf`.
5. Answer the venue questions. The ones that need a real decision:
   - *Prior/concurrent submission*: no.
   - *Conflicts of interest*: your own affiliations.
   - *Code/data availability*: yes — the repository, once you decide whether to
     make it public. It currently has no git remote.
6. Submit, then check the anonymous preview OpenReview renders back to you.

## Before you click submit

- **The repository is not public and has no remote.** If you say code is
  available, publish it first, and check the history — it contains the
  superseded results and the corrections, which is fine and arguably good, but
  it is your call.
- **The author block in `main.tex` is commented out on purpose.** Restore it
  together with `\usepackage[accepted]{tmlr}` only for the camera-ready.
- `\def\openreview{XXXXXX}` is a placeholder that only renders under
  `[accepted]`. Set it to the assigned ID at camera-ready time.

## What to expect

TMLR judges two things: whether the claims are supported by the evidence, and
whether some subset of the community would be interested. It does *not* judge
novelty or impact. That suits this paper, whose strength is careful measurement
and whose weakness is a classical core identity and a single domain.

A realistic outcome is a first round asking for external validity — does the
result hold on the sliding-tile puzzle or in classical planning. That is a fair
request and the honest answer is that we have not tested it; the apparatus for a
second domain is roughly a week of engineering, mostly a general permutation
ranking function to replace the base-24 indexer.
