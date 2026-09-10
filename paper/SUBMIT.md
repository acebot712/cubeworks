# Submitting

> **Status, 2026-09-10.** The arXiv path is **unblocked**. TMLR desk-rejected
> submission 11461 on 23 August without review, citing reviewer bandwidth rather
> than content. The plan is an arXiv preprint now, then a search venue such as
> SoCS. The TMLR notes are kept further down because a revised resubmission
> remains possible.

Everything is built and checked. What remains needs your arXiv account, so you
do it. No assistant types your credentials or accepts a licence in your name.

## The category, and the endorsement that is still blocking

Submit with **`cs.LG` as the primary**, cross-listing `cs.AI`.

`cs.LG` is an honest primary rather than a door that happened to be open. Learned
heuristics are the object of study and the central claim is a statistical law
about how a learned predictor's values distribute within and between shells.
`cs.AI` is the more precise home for the search lineage, ACM class I.2.8 is
literally Problem Solving, Control Methods and Search, hence the cross-list.

**Both categories are gated, and this is the live blocker.** Tested by clicking
Continue on the submission form, which is the only test that means anything: the
gate fires on submit, not on choosing a category from the dropdown. Selecting
`cs.LG` and seeing nothing happen is a false negative, and this document
previously recorded one.

The cause is not the publication record. The author has three arXiv papers,
including one with `cs.LG` as its primary category and the author first on it.
None of them is linked to the arXiv account, so arXiv sees a first-time submitter
and auto-endorsement has nothing to read. The account page says as much, listing
no articles and asking whether there are any "not listed here".

Three routes, and the first two are worth running in parallel with the third:

1. **Claim the papers with a paper password.** Fastest, potentially immediate. The
   submission confirmation e-mail for each paper carries a paper password; the
   *Claim Ownership with a password* form takes it. If the password is lost, the
   *Recover Paper Password* form sends it to whoever submitted the paper, which
   may be a co-author rather than the author.
2. **Claim Authorship form** (`/auth/request-ownership`), which takes up to ten
   arXiv IDs at once. arXiv's own warning: it "requires manual intervention, it
   make take several days for you to get a response, so you should treat this only
   as a last resort".
3. **An endorsement request** to a qualified arXiv submitter. The endorser pool is
   wider than it looks: for `cs.LG` they need three papers in ANY `cs.*` category,
   not `cs.LG` specifically.

**Routes 1 and 2 are the most promising and are not guaranteed.** arXiv's help
says authority records "support the endorsement system" but never states that
owning a paper in a category confers endorsement for it. Nobody should treat the
claim as a certain fix, which is why route 3 runs alongside rather than after.

## Files

| what | where |
|---|---|
| **the source bundle you upload** | `paper/arxiv-submission.tar.gz` |
| the abstract, to paste into the metadata form | `paper/abstract.txt`, 1887 characters, under arXiv's 1920 cap |
| the PDF, for your own reading | `paper/main.pdf`, 17 pages |
| supplementary code, TMLR only | `paper/supplementary-anonymous.zip` |
| LaTeX source for Overleaf, TMLR only | `paper/tmlr-submission.zip` |

**Upload the tarball, not the PDF.** arXiv declines PDFs produced from TeX when
the source exists. The manifest is derived from `main.tex`'s own `\input` and
`\includegraphics`, not hand-kept, and it was last verified by extracting it to a
clean directory and building it there with nothing else present.

Rebuild it if any table or figure changes:

```bash
cd paper && ../.venv-mlx/bin/python build_tables.py && ../.venv-mlx/bin/python make_abstract.py
```

`make_abstract.py` regenerates `abstract.txt` from `main.tex` and fails loudly if
it exceeds the cap or leaves a LaTeX control sequence where a number belongs.

## The arXiv form

**Primary category**: `cs.LG`. **Cross-list**: `cs.AI`.

**Title**:

    Learned Heuristics Decay With Distance to Goal: So Do Pattern Databases, by the Same Two-Moment Law

**Abstract**: paste `paper/abstract.txt` verbatim. It contains no LaTeX.

**Authors**: your own name and affiliation. Unlike TMLR this is not anonymous,
and `main.tex` currently has the author block commented out for TMLR's sake. That
does not matter for arXiv, which takes authorship from the metadata form rather
than the PDF, so there is nothing to uncomment before uploading.

**Comments**: optional. A line naming the code repository is worth adding, since
the repository is public and the paper already cites it at
`https://github.com/acebot712/cubeworks`.

**Licence**: your choice, and yours alone to accept. arXiv's default
non-exclusive licence is the least restrictive to your later options; a CC licence
is fine too and is what most preprints carry.

**Announcement is permanent.** arXiv states that announced content is archival and
cannot be removed. Read the processed PDF arXiv renders back before you click the
final confirmation, not after.

## After it is announced

1. Add the arXiv ID to `README.md` and to the repository description.
2. `eval/frames/` stays out of the repository and out of every archive,
   permanently. See `eval/README.md`.

---

# If you resubmit to TMLR

Kept because a revised resubmission remains possible. **Recheck the page figures
before using this**: the paper has grown since the August submission, and the
sliding-tile section is new.

## The form, field by field

**Title** and **Abstract**: as above.

**Authors**: your OpenReview profile is prefilled. Authors go here and never in
the PDF; OpenReview hides them from reviewers and reveals them on acceptance.

**PDF**: `paper/main.pdf`.

**Beyond PDF**: LEAVE EMPTY. This is not the supplementary field. It takes a ZIP
of an interactive webpage and only applies if you pick the Beyond PDF submission
type, which we are not.

**Submission Type**: **Long submission (more than 12 pages of main content).**
Main content is everything before references and appendices. It was 13 pages at
the August submission and the paper is longer now, so this stays Long. TMLR notes
that review may take longer for long submissions.

**Supplementary Material**: `paper/supplementary-anonymous.zip`. This is the field
for it, not Beyond PDF. The form warns supplementary material is visible to
reviewers and the public and must be anonymised; this archive is.

**Previous TMLR Submission Url**: submission 11461, if the resubmission is
presented as a revision of it. Leave empty otherwise.

**Changes Since Last Submission**: the sliding-tile domain is the substantive
addition. See below.

**Competing Interests** (required): answer honestly for the last 36 months. `N/A`
is acceptable if you have none. The form asks specifically about engagements with
commercial companies or startups, sabbaticals, employments, stipends, honorariums,
and donated hardware or cloud computing.

**Human Subjects Reporting** (required): `N/A`.

**License, Readers, Signatures**: fixed by the venue. Leave them.

There is no keywords field, no "prior or concurrent submission" question and no
"code and data availability" question on this form. Do not go looking for them.

## Do not

- **Do not link the GitHub repo** on TMLR, even if a field seems to invite a URL.
  It is under your own account and would identify you. The supplementary archive
  exists precisely so you do not have to. This does not apply to arXiv, which is
  not anonymous.
- **Do not put the zip in Beyond PDF.** It belongs in Supplementary Material.
- **Do not restore the author block** in `main.tex` for TMLR. The source is not
  the PDF, and reviewers may receive the source.

## At camera-ready, after acceptance

1. Uncomment the `\author{...}` block in `main.tex` and fill in your details.
2. Change line 2 to `\usepackage[accepted]{tmlr}`.
3. Set `\def\openreview{...}` to the assigned ID (currently `XXXXXX`; it renders
   only under `[accepted]`).

## What to expect

TMLR judges two things: whether the claims are supported by evidence, and whether
some subset of the community would be interested. It does not judge novelty or
impact. That suits this paper, whose strength is careful measurement.

The obvious first-round request, external validity, **has since been answered and
the answer is in the paper.** The sliding tile was built and measured through the
same code path, and it bounds the law rather than confirming it: mean absolute
error is 0.0117 on the cube over 252 observations against 0.0429 on the tile over
239, nearly four times larger, which is the figure the paper quotes. Four
explanations were tested and rejected, three of them backwards from prediction.
Skewness carries partial signal, so the finding is reframed as skew-limited rather
than domain-limited, with a residual above |skew| = 1 left unexplained. A referee asking
for a second domain will find one, and will find it does not fully cooperate,
which is stated rather than buried.
