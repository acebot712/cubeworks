# Related work and positioning

Written against verified primary sources. Every claim about a prior paper below
was checked against its text, not its abstract. Where our contribution is
narrower than it first appeared, that is stated here rather than discovered by a
reviewer.

---

## 1. The nearest prior work: Goal Distance Rank Correlation

**Wilt & Ruml, "Effective Heuristics for Suboptimal Best-First Search", JAIR
57:273–306, 2016.**

This is the closest published work to ours and the one a reviewer will reach for
first. It must be confronted directly in the introduction, not buried.

Wilt & Ruml define **Goal Distance Rank Correlation (GDRC)**: Kendall's τ between
a heuristic `h(n)` and the *exact* edge-count distance-to-goal `d*(n)`, computed
offline by backward breadth-first search. Their §4.3: *"We will call this type of
metric the Goal Distance Rank Correlation (GDRC) and, unless otherwise noted,
compute it using Kendall's tau."*

Everything we might have claimed as novel about "a ranking measure against exact
ground truth" is theirs:

- it is **search-independent**: computed without running a forward search;
- it is a **ranking** measure, not a magnitude-error measure;
- it **predicts search cost**. Their p.293: *"when the GDRC is below roughly 0.4,
  greedy best-first search performs very poorly, but as the GDRC increases, the
  average number of expansions done by greedy best-first search decreases."*
- it is used **prospectively to construct** heuristics (§5, "Building a Heuristic
  by Searching on GDRC").

They also anticipate, in 2016, the "only ordering matters" position usually
attributed to Chrestien et al. (2023), with a scale-invariance argument (§4.4),
and they *empirically reject* h-vs-h\* correlation as a predictor of search cost.

**What is left for us.** GDRC is a **single pooled scalar over all states**. Their
Algorithm 1 samples 10% of the nodes found by backward BFS and returns one τ.
A full-text search of the paper for `stratif`, `shell`, `bucket`, `binned`,
`each depth`, and `conditional` returns zero hits; Tables 4 and 5 report exactly
one coefficient per domain. The one object resembling a decomposition is Figure
10, an unbinned h-vs-d\* scatter used qualitatively.

Pooling is a reasonable choice for the heuristics they study: pattern databases,
whose quality has no particular reason to vary with distance from the goal. It is
the wrong choice for a heuristic trained by bootstrapping, whose accuracy is
anchored at the goal and degrades outward by construction. Our measurement is
GDRC **stratified by true distance**, and §4 shows two heuristics on the same
problem whose pooled τ ranks one above the other while their profiles cross.

We therefore claim a **refinement of an existing measure applied to a class of
heuristics it was not designed for**, not a new measure.

### 1.1 A structural blind spot: τ is diluted by the pairs search never makes

There is a second, sharper problem with the pooled statistic, and it concerns
*which pairs of states enter the coefficient*.

Kendall's τ is computed over all pairs. In a state distribution spanning
distances 1 to 9, most randomly drawn pairs are far apart, and far-apart pairs
are easy to order correctly. A search never makes those comparisons. Beam search,
greedy best-first and A\* all compare a node against its siblings, which are one
move apart and therefore in adjacent distance shells. **The comparison a search
depends on is the hardest one, and it is the one τ dilutes most.**

The effect is large enough to invert a verdict. Across three seeds of our deepest
variant (rung k = 4, `oi-q3`, diameter 12, identical data and budget), one run
degenerated:

| seed | pooled τ | span of `h` | adjacent-shell accuracy |
| ---- | -------- | ----------- | ----------------------- |
| 0    | **+0.660** | 0.004 | **0.616** |
| 1    | +0.936 | 10.256 | 0.932 |
| 2    | +0.936 | 10.839 | 0.925 |

Seed 0's τ of +0.660 sits in the same range as the pattern database's +0.693 on
the same states, so the pooled statistic rates a barely-functional heuristic
alongside a working classical one. Resolving that same heuristic by pair
separation shows why:

| shell separation | 1 | 2 | 4 | 8 |
| ---------------- | - | - | - | - |
| ordering accuracy | **0.616** | 0.741 | 0.886 | 0.993 |

It orders distant states almost perfectly and adjacent ones barely above chance.
τ reports the average over that range; search experiences only the left-hand
column.

**A correction we make explicitly.** We first attributed this to Kendall's τ
being scale-invariant (seed 0's outputs span only 0.004) and that explanation
is wrong. A heuristic with a small span and clean ordering works perfectly well;
scale-invariance is a *feature* here, and Wilt & Ruml rely on it deliberately
(§4.4). What actually degrades seed 0 is that its within-shell spread
(sd ≈ 6 × 10⁻⁴) exceeds its between-shell gap (≈ 2 × 10⁻⁴), so adjacent shells
overlap. Span alone is not a diagnostic and we do not propose it as one; it is
recorded only as context. The diagnostic is the adjacent-shell profile itself.

One caution against over-reading the case. Seed 0 still solved 73.5% of its
evaluation scrambles. That is beam search brute-forcing the problem: 12
generators at beam width 100 over depth 60 touches an appreciable fraction of a
255,024-state space, not the heuristic contributing. It is a further reason
solve rate alone is a poor instrument for judging a heuristic, and a reader must
not take 73.5% as evidence that this heuristic partially worked.

---

## 2. Predicting search cost from a heuristic: the classical line

**Korf, Reid & Edelkamp, "Time complexity of iterative-deepening-A\*", AIJ
129:199–218, 2001; Korf, AAAI-07.**

The KRE formula `E(N, d, P) = Σ_i N_i · P(d − i)` predicts IDA\* node expansions
from the search depth, brute-force node counts, and the heuristic's cumulative
*value* distribution. It is accurate to within 1% on Rubik's Cube and the sliding
puzzles at depths 10–50, independently replicated by Zahavi et al. (JAIR 37) with
prediction/actual ratios of 0.93–0.99.

So "an intrinsic property of a heuristic predicts search cost" is long settled.

Two things distinguish it from what we measure. First, KRE predicts from a
**magnitude-value distribution**, and `h*` appears nowhere in its characterisation
of a heuristic: Korf states the omission deliberately: *"the characterization of
a heuristic function in terms of its distribution is not a measure of the accuracy
of the function. In particular, it says nothing about the correlation of heuristic
values with actual costs."* Two heuristics with identical value histograms but
different state-to-value assignments, one a perfect ranking of `h*`, one shuffled: receive **identical KRE predictions**. The formula is provably ranking-blind.

Second, and importantly for us, KRE **contradicts** a framing we initially
adopted. Its cost is proportional to the brute-force node counts `N_i`, and the
paper proves *"the asymptotic heuristic branching factor is the same as the
brute-force branching factor"*, so cost scales as `b^(d−k)`. Any claim that
required search budget is *independent of state-space growth* is inconsistent with
this. **We do not make that claim.** Our depth experiment (§3) holds cardinality
exactly constant and varies depth and branching together; it can rule size out as
the operative variable in that design, and nothing more.

**Zahavi et al.'s Conditional Distribution Prediction** conditions on parent and
grandparent heuristic value, node type and operator: never on `h*`. **CDP and
KRE stratify by depth in the brute-force search tree** (distance from the *start*),
which is a different variable from distance to the *goal*.

---

## 3. Why ranking is the right object

**Chrestien, Pevný, Edelkamp & Komenda, NeurIPS 2023** prove that forward search
with `f(s) = αg(s) + βh(s)` is strictly optimally efficient *iff* `h` is a perfect
ranking, concluding that *"the absolute value of the heuristic is not important"*.
They credit the observation to **Xu & Fern (2007, 2009)** for beam search and to
earlier work for greedy best-first search.

This motivates measuring ranking rather than error, but it is prior art for the
principle, not for any measurement. We cite it as justification for the choice of
quantity and claim nothing from it.

---

## 4. Learned heuristics, and what is known to go wrong with them

**Agostinelli, McAleer, Shmakov & Baldi, "Solving the Rubik's cube with deep
reinforcement learning and search", Nature Machine Intelligence 1:356–363, 2019**
introduces Deep Approximate Value Iteration, the training procedure we study.
**Chervov et al. (CayleyPy, NeurIPS 2025 Spotlight)** extend learned solvers to
the 4×4×4 and 5×5×5 cube; their trained weights are not publicly released, which
is why our external comparison uses DeepCubeA's and EfficientCube's released 3×3
checkpoints instead.

Three known failure modes are adjacent to ours and must be distinguished:

**Distributional shift (Hadar, Agostinelli & Shperberg, AAAI-26).** *"the search
process favors nodes with lower heuristic values, leading to a significant
overrepresentation of states with lower heuristic estimates"*, causing systematic
underestimation in the region search visits. This is a *train/deploy* mismatch
driven by the search's own sampling. Our profile is measured on states drawn
independently of any search, so the decay we report is a property of the trained
function rather than of what a search chooses to look at. The two effects are
compatible and probably compound.

**Heuristic depression (Hernández & Baier, JAIR 43:523–570, 2012).** A *bounded
region* of the state space in which the heuristic misleads, trapping *real-time*
search. Full-text counts of that paper: `neural` = 0, `beam` = 0, `offline` = 0.
Depression is local and geometric; what we measure is a global, monotone
degradation as a function of distance from the goal. A reviewer will raise this
objection, and §4 answers it by showing the effect is monotone in `d` across the
whole space rather than confined to a region.

**Value monotonicity (Czechowski et al., kSubS, NeurIPS 2021).** Quantifies how a
learned value function's ordering degrades as a function of the **step spacing
between the states being compared** along a path, and reports a signal-to-noise
ratio at one fixed spacing. This is the nearest quantitative statement to ours and
is genuinely close. It differs in the conditioning variable: spacing between two
compared states, versus absolute distance from the goal. Ours can therefore
exhibit a *profile* (a curve over `d`) where theirs yields a single number.

**Zawalski et al. (arXiv:2406.03361)** observe qualitatively that learned value
functions fail on states far from the goal, but probe this by injecting
*distance-independent Gaussian noise* into a trained value function, with no
comparison against ground truth and no stratification. Every domain in that paper
is fixed at a single scale.

---

## 5. Scaling, capacity and admissibility

**Pendurkar, Huang, Koenig & Sharon (TMLR 2023)** show that the parameters needed
to approximate a heuristic grow exponentially in instance size, and prove that
*"if P≠NP, then it is impossible to calculate each value of a high-precision
approximation of the completely informed heuristic function in time polynomial in
the instance sizes."* Their sweep varies *instance size* across pancake, TSP and
Blocks World, with branching factor changing along with it. Ours holds branching,
encoding, architecture and budget fixed and varies only the projection, which is
what lets us separate size from depth in §3, but their result is the stronger
statement about representational capacity and we do not contest it.

**Futuhi & Sturtevant (ICLR 2026)** publish ordinal/classification value losses,
PAC-dimension bounds for learned heuristics, and *near-admissible* heuristics with
measured overestimation rates of 3×10⁻⁷ to 9×10⁻⁵: nonzero, and self-described as
"near-admissible". Sound admissibility from a learned heuristic remains
unachieved. Their ordinal losses are a natural remedy to test against the decay we
report, and we treat them as a baseline rather than as a contribution of ours.

**Arfaee, Zilles & Holte (AIJ 175:2075–2098, 2011)** bootstrap a heuristic from a
weak one using *backward random walks of increasing length* when the initial
heuristic solves nothing. That is the depth curriculum we use. It is prior art;
we adopt it and claim nothing for it.

---

## 6. What we claim, and what we do not

**We claim:**

1. **Ranking quality decays with distance from the goal for every informative
   heuristic on this family** (learned and classical alike) and the size of
   the decay is predicted by heuristic strength (r = +0.588, n = 39,
   p = 8.3e-5), not by training method. Controlling for strength leaves no
   detectable difference (Welch t = +0.72, p = 0.48; difference +0.023, 95% CI
   [−0.039, +0.084] against a mean decay of 0.154), and matched on identical
   states the abstraction decays more in 13 of 18 pairs (§4).

   > **Superseded.** An earlier version of this claim asserted that learned
   > heuristics decay *while pattern databases do not*. That comparison used a
   > single abstraction, rung k=2, whose pooled τ on rung k=6 is +0.570: near
   > enough to chance that it is flat for want of accuracy to lose. Reporting
   > every available abstraction reverses the reading: the steepest decay on
   > that rung belongs to PDB(k=5), the strongest heuristic in the comparison.
   > Retained here rather than deleted, because the failure mode generalises:
   > a profile comparison is only meaningful between heuristics matched on
   > overall strength.

2. This structure is **invisible to a pooled measure**, and now demonstrably so
   within GDRC's own original scope, since the pattern databases it was designed
   around have distance-dependent profiles too.
3. A pooled τ is **diluted by pairs of states no search ever compares**. Ordering
   accuracy on our degenerate seed runs 0.616 for adjacent shells and 0.993 for
   shells eight apart; τ averages these to +0.660, which is indistinguishable
   from a working pattern database's +0.693. Search only ever compares
   near-neighbours, so the adjacent-shell figure is the one that matters (§1.1).
4. On a family where state-space cardinality is held **exactly constant** and only
   the diameter varies, performance degrades, so cardinality is not the operative
   variable in that design (§3). The decay itself grows with diameter: 0.108 ±
   0.008 at diameter 6, rising to 0.259 ± 0.029 at diameter 10, all at 255,024
   states.
5. At the deepest variants, training becomes **unreliable rather than uniformly
   worse**: seeds disagree, one collapsing entirely while its siblings succeed.
   The same pattern appears across the size ladder, where seed variance peaks in
   the transition region (±23.5 percentage points at k = 10) and is near zero
   where the method clearly works or clearly fails.

**We do not claim:**

- that a ranking statistic predicting search cost is novel (Wilt & Ruml 2016);
- that predicting search cost from an intrinsic heuristic property is novel
  (Korf–Reid–Edelkamp);
- that required search budget is independent of state-space size: KRE implies
  otherwise, and our design cannot separate depth from branching factor;
- that the profile distinguishes learned heuristics from classical ones: our
  own strength-controlled test refutes this, and the null result is reported
  with its confidence interval rather than as a proof of equality;
- any admissibility or optimality guarantee for a learned heuristic;
- a new solver, or state-of-the-art performance on any benchmark.

---

## 7. Future work we deliberately do not attempt here

Conformal calibration of a learned heuristic into a sound probabilistic lower
bound is attractive, but the template is published: **Clarke & Stellato
(arXiv:2602.01476)** conformally calibrate a learned optimality-gap estimator
inside MIP branch-and-bound for a distribution-free bounded-suboptimality
guarantee, and their earlier work conformally lower-bounds the true optimum. What
remains open is a **per-state** bound `h_lower(s) ≤ h*(s)` consumed inside node
ordering, rather than a per-instance certificate used as a stopping rule, and the
obstacle is that per-state marginal guarantees do not compose over the many states
a search expands. We flag this as open rather than solved.
