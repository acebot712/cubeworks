# Decay is first-minus-last measured shell, and is therefore not comparable across tasks

Status: accepted

Decay is defined as `accs[0] - accs[-1]`, the ordering accuracy at a Profile's
first measured shell minus its last. The endpoints are whichever shells survived
the minimum-population filter, so a task measured over 20 shells has far more
room to fall than one measured over 4, and two Decays are only comparable when
computed over the same shells. We keep the definition because it is what every
committed number and every figure already means, and normalising it now would
silently restate published results.

## Consequences

Any analysis that pools Decay across tasks is invalid unless shell coverage is
matched. This is not hypothetical: adding the sliding-tile domain put 18-to-20
shell profiles into `strength_control.py`'s pool alongside 4-to-10 shell cube
ones, and the mean Decay differed by 2.5x for reasons of measurement range alone
(cube 0.155 over a median 7 shells, tile 0.390 over 20). That inverted the
paper's matched-strength null from Welch p = 0.478 to p = 0.042, purely as a
pooling artifact. Split by domain the cube reproduces the null at p = 0.924,
while the tile's 13 observations carry a decay-versus-strength correlation of the
opposite sign.

`strength_control.py` therefore reports per domain and never pools, the same
discipline `dprime_law.py` already applies to the two-moment law.

## Considered and rejected

**Normalise by shell count**, reporting decay per shell. Comparable across tasks,
but it is a different quantity from the one in every published figure and table,
and the relationship between accuracy and distance is not linear, so dividing by
shell count does not actually make two profiles commensurable.

**Fix a common shell range** across all tasks. Comparable, but it would discard
most of the sliding tile's range, which is where the interesting behaviour is,
and the common range across cube and tile is only about four shells.
