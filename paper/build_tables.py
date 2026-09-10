#!/usr/bin/env python
"""Regenerate every LaTeX table from eval/results/*.json.

No number in the paper is typed by hand. Re-run a measurement and the table
that reports it changes with it.
"""
import json, pathlib, numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
ROOT = pathlib.Path(__file__).resolve().parent.parent
R, T = ROOT/'eval'/'results', pathlib.Path(__file__).parent/'tables'
T.mkdir(exist_ok=True)
def w(n, s): (T/f"{n}.tex").write_text(s); print(f"  tables/{n}.tex")

d = json.load(open(R/'ladder.json')); runs = d['runs']
ks = sorted({r['k'] for r in runs}); bs = sorted({r['budget'] for r in runs})
L = [r"\begin{tabular}{rrrrr}", r"\toprule",
     r"$k$ & states & \multicolumn{3}{c}{solve rate at training budget} \\",
     r"\cmidrule(l){3-5}", " & & " + " & ".join(f"{b//1000}k" for b in bs) + r" \\", r"\midrule"]
for k in ks:
    st = next(r['states'] for r in runs if r['k'] == k); m, e = f"{st:.2e}".split("e")
    cells = []
    for b in bs:
        v = [r['solve_rate'] for r in runs if r['k'] == k and r['budget'] == b]
        cells.append(f"{np.mean(v)*100:.1f}\\,$\\pm$\\,{np.std(v)*100:.1f}" if v else "--")
    L.append(f"{k} & ${m}\\!\\times\\!10^{{{int(e)}}}$ & " + " & ".join(cells) + r" \\")
w("ladder", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

d = json.load(open(R/'depth-vs-size.json')); by = {}
for r in d['runs']: by.setdefault(r['moveset'], []).append(r)
nm = {'all': 63, 'oi-q': 24, 'oi-q4': 16, 'oi-q3': 12}
# Ten seeds per row, so per-seed columns no longer fit and are not the point:
# what matters is that the only sub-100% cell is one run whose curriculum never
# advanced, and that removing it leaves every diameter identical.
L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
     r"generators & \#moves & diameter & seeds & solve rate & excl.\ deadlocked \\",
     r"\midrule"]
ndead = 0
for ms in ('all', 'oi-q', 'oi-q4', 'oi-q3'):
    rs = by.get(ms) or []
    if not rs: continue
    v = np.array([x['solve_rate'] for x in rs]) * 100
    ok = np.array([x['solve_rate'] for x in rs if not x.get('deadlocked')]) * 100
    nd = sum(1 for x in rs if x.get('deadlocked')); ndead += nd
    cell = f"{ok.mean():.1f}\\,$\\pm$\\,{ok.std():.1f}" if nd else "--"
    L.append(f"\\texttt{{{ms}}} & {nm[ms]} & {rs[0]['diameter']} & {len(v)} & "
             f"{v.mean():.1f}\\,$\\pm$\\,{v.std():.1f} & {cell} \\\\")
L += [r"\midrule", r"\multicolumn{6}{l}{"
      f"{ndead} of {len(d['runs'])} runs never advanced past the first curriculum "
      r"level; see Section~\ref{sec:depth}} \\"]
w("depth", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

rows = json.load(open(R/'e5-prediction.json'))['rows']
ok = [r for r in rows if r['w90'] is not None]; y = np.log([r['w90'] for r in ok])
lbl = {'gdrc': r"pooled GDRC \citep{wilt2016}", 'acc_deepest': "per-shell, deepest",
       'acc_min': "per-shell, minimum", 'acc_mean': "per-shell, mean"}
L = [r"\begin{tabular}{lrrr}", r"\toprule",
     r"predictor & Pearson $r$ & Spearman $\rho$ & Kendall $\tau$ \\", r"\midrule"]
for key in ('gdrc', 'acc_deepest', 'acc_min', 'acc_mean'):
    x = np.array([r[key] for r in ok])
    b, e = (r"\textbf{", "}") if key == 'acc_mean' else ("", "")
    L.append(f"{b}{lbl[key]}{e} & {b}{pearsonr(x,y).statistic:+.3f}{e} & "
             f"{b}{spearmanr(x,y).statistic:+.3f}{e} & {b}{kendalltau(x,y).statistic:+.3f}{e} \\\\")
w("e5", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

hs = json.load(open(R/'profile-wings-k6_s0.json'))['heuristics']
rows = sorted(((n, h['gdrc'], [r['acc'] for r in h['profile']]) for n, h in hs.items()),
              key=lambda t: t[1])
L = [r"\begin{tabular}{lrrrr}", r"\toprule",
     r"heuristic & pooled GDRC & acc.\ at $d{=}1$ & at $d{=}6$ & decay \\", r"\midrule"]
for n, g, a in rows:
    b, e = (r"\textbf{", "}") if n == 'learned' else ("", "")
    nm = n.replace('PDB(k=', r'PDB $k{=}$').replace(')', '')
    L.append(f"{b}{nm}{e} & {b}{g:+.3f}{e} & {b}{a[0]:.3f}{e} & {b}{a[-1]:.3f}{e} & "
             f"{b}{a[0]-a[-1]:+.3f}{e} \\\\")
w("matched", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# The cube explicitly, not whatever sits at the top level. Decay is not
# comparable across tasks with different shell coverage, so this control is
# reported per domain; the paper's claim is the cube one. See docs/adr/0001.
s = json.load(open(R/'strength-control.json'))['by_domain']['cube']
rt, mp = s['residual_test'], s['matched_pairs']
L = [r"\begin{tabular}{lrrr}", r"\toprule",
     r"heuristic class & $n$ & mean GDRC & mean decay \\", r"\midrule"]
for kd in ('random', 'PDB', 'learned'):
    k = s['by_kind'][kd]
    L.append(f"{kd} & {k['n']} & {k['mean_gdrc']:+.3f} & "
             f"{k['mean_decay']:+.3f}\\,$\\pm$\\,{k['sd_decay']:.3f} \\\\")
pm, pe = f"{s['decay_vs_strength_p']:.1e}".split("e")
L += [r"\midrule", r"\multicolumn{4}{l}{decay vs.\ strength: "
      f"$r = {s['decay_vs_strength_r']:+.3f}$ over {s['n_obs']} observations "
      f"($p = {pm}\\!\\times\\!10^{{{int(pe)}}}$)" r"} \\",
      r"\multicolumn{4}{l}{residual about that fit: learned "
      f"${rt['learned_resid']:+.4f}$, PDB ${rt['pdb_resid']:+.4f}$; "
      f"Welch $t = {rt['t']:+.2f}$, $p = {rt['p']:.2f}$" r"} \\",
      r"\multicolumn{4}{l}{\quad difference "
      f"${rt['diff']:+.4f}$, 95\\% CI $[{rt['ci95'][0]:+.3f}, {rt['ci95'][1]:+.3f}]$" r"} \\",
      r"\multicolumn{4}{l}{matched on identical states: PDB decays more in "
      f"{mp['pdb_decays_more']} of {mp['n']} pairs" r"} \\"]
w("strength", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

law = json.load(open(R/'dprime-law.json')); fp = law['free_parameter_check']
L = [r"\begin{tabular}{lrrrr}", r"\toprule",
     r"sample & $n$ & mean $|$error$|$ & noise floor & max $|$error$|$ \\", r"\midrule"]
def lawrow(lbl, d, ind=False):
    if not d: return
    pre = r"\quad " if ind else ""
    L.append(f"{pre}{lbl} & {d['n']} & {d['mae']:.4f} & {d.get('noise_floor', 0):.4f} & "
             + (f"{d['max_err']:.3f}" if 'max_err' in d else "--") + r" \\")
lawrow(r"exhaustive ground truth, $k\!\le\!6$", law['in_sample_split'])
L.append(r"\quad \emph{same, moments and accuracy from one sample} & "
         f"{law['in_sample']['n']} & {law['in_sample']['mae']:.4f} & "
         f"{law['in_sample'].get('noise_floor',0):.4f} & "
         f"{law['in_sample']['max_err']:.3f}" r" \\")
L.append(r"\midrule")
lawrow(r"\emph{held out}: probed ground truth, $k\!=\!8$", law.get('held_out_split') or law['held_out'])
lawrow(r"\emph{held out}: other training objectives", law.get('held_out_objectives_split'))
L += [r"\midrule",
      r"\multicolumn{5}{l}{Two free parameters, $\Phi(d'/s + b)$, give mean $|$error$|$ "
      f"{fp['mae_two_free_params']:.4f} at $s = {fp['free_scale']:.3f}$," r"} \\",
      r"\multicolumn{5}{l}{\quad against " f"{fp['mae_parameter_free']:.4f} at the "
      f"parameter-free $s = \\sqrt{{2}} = {fp['theory_scale']:.3f}$" r"} \\"]
w("law", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

L = [r"\begin{tabular}{rrrr}", r"\toprule",
     r"$k$ & states & diameter & mean distance \\", r"\midrule"]
for k in range(2, 7):
    x = json.load(open(R/f'exact-k{k}.json'))
    L.append(f"{k} & {x['states']:,} & {x['diameter']} & {x['mean_distance']:.2f} \\\\".replace(",", "{,}"))
w("exact", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# --- the loss comparison, main arms + the penalty-weight sweep -----------------
runs = (json.load(open(R/'p3-losses.json'))['runs']
        + json.load(open(R/'p3-lambda.json'))['runs'])
k6 = [r for r in runs if r['task'] == 'wings-k6' and not r.get('deadlocked')]
NAME = {'l2': r"squared error (baseline)", 'rank': r"pairwise rank, uniform pairs",
        'rank-adj': r"pairwise rank, adjacent pairs only",
        'dprime': r"\quad $+$ variance penalty, $\lambda=1$",
        'dprime:3': r"\quad $\lambda=3$", 'dprime:10': r"\quad $\lambda=10$",
        'dprime:30': r"\quad $\lambda=30$"}
L = [r"\begin{tabular}{lrrrr}", r"\toprule",
     r"objective & solve rate & decay & beam width & curriculum \\", r"\midrule"]
for a in ('l2', 'rank', 'rank-adj', 'dprime', 'dprime:3', 'dprime:10', 'dprime:30'):
    g = [r for r in k6 if r['arm'] == a]
    if not g: continue
    ws = [r['w90'] for r in g if r.get('w90')]
    b, e = (r"\textbf{", "}") if a == 'l2' else ("", "")
    if a == 'dprime': L.append(r"\midrule")
    L.append(f"{b}{NAME[a]}{e} & {b}{np.mean([r['solve_rate'] for r in g])*100:.1f}\\%{e} & "
             f"{b}{np.mean([r['decay'] for r in g]):.3f}{e} & "
             f"{b}{(np.mean(ws) if ws else float('nan')):.1f}{e} & "
             f"{b}{np.mean([r['k_cur'] for r in g]):.0f}{e} \\\\")
w("losses", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# --- the noise-injection falsification test ----------------------------------
nd = json.load(open(R/'noise-dose.json'))
TAIL = {'uniform': 'lighter than normal', 'gaussian': 'normal (the assumption)',
        'laplace': 'heavier, finite variance', 'cauchy': r"\textbf{no finite variance}"}
L = [r"\begin{tabular}{llrrr}", r"\toprule",
     r"noise family & tails & $n$ & measured & forward \\", r"\midrule"]
for fam in ('uniform', 'gaussian', 'laplace', 'cauchy'):
    g = [r for r in nd['rows'] if r['family'] == fam and r['lam'] > 0]
    if not g: continue
    fw = [r['forward_mae'] for r in g if r['forward_mae'] is not None]
    b, e = (r"\textbf{", "}") if fam == 'cauchy' else ("", "")
    L.append(f"{b}{fam}{e} & {TAIL[fam]} & {len(g)} & {b}{np.mean([r['mae'] for r in g]):.4f}{e} & "
             + (f"{np.mean(fw):.4f}" if fw else r"\emph{undefined}") + r" \\")
L += [r"\midrule",
      r"\multicolumn{5}{l}{Mean absolute error of Equation~\ref{eq:law}, over "
      f"$\\lambda \\in \\{{{', '.join(str(x) for x in nd['lambdas'][1:])}\\}}$ "
      r"in units of the} \\",
      r"\multicolumn{5}{l}{unperturbed within-shell spread. \emph{Forward} uses only "
      r"the $\lambda\!=\!0$ moments.} \\"]
w("noise", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))


# ---------------------------------------------------------------- second domain
# Two panels rather than two tables: the top says the law is worse on the
# sliding tile, the bottom says that is not a fact about the domain. Split
# apart, a reader can take the first without the second.
cd = json.load(open(R/'cross-domain-law.json'))
law2 = json.load(open(R/'dprime-law.json'))
NAME = {"wings-k6": r"cube, $k\!=\!6$", "tile-3x3": r"sliding tile, $3\times3$",
        "tile-2x4": r"sliding tile, $2\times4$"}
L = [r"\begin{tabular}{lrrrr}", r"\toprule",
     r"task & $n$ & median $|$skew$|$ & mean $|$error$|$ & bias \\", r"\midrule"]
for k, v in cd['by_task'].items():
    L.append(f"{NAME.get(k, k)} & {v['n']} & {v['median_skew']:.3f} & "
             f"{v['mae']:.4f} & {v['bias']:+.4f}" r" \\")
L += [r"\midrule",
      r"\multicolumn{5}{l}{\emph{Matched on skew, the domain gap closes below "
      r"$|\mathrm{skew}| = 1$:}} \\"]
sk = cd['stratified']['skew']
L.append(r"$|$skew$|$ range & cube $n$ & cube $|$err$|$ & tile $n$ & tile $|$err$|$ \\")
for b in sk['bins']:
    if not (b['cube_n'] and b['tile_n']):
        continue
    hi = r"$\infty$" if b['hi'] > 1e8 else f"{b['hi']:g}"
    L.append(f"{b['lo']:g} to {hi} & {b['cube_n']} & {b['cube_mae']:.4f} & "
             f"{b['tile_n']} & {b['tile_mae']:.4f}" r" \\")
L += [r"\midrule",
      r"\multicolumn{5}{l}{Reweighting the cube to the tile's skew mix gives "
      f"{sk['cube_reweighted_to_tile']:.4f}, against the tile's " 
      f"{sk['tile_mae']:.4f}." r"} \\"]
w("domain", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# The class split, which is the reason the tile exposes the boundary at all.
bd = law2.get('by_domain_split') or {}
if bd:
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{cube} & \multicolumn{2}{c}{sliding tile} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
         r"heuristic class & $n$ & mean $|$error$|$ & $n$ & mean $|$error$|$ \\",
         r"\midrule"]
    rows = cd.get('by_class') or {}
    for cls in ("learned", "PDB", "random"):
        c, t = rows.get(cls, {}).get('cube'), rows.get(cls, {}).get('tile')
        if not (c and t):
            continue
        L.append(f"{cls} & {c['n']} & {c['mae']:.4f} & {t['n']} & {t['mae']:.4f}" r" \\")
    if len(L) > 6:
        w("domainclass", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# ------------------------------------------------- pre-registered search cost
cp = json.load(open(R/'cost-predict.json'))
NICE = {"tile-2x4/all": r"sliding tile $2\times4$", "tile-3x3/all": r"sliding tile $3\times3$",
        "wings-k4/all": r"cube $k\!=\!4$, all", "wings-k4/oi-q": r"cube $k\!=\!4$, oi-q",
        "wings-k4/oi-q3": r"cube $k\!=\!4$, oi-q3", "wings-k4/oi-q4": r"cube $k\!=\!4$, oi-q4"}
prim = cp['median_expansions']
L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
     r"task & $n$ & $r$ profile & $r$ GDRC & difference & 95\% CI \\", r"\midrule"]
for k, v in prim['per_task'].items():
    L.append(f"{NICE.get(k, k)} & {v['n']} & {v['r_predictor']:+.3f} & {v['r_gdrc']:+.3f} & "
             f"{v['gap']:+.3f} & $[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]$ \\\\")
c = prim['combined']
L += [r"\midrule",
      r"\multicolumn{6}{l}{combined difference "
      f"${c['mean']:+.4f}$, 95\\% CI $[{c['lo']:+.4f}, {c['hi']:+.4f}]$; positive in "
      f"{prim['tasks_positive']} of {prim['n_tasks']} tasks, 4 required" r"} \\"]
g = cp['geomean_expansions']['combined']
gp = cp['geomean_expansions']['tasks_positive']
L += [r"\multicolumn{6}{l}{\emph{robustness check}, geometric mean: "
      f"${g['mean']:+.4f}$, $[{g['lo']:+.4f}, {g['hi']:+.4f}]$; positive in {gp} of 6" r"} \\"]
w("costpredict", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# ------------------------------------------------- numbers quoted in the prose
# Some figures belong in a sentence rather than a table, and those were the last
# hand-typed numbers in the paper. They are macros now, so re-running the
# measurement moves the prose the same way it moves a table.
tt = json.load(open(R/'tau-theory.json'))
lw = json.load(open(R/'dprime-law.json'))
inf = [r['inflation'] for r in tt['tie_inflation']]
if 'cube' not in lw.get('moments', {}):
    raise SystemExit("dprime-law.json predates the per-domain split of moments, so "
                     "its gap figures pool the cube with the sliding tile. A gap is "
                     "measured in moves and a move is not the same thing in two "
                     "state spaces. Re-run dprime_law.py first.")
cube = lw['moments']['cube']
NUMS = {
    # tau-b tie inflation, from the analysis that builds its own cube sample
    'tieinflationmean': f"{np.mean(inf):.3f}",
    'tieinflationmin':  f"{min(inf):.3f}",
    'tieinflationmax':  f"{max(inf):.3f}",
    'tiefreeceiling':   f"{tt['ceiling_verified']['analytic_tie_free_ceiling']:.3f}",
    # median between-shell gap per class, CUBE only: a gap is measured in moves,
    # and a move is not the same thing in two state spaces
    'gaplearned':       f"{cube['learned']['median_gap']:.3f}",
    'gapabstraction':   f"{cube['PDB']['median_gap']:.3f}",
}
w("numbers", "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in NUMS.items()))
