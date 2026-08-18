// Every figure here is drawn from a file produced by an actual run:
//   eval/results/*.json   <- scripts/bench-solver.mjs, scripts/collect-metrics.mjs
// The two schematics (Fig 1, Fig 2) are the exception and are labelled as such,
// because they illustrate a mechanism rather than report a measurement.
import { Axes, C, Dots, Figure, Label, Line, Mono, Pow, linear, log, logTicks, ticks } from './plot.jsx';
import TrainingPanel, { useTraining } from './TrainingPanel.jsx';

import beam from '../../eval/results/centres-beam-width.json';
import centres from '../../eval/results/centres-greedy-vs-learned.json';
import full from '../../eval/results/full-solve.json';
import trainWingsFile from '../../eval/results/training-wings.json';

// Opt-in results that may not have been generated yet. A glob resolves to an
// empty object rather than a build error.
const optional = (name) => Object.values(
  import.meta.glob('../../eval/results/*.json', { eager: true }),
).length
  ? (Object.entries(import.meta.glob('../../eval/results/*.json', { eager: true }))
      .find(([p]) => p.endsWith(`/${name}.json`))?.[1]?.default ?? null)
  : null;

const pdb = optional('pdb-depths');
const ladderExact = optional('ladder-exact');   // exact.py, rungs k<=6
const ladder = optional('ladder');              // sweep.py, the full experiment
const figDepth = optional('fig-depth');         // depth_vs_size.py: size held constant
// Read straight from the profiles.py output rather than a hand-shaped copy, so
// the figure and paper/tables/matched.tex cannot drift apart.
const figProfile = optional('profile-wings-k6_s0');

const W = 720;
const H = 380;
const M = { l: 64, r: 24, t: 24, b: 52 };
// `inset` reserves room on the right for a second axis or for value labels that
// would otherwise run off the edge.
const px = (inset = 0) => [M.l, W - M.r - inset];
const py = (inset = 0) => [H - M.b - inset, M.t];

const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;

const figStrength = optional('strength-control'); // strength_control.py
const figLaw = optional('dprime-law');            // dprime_law.py

// ---------------------------------------------------------------------------
// Fig 1: what the pipeline is
function PipelineFigure() {
  const stages = [
    { t: 'Camera frame', s: ['640 px'], c: C.mid },
    { t: 'Face detection', s: ['comb filter', '+ SAM / VLM'], c: C.blue },
    { t: 'Homography', s: ['unit square', '→ quad'], c: C.blue },
    { t: 'Clustering', s: ['balanced', 'min-cost flow'], c: C.sky },
    { t: 'Labelling', s: ['720 perms', '+ chirality'], c: C.sky },
    { t: 'Solve', s: ['learned centres', '→ reduction'], c: C.green },
  ];
  const gap = 26;
  const bw = (W - M.l - M.r - gap * (stages.length - 1)) / stages.length;
  const at = (i) => M.l + i * (bw + gap);
  const bar = (i0, i1, colour, text) => (
    <g>
      <Label x={at(i0)} y={62} size={12} bold fill={colour}>{text}</Label>
      <line x1={at(i0)} y1={74} x2={at(i1) + bw} y2={74} stroke={colour} strokeWidth="2" />
    </g>
  );
  return (
    <g>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill={C.mid} />
        </marker>
      </defs>
      {bar(0, 4, C.blue, 'Perception: classical, verifiable')}
      {bar(5, 5, C.green, 'Search: learned')}
      {stages.map((st, i) => {
        const x = at(i);
        return (
          <g key={st.t}>
            <rect x={x} y={106} width={bw} height={84} rx="8"
              fill={`${st.c}14`} stroke={st.c} strokeWidth="1.5" />
            <Label x={x + bw / 2} y={134} anchor="middle" size={11.5} bold>{st.t}</Label>
            {st.s.map((ln, j) => (
              <Mono key={ln} x={x + bw / 2} y={154 + j * 14} anchor="middle" size={9.5}>{ln}</Mono>
            ))}
            {i < stages.length - 1 && (
              <path d={`M${x + bw + 6} 148 L${x + bw + gap - 7} 148`} stroke={C.mid}
                strokeWidth="1.5" markerEnd="url(#arrow)" />
            )}
          </g>
        );
      })}
      <Mono x={M.l} y={228} size={11}>
        Every stage verifies its own output and falls back rather than emitting an unverified result.
      </Mono>
      <Mono x={M.l} y={248} size={11}>
        {`The solve stage is the only learned component: ${centres.summary.learned.mean.toFixed(1)} moves on the centres`}
      </Mono>
      <Mono x={M.l} y={266} size={11}>
        {`where the hand-written solver spends ${centres.summary.greedy.mean.toFixed(1)} (Figure 5).`}
      </Mono>
    </g>
  );
}

// ---------------------------------------------------------------------------
// Fig 2: how DAVI actually works
function DaviFigure() {
  const cx = 158, cy = 218;
  const ring = [38, 76, 114, 152];
  return (
    <g>
      <Label x={cx} y={36} anchor="middle" size={12} bold>value spreads outward from solved</Label>
      {ring.map((r, i) => (
        <circle key={r} cx={cx} cy={cy} r={r} fill="none" stroke={C.faint} strokeWidth="1.5"
          strokeDasharray={i > 1 ? '4 4' : undefined} />
      ))}
      {/* one radial axis carries every label, so nothing reads as a stack */}
      <path d={`M${cx} ${cy} L${cx} ${cy - ring.at(-1) - 16}`} stroke={C.mid}
        strokeWidth="1.25" markerEnd="url(#arrow2)" />
      <defs>
        <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill={C.mid} />
        </marker>
      </defs>
      <circle cx={cx} cy={cy} r="7" fill={C.green} />
      <Mono x={cx + 12} y={cy + 4} size={10} fill={C.green}>solved · J = 0</Mono>
      {ring.map((r, i) => (
        <g key={r}>
          <line x1={cx - 4} x2={cx + 4} y1={cy - r} y2={cy - r} stroke={C.mid} strokeWidth="1.25" />
          <Mono x={cx + 12} y={cy - r + 4} size={10} over>{`J ≈ ${i + 1}`}</Mono>
        </g>
      ))}
      <Mono x={cx} y={cy + ring.at(-1) + 26} anchor="middle" size={10}>
        one shell per target-network refresh
      </Mono>

      <g transform="translate(360 78)">
        <Label x={0} y={0} size={13} bold>The update</Label>
        <Mono x={0} y={26} size={12} fill={C.ink}>J(s) = 0                    if s solved</Mono>
        <Mono x={0} y={46} size={12} fill={C.ink}>J(s) = min  1 + J    (s′)   otherwise</Mono>
        <Mono x={52} y={54} size={9}>a</Mono>
        <Mono x={92} y={54} size={9}>target</Mono>

        <Label x={0} y={92} size={13} bold>Why the data is free</Label>
        <Mono x={0} y={114} size={11}>scramble k moves from solved → a labelled state.</Mono>
        <Mono x={0} y={130} size={11}>no dataset to collect, no annotation, unlimited.</Mono>

        <Label x={0} y={168} size={13} bold>Why a curriculum</Label>
        <Mono x={0} y={190} size={11}>only states near solved have a grounded target.</Mono>
        <Mono x={0} y={206} size={11}>sampling k~U(1,40) over 63 moves put 2.5% of the</Mono>
        <Mono x={0} y={222} size={11}>batch there; the run collapsed to a constant</Mono>
        <Mono x={0} y={238} size={11} fill={C.red}>J = 15.33 at every depth. Growing k fixed it.</Mono>

        <Label x={0} y={276} size={13} bold>What it cannot give you</Label>
        <Mono x={0} y={298} size={11}>a learned heuristic is not admissible, so the</Mono>
        <Mono x={0} y={314} size={11}>result is near-optimal: never proven minimal.</Mono>
      </g>
    </g>
  );
}

// ---------------------------------------------------------------------------
// Fig 3, the value function taking shape
function ValueShapeFigure({ rows }) {
  if (!rows.length) return <Label x={M.l} y={H / 2}>no training metrics yet</Label>;
  const maxK = Math.max(...rows.flatMap((r) => r.probe.map(([k]) => k)));
  const maxJ = Math.max(2, ...rows.flatMap((r) => r.probe.map(([, j]) => j)));
  const x = linear(0, maxK, ...px());
  const y = linear(0, Math.ceil(maxJ), ...py());
  // a handful of evenly spaced checkpoints, oldest faintest
  const pick = rows.length <= 6 ? rows
    : Array.from({ length: 6 }, (_, i) => rows[Math.round(i * (rows.length - 1) / 5)]);
  return (
    <Axes x={x} y={y} pad={6} xLabel="scramble depth k (random moves from solved)"
      yLabel="mean predicted cost-to-go  J">
      {/* J = k is the ceiling a perfect estimator would sit under: a k-move
          scramble is solvable in at most k, usually fewer once moves cancel */}
      <Line pts={[[x(0), y(0)], [x(Math.min(maxK, Math.ceil(maxJ))), y(Math.min(maxK, Math.ceil(maxJ)))]]}
        stroke={C.faint} width={1.5} dash="5 4" />
      <Mono x={x(Math.min(maxK, Math.ceil(maxJ))) - 6} y={y(Math.min(maxK, Math.ceil(maxJ))) - 8}
        anchor="end" size={10}>J = k</Mono>
      {pick.map((r, i) => {
        const t = i / Math.max(pick.length - 1, 1);
        const pts = r.probe.map(([k, j]) => [x(k), y(j)]);
        return (
          <g key={r.step} opacity={0.28 + 0.72 * t}>
            <Line pts={pts} stroke={C.blue} width={i === pick.length - 1 ? 2.5 : 1.6} />
            {i === pick.length - 1 && <Dots pts={pts} fill={C.blue} r={3} />}
          </g>
        );
      })}
      <Mono x={x(maxK) - 4} y={y(pick.at(-1).probe.at(-1)[1]) - 10} anchor="end" size={10} fill={C.blue}>
        {`step ${pick.at(-1).step.toLocaleString()}`}
      </Mono>
      <Mono x={x(0) + 8} y={y(0) - 10} size={10}>
        {`step ${pick[0].step.toLocaleString()}`}
      </Mono>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
// Fig 4: loss and curriculum together
function TrainingFigure({ rows }) {
  if (!rows.length) return <Label x={M.l} y={H / 2}>no training metrics yet</Label>;
  const R = 52;                               // room for the curriculum axis
  const maxStep = Math.max(...rows.map((r) => r.step));
  const x = linear(0, maxStep, ...px(R));
  const lo = Math.min(...rows.map((r) => r.loss));
  const hi = Math.max(...rows.map((r) => r.loss));
  const y = log(lo * 0.9, hi * 1.1, ...py());
  const maxK = Math.max(...rows.map((r) => r.k ?? 0));
  const y2 = linear(0, maxK || 1, ...py());
  const right = W - M.r - R;
  return (
    <Axes x={x} y={y} pad={6} xLabel="training step" yLabel="training loss  (log scale)"
      yTicks={logTicks(lo * 0.9, hi * 1.1)}
      fmtX={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)}
      fmtY={(v) => (v < 0.01 ? v.toExponential(0) : String(+v.toPrecision(2)))}>
      <Line pts={rows.map((r) => [x(r.step), y2(r.k ?? 0)])} stroke={C.orange} width={2} />
      <Line pts={rows.map((r) => [x(r.step), y(r.loss)])} stroke={C.blue} width={2} />
      {ticks(0, maxK || 1, 4).map((t) => (
        <g key={t}>
          <line x1={right} x2={right + 5} y1={y2(t)} y2={y2(t)} stroke={C.orange} strokeWidth="1.25" />
          <Mono x={right + 9} y={y2(t) + 4} size={10} fill={C.orange}>{t}</Mono>
        </g>
      ))}
      <text transform={`translate(${right + 40} ${(py()[0] + py()[1]) / 2}) rotate(-90)`}
        textAnchor="middle"
        style={{ font: '600 12px "Space Grotesk", sans-serif', fill: C.orange }}>
        curriculum depth k
      </text>
      <Mono x={x(maxStep) - 8} y={y(lo) - 12} anchor="end" size={11} fill={C.blue} over>
        loss
      </Mono>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
// Fig 5: greedy vs learned, paired
function CentresFigure() {
  const rows = centres.rows.filter((r) => r.learned !== null);
  const all = rows.flatMap((r) => [r.greedy, r.learned]);
  // A common range on both axes so the diagonal means what it looks like. Zero
  // is not the reference here (the diagonal is) so the range starts at the data.
  const lo = Math.max(0, Math.floor((Math.min(...all) - 3) / 5) * 5);
  const hi = Math.ceil((Math.max(...all) + 3) / 5) * 5;
  const x = linear(lo, hi, ...px());
  const y = linear(lo, hi, ...py());
  return (
    <Axes x={x} y={y} pad={6}
      xLabel="moves used by the hand-written greedy solver"
      yLabel="moves used by the learned beam search">
      <Line pts={[[x(lo), y(lo)], [x(hi), y(hi)]]} stroke={C.faint} width={1.5} dash="5 4" />
      <Mono x={x(hi) - 8} y={y(hi) + 18} anchor="end" size={10}>equal: above this line is worse</Mono>
      <Dots pts={rows.map((r) => [x(r.greedy), y(r.learned)])} fill={C.blue} r={4} />
      {/* parked in the gap between the cloud and the axis, not on top of it */}
      <Label anchor="middle" size={13} bold fill={C.blue}
        x={x((Math.min(...rows.map((r) => r.greedy)) + Math.max(...rows.map((r) => r.greedy))) / 2)}
        y={y(lo + (Math.min(...rows.map((r) => r.learned)) - lo) * 0.42)}>
        {`all ${rows.length}/${centres.n} below the line`}
      </Label>
      <g transform={`translate(${x(lo) + 12} ${y(hi) + 22})`}>
        <Mono x={0} y={0} size={11} fill={C.ink}>
          {`greedy  mean ${centres.summary.greedy.mean.toFixed(1)}   worst ${centres.summary.greedy.max}`}
        </Mono>
        <Mono x={0} y={17} size={11} fill={C.blue}>
          {`learned mean ${centres.summary.learned.mean.toFixed(1)}   worst ${centres.summary.learned.max}`
            + `   (${((1 - centres.summary.learned.mean / centres.summary.greedy.mean) * 100).toFixed(0)}% shorter)`}
        </Mono>
      </g>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
// Fig 6: what beam width buys, and what it costs
function BeamFigure() {
  const R = 44;
  const rows = beam.rows;
  const solved = rows.filter((r) => r.mean !== null);
  const x = log(rows[0].width, rows.at(-1).width, ...px(R));
  const y = linear(0, Math.ceil(Math.max(...solved.map((r) => r.mean)) + 3), ...py());
  const yRate = linear(0, 1, ...py());
  const right = W - M.r - R;
  return (
    <Axes x={x} y={y} pad={6} xLabel="beam width (states kept per depth, log scale)"
      yLabel="mean solution length (moves)"
      xTicks={rows.map((r) => r.width)}>
      <Line pts={rows.map((r) => [x(r.width), yRate(r.solved / r.n)])} stroke={C.orange} width={2} />
      <Dots pts={rows.map((r) => [x(r.width), yRate(r.solved / r.n)])} fill={C.orange} r={3} />
      <Line pts={solved.map((r) => [x(r.width), y(r.mean)])} stroke={C.blue} width={2.5} />
      <Dots pts={solved.map((r) => [x(r.width), y(r.mean)])} fill={C.blue} r={4} />
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line x1={right} x2={right + 5} y1={yRate(t)} y2={yRate(t)} stroke={C.orange} strokeWidth="1.25" />
          <Mono x={right + 9} y={yRate(t) + 4} size={10} fill={C.orange}>{`${t * 100}%`}</Mono>
        </g>
      ))}
      <Mono x={x(rows.at(-1).width) - 6} y={yRate(1) - 12} anchor="end" size={11} fill={C.orange} over>
        fraction solved
      </Mono>
      <Mono x={x(solved[0].width) + 8} y={y(solved[0].mean) - 12} size={11} fill={C.blue} over>
        solution length
      </Mono>
      {/* the last two points cost an order of magnitude of search for ~0.1 moves */}
      {[10, 100, 800].map((w) => (
        <Mono key={w} x={x(w)} y={y(0) - 40} anchor="middle" size={10} over>
          {`${(rows.find((r) => r.width === w).ms / 1000).toFixed(1)} s`}
        </Mono>
      ))}
      <Mono x={M.l + 6} y={y(0) - 40} size={10}>time per solve:</Mono>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
// Fig 7, where the moves actually go
function PhasesFigure() {
  const names = ['Centers', 'Edge pairing', 'Parity', '3×3 finish'];
  const cols = [C.blue, C.sky, C.orange, C.green];
  const bars = [
    { label: 'greedy centres', vals: names.map((n) => full.phases.off[n] ?? 0) },
    { label: 'learned centres', vals: names.map((n) => full.phases.on[n] ?? 0) },
  ];
  const hi = Math.max(...bars.map((b) => b.vals.reduce((s, v) => s + v, 0)));
  const R = 92;                                 // room for the "N total" labels
  const x = linear(0, hi, ...px(R));
  const bh = 54, baseline = 266;
  return (
    <g>
      {bars.map((b, bi) => {
        let acc = 0;
        const top = 74 + bi * (bh + 46);
        return (
          <g key={b.label}>
            <Label x={M.l} y={top - 9} size={12} bold>{b.label}</Label>
            {b.vals.map((v, i) => {
              const x0 = x(acc); acc += v;
              const wSeg = x(acc) - x0;
              return (
                <g key={names[i]}>
                  <rect x={x0} y={top} width={wSeg} height={bh} fill={cols[i]} />
                  <Mono x={(x0 + x(acc)) / 2} y={top + bh / 2 + 4} anchor="middle"
                    size={12} fill="#fff">{v.toFixed(0)}</Mono>
                  {bi === 1 && (
                    <Mono x={(x0 + x(acc)) / 2} y={top + bh + 16} anchor="middle"
                      size={10} fill={cols[i]}>{wSeg > 54 ? names[i] : ''}</Mono>
                  )}
                </g>
              );
            })}
            <Mono x={x(acc) + 12} y={top + bh / 2 + 4} size={13} fill={C.ink}>
              {`${acc.toFixed(0)} total`}
            </Mono>
          </g>
        );
      })}
      <line x1={M.l} y1={baseline} x2={W - M.r - R} y2={baseline} stroke={C.ink} strokeWidth="1.25" />
      {ticks(0, hi, 5).map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={baseline} y2={baseline + 5} stroke={C.ink} strokeWidth="1.25" />
          <Mono x={x(t)} y={baseline + 18} anchor="middle" size={10}>{t}</Mono>
        </g>
      ))}
      <Label x={(M.l + W - M.r - R) / 2} y={baseline + 38} anchor="middle" size={12} bold>
        mean moves over the full solve
      </Label>
      <Mono x={M.l} y={baseline + 62} size={11}>
        {`${full.n} scrambles. The learned stage removes `
          + `${(full.summary.off - full.summary.on).toFixed(0)} moves overall `
          + `(${((1 - full.summary.on / full.summary.off) * 100).toFixed(0)}%), though parity rises`}
      </Mono>
      <Mono x={M.l} y={baseline + 80} size={11}>
        {`from ${full.phases.off.Parity.toFixed(0)} to ${full.phases.on.Parity.toFixed(0)}: `
          + 'a different centre solution leaves a different parity case.'}
      </Mono>
    </g>
  );
}

// ---------------------------------------------------------------------------
// Fig 8, the pattern database, and why it does not certify anything
function PdbFigure() {
  if (!pdb) {
    return (
      <g>
        <Label x={M.l} y={H / 2 - 10} size={13} bold>not built</Label>
        <Mono x={M.l} y={H / 2 + 14} size={11}>
          run `node scripts/bench-solver.mjs --pdb`, the BFS covers 51.5M states and takes a few minutes
        </Mono>
      </g>
    );
  }
  const B = 56;                              // room for the notes under the axis
  const hist = pdb.histogram.filter((h) => h.count > 0);
  const top = Math.max(...hist.map((h) => h.count));
  const x = linear(-0.5, pdb.maxDepth + 0.5, ...px());
  const y = log(1, top, ...py(B));
  const decades = [];
  for (let e = 0; e <= Math.floor(Math.log10(top)); e++) decades.push(10 ** e);
  const bw = (x(1) - x(0)) * 0.72;
  return (
    <>
      <Axes x={x} y={y} pad={6} xTicks={hist.map((h) => h.depth)} yTicks={decades}
        xLabel="exact distance from solved (moves)"
        yLabel="arrangements (log scale)"
        fmtY={(v) => (v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}k` : v)}>
        {hist.map((h) => (
          <rect key={h.depth} x={x(h.depth) - bw / 2} y={y(h.count)}
            width={bw} height={y(1) - y(h.count)} fill={C.purple} />
        ))}
      </Axes>
      <Mono x={M.l} y={H - 34} size={11} fill={C.ink}>
        {`All ${pdb.reached.toLocaleString()} arrangements of the U and D centres, `
          + `solved exactly by breadth-first search.`}
      </Mono>
      <Mono x={M.l} y={H - 16} size={11} fill={C.red}>
        {`The deepest is ${pdb.maxDepth} moves, so this bound can never certify more than `
          + `${pdb.maxDepth} of a ~${full.summary.on.toFixed(0)}-move solve.`}
      </Mono>
    </>
  );
}

// ---------------------------------------------------------------------------
// Fig 9, the scale that makes the 4x4 hard
function ScaleFigure() {
  const items = [
    { t: '4×4 centres', v: 3.25e15, note: 'learned, solved here', c: C.green },
    { t: '3×3 cube', v: 4.33e19, note: "DeepCubeA; God's number is 20", c: C.sky },
    { t: '4×4 edge wings', v: 3.10e23, note: 'in training', c: C.orange },
    { t: '4×4 cube', v: 7.40e45, note: 'no optimal solver exists', c: C.red },
  ];
  const labelR = M.l + 112;                   // row labels live outside the bars
  const x0 = labelR + 14;
  const x = log(1e14, 1e47, x0, W - M.r - 186);
  const bh = 34, gap = 22, baseline = 292;
  return (
    <g>
      {items.map((it, i) => {
        const top = 72 + i * (bh + gap);
        const [mant, exp] = it.v.toExponential(2).split('e+');
        return (
          <g key={it.t}>
            <Label x={labelR} y={top + bh / 2 + 5} anchor="end" size={13} bold>{it.t}</Label>
            <rect x={x0} y={top} width={x(it.v) - x0} height={bh} rx="3" fill={it.c} />
            <Pow x={x(it.v) + 12} y={top + bh / 2 - 2} size={12} fill={C.ink}
              prefix={`${mant} ×`} exp={exp} />
            <Mono x={x(it.v) + 12} y={top + bh / 2 + 14} size={10}>{it.note}</Mono>
          </g>
        );
      })}
      <line x1={x0} y1={baseline} x2={W - M.r - 186} y2={baseline} stroke={C.ink} strokeWidth="1.25" />
      {[1e15, 1e20, 1e25, 1e30, 1e35, 1e40, 1e45].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={baseline} y2={baseline + 5} stroke={C.ink} strokeWidth="1.25" />
          <Pow x={x(t)} y={baseline + 19} anchor="middle" size={10} exp={Math.log10(t)} />
        </g>
      ))}
      <Label x={(x0 + W - M.r - 186) / 2} y={baseline + 42} anchor="middle" size={12} bold>
        reachable states (log scale)
      </Label>
      <Mono x={M.l} y={44} size={11}>
        {`the wings are ${(3.10e23 / 3.25e15).toExponential(1).replace('e+7', ' × 10⁷')} times the centres; `
          + 'the full 4×4 is another 10²² beyond the wings'}
      </Mono>
    </g>
  );
}

// ---------------------------------------------------------------------------
// Fig 10, the control that makes the ladder interpretable.
// If depth grew as fast as size, a failure at the top of the ladder would be
// ambiguous. It does not: exact BFS shows depth adding one move per rung while
// the space multiplies by ~20.
function DepthControlFigure() {
  if (!ladderExact) {
    return (
      <g>
        <Label x={M.l} y={H / 2 - 10} size={13} bold>not built</Label>
        <Mono x={M.l} y={H / 2 + 14} size={11}>
          run `mlx-solver/exact.py --k 2..6`: exhaustive BFS, ~22 min at k=6
        </Mono>
      </g>
    );
  }
  const rows = ladderExact.rungs;
  const fit = ladderExact.fits;
  const K_TOP = 24;
  const x = log(1e2, 1e24, ...px(60));
  const y = linear(0, 28, ...py());
  const sizeAt = (k) => rows.find((r) => r.k === k)?.states;
  // extrapolate the fits across the whole ladder
  const extrap = (f) => [2, K_TOP].map((k) => [k, f.slope * k + f.intercept]);
  const P24 = (k) => {
    let n = 1;
    for (let i = 0; i < k; i++) n *= 24 - i;
    return k >= 23 ? n / 2 : n;
  };
  return (
    <Axes x={x} y={y} pad={6}
      xLabel="reachable states (log scale)"
      yLabel="moves from solved"
      xTicks={[1e2, 1e6, 1e10, 1e14, 1e18, 1e22]}
      fmtX={(v) => `10^${Math.round(Math.log10(v))}`}>
      {[['diameter', C.red], ['mean_distance', C.blue]].map(([key, col]) => (
        <g key={key}>
          <Line dash="5 4" width={1.5} stroke={col}
            pts={extrap(fit[key]).map(([k, v]) => [x(P24(k)), y(v)])} />
          <Line width={2.5} stroke={col}
            pts={rows.map((r) => [x(r.states), y(key === 'diameter' ? r.diameter : r.mean_distance)])} />
          <Dots r={4} fill={col}
            pts={rows.map((r) => [x(r.states), y(key === 'diameter' ? r.diameter : r.mean_distance)])} />
        </g>
      ))}
      <Mono x={x(sizeAt(6)) + 14} y={y(8) - 6} size={11} fill={C.red} over>diameter</Mono>
      <Mono x={x(sizeAt(6)) + 14} y={y(6.4) + 16} size={11} fill={C.blue} over>mean distance</Mono>

      {/* the whole argument of the figure, stated on it. Solid = enumerated,
          dashed = the fit carried across the rungs too large to enumerate. */}
      <Mono x={M.l + 8} y={M.t + 16} size={11} fill={C.ink}>
        solid: measured exactly by BFS · dashed: the fit extended
      </Mono>
      <Mono x={M.l + 8} y={M.t + 34} size={11} fill={C.green}>
        {`diameter = k + 2 exactly (R² = ${fit.diameter.r2.toFixed(5)})`}
      </Mono>
      <Mono x={W - M.r - 60} y={y(3.4)} anchor="end" size={11} fill={C.ink}>
        the space grows 21 orders of magnitude
      </Mono>
      <Mono x={W - M.r - 60} y={y(2.0)} anchor="end" size={11} fill={C.ink}>
        while depth grows only from 4 to ~26
      </Mono>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
// Fig 11, the experiment itself: does a learned heuristic degrade with size?
function LadderFigure() {
  if (!ladder) {
    return (
      <g>
        <Label x={M.l} y={H / 2 - 10} size={13} bold>sweep not run yet</Label>
        <Mono x={M.l} y={H / 2 + 14} size={11}>
          run `mlx-solver/sweep.py --steps N --seeds 3`: 11 rungs, ~10 GPU-hours
        </Mono>
      </g>
    );
  }
  const runs = ladder.runs;
  const x = log(1e2, 1e24, ...px(20));
  const y = linear(0, 1, ...py());
  // group seeds by rung so the spread is visible rather than averaged away
  const byK = [...new Set(runs.map((r) => r.k))].sort((a, b) => a - b)
    .map((k) => ({ k, rs: runs.filter((r) => r.k === k) }));
  return (
    <Axes x={x} y={y} pad={6}
      xLabel="reachable states (log scale)" yLabel="fraction of scrambles solved"
      xTicks={[1e2, 1e6, 1e10, 1e14, 1e18, 1e22]}
      yTicks={[0, 0.25, 0.5, 0.75, 1]}
      fmtX={(v) => `10^${Math.round(Math.log10(v))}`}
      fmtY={(v) => `${v * 100}%`}>
      {/* Wilson intervals, a solve rate from n=200 is a range, not a point */}
      {byK.flatMap(({ rs }) => rs.map((r, i) => (
        <line key={`${r.k}-${i}`} x1={x(r.states)} x2={x(r.states)}
          y1={y(r.ci95[0])} y2={y(r.ci95[1])} stroke={C.faint} strokeWidth="1.5" />
      )))}
      <Line width={2.5} stroke={C.blue}
        pts={byK.map(({ k, rs }) => [x(rs[0].states), y(mean(rs.map((r) => r.solve_rate)))])} />
      <Dots r={4} fill={C.blue}
        pts={byK.map(({ rs }) => [x(rs[0].states), y(mean(rs.map((r) => r.solve_rate)))])} />
      {/* where ground truth exists, the stronger claim */}
      {byK.filter(({ rs }) => rs[0].optimality).length > 0 && (
        <>
          <Line width={2} stroke={C.green} dash="4 3"
            pts={byK.filter(({ rs }) => rs[0].optimality)
              .map(({ rs }) => [x(rs[0].states), y(mean(rs.map((r) => r.optimality.exact_rate)))])} />
          <Mono x={M.l + 10} y={y(0.5)} size={11} fill={C.green} over>
            provably optimal
          </Mono>
        </>
      )}
      <Mono x={M.l + 8} y={M.t + 16} size={11} fill={C.ink}>
        {`${ladder.seeds} seeds per rung, ${ladder.eval_n} scrambles each, `
          + `identical ${ladder.steps.toLocaleString()}-step budget throughout`}
      </Mono>
      <Mono x={M.l + 8} y={M.t + 34} size={11} fill={C.faint}>
        vertical bars are 95% Wilson intervals
      </Mono>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
// Fig 12, the confound broken. Every bar is the SAME 255,024 states; only the
// generating set changes, and with it the diameter. Size cannot explain a
// difference across bars because size never differs.
function DepthFigure() {
  if (!figDepth) {
    return (
      <g>
        <Label x={M.l} y={H / 2 - 10} size={13} bold>not built</Label>
        <Mono x={M.l} y={H / 2 + 14} size={11}>run `mlx-solver/depth_vs_size.py --seeds 3`</Mono>
      </g>
    );
  }
  const rows = figDepth.rows;
  const x = linear(0, rows.length, ...px());
  const y = linear(0, 1, ...py());
  const bw = (x(1) - x(0)) * 0.5;
  return (
    <Axes x={x} y={y} pad={6} xTicks={[]} yTicks={[0, 0.25, 0.5, 0.75, 1]}
      yLabel="fraction of scrambles solved" fmtY={(v) => `${v * 100}%`}>
      {rows.map((r, i) => {
        const cx = x(i + 0.5);
        // Colour by what the condition does when training succeeds. Marking the
        // whole bar as failed because one run deadlocked at the first curriculum
        // level is the reading this experiment was re-run to remove.
        const clean = r.mean_rate_excl_deadlock ?? r.mean_rate;
        const col = clean < 0.99 ? C.red : C.blue;
        const lo = Math.min(...r.rates);
        return (
          <g key={r.moveset}>
            <rect x={cx - bw / 2} y={y(clean)} width={bw}
              height={y(0) - y(clean)} fill={col} opacity="0.85" />
            {/* Ten seeds, nearly all exactly at 1.0 and drawn on top of each
                other. The one outlier is the deadlocked run and is marked as
                such rather than left to read as a hard instance. */}
            {r.rates.map((v, j) => (
              <circle key={j} cx={cx} cy={y(v)} r="3.5" fill={C.paper}
                stroke={C.ink} strokeWidth="1.5" />
            ))}
            {r.n_deadlocked > 0 && (() => {
              // annotate inward, on the last bar an outward label runs off the plot
              const right = i >= rows.length - 1;
              const tx = right ? cx - bw / 2 - 10 : cx + 10;
              const anchor = right ? 'end' : 'start';
              return (
                <g>
                  <Mono x={tx} y={y(lo) + 4} size={9.5} fill={C.red} anchor={anchor}>
                    {`${r.n_deadlocked} run never left`}
                  </Mono>
                  <Mono x={tx} y={y(lo) + 16} size={9.5} fill={C.red} anchor={anchor}>
                    curriculum level 1
                  </Mono>
                </g>
              );
            })()}
            <Mono x={cx} y={y(0) + 18} anchor="middle" size={11} fill={C.ink}>
              {`diam ${r.diameter}`}
            </Mono>
            <Mono x={cx} y={y(0) + 33} anchor="middle" size={9.5}>{r.moveset}</Mono>
            <Mono x={cx} y={y(r.max_rate) - 10} anchor="middle" size={10} fill={col} over>
              {`${(clean * 100).toFixed(0)}%`}
            </Mono>
          </g>
        );
      })}
      <Mono x={M.l} y={392} size={11} fill={C.green}>
        {`state space fixed at ${figDepth.states.toLocaleString()} in every bar: verified by exhaustive BFS`}
      </Mono>
      <Mono x={M.l} y={412} size={11} fill={C.ink}>
        {`${figDepth.seeds} seeds each (open circles). Bars show the rate excluding deadlocked runs:`}
      </Mono>
      <Mono x={M.l} y={430} size={11} fill={C.ink}>
        doubling the diameter changes nothing, every other seed solves everything
      </Mono>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
// Fig 13, the mechanism, and what a pooled scalar hides.
function ProfileFigure() {
  if (!figProfile) {
    return (
      <g>
        <Label x={M.l} y={H / 2 - 10} size={13} bold>not built</Label>
        <Mono x={M.l} y={H / 2 + 14} size={11}>run `mlx-solver/profiles.py`</Mono>
      </g>
    );
  }
  const hs = figProfile.heuristics;
  const all = Object.values(hs).flatMap((h) => h.profile.map((r) => r.d));
  const x = linear(Math.min(...all) - 0.4, Math.max(...all) + 0.4, ...px(90));
  const y = linear(0.4, 1.0, ...py());
  // Every abstraction is shown, shaded by strength. Comparing a learned
  // heuristic against ONE weak abstraction is what produced the earlier,
  // mistaken conclusion that only learned heuristics decay.
  const pdbs = Object.keys(hs).filter((n) => n.startsWith('PDB')).sort();
  const colour = { learned: C.blue, random: C.faint };
  pdbs.forEach((n, i) => { colour[n] = C.purple; });
  const alphaOf = (n) => (n.startsWith('PDB')
    ? 0.35 + 0.65 * ((pdbs.indexOf(n) + 1) / pdbs.length) : 1);
  // Six curves end within a narrow band, so a label at each endpoint would
  // overlap. Place them top-down at their own endpoint, pushed apart only as
  // far as needed: order still matches the curves, so no leader lines.
  const labelY = {};
  let floor = -Infinity;
  Object.entries(hs)
    .sort((a, b) => b[1].profile.at(-1).acc - a[1].profile.at(-1).acc)
    .forEach(([name, h]) => {
      floor = Math.max(y(h.profile.at(-1).acc), floor + 24);
      labelY[name] = floor;
    });
  return (
    <Axes x={x} y={y} pad={6}
      xLabel="true distance from goal  d   (exact, by breadth-first search)"
      yLabel="P[ h(s) < h(s') ]  for s at d, s' at d+1"
      xTicks={[...new Set(all)].sort((a, b) => a - b)}
      yTicks={[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
      fmtY={(v) => v.toFixed(1)}>
      {/* chance, a heuristic at this line cannot order anything */}
      <Line pts={[[x(Math.min(...all) - 0.4), y(0.5)], [x(Math.max(...all) + 0.4), y(0.5)]]}
        stroke={C.faint} width={1.5} dash="4 4" />
      <Mono x={x(Math.min(...all) - 0.3)} y={y(0.5) - 7} size={10}>chance</Mono>
      {Object.entries(hs).map(([name, h]) => {
        const pts = h.profile.map((r) => [x(r.d), y(r.acc)]);
        const last = h.profile.at(-1);
        return (
          <g key={name}>
            <g opacity={alphaOf(name)}>
              <Line pts={pts} stroke={colour[name]} width={name === 'learned' ? 2.8 : 2} />
              <Dots pts={pts} fill={colour[name]} r={3.2} />
            </g>
            <Mono x={x(last.d) + 10} y={labelY[name]} size={11} fill={colour[name]}>
              {name}
            </Mono>
            <Mono x={x(last.d) + 10} y={labelY[name] + 12} size={9} fill={colour[name]}>
              {`τ ${h.gdrc >= 0 ? '+' : ''}${h.gdrc.toFixed(2)}`}
            </Mono>
          </g>
        );
      })}
      <Mono x={M.l + 8} y={y(0.455)} size={11} fill={C.ink}>
        every informative heuristic decays; only the random control is flat
      </Mono>
      <Mono x={M.l + 8} y={y(0.425)} size={11} fill={C.red}>
        the strongest heuristic here is a pattern database, and it decays most
      </Mono>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
// Fig 14: decay against strength, which is the confound that broke the
// original version of the profile claim
function StrengthFigure() {
  if (!figStrength) {
    return (
      <g>
        <Label x={M.l} y={H / 2 - 10} size={13} bold>not built</Label>
        <Mono x={M.l} y={H / 2 + 14} size={11}>run `mlx-solver/strength_control.py`</Mono>
      </g>
    );
  }
  const obs = figStrength.observations;
  const { slope, intercept } = figStrength.fit;
  const rt = figStrength.residual_test;
  const mp = figStrength.matched_pairs;
  const x = linear(-0.05, 1.0, ...px(96));
  const y = linear(-0.05, 0.4, ...py());
  const style = {
    learned: { fill: C.blue, r: 4 },
    PDB: { fill: C.purple, r: 3.4 },
    random: { fill: C.faint, r: 3 },
  };
  return (
    <Axes x={x} y={y} pad={6}
      xLabel="pooled GDRC   (Kendall τ against exact d*)"
      yLabel="decay: fall in per-shell accuracy"
      xTicks={[0, 0.25, 0.5, 0.75, 1.0]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4]}
      fmtX={(v) => v.toFixed(2)} fmtY={(v) => v.toFixed(1)}>
      {/* the fit is the null the paper now tests against: if the training
          method mattered, learned points would sit above this line */}
      <Line pts={[[x(-0.05), y(slope * -0.05 + intercept)],
        [x(1.0), y(slope * 1.0 + intercept)]]}
        stroke={C.mid} width={1.5} dash="5 4" />
      {['random', 'PDB', 'learned'].map((kd) => (
        <Dots key={kd} pts={obs.filter((o) => o.kind === kd).map((o) => [x(o.gdrc), y(o.decay)])}
          fill={style[kd].fill} r={style[kd].r} />
      ))}
      {['learned', 'PDB', 'random'].map((kd, i) => (
        <g key={kd}>
          <circle cx={W - M.r - 92} cy={M.t + 12 + i * 20} r={style[kd].r}
            fill={style[kd].fill} />
          <Mono x={W - M.r - 82} y={M.t + 16 + i * 20} size={11} fill={style[kd].fill}>
            {`${kd} (n=${figStrength.by_kind[kd].n})`}
          </Mono>
        </g>
      ))}
      <Mono x={M.l + 10} y={y(0.375)} size={11} fill={C.ink}>
        {`decay rises with strength: r = ${figStrength.decay_vs_strength_r >= 0 ? '+' : ''}${figStrength.decay_vs_strength_r.toFixed(2)}`}
      </Mono>
      <Mono x={M.l + 10} y={y(0.345)} size={11} fill={C.red}>
        {`residuals do not separate: Welch t = ${rt.t >= 0 ? '+' : ''}${rt.t.toFixed(2)}, p = ${rt.p.toFixed(2)}`}
      </Mono>
      <Mono x={M.l + 10} y={y(0.315)} size={11} fill={C.red}>
        {`matched on identical states, the PDB decays more in ${mp.pdb_decays_more} of ${mp.n} pairs`}
      </Mono>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
// Fig 15, the profile is a two-moment quantity
function LawFigure() {
  if (!figLaw) {
    return (
      <g>
        <Label x={M.l} y={H / 2 - 10} size={13} bold>not built</Label>
        <Mono x={M.l} y={H / 2 + 14} size={11}>run `mlx-solver/dprime_law.py`</Mono>
      </g>
    );
  }
  // Split-sample points only: moments from half of each shell, accuracy from the
  // other half. Plotting the shared-sample version would show a tighter cloud
  // that partly reflects both axes being computed from the same states.
  const obs = figLaw.observations.filter((o) => o.split_acc !== undefined);
  const x = linear(0.45, 1.02, ...px(20));
  const y = linear(0.45, 1.02, ...py());
  const style = {
    learned: { fill: C.blue, r: 3.4 },
    PDB: { fill: C.purple, r: 3.0 },
    random: { fill: C.faint, r: 2.6 },
  };
  const ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  return (
    <Axes x={x} y={y} pad={6}
      xLabel="predicted   Φ( gap / (spread · √2) ), no fitted parameters"
      yLabel="measured ordering accuracy"
      xTicks={ticks} yTicks={ticks}
      fmtX={(v) => v.toFixed(1)} fmtY={(v) => v.toFixed(1)}>
      {/* y = x. Every point that lands on it is a shell whose ordering accuracy
          was predicted from two moments of the value distribution. */}
      <Line pts={[[x(0.45), y(0.45)], [x(1.02), y(1.02)]]}
        stroke={C.mid} width={1.5} dash="5 4" />
      {['random', 'PDB', 'learned'].map((kd) => (
        <Dots key={kd}
          pts={obs.filter((o) => o.kind === kd).map((o) => [x(o.split_pred), y(o.split_acc)])}
          fill={style[kd].fill} r={style[kd].r} />
      ))}
      {['learned', 'PDB', 'random'].map((kd, i) => (
        <g key={kd}>
          <circle cx={M.l + 18} cy={M.t + 14 + i * 20} r={style[kd].r} fill={style[kd].fill} />
          <Mono x={M.l + 28} y={M.t + 18 + i * 20} size={11} fill={style[kd].fill}>
            {`${kd}`}
          </Mono>
        </g>
      ))}
      <Mono x={x(0.62)} y={y(0.53)} size={11} fill={C.ink}>
        {`${figLaw.in_sample_split.n} shells, split-sample mean |error| ${figLaw.in_sample_split.mae.toFixed(4)}`}
      </Mono>
      <Mono x={x(0.62)} y={y(0.49)} size={11} fill={C.red}>
        {`estimator's own noise floor: ${figLaw.in_sample_split.noise_floor.toFixed(4)}`}
      </Mono>
    </Axes>
  );
}

// ---------------------------------------------------------------------------
async function exportAll() {
  const figs = [...document.querySelectorAll('figure svg')].map((el) => {
    const svg = el.cloneNode(true);
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    return { n: el.id.replace('fig-', ''), svg: new XMLSerializer().serializeToString(svg) };
  });
  const r = await fetch('/__figures', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(figs),
  });
  const j = await r.json();
  window.alert(j.ok ? `wrote ${j.written.length} figures to eval/figures/` : `failed: ${j.error}`);
}

export default function FiguresApp() {
  const training = useTraining();
  // Live rows when the dev endpoint answers, otherwise the checked-in file.
  const wingRows = training?.tasks?.find((t) => t.task === 'wings')?.rows
    ?? trainWingsFile.rows;
  const w = wingRows.at(-1);
  return (
    <main>
      <header>
        <h1>CUBEWORKS</h1>
        <p className="sub">
          Figures for the perception and search stack. Every plot is rendered from a
          results file written by an actual run: <code>scripts/bench-solver.mjs</code> and{' '}
          <code>scripts/collect-metrics.mjs</code>, so re-running the benchmarks moves
          the figures. The two schematics are labelled as such.
        </p>
      </header>

      <TrainingPanel data={training} />

      <Figure n={1} width={W} height={290} title="System overview"
        caption="Schematic. The perception stages are classical and verifiable; the only learned component in the critical path is the centres search. Each stage checks its own output and falls back rather than emitting an unverified result.">
        <PipelineFigure />
      </Figure>

      <Figure n={2} width={W} height={420} title="Deep Approximate Value Iteration"
        caption="Schematic. The method is DeepCubeA's, not AlphaZero's: a cube is a single-agent shortest-path problem, so there is no adversary and no self-play. Value propagates outward from the solved state, roughly one shell per target-network refresh.">
        <DaviFigure />
      </Figure>

      <Figure n={3} title="The value function takes shape (edge wings)"
        caption={`Mean predicted cost-to-go against true scramble depth, at ${Math.min(6, wingRows.length)} checkpoints from earliest (faint) to latest (bold). A network that has collapsed shows a flat line here, which a falling training loss will not reveal, an earlier run without a curriculum sat at J = 15.33 for every depth. Curves sit under J = k because random scrambles partly cancel.`}>
        <ValueShapeFigure rows={wingRows} />
      </Figure>

      <Figure n={4} title="Training loss and curriculum depth"
        caption={`Loss (blue, log scale, left) against curriculum depth k (orange, right). Depth advances only once the loss at the current depth falls below a threshold, so the two rise together: each new depth is harder than the one before, and the loss climbing is the curriculum working rather than a failure. ${w ? `Through step ${w.step.toLocaleString()}, k = ${w.k}.` : ''} Loss alone is a poor progress signal here, the collapsed run in Figure 2 reached a lower loss than this one while being useless to search. Figure 3 is the signal that matters.`}>
        <TrainingFigure rows={wingRows} />
      </Figure>

      <Figure n={5} title="Learned centres search vs the hand-written solver"
        caption={`${centres.n} random scrambles, each solved both ways; every solution replayed on the cube before being counted. Points below the diagonal are scrambles the network shortened. Mean ${centres.summary.greedy.mean.toFixed(1)} → ${centres.summary.learned.mean.toFixed(1)} moves.`}>
        <CentresFigure />
      </Figure>

      <Figure n={6} title="What beam width buys"
        caption={`Solution length (blue) and fraction solved (orange) against beam width, ${beam.n} scrambles per width. Below width 10 the search mostly fails outright; past 300 it buys little. The shipped solver escalates 100 → 300 → 800 and keeps the greedy result if all three fail, so the learned stage can only shorten a solve.`}>
        <BeamFigure />
      </Figure>

      <Figure n={7} width={W} height={370} title="Where the moves go"
        caption={`Mean moves per phase over ${full.n} full solves. Replacing the greedy centres stage cuts the total from ${full.summary.off.toFixed(0)} to ${full.summary.on.toFixed(0)} moves. Edge pairing is now the largest single cost, which is why the wings are the next target.`}>
        <PhasesFigure />
      </Figure>

      <Figure n={8} title="Pattern database: an exact bound that is too weak"
        caption="Exact distance-to-solved for every arrangement of the U and D centres, by breadth-first search. This is a genuinely admissible heuristic (it can never overestimate) which is exactly what an optimality proof needs. It is also far too shallow to certify anything about a full solve, which is the honest reason this project does not claim minimal solutions.">
        <PdbFigure />
      </Figure>

      <Figure n={9} width={W} height={362} title="Why the 4×4 is not the 3×3"
        caption="Reachable states, log scale. DeepCubeA solved the 3×3 with billions of training states and days of multi-GPU time. The 4×4 centres are four orders of magnitude smaller than that and train in minutes; the wings are four orders larger; the full cube is another twenty-two beyond. No optimal 4×4 solver exists, and none is close."
      >
        <ScaleFigure />
      </Figure>

      <Figure n={10} title="The depth control"
        caption="Exact breadth-first search on the five ladder rungs small enough to enumerate. Depth is the obvious confound for a scaling study (bigger spaces are usually deeper ones) and here it is measured rather than assumed. Diameter is exactly k+2; over the whole ladder the state space grows by 21 orders of magnitude while depth grows from 4 to about 26.">
        <DepthControlFigure />
      </Figure>

      <Figure n={11} title="Where a learned heuristic stops working"
        caption="The experiment. Identical DAVI, identical network, identical budget, on sub-problems spanning 21 orders of magnitude with branching factor pinned at 63. The dashed green line is the stronger claim (solutions verified optimal against exhaustive ground truth) which is only available where the space is small enough to enumerate, and that limit is itself part of the result.">
        <LadderFigure />
      </Figure>

      <Figure n={12} width={W} height={430} title="Neither depth nor size, over this range"
        caption="Every bar is the same 255,024 states: restricting the generating set leaves the group unchanged (a half turn is two quarter turns) and moves only the diameter, verified by exhaustive BFS reaching all 255,024 in each case. Ten seeds per condition; open circles are individual seeds. Doubling the diameter from 6 to 12 changes nothing: every seed solves every scramble, with one exception. An earlier version of this figure had three seeds and read that exception as a depth effect, but the checkpoint shows the run never advanced past the first curriculum level, so it trained only within two moves of the goal and never saw depth at all. At ten seeds the failure rate is 1 in 40 and is not associated with diameter (Fisher p ≈ 0.23); excluding it, every cell is 100.0 ± 0.0. The design still rules out cardinality, which is held exactly constant, and it never could have separated depth from branching factor since fewer generators means both.">
        <DepthFigure />
      </Figure>

      <Figure n={13} title="What a pooled correlation hides"
        caption="Per-shell ranking accuracy against exact ground truth on rung k=6, for the learned heuristic and for every pattern database available on this rung, the exact distance table of a smaller rung, admissible here by abstraction. Each is summarised by a single Goal Distance Rank Correlation (Wilt &amp; Ruml, JAIR 2016). Every informative heuristic is near-perfect adjacent to the goal and decays outward; only the random control is flat, because it starts at chance and has nowhere to fall from. An earlier version of this figure showed PDB(k=2) alone and read its flatness as a property of classical heuristics; the stronger abstractions show it was a property of being near chance. The decay is not a signature of bootstrapping, the steepest curve here belongs to a pattern database.">
        <ProfileFigure />
      </Figure>

      <Figure n={15} title="The profile is a two-moment quantity"
        caption="Each point is one heuristic at one true-distance shell. The horizontal axis is what the shell's ordering accuracy should be if the two value distributions were normal with equal variance: Φ(gap / (spread·√2)), with nothing fitted. The vertical axis is what was measured. Crucially the two axes use disjoint halves of each shell: moments are estimated on one half and accuracy measured on the other, because computing both from the same states manufactures agreement: run that version on a heuristic with no signal at all and it reports a correlation of +0.95 where the truth is zero. The dashed line is y = x. It holds for learned networks, for integer-valued pattern databases with heavy ties, and for a random control, across every rung and diameter, and on a rung 300× larger whose ground truth comes from a different apparatus. The residual is about six times the estimator's own binomial noise, so this is a good approximation rather than an exact law.">
        <LawFigure />
      </Figure>

      <Figure n={14} title="The decay tracks strength, not training method"
        caption="Every distinct heuristic measured here (22 learned checkpoints, 12 pattern databases, 5 random controls) plotted as pooled GDRC against how far its per-shell ordering accuracy falls. A profile file is written per learned checkpoint and re-measures the same abstractions on the same states, so those repeats are collapsed; counting them would have inflated n from 39 to 98 and every p-value with it. Decay rises with strength (r = +0.59), which is what makes a weak baseline look deceptively flat. The line is the least-squares fit; the test that matters is whether learned points sit above it, and they do not (Welch t = +0.72, p = 0.48 on the residuals, difference +0.023 with 95% CI [−0.039, +0.084] against a mean decay of 0.154). Matched on the same states at the nearest available strength, the pattern database decays more in 13 of 18 pairs and the two means are identical to three decimals.">
        <StrengthFigure />
      </Figure>

      <footer>
        <p>
          Regenerate the data: <code>node scripts/bench-solver.mjs --pdb</code> then{' '}
          <code>node scripts/collect-metrics.mjs</code>. Ladder:{' '}
          <code>mlx-solver/exact.py</code> then <code>mlx-solver/sweep.py</code>.
        </p>
        <p>
          <button type="button" className="export" onClick={exportAll}>
            Write all figures to eval/figures/
          </button>
        </p>
      </footer>
    </main>
  );
}
