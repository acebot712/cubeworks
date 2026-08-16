// Measure the solver, and write the numbers the figures page plots.
//
// Every figure in figures.html is drawn from a file this script produces. That
// is deliberate: a chart built from numbers typed in by hand is a drawing, not
// a measurement, and there is no way to tell the two apart once it is rendered.
// Re-run this and the figures move.
//
//   node scripts/bench-solver.mjs            # the standard set
//   node scripts/bench-solver.mjs --pdb      # ...plus the pattern database BFS
//
// Results land in eval/results/*.json.
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { applyMoves, solvedState } from '../src/cube/state.js';
import { randomScramble } from '../src/cube/scramble.js';
import { solveCenters } from '../src/solver/centers.js';
import { solveCentersLearned, solveCentersLearnedVerified } from '../src/solver/centers-learned.js';
import { centersFromState } from '../src/solver/tables.js';
import { solve4x4 } from '../src/solver/index.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'eval', 'results');

const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];

function save(name, data) {
  mkdirSync(OUT, { recursive: true });
  const path = join(OUT, `${name}.json`);
  writeFileSync(path, `${JSON.stringify({ generated: new Date().toISOString(), ...data }, null, 2)}\n`);
  console.log(`  -> eval/results/${name}.json`);
}

// A deterministic PRNG so the same scrambles are measured every run and the
// figures only move when the solver does.
function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function scrambles(n, seed = 12345, len = 45) {
  const rand = mulberry32(seed);
  return Array.from({ length: n }, () => applyMoves(solvedState(), randomScramble(len, rand)));
}

const centresSolved = (s) => {
  const c = centersFromState(s);
  for (let i = 0; i < 24; i++) if (c[i] !== ((i / 4) | 0)) return false;
  return true;
};

// --- 1. greedy vs learned centres ------------------------------------------
function benchCentres(n = 40) {
  console.log(`centres: greedy vs learned value network, ${n} scrambles`);
  const states = scrambles(n);
  const rows = [];
  for (const [i, s] of states.entries()) {
    const t0 = performance.now();
    const g = solveCenters(s);
    const tg = performance.now() - t0;
    const t1 = performance.now();
    const l = solveCentersLearnedVerified(s);
    const tl = performance.now() - t1;
    // never record a solution without replaying it
    if (!centresSolved(g.state)) throw new Error(`greedy failed on scramble ${i}`);
    if (l && !centresSolved(applyMoves(s, l.moves))) throw new Error(`learned failed on ${i}`);
    rows.push({ greedy: g.moves.length, learned: l ? l.moves.length : null,
                greedyMs: tg, learnedMs: tl, width: l?.width ?? null });
  }
  const g = rows.map((r) => r.greedy);
  const l = rows.filter((r) => r.learned !== null).map((r) => r.learned);
  console.log(`  greedy  mean ${mean(g).toFixed(1)}  median ${median(g)}  worst ${Math.max(...g)}`);
  console.log(`  learned mean ${mean(l).toFixed(1)}  median ${median(l)}  worst ${Math.max(...l)}`
            + `   (${l.length}/${n} solved, ${mean(rows.map(r=>r.learnedMs)).toFixed(0)} ms each)`);
  save('centres-greedy-vs-learned', {
    n, rows,
    summary: { greedy: { mean: mean(g), median: median(g), max: Math.max(...g), min: Math.min(...g) },
               learned: { mean: mean(l), median: median(l), max: Math.max(...l), min: Math.min(...l),
                          solved: l.length } },
  });
}

// --- 2. how much does beam width buy? --------------------------------------
function benchBeamWidth(widths = [1, 5, 10, 25, 50, 100, 300, 800], n = 15) {
  console.log(`\ncentres: beam width sweep, ${n} scrambles per width`);
  const states = scrambles(n, 999);
  const rows = [];
  for (const width of widths) {
    const lens = [];
    let nodes = 0, fails = 0;
    const t0 = performance.now();
    for (const s of states) {
      const r = solveCentersLearned(s, { width });
      if (!r || !centresSolved(applyMoves(s, r.moves))) { fails++; continue; }
      lens.push(r.moves.length);
      nodes += r.nodes;
    }
    const ms = (performance.now() - t0) / n;
    rows.push({ width, solved: lens.length, n, mean: lens.length ? mean(lens) : null,
                nodes: nodes / Math.max(lens.length, 1), ms });
    console.log(`  width ${String(width).padStart(4)}  solved ${lens.length}/${n}  `
      + `mean ${lens.length ? mean(lens).toFixed(1) : '  - '}  ${ms.toFixed(0)} ms  `
      + `${(nodes / Math.max(lens.length, 1) / 1000).toFixed(0)}k nodes`);
  }
  save('centres-beam-width', { n, rows });
}

// --- 3. the whole solve, both ways -----------------------------------------
function benchFullSolve(n = 25) {
  console.log(`\nfull solve: learned centres on vs off, ${n} scrambles`);
  const states = scrambles(n, 4242);
  const rows = [];
  for (const s of states) {
    const off = solve4x4(s, { learnedCenters: false });
    const on = solve4x4(s, { learnedCenters: true });
    const phase = (r) => Object.fromEntries(r.phases.map((p) => [p.name, p.moves.length]));
    rows.push({ off: off.totalMoves, on: on.totalMoves, offPhases: phase(off), onPhases: phase(on) });
  }
  const off = rows.map((r) => r.off), on = rows.map((r) => r.on);
  console.log(`  greedy centres  mean ${mean(off).toFixed(1)} moves`);
  console.log(`  learned centres mean ${mean(on).toFixed(1)} moves   `
            + `(${((1 - mean(on) / mean(off)) * 100).toFixed(0)}% shorter)`);

  // per-phase averages, for the stacked breakdown
  const names = ['Centers', 'Edge pairing', 'Parity', '3×3 finish'];
  const byPhase = (key) => Object.fromEntries(
    names.map((nm) => [nm, mean(rows.map((r) => r[key][nm] ?? 0))]));
  save('full-solve', {
    n, rows,
    summary: { off: mean(off), on: mean(on) },
    phases: { off: byPhase('offPhases'), on: byPhase('onPhases') },
  });
}

// --- 4. pattern database depth distribution --------------------------------
// The honest picture of why this approach was abandoned: the table is complete,
// but it bottoms out far too shallow to certify anything about a ~130-move solve.
async function benchPdb() {
  console.log('\npattern database: BFS over all 51,482,970 U+D centre arrangements');
  const { buildPdb, PDB_SIZE } = await import('../src/solver/pdb.js');
  const t0 = performance.now();
  const levels = [];
  const { dist, maxDepth, reached } = buildPdb({
    onProgress: ({ depth, seen, frontier }) => {
      levels.push({ depth: depth - 1, frontier });
      console.log(`  depth ${String(depth - 1).padStart(2)}  frontier ${frontier.toLocaleString()}`
                + `  seen ${seen.toLocaleString()}`);
    },
  });
  const hist = new Array(maxDepth + 1).fill(0);
  for (let i = 0; i < dist.length; i++) if (dist[i] !== 255) hist[dist[i]]++;
  const seconds = (performance.now() - t0) / 1000;
  console.log(`  built in ${seconds.toFixed(0)}s   max depth ${maxDepth}   `
            + `reached ${reached.toLocaleString()}/${PDB_SIZE.toLocaleString()}`);
  save('pdb-depths', { size: PDB_SIZE, reached, maxDepth, seconds,
                       histogram: hist.map((count, depth) => ({ depth, count })) });
}

const args = new Set(process.argv.slice(2));
benchCentres();
benchBeamWidth();
benchFullSolve();
if (args.has('--pdb')) await benchPdb();
else console.log('\n(skipping the pattern-database BFS; pass --pdb to include it)');
