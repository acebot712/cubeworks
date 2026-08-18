// Score the MLX acquisition service on the same real frames as the classical
// detector, so the two are directly comparable.
//
// The model only has to ANSWER ONE QUESTION: roughly where is the cube? Exact
// corners come from the existing refinement running inside its box, so this
// measures box recall and box IoU, not corner precision.
//
// Usage:  npm run eval:mlx  [-- --port 8765] [--verbose]
import fs from 'node:fs';
import path from 'node:path';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf(`--${k}`); return i >= 0 ? argv[i + 1] : d; };
const PORT = Number(arg('port', 8765));
const VERBOSE = argv.includes('--verbose');
const IOU_PASS = Number(arg('iou', 0.5));

const LABELS = 'eval/labels.json';
if (!fs.existsSync(LABELS)) { console.error('No dataset: capture frames first.'); process.exit(1); }
const labels = JSON.parse(fs.readFileSync(LABELS, 'utf8'));

const health = await fetch(`http://127.0.0.1:${PORT}/health`).then((r) => r.json()).catch(() => null);
if (!health) {
  console.error(`No MLX service on port ${PORT}. Start it with:

  ./.venv-mlx/bin/python mlx-detect/server.py
`);
  process.exit(1);
}
console.log(`\nMLX ACQUISITION EVAL: model ${health.model}`);

// axis-aligned IoU: the model returns a box, so compare against the labelled
// quad's bounding box rather than the quad itself
const bboxOf = (corners) => {
  const xs = corners.map((p) => p[0]), ys = corners.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
};
function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
  if (x1 <= x0 || y1 <= y0) return 0;
  const inter = (x1 - x0) * (y1 - y0);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

const results = [];
for (const l of labels) {
  const jpeg = fs.readFileSync(path.join('eval/frames', `${l.id}.jpg`)).toString('base64');
  const r = await fetch(`http://127.0.0.1:${PORT}/detect`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ image: `data:image/jpeg;base64,${jpeg}` }),
  }).then((x) => x.json()).catch((e) => ({ error: String(e) }));

  const positive = !!l.corners;
  const overlap = positive && r.found ? iou(r.box, bboxOf(l.corners)) : null;
  results.push({ id: l.id, tags: l.tags, positive, found: !!r.found, iou: overlap, ms: r.ms || 0, raw: r.raw, error: r.error });
  process.stdout.write('.');
}
process.stdout.write('\n');

const pos = results.filter((r) => r.positive);
const neg = results.filter((r) => !r.positive);
const hits = pos.filter((r) => r.found && r.iou >= IOU_PASS);
const missed = pos.filter((r) => !r.found);
const off = pos.filter((r) => r.found && r.iou < IOU_PASS);
const fp = neg.filter((r) => r.found);
const pct = (n, d) => (d ? `${((n / d) * 100).toFixed(1)}%` : ': ');
const med = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : 0);

console.log('─'.repeat(64));
console.log(`  with a cube        ${String(pos.length).padStart(4)}`);
console.log(`    box found + right  ${String(hits.length).padStart(4)}   ${pct(hits.length, pos.length)}`);
console.log(`    no box returned    ${String(missed.length).padStart(4)}   ${pct(missed.length, pos.length)}`);
console.log(`    box in wrong place ${String(off.length).padStart(4)}   ${pct(off.length, pos.length)}`);
console.log(`  without a cube     ${String(neg.length).padStart(4)}`);
console.log(`    false positives    ${String(fp.length).padStart(4)}   ${pct(fp.length, neg.length)}`);
console.log('─'.repeat(64));
console.log(`  median box IoU           ${med(pos.filter((r) => r.found).map((r) => r.iou)).toFixed(3)}`);
console.log(`  median latency           ${med(results.map((r) => r.ms))} ms   (acquisition only: tracking stays 3.2ms)`);

const bad = [...missed, ...off, ...fp];
if (bad.length) {
  console.log('\n  FAILURES');
  for (const r of bad) {
    const why = !r.positive ? 'false positive' : !r.found ? 'no box' : `box off (IoU ${r.iou.toFixed(2)})`;
    console.log(`    ${r.id}  ${why}${r.error ? `  ${r.error}` : ''}`);
    if (VERBOSE && r.raw) console.log(`          model said: ${r.raw.replace(/\n/g, ' ')}`);
  }
}
console.log('');
