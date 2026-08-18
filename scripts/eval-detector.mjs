// Measure the detector against REAL captured frames.
//
// The unit tests are synthetic: images generated from the same assumptions the
// detector encodes, which is circular and stayed green through every real-world
// failure so far. This runs the same detector over frames from an actual camera
// and reports what it actually does: including which conditions it fails in.
//
// Usage:  npm run eval  [-- --threshold 0.5] [--verbose]
import fs from 'node:fs';
import path from 'node:path';
import { detectFace } from '../src/scan/detect/search.js';
import { ACQUIRE_SCORE } from '../src/scan/detect/tracker.js';
import { DETECT_W } from '../src/scan/detect/frame.js';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf(`--${k}`); return i >= 0 ? argv[i + 1] : d; };
const ACQUIRE = Number(arg('threshold', ACQUIRE_SCORE));
const VERBOSE = argv.includes('--verbose');
const IOU_PASS = Number(arg('iou', 0.6));
const WIDTH = Number(arg('width', DETECT_W));

const LABELS = 'eval/labels.json';
if (!fs.existsSync(LABELS)) {
  console.error(`No dataset yet: ${LABELS} does not exist.

Capture some frames first:
  1. npm run dev
  2. open http://localhost:5183/capture.html
  3. label real frames (and negatives!), then re-run npm run eval
`);
  process.exit(1);
}
const labels = JSON.parse(fs.readFileSync(LABELS, 'utf8'));

// ---------- polygon helpers (no dependencies) ----------
const area = (poly) => {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
};

// Sutherland–Hodgman: clip `subject` against convex `clip`
function clipPoly(subject, clip) {
  let out = subject;
  for (let i = 0; i < clip.length; i++) {
    const A = clip[i], B = clip[(i + 1) % clip.length];
    const input = out;
    out = [];
    const side = (p) => (B[0] - A[0]) * (p[1] - A[1]) - (B[1] - A[1]) * (p[0] - A[0]);
    for (let j = 0; j < input.length; j++) {
      const P = input[j], Q = input[(j + 1) % input.length];
      const sp = side(P), sq = side(Q);
      if (sp <= 0) out.push(P);
      if ((sp < 0 && sq > 0) || (sp > 0 && sq < 0)) {
        const t = sp / (sp - sq);
        out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]);
      }
    }
    if (!out.length) return [];
  }
  return out;
}

function ensureCW(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    a += x1 * y2 - x2 * y1;
  }
  return a > 0 ? [...poly].reverse() : poly;
}

function iou(a, b) {
  const A = ensureCW(a), B = ensureCW(b);
  const inter = clipPoly(A, B);
  if (!inter.length) return 0;
  const ai = area(inter);
  return ai / (area(A) + area(B) - ai);
}

// the detector reports centre/size/angle; turn that into a quad
function quadOf(d) {
  const c = Math.cos(d.theta), s = Math.sin(d.theta), h = d.size / 2;
  return [[-h, -h], [h, -h], [h, h], [-h, h]].map(([u, v]) => [d.cx + u * c - v * s, d.cy + u * s + v * c]);
}

// Stored buffers are high-resolution; box-downsample to the width the detector
// actually works at. Keeping the two separate means changing DETECT_W re-scores
// the existing dataset instead of invalidating it.
function loadFrame(id, w, h, target) {
  const buf = fs.readFileSync(path.join('eval/frames', `${id}.bin`));
  if (!target || target >= w) {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0, j = 0; i < w * h; i++, j += 3) {
      data[i * 4] = buf[j]; data[i * 4 + 1] = buf[j + 1]; data[i * 4 + 2] = buf[j + 2]; data[i * 4 + 3] = 255;
    }
    return { data, width: w, height: h };
  }
  const tw = target, th = Math.max(1, Math.round((h * target) / w));
  const data = new Uint8ClampedArray(tw * th * 4);
  const sx = w / tw, sy = h / th;
  for (let y = 0; y < th; y++) {
    const y0 = Math.floor(y * sy), y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
    for (let x = 0; x < tw; x++) {
      const x0 = Math.floor(x * sx), x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
      let r = 0, g = 0, b = 0, n = 0;
      for (let yy = y0; yy < y1 && yy < h; yy++) {
        for (let xx = x0; xx < x1 && xx < w; xx++) {
          const j = (yy * w + xx) * 3;
          r += buf[j]; g += buf[j + 1]; b += buf[j + 2]; n++;
        }
      }
      const o = (y * tw + x) * 4;
      data[o] = r / n; data[o + 1] = g / n; data[o + 2] = b / n; data[o + 3] = 255;
    }
  }
  return { data, width: tw, height: th };
}

// ---------- run ----------
const results = [];
for (const l of labels) {
  const frame = loadFrame(l.id, l.detW, l.detH, WIDTH);
  const t0 = process.hrtime.bigint();
  const d = detectFace(frame);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  const found = !!(d && d.score >= ACQUIRE);

  let score = d ? d.score : 0;
  let overlap = null;
  if (l.corners) {
    const truth = l.corners.map(([x, y]) => [x * frame.width, y * frame.height]);
    overlap = found ? iou(quadOf(d), truth) : 0;
  }
  results.push({ id: l.id, tags: l.tags, positive: !!l.corners, found, score, iou: overlap, ms });
}

// ---------- report ----------
const pos = results.filter((r) => r.positive);
const neg = results.filter((r) => !r.positive);
const hits = pos.filter((r) => r.found && r.iou >= IOU_PASS);
const missed = pos.filter((r) => !r.found);
const mislocated = pos.filter((r) => r.found && r.iou < IOU_PASS);
const falsePos = neg.filter((r) => r.found);

const pct = (n, d) => (d ? `${((n / d) * 100).toFixed(1)}%` : ': ');
const med = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : 0);

console.log(`\nDETECTOR EVAL: ${results.length} real frames  (width ${WIDTH}, acquire ${ACQUIRE}, IoU pass ${IOU_PASS})`);
console.log('─'.repeat(64));
console.log(`  with a cube        ${String(pos.length).padStart(4)}`);
console.log(`    detected + located ${String(hits.length).padStart(4)}   ${pct(hits.length, pos.length)}`);
console.log(`    missed entirely    ${String(missed.length).padStart(4)}   ${pct(missed.length, pos.length)}`);
console.log(`    found but off      ${String(mislocated.length).padStart(4)}   ${pct(mislocated.length, pos.length)}`);
console.log(`  without a cube     ${String(neg.length).padStart(4)}`);
console.log(`    false positives    ${String(falsePos.length).padStart(4)}   ${pct(falsePos.length, neg.length)}`);
console.log('─'.repeat(64));
if (pos.length) {
  const ious = pos.filter((r) => r.found).map((r) => r.iou);
  console.log(`  median IoU (when found)  ${med(ious).toFixed(3)}`);
  console.log(`  median score on cubes    ${med(pos.map((r) => r.score)).toFixed(3)}`);
}
if (neg.length) console.log(`  max score on negatives   ${Math.max(...neg.map((r) => r.score)).toFixed(3)}`);
console.log(`  median detect time       ${med(results.map((r) => r.ms)).toFixed(1)} ms`);

// per-tag breakdown: this is how you learn WHICH conditions break it
const tagSet = new Set();
for (const r of results) for (const t of String(r.tags || '').split(',').map((s) => s.trim()).filter(Boolean)) tagSet.add(t);
if (tagSet.size) {
  console.log('\n  BY CONDITION');
  for (const t of [...tagSet].sort()) {
    const rs = results.filter((r) => String(r.tags || '').split(',').map((s) => s.trim()).includes(t));
    const p = rs.filter((r) => r.positive);
    const ok = p.filter((r) => r.found && r.iou >= IOU_PASS);
    const fp = rs.filter((r) => !r.positive && r.found);
    const bits = [];
    if (p.length) bits.push(`${ok.length}/${p.length} found`);
    if (rs.length - p.length) bits.push(`${fp.length}/${rs.length - p.length} false-fire`);
    console.log(`    ${t.padEnd(22)} ${bits.join('  ·  ')}`);
  }
}

const bad = [...missed, ...mislocated, ...falsePos];
if (bad.length) {
  console.log(`\n  FAILURES: open eval/frames/<id>.jpg to see what it saw`);
  for (const r of bad.slice(0, VERBOSE ? 999 : 15)) {
    const why = !r.positive ? `false positive (score ${r.score.toFixed(3)})`
      : !r.found ? `missed (score ${r.score.toFixed(3)})`
      : `mislocated (IoU ${r.iou.toFixed(2)})`;
    console.log(`    ${r.id}  ${why}${r.tags ? `  [${r.tags}]` : ''}`);
  }
  if (!VERBOSE && bad.length > 15) console.log(`    …and ${bad.length - 15} more (--verbose)`);
}
console.log('');

// non-zero exit if the real-world numbers are poor, so this can gate a commit
const recall = pos.length ? hits.length / pos.length : 1;
const fpr = neg.length ? falsePos.length / neg.length : 0;
process.exit(recall >= 0.9 && fpr <= 0.02 ? 0 : 1);
