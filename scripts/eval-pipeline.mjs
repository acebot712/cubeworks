// End-to-end evaluation of the full acquisition pipeline on real frames.
//
// Compares, per frame:
//   classical  — the in-browser search alone, no service
//   best-of    — every proposal the service makes (the VLM's box and each of
//                SAM's masks), each polished by the corner descent, with the
//                cube-face scorer picking the winner
//
// The scorer is the arbiter because neither model knows what a cube FACE is:
// SAM's best mask is usually the whole 3D cube, whose silhouette includes the
// side faces.
//
// Usage:  npm run eval:pipeline  [-- --port 8765] [--verbose]
import fs from 'node:fs';
import path from 'node:path';
import { detectFace, detectFaceInRoi } from '../src/scan/detect/search.js';
import { scoreQuad } from '../src/scan/detect/score.js';
import { ACQUIRE_SCORE } from '../src/scan/detect/tracker.js';
import { DETECT_W } from '../src/scan/detect/frame.js';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf(`--${k}`); return i >= 0 ? argv[i + 1] : d; };
const PORT = Number(arg('port', 8765));
const WIDTH = Number(arg('width', DETECT_W));
const VERBOSE = argv.includes('--verbose');

const labels = JSON.parse(fs.readFileSync('eval/labels.json', 'utf8'));
const health = await fetch(`http://127.0.0.1:${PORT}/health`).then((r) => r.json()).catch(() => null);
if (!health) {
  console.error(`No service on ${PORT}. Start it with:  npm run mlx\n`);
  process.exit(1);
}

function loadFrame(l, target) {
  const buf = fs.readFileSync(path.join('eval/frames', `${l.id}.bin`));
  const w = l.detW, h = l.detH;
  const tw = Math.min(target, w), th = Math.round((h * tw) / w);
  const data = new Uint8ClampedArray(tw * th * 4);
  const sx = w / tw, sy = h / th;
  for (let y = 0; y < th; y++) {
    const y0 = Math.floor(y * sy), y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
    for (let x = 0; x < tw; x++) {
      const x0 = Math.floor(x * sx), x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
      let r = 0, g = 0, b = 0, n = 0;
      for (let yy = y0; yy < y1 && yy < h; yy++) {
        for (let xx = x0; xx < x1 && xx < w; xx++) {
          const j = (yy * w + xx) * 3; r += buf[j]; g += buf[j + 1]; b += buf[j + 2]; n++;
        }
      }
      const o = (y * tw + x) * 4;
      data[o] = r / n; data[o + 1] = g / n; data[o + 2] = b / n; data[o + 3] = 255;
    }
  }
  return { data, width: tw, height: th };
}

const area = (p) => {
  let a = 0;
  for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; a += x1 * y2 - x2 * y1; }
  return Math.abs(a) / 2;
};
function clipPoly(s, c) {
  let o = s;
  for (let i = 0; i < c.length; i++) {
    const A = c[i], B = c[(i + 1) % c.length], inp = o; o = [];
    const side = (p) => (B[0] - A[0]) * (p[1] - A[1]) - (B[1] - A[1]) * (p[0] - A[0]);
    for (let j = 0; j < inp.length; j++) {
      const P = inp[j], Q = inp[(j + 1) % inp.length], sp = side(P), sq = side(Q);
      if (sp <= 0) o.push(P);
      if ((sp < 0 && sq > 0) || (sp > 0 && sq < 0)) {
        const t = sp / (sp - sq); o.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]);
      }
    }
    if (!o.length) return [];
  }
  return o;
}
const cw = (p) => {
  let a = 0;
  for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; a += x1 * y2 - x2 * y1; }
  return a > 0 ? [...p].reverse() : p;
};
const iou = (a, b) => {
  const I = clipPoly(cw(a), cw(b));
  if (!I.length) return 0;
  const ai = area(I);
  return ai / (area(cw(a)) + area(cw(b)) - ai);
};
const boxOf = (q) => {
  const xs = q.map((p) => p[0]), ys = q.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
};

console.log(`\nPIPELINE EVAL — ${health.model}`);
console.log(`  segmentation: ${health.segment ? 'on' : 'OFF'}   width ${WIDTH}   acquire ${ACQUIRE_SCORE}\n`);
console.log('id   | classical         | best-of pipeline                    | winner');
console.log('-----|-------------------|-------------------------------------|--------');

let nClass = 0, nBest = 0, tSum = 0, nCalls = 0;
const ious = { classical: [], best: [] };
for (const l of labels.filter((x) => x.corners)) {
  const f = loadFrame(l, WIDTH);
  const truth = l.corners.map(([x, y]) => [x * f.width, y * f.height]);
  const jpeg = fs.readFileSync(path.join('eval/frames', `${l.id}.jpg`)).toString('base64');
  const t0 = Date.now();
  const r = await fetch(`http://127.0.0.1:${PORT}/detect`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ image: `data:image/jpeg;base64,${jpeg}` }),
  }).then((x) => x.json()).catch((e) => ({ error: String(e) }));
  tSum += Date.now() - t0; nCalls++;

  const classical = detectFace(f);
  if (classical && classical.score >= ACQUIRE_SCORE) nClass++;
  if (classical) ious.classical.push(iou(classical.corners, truth));

  const props = [];
  if (classical) props.push({ tag: 'classical', d: classical });
  if (r.box) {
    const mb = [r.box[0] * f.width, r.box[1] * f.height, r.box[2] * f.width, r.box[3] * f.height];
    const h = detectFaceInRoi(f, mb, 0.18);
    if (h) props.push({ tag: 'vlm', d: h });
  }
  for (const [i, cand] of (r.candidates || []).entries()) {
    const q = cand.corners.map(([x, y]) => [x * f.width, y * f.height]);
    props.push({ tag: `sam${i}raw`, d: { score: scoreQuad(f, q), corners: q } });
    const h = detectFaceInRoi(f, boxOf(q), 0.10);
    if (h) props.push({ tag: `sam${i}`, d: h });
  }
  props.sort((a, b) => b.d.score - a.d.score);
  const best = props[0];
  if (best && best.d.score >= ACQUIRE_SCORE) nBest++;
  if (best) ious.best.push(iou(best.d.corners, truth));

  const fmt = (d) => (d ? `${d.score.toFixed(2)}, IoU ${iou(d.corners, truth).toFixed(2)}` : 'none');
  console.log(`${l.id} | ${fmt(classical).padEnd(17)} | ${fmt(best && best.d).padEnd(35)} | ${best ? best.tag : '—'}`);
  if (VERBOSE) for (const p of props.slice(0, 6)) console.log(`     ${p.tag.padEnd(12)} ${fmt(p.d)}`);
}

const med = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : 0);
const n = labels.filter((x) => x.corners).length;
console.log('\n' + '─'.repeat(64));
console.log(`  above acquire threshold   classical ${nClass}/${n}   best-of ${nBest}/${n}`);
console.log(`  median IoU                classical ${med(ious.classical).toFixed(3)}   best-of ${med(ious.best).toFixed(3)}`);
console.log(`  mean service latency      ${Math.round(tSum / Math.max(1, nCalls))} ms  (acquisition only)\n`);
