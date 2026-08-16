// Palette-free colour resolution for a scanned cube.
//
// The scanner must not assume what the six colours ARE. Cube brands ship wildly
// different palettes — pastel, neon, "bright" stickerless, purple in place of
// blue, custom sticker sets — and a fixed hue table silently fails on all of
// them. What IS universal is the structure:
//
//   * a 4x4 cube has exactly 16 stickers of each of 6 colours.
//
// So we cluster the 96 RGB samples into 6 balanced groups with no reference
// colours at all. Naming those groups is a separate problem solved in
// resolve.js by the cube's piece geometry (NOT by centres — on a 4x4 the
// centre blocks are scrambled like any other piece). The solver only ever sees
// face letters; colour names and hexes become a display concern derived from
// the measured pixels.
//
// A colour cast (warm bulbs, cool daylight) shifts every sticker together, so
// clustering in chromaticity space absorbs it for free.

// chromaticity (lighting-robust) + damped brightness
function feat(rgb) {
  const [r, g, b] = rgb;
  const s = r + g + b + 1e-6;
  return [r / s, g / s, b / s, (Math.max(r, g, b) / 255) * 0.35];
}

function dist(a, b) {
  let d = 0;
  for (let i = 0; i < 4; i++) { const t = a[i] - b[i]; d += t * t; }
  return Math.sqrt(d);
}

// Deterministic farthest-point seeding (k-means++ without the randomness, so
// runs are reproducible and testable). Seeds come from the DATA, never from a
// reference palette.
function seedCenters(feats, k) {
  const mean = [0, 0, 0, 0];
  for (const f of feats) for (let d = 0; d < 4; d++) mean[d] += f[d] / feats.length;
  let first = 0, best = -1;
  for (let i = 0; i < feats.length; i++) {
    const q = dist(feats[i], mean);
    if (q > best) { best = q; first = i; }
  }
  const centers = [feats[first].slice()];
  while (centers.length < k) {
    let pick = 0, far = -1;
    for (let i = 0; i < feats.length; i++) {
      let near = Infinity;
      for (const c of centers) near = Math.min(near, dist(feats[i], c));
      if (near > far) { far = near; pick = i; }
    }
    centers.push(feats[pick].slice());
  }
  return centers;
}

// Optimal balanced assignment: every cluster takes exactly `per` items at the
// lowest total cost. This is a transportation problem, solved exactly by
// min-cost flow. A greedy pass is much cheaper but order-dependent, and on
// palettes with two close colours (neon white vs neon yellow) it wedges items
// into the wrong group and never recovers — the 16-per-colour constraint only
// corrects errors if it is applied optimally.
function balancedAssign(costs, per) {
  const N = costs.length, K = costs[0].length;
  const S = N + K, T = N + K + 1, V = N + K + 2;
  const head = new Array(V).fill(-1);
  const to = [], nxt = [], cap = [], cost = [];
  const add = (u, v, c, w) => {
    to.push(v); cap.push(c); cost.push(w); nxt.push(head[u]); head[u] = to.length - 1;
    to.push(u); cap.push(0); cost.push(-w); nxt.push(head[v]); head[v] = to.length - 1;
  };
  for (let i = 0; i < N; i++) add(S, i, 1, 0);
  for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) add(i, N + k, 1, costs[i][k]);
  for (let k = 0; k < K; k++) add(N + k, T, per, 0);

  const pot = new Array(V).fill(0);
  const distv = new Array(V), inq = new Array(V), prevE = new Array(V);
  for (let f = 0; f < N; f++) {
    distv.fill(Infinity); inq.fill(false); prevE.fill(-1);
    distv[S] = 0;
    const q = [S]; inq[S] = true;
    while (q.length) { // SPFA: robust to the zero-cost edges here
      const u = q.shift(); inq[u] = false;
      for (let e = head[u]; e !== -1; e = nxt[e]) {
        if (cap[e] <= 0) continue;
        const v = to[e], w = cost[e] + pot[u] - pot[v];
        if (distv[u] + w < distv[v] - 1e-12) {
          distv[v] = distv[u] + w; prevE[v] = e;
          if (!inq[v]) { inq[v] = true; q.push(v); }
        }
      }
    }
    for (let v = 0; v < V; v++) if (distv[v] < Infinity) pot[v] += distv[v];
    for (let v = T; v !== S; v = to[prevE[v] ^ 1]) { cap[prevE[v]]--; cap[prevE[v] ^ 1]++; }
  }

  const assign = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    for (let e = head[i]; e !== -1; e = nxt[e]) {
      if (to[e] >= N && to[e] < N + K && cap[e] === 0) { assign[i] = to[e] - N; break; }
    }
  }
  return assign;
}

// rgbs: 96 x [r,g,b]
// -> { assign: 96 cluster ids 0..5, centroids: 6 x [r,g,b], conf: 96, alt: 96 }
export function clusterStickers(rgbs) {
  const N = rgbs.length;
  const per = N / 6;
  const raw = rgbs.map(feat);

  // Normalise by the cube's OWN colour spread, so a washed-out macaron cube is
  // clustered as confidently as a neon one: what matters is separation relative
  // to the spread present, not absolute vividness.
  //
  // The three chromaticity axes share ONE scale, because they are commensurate
  // and rescaling them independently would distort the colour geometry.
  // Brightness gets its own scale but floored — on a cube whose colours are all
  // equally bright that axis carries nothing but sensor noise, and normalising
  // it independently would amplify that noise until it drowned the real signal.
  const mean = [0, 0, 0, 0], sd = [0, 0, 0, 0];
  for (const f of raw) for (let d = 0; d < 4; d++) mean[d] += f[d] / N;
  for (const f of raw) for (let d = 0; d < 4; d++) sd[d] += (f[d] - mean[d]) ** 2 / N;
  for (let d = 0; d < 4; d++) sd[d] = Math.sqrt(sd[d]);
  const sChroma = Math.max((sd[0] + sd[1] + sd[2]) / 3, 1e-4);
  const sVal = Math.max(sd[3], 0.6 * sChroma);
  const scale = [sChroma, sChroma, sChroma, sVal / 0.7];
  const feats = raw.map((f) => f.map((v, d) => (v - mean[d]) / scale[d]));

  let centers = seedCenters(feats, 6);
  let assign = new Array(N).fill(-1);

  for (let iter = 0; iter < 16; iter++) {
    const costs = feats.map((f) => centers.map((c) => dist(f, c)));
    const next = balancedAssign(costs, per);
    const converged = next.every((v, i) => v === assign[i]);
    assign = next;
    const sum = [0, 1, 2, 3, 4, 5].map(() => [0, 0, 0, 0]);
    for (let i = 0; i < N; i++) for (let d = 0; d < 4; d++) sum[assign[i]][d] += feats[i][d];
    centers = sum.map((s) => s.map((v) => v / per));
    if (converged) break;
  }

  // centroids in RGB, for display: show the user their own cube's colours
  const centroids = [0, 1, 2, 3, 4, 5].map(() => [0, 0, 0]);
  for (let i = 0; i < N; i++) for (let d = 0; d < 3; d++) centroids[assign[i]][d] += rgbs[i][d] / per;

  const alt = new Array(N);
  const conf = feats.map((f, i) => {
    const chosen = dist(f, centers[assign[i]]);
    let other = Infinity, otherK = -1;
    for (let k = 0; k < 6; k++) {
      if (k === assign[i]) continue;
      const d = dist(f, centers[k]);
      if (d < other) { other = d; otherK = k; }
    }
    alt[i] = otherK;
    return Math.max(0, Math.min(1, ((other - chosen) / (other + chosen + 1e-6)) * 2.2));
  });
  return { assign, centroids: centroids.map((c) => c.map(Math.round)), conf, alt };
}

// Nearest canonical name, for labels only — never for decisions.
const CANON = [
  { key: 'W', rgb: [242, 243, 245] }, { key: 'Y', rgb: [255, 208, 40] },
  { key: 'G', rgb: [35, 177, 90] }, { key: 'B', rgb: [44, 107, 232] },
  { key: 'R', rgb: [232, 64, 42] }, { key: 'O', rgb: [255, 122, 26] },
];
export function nameCentroids(centroids) {
  const cf = centroids.map(feat);
  const used = new Set();
  const out = new Array(centroids.length).fill('W');
  const pairs = [];
  cf.forEach((f, i) => CANON.forEach((c) => pairs.push({ i, key: c.key, d: dist(f, feat(c.rgb)) })));
  pairs.sort((a, b) => a.d - b.d);
  const done = new Set();
  for (const p of pairs) {
    if (done.has(p.i) || used.has(p.key)) continue;
    out[p.i] = p.key; used.add(p.key); done.add(p.i);
  }
  return out;
}
