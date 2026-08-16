// Dev-only endpoint that writes the rendered figures to eval/figures/ as
// standalone SVG files.
//
// The per-figure download button covers one figure at a time through the
// browser's download flow. This writes the whole set in one call, which is what
// you actually want when the numbers have moved and every figure needs
// regenerating — and it puts them on disk as real files rather than in a
// downloads folder.
import fs from 'node:fs';
import path from 'node:path';

const DIR = path.join('eval', 'figures');
const MLX = 'mlx-solver';

// The trainer already computes the rate and ETA and prints them, so parse its
// own line rather than recomputing from timestamps — one source of truth, and
// it keeps working across a resume (where elapsed restarts at zero).
//   "  step  36,000/150,000  k 40  loss   0.0959  J spread   9.00    3.9 it/s  ETA 8.1h"
const LINE = /step\s+([\d,]+)\/([\d,]+)\s+k\s+(\d+)\s+loss\s+([\d.]+)\s+J spread\s+([-\d.]+)\s+([\d.]+) it\/s\s+ETA\s+(\S+)/;
const num = (s) => Number(s.replace(/,/g, ''));

function readTraining(task) {
  const log = path.join(MLX, `${task}_train.log`);
  const metrics = path.join(MLX, `metrics_${task}.jsonl`);
  if (!fs.existsSync(metrics)) return null;

  const rows = fs.readFileSync(metrics, 'utf8').trim().split('\n')
    .filter(Boolean).map((l) => JSON.parse(l));
  // a resumed run re-appends from its checkpoint step, so later entries win
  const merged = [...new Map(rows.map((r) => [r.step, r])).values()]
    .sort((a, b) => a.step - b.step);

  const out = { task, rows: merged };

  if (fs.existsSync(log)) {
    const text = fs.readFileSync(log, 'utf8');
    const hits = text.split('\n').map((l) => LINE.exec(l)).filter(Boolean);
    const m = hits.at(-1);
    if (m) {
      out.live = { step: num(m[1]), total: num(m[2]), k: num(m[3]),
                   loss: Number(m[4]), spread: Number(m[5]),
                   rate: Number(m[6]), eta: m[7] };
    }
    // how long since the trainer last wrote — a dead run goes quiet, and a
    // stale panel that still says "running" is worse than no panel
    out.ageMs = Date.now() - fs.statSync(log).mtimeMs;
  }
  const meta = path.join(MLX, `ckpt_${task}.json`);
  if (fs.existsSync(meta)) out.meta = JSON.parse(fs.readFileSync(meta, 'utf8'));
  return out;
}

export default function figuresPlugin() {
  return {
    name: 'cubeworks-figures',
    apply: 'serve',
    configureServer(server) {
      // Live training state, polled by the figures page. Reads the files the
      // trainer is already writing, so nothing has to be running for this to
      // answer — it just reports an older `ageMs`.
      server.middlewares.use('/__training', (req, res) => {
        res.setHeader('content-type', 'application/json');
        res.setHeader('cache-control', 'no-store');
        try {
          const tasks = ['wings', 'centers'].map(readTraining).filter(Boolean);
          res.end(JSON.stringify({ tasks, now: Date.now() }));
        } catch (e) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: String(e) }));
        }
      });

      server.middlewares.use('/__figures', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }
        let body = '';
        req.on('data', (c) => { body += c; });
        req.on('end', () => {
          try {
            const figs = JSON.parse(body);
            fs.mkdirSync(DIR, { recursive: true });
            const written = figs.map(({ n, svg }) => {
              const file = path.join(DIR, `figure-${n}.svg`);
              fs.writeFileSync(file, `<?xml version="1.0" encoding="UTF-8"?>\n${svg}\n`);
              return file;
            });
            res.setHeader('content-type', 'application/json');
            res.end(JSON.stringify({ ok: true, written }));
          } catch (e) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(e) }));
          }
        });
      });
    },
  };
}
