// Fold the trainer's append-only metrics log into JSON the figures page can
// import. Safe to run while training is still going, it just picks up whatever
// has been written so far, so the figures track a run in progress.
//
//   node scripts/collect-metrics.mjs
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

for (const task of ['centers', 'wings']) {
  const src = join(ROOT, 'mlx-solver', `metrics_${task}.jsonl`);
  if (!existsSync(src)) {
    console.log(`  (no metrics_${task}.jsonl yet: skipping)`);
    continue;
  }
  const rows = readFileSync(src, 'utf8').trim().split('\n')
    .filter(Boolean).map((l) => JSON.parse(l));
  // A resumed run re-appends from the checkpoint step, so later entries win.
  const byStep = new Map(rows.map((r) => [r.step, r]));
  const merged = [...byStep.values()].sort((a, b) => a.step - b.step);

  const meta = existsSync(join(ROOT, 'mlx-solver', `ckpt_${task}.json`))
    ? JSON.parse(readFileSync(join(ROOT, 'mlx-solver', `ckpt_${task}.json`), 'utf8'))
    : {};

  mkdirSync(join(ROOT, 'eval', 'results'), { recursive: true });
  const out = join(ROOT, 'eval', 'results', `training-${task}.json`);
  writeFileSync(out, `${JSON.stringify({ task, meta, rows: merged }, null, 2)}\n`);
  const last = merged.at(-1);
  console.log(`  ${task}: ${merged.length} checkpoints, through step ${last.step.toLocaleString()}`
    + ` (k=${last.k ?? '-'}, loss ${last.loss.toFixed(4)}) -> eval/results/training-${task}.json`);
}
