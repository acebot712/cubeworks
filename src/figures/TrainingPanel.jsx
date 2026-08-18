// Live training status, polled from the dev server's /__training endpoint.
//
// The trainer writes a checkpoint every few minutes and prints its own rate and
// ETA; this reads those files rather than recomputing anything, so the number
// here is the same number the terminal shows. Nothing needs to be running for
// the panel to render, if the trainer has stopped, it says so instead of
// quietly showing a stale ETA as if it were live.
import { useEffect, useState } from 'react';
import { C, Line, linear } from './plot.jsx';

const fmtAge = (ms) => {
  const s = ms / 1000;
  if (s < 90) return `${s.toFixed(0)}s ago`;
  if (s < 5400) return `${(s / 60).toFixed(0)} min ago`;
  return `${(s / 3600).toFixed(1)}h ago`;
};

export function useTraining(everyMs = 15000) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let alive = true;
    const pull = () => fetch('/__training')
      .then((r) => r.json())
      .then((j) => { if (alive) setData(j); })
      .catch(() => {});       // dev endpoint absent (built page): panel hides
    pull();
    const id = setInterval(pull, everyMs);
    return () => { alive = false; clearInterval(id); };
  }, [everyMs]);
  return data;
}

// A compact trace of the last N checkpoints, enough to see a direction.
function Spark({ values, colour, w = 128, h = 30 }) {
  if (values.length < 2) return null;
  const lo = Math.min(...values), hi = Math.max(...values);
  const x = linear(0, values.length - 1, 1, w - 1);
  const y = linear(lo, hi === lo ? lo + 1 : hi, h - 2, 2);
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <Line pts={values.map((v, i) => [x(i), y(v)])} stroke={colour} width={1.5} />
    </svg>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <span className="stat-l">{label}</span>
      <span className="stat-v">{value}</span>
      {sub && <span className="stat-s">{sub}</span>}
    </div>
  );
}

function Row({ t }) {
  const { live, rows, ageMs } = t;
  if (!live) return null;

  // The checkpoint period tells us how quiet is too quiet: a run writing every
  // ~8 minutes that has said nothing for 20 is not running any more.
  const period = rows.length >= 2
    ? (rows.at(-1).elapsed - rows.at(-2).elapsed) * 1000
    : 10 * 60_000;
  const running = ageMs != null && ageMs < Math.max(period * 2.5, 120_000);
  const done = live.step >= live.total;
  const pct = (live.step / live.total) * 100;

  const tail = rows.slice(-24);
  const spread = (r) => (r.probe.at(-1)[1] - r.probe[0][1]);

  return (
    <div className="run">
      <div className="run-head">
        <span className={`dot ${done ? 'done' : running ? 'live' : 'stale'}`} />
        <strong>{t.task}</strong>
        <span className="run-sub">
          {t.meta?.hidden ? `hidden ${t.meta.hidden.join(', ')}` : ''}
        </span>
        <span className="run-when">
          {done ? 'finished' : running ? `updated ${fmtAge(ageMs)}` : `stopped: last wrote ${fmtAge(ageMs)}`}
        </span>
      </div>

      <div className="bar"><span style={{ width: `${pct}%` }} /></div>
      <div className="bar-l">
        <span>{live.step.toLocaleString()} / {live.total.toLocaleString()} steps</span>
        <span>{pct.toFixed(1)}%</span>
      </div>

      <div className="stats">
        <Stat label="ETA" value={done ? ': ' : live.eta} sub={`${live.rate} it/s`} />
        <Stat label="curriculum k" value={live.k} sub={`of ${t.meta?.kmax ?? '?'}`} />
        <Stat label="J spread" value={live.spread.toFixed(2)}
          sub={<Spark values={tail.map(spread)} colour={C.blue} w={96} h={22} />} />
        <Stat label="loss" value={live.loss.toFixed(4)}
          sub={<Spark values={tail.map((r) => r.loss)} colour={C.orange} w={96} h={22} />} />
      </div>
    </div>
  );
}

export default function TrainingPanel({ data }) {
  const runs = (data?.tasks ?? []).filter((t) => t.live);
  if (!runs.length) return null;
  return (
    <section className="panel">
      <h2>Training</h2>
      {runs.map((t) => <Row key={t.task} t={t} />)}
      <p className="panel-note">
        Polled every 15s from the trainer's own log, the ETA here is the one it
        prints. Figures 3 and 4 below track the same run.
      </p>
    </section>
  );
}
