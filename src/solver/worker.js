// Web worker: runs the full 4x4 solve off the UI thread.
import { solve4x4 } from './index.js';
import { initThreeSolver } from './three.js';

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'warmup') {
    try { initThreeSolver(); } catch {}
    self.postMessage({ type: 'ready' });
    return;
  }
  if (msg.type === 'solve') {
    try {
      const state = Uint8Array.from(msg.state);
      const result = solve4x4(state);
      self.postMessage({ type: 'solution', id: msg.id, phases: result.phases, totalMoves: result.totalMoves });
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, message: err && err.message ? err.message : String(err) });
    }
  }
};
