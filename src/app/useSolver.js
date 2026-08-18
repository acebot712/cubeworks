// The solver worker and everything that can go wrong with it. Every failure
// path ends in a message the Solve screen can act on: a worker that never loads
// posts nothing at all, which used to leave the screen spinning forever.
import { useCallback, useEffect, useRef, useState } from 'react';

const WATCHDOG_MS = 45000;
const SLOW_MESSAGE =
  'The solver is taking much longer than expected. This usually means the scanned '
  + 'state is valid but unusually hard: try again, or re-scan.';

export function useSolver() {
  const workerRef = useRef(null);
  const [solution, setSolution] = useState(null); // {phases, flat, totalMoves, stateKey}
  const [solving, setSolving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const w = new Worker(new URL('../solver/worker.js', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.postMessage({ type: 'warmup' });

    const fail = (message) => { setSolving(false); setError(message); };
    w.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === 'solution') {
        setSolution({
          phases: msg.phases,
          flat: msg.phases.flatMap((p) => p.moves),
          totalMoves: msg.totalMoves,
          stateKey: msg.id,
        });
        setSolving(false);
        setError(null);
      } else if (msg.type === 'error') {
        fail(msg.message);
      }
    };
    w.onerror = (e) => fail((e && e.message) || 'The solver failed to start.');
    w.onmessageerror = () => fail('The solver sent a message this browser could not read.');
    return () => w.terminate();
  }, []);

  // no solver run should outlive this without saying something
  useEffect(() => {
    if (!solving) return;
    const id = setTimeout(() => { setSolving(false); setError(SLOW_MESSAGE); }, WATCHDOG_MS);
    return () => clearTimeout(id);
  }, [solving]);

  // Ask for a solution unless the one in hand already answers this state.
  // -> true if a run started
  const solve = useCallback((stateKey, state) => {
    if (solution && solution.stateKey === stateKey) return false;
    setSolving(true);
    setError(null);
    workerRef.current.postMessage({ type: 'solve', id: stateKey, state: Array.from(state) });
    return true;
  }, [solution]);

  // The cube changed under us (an edit, a re-scan, a reset), so whatever we
  // have no longer describes it.
  const discard = useCallback(() => { setSolution(null); setError(null); }, []);

  return { solution, solving, error, solve, discard, restore: setSolution };
}
