// Solution playback: the transport, its timer, and the keyboard shortcuts.
import { useCallback, useEffect, useRef, useState } from 'react';

const BASE_INTERVAL_MS = 750;
const MIN_INTERVAL_MS = 340;

export function usePlayback(solution) {
  const [moveIdx, setMoveIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const total = solution ? solution.flat.length : 0;
  const latest = useRef({ total });
  latest.current = { total };

  const step = useCallback((delta) => {
    setMoveIdx((i) => Math.max(0, Math.min(latest.current.total, i + delta)));
    if (delta < 0) setPlaying(false);
  }, []);

  const seek = useCallback((i) => { setMoveIdx(i); setPlaying(false); }, []);
  const restart = useCallback(() => { setMoveIdx(0); setPlaying(false); }, []);

  useEffect(() => {
    if (!playing || !total) return undefined;
    const id = setInterval(() => {
      setMoveIdx((i) => {
        if (i >= latest.current.total) { setPlaying(false); return i; }
        return i + 1;
      });
    }, Math.max(MIN_INTERVAL_MS, BASE_INTERVAL_MS / speed));
    return () => clearInterval(id);
  }, [playing, speed, total]);

  // `enabled` keeps the shortcuts off the Scan and Confirm screens.
  const bindKeys = useCallback((enabled) => {
    if (!enabled) return undefined;
    const onKey = (e) => {
      if (e.code === 'Space') { e.preventDefault(); setPlaying((p) => !p); }
      else if (e.key === 'ArrowRight') step(1);
      else if (e.key === 'ArrowLeft') step(-1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step]);

  return { moveIdx, setMoveIdx: seek, playing, setPlaying, speed, setSpeed, step, restart, bindKeys };
}
