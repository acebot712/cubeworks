// Saving and resuming a scan. localStorage is best-effort: a browser that
// refuses it (private mode, quota) must not break the app, so every access is
// guarded and a failure simply means no resume.
import { useEffect, useRef, useState } from 'react';

const KEY = 'cubeworks4-v1';

function read() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw);
    return saved && saved.rawCaptures && Object.keys(saved.rawCaptures).length ? saved : null;
  } catch { return null; }
}

export function clearSaved() {
  try { localStorage.removeItem(KEY); } catch { /* nothing to clean up */ }
}

// onRestore(saved) runs once on mount if there is something to pick up.
// snapshot is written back on every change after that.
export function usePersistence(snapshot, onRestore) {
  const [resumed, setResumed] = useState(false);
  const restoring = useRef(true);

  useEffect(() => {
    const saved = read();
    if (saved) { onRestore(saved); setResumed(true); }
    restoring.current = false;
  }, [onRestore]);

  // Skip the mount pass: it still holds the empty pre-restore snapshot and
  // would overwrite the very scan we are about to resume.
  useEffect(() => {
    if (restoring.current) return;
    try { localStorage.setItem(KEY, JSON.stringify(snapshot)); } catch { /* nothing to save to */ }
  }, [snapshot]);

  return { resumed, dismissResume: () => setResumed(false) };
}
