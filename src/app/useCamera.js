// Camera lifecycle: acquire the stream once, keep it attached across screen
// changes that remount the <video>, and report why it failed.
import { useCallback, useEffect, useRef, useState } from 'react';

const CONSTRAINTS = { video: { width: 1280, height: 720 } };

export function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState(false);
  const [denied, setDenied] = useState(false);
  const [retries, setRetries] = useState(0);

  const attach = useCallback((stream) => {
    const v = videoRef.current;
    if (!v || v.srcObject === stream) return;
    v.srcObject = stream;
    const p = v.play();
    if (p && p.catch) p.catch(() => {});
  }, []);

  const start = useCallback(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError(true); setDenied(false); return;
    }
    setRetries((n) => n + 1);
    navigator.mediaDevices.getUserMedia(CONSTRAINTS)
      .then((stream) => {
        streamRef.current = stream;
        setError(false);
        attach(stream);
      })
      .catch(() => { setError(true); setDenied(true); });
  }, [attach]);

  useEffect(() => {
    start();
    return () => {
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    };
  }, [start]);

  // re-attach after a screen change remounts the element
  useEffect(() => {
    if (streamRef.current) attach(streamRef.current);
  });

  return { videoRef, error, denied, retries, start };
}
