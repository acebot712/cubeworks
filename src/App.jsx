// CUBEWORKS — 4x4 cube AR solver.
//
// Composition root. Three stages: 01 Scan (camera, guided capture), 02 Confirm
// (fix uncertain stickers, validate), 03 Solve (real reduction solution,
// animated 3D guide). Everything substantial lives in a hook (app/) or a
// screen (ui/screens/); this file only wires them together and owns the
// handful of things that genuinely span all three stages.
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { FACES } from './cube/geometry.js';
import { applyMoves } from './cube/state.js';
import { stateFromColors } from './cube/colorState.js';
import { CAPTURE_STEPS } from './scan/orientations.js';
import { sampleCaptures } from './scan/sampleCube.js';
import { COLOR_OF_FACE } from './state/colors.js';
import { useCamera } from './app/useCamera.js';
import { useCubeScan } from './app/useCubeScan.js';
import { useScanLoop } from './app/useScanLoop.js';
import { useSolver } from './app/useSolver.js';
import { usePlayback } from './app/usePlayback.js';
import { usePersistence, clearSaved } from './app/usePersistence.js';
import { colorLookup } from './ui/cubeFaces.jsx';
import { ACCENT, ACCENT_INK, BG, INK } from './ui/theme.js';
import { Btn } from './ui/primitives.jsx';
import AppHeader from './ui/AppHeader.jsx';
import ScanScreen from './ui/screens/ScanScreen.jsx';
import ReviewScreen from './ui/screens/ReviewScreen.jsx';
import SolveScreen from './ui/screens/SolveScreen.jsx';

export default function App() {
  const [screen, setScreen] = useState('scan');
  const [labels, setLabels] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [awaitingTurn, setAwaitingTurn] = useState(false);
  const [scanning, setScanning] = useState(false);

  const camera = useCamera();
  const scan = useCubeScan();
  const solver = useSolver();
  const playback = usePlayback(solver.solution);

  const cube = {
    colors: scan.colors, conf: scan.conf, palette: scan.palette,
    capturedCount: scan.capturedCount, allCaptured: scan.allCaptured,
    complete: scan.complete, validation: scan.validation, counts: scan.counts,
  };
  const step = CAPTURE_STEPS[stepIdx] || CAPTURE_STEPS[0];

  // ---------- scanning ----------
  // A locked face advances the loop to the next one still to capture, then
  // waits: the user turns the cube and says so.
  const onLock = useCallback((capture) => {
    scan.recordCapture(step.key, capture);
    const captured = new Set([...Object.keys(scan.rawCaptures), step.key]);
    const remaining = CAPTURE_STEPS
      .map((s, i) => i)
      .filter((i) => !captured.has(CAPTURE_STEPS[i].key));
    if (!remaining.length) { setAwaitingTurn(false); return; }
    const next = remaining.find((i) => i > stepIdx);
    setStepIdx(next !== undefined ? next : remaining[0]);
    setAwaitingTurn(true);
  }, [scan, step, stepIdx]);

  const loop = useScanLoop({
    videoRef: camera.videoRef,
    enabled: screen === 'scan' && !camera.error,
    active: scanning && !scan.allCaptured && !awaitingTurn,
    onLock,
  });

  // ---------- derived ----------
  const stateKey = cube.colors.join('');
  const solveReady = cube.validation.ok || !!(solver.solution && solver.solution.stateKey === stateKey);

  // the cube after each move of the solution, for the 3D view
  const solveStates = useMemo(() => {
    if (!solver.solution) return null;
    const start = stateFromColors(cube.colors);
    if (!start) return null;
    const list = [start];
    for (const move of solver.solution.flat) list.push(applyMoves(list[list.length - 1], [move]));
    return list;
  }, [solver.solution, cube.colors]);

  const colorOf = colorLookup(cube.palette);

  // ---------- persistence ----------
  const snapshot = useMemo(() => ({
    screen, stepIdx, labels,
    rawCaptures: scan.rawCaptures, manualColors: scan.manualColors, repair: scan.repair,
    moveIdx: playback.moveIdx, solution: solver.solution,
  }), [screen, stepIdx, labels, scan.rawCaptures, scan.manualColors, scan.repair, playback.moveIdx, solver.solution]);

  const restoreScan = scan.restore, restoreSolution = solver.restore, seek = playback.setMoveIdx;
  const restore = useCallback((saved) => {
    restoreScan(saved);
    setStepIdx(saved.stepIdx || 0);
    setLabels(!!saved.labels);
    seek(saved.moveIdx || 0);
    // A solve in flight is not resumable: the worker request died with the
    // page, so land on Confirm rather than an eternal spinner.
    if (saved.solution) restoreSolution(saved.solution);
    setScreen(saved.screen === 'solve' && !saved.solution ? 'review' : saved.screen || 'scan');
  }, [restoreScan, restoreSolution, seek]);

  const { resumed, dismissResume } = usePersistence(snapshot, restore);

  const bindKeys = playback.bindKeys;
  useEffect(() => bindKeys(screen === 'solve' && !!solver.solution), [bindKeys, screen, solver.solution]);

  // ---------- actions ----------
  const startScanning = useCallback(() => { loop.restart(); setScanning(true); }, [loop]);

  const selectStep = useCallback((index) => {
    setStepIdx(index);
    setAwaitingTurn(false);
    setScanning(true);
    loop.restart();
  }, [loop]);

  const goReview = useCallback(() => setScreen('review'), []);

  const requestSolve = useCallback(() => {
    const state = stateFromColors(cube.colors);
    if (!state) return;
    setScreen('solve');
    if (solver.solve(stateKey, state)) playback.restart();
  }, [cube.colors, stateKey, solver, playback]);

  const rescanFace = useCallback((faceKey) => {
    scan.clearFace(faceKey);
    solver.discard();
    selectStep(CAPTURE_STEPS.findIndex((s) => s.key === faceKey));
    setScreen('scan');
  }, [scan, solver, selectStep]);

  const useSampleCube = useCallback(() => {
    scan.loadCaptures(sampleCaptures());
    solver.discard();
    setStepIdx(CAPTURE_STEPS.length - 1);
    setAwaitingTurn(false);
    loop.restart();
  }, [scan, solver, loop]);

  const hardReset = useCallback(() => {
    scan.reset();
    solver.discard();
    playback.restart();
    loop.restart();
    loop.clearManualFrame();
    setScreen('scan');
    setStepIdx(0);
    setAwaitingTurn(false);
    setScanning(false);
    dismissResume();
    clearSaved();
  }, [scan, solver, playback, loop, dismissResume]);

  const goTo = useCallback((target) => {
    if (target === 'solve') requestSolve();
    else setScreen(target);
  }, [requestSolve]);

  const actions = {
    goReview, requestSolve, rescanFace, useSampleCube, startScanning, selectStep,
    startOver: hardReset,
    goScan: () => setScreen('scan'),
    continueScan: () => { loop.restart(); setAwaitingTurn(false); },
    placeManualFrame: loop.placeManualFrame,
    lockNow: loop.lockNow,
    retrySolve: () => { solver.discard(); requestSolve(); },
    invalidateSolution: solver.discard,
    reapplyRepair: scan.reapplyRepair,
    undoRepair: scan.dismissRepair,
  };

  return (
    <div style={{ height: '100vh', minHeight: 800, display: 'flex', flexDirection: 'column', background: BG, color: INK, fontFamily: "'Space Grotesk', system-ui, sans-serif", overflow: 'hidden' }}>
      <AppHeader
        screen={screen}
        solveReady={solveReady}
        labels={labels}
        onGo={goTo}
        onToggleLabels={() => setLabels((l) => !l)}
        onReset={hardReset}
      />

      {resumed && (
        <ResumeBar
          capturedCount={cube.capturedCount}
          onContinue={() => { dismissResume(); if (!cube.allCaptured) startScanning(); }}
          onStartOver={hardReset}
        />
      )}

      {screen === 'scan' && (
        <ScanScreen
          camera={camera} scan={scan} loop={loop} cube={cube}
          step={step} stepIdx={stepIdx} awaitingTurn={awaitingTurn} scanning={scanning}
          labels={labels} actions={actions}
        />
      )}

      {screen === 'review' && (
        <ReviewScreen cube={cube} scan={scan} labels={labels} actions={actions} />
      )}

      {screen === 'solve' && (
        <SolveScreen
          videoRef={camera.videoRef}
          solver={solver} playback={playback} cube={cube} states={solveStates}
          labels={labels} colorOf={(code) => colorOf(COLOR_OF_FACE[FACES[code]])}
          labelOf={(code) => COLOR_OF_FACE[FACES[code]]}
          actions={actions}
        />
      )}
    </div>
  );
}

function ResumeBar({ capturedCount, onContinue, onStartOver }) {
  return (
    <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 14, padding: '11px 18px', background: 'rgba(79,227,193,0.08)', borderBottom: '1px solid rgba(79,227,193,0.2)', animation: 'cw-fade .3s ease' }}>
      <div style={{ fontSize: 12.5, color: '#B6F2E5' }}>
        Picking up where you left off — {capturedCount} of 6 faces already scanned.
      </div>
      <div style={{ flex: 1 }} />
      <Btn onClick={onContinue} style={{ padding: '5px 12px', borderRadius: 7, fontSize: 12, fontWeight: 600, background: ACCENT, color: ACCENT_INK }}>Continue</Btn>
      <Btn onClick={onStartOver} style={{ padding: '5px 12px', borderRadius: 7, fontSize: 12, color: '#9AA2AC', border: '1px solid rgba(255,255,255,0.12)' }}>Start over</Btn>
    </div>
  );
}
