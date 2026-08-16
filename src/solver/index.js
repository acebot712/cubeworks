// Full 4x4 solve: centers -> edge pairing -> parity fixes -> 3x3 Kociemba.
// Returns phases with user-facing WCA moves (wide turns, no bare inner
// slices) plus a verification flag computed by replaying everything.
import { simplifyMoves, toUserMoves } from '../cube/notation.js';
import { applyMoves } from '../cube/state.js';
import { solveCenters } from './centers.js';
import { solveCentersLearnedVerified } from './centers-learned.js';
import { solveEdges } from './edges.js';
import { solveThreeStage, initThreeSolver } from './three.js';

export function isSolved(state) {
  for (let i = 0; i < 96; i++) if (state[i] !== ((i / 16) | 0)) return false;
  return true;
}

// opts.learnedCenters (default true): use the trained value network for the
// centres stage. Measured on real scrambles it spends ~15 moves where the
// greedy solver spends ~36, and it verifies its own output, falling back to
// greedy if the search fails — so it can only shorten the solve.
export function solve4x4(state96, { learnedCenters = true } = {}) {
  initThreeSolver();
  const phases = [];
  let state = state96;

  const centers = (learnedCenters && solveCentersLearnedVerified(state)) || solveCenters(state);
  state = centers.state;
  const edges = solveEdges(state);
  state = edges.state;
  const three = solveThreeStage(state);
  state = three.state;

  if (!isSolved(state)) throw new Error('internal error: pipeline did not solve the cube');

  const centersUser = toUserMoves(simplifyMoves(centers.moves));
  const edgesUser = toUserMoves(simplifyMoves(edges.moves));
  const parityUser = toUserMoves(simplifyMoves(three.parityMoves));
  const threeUser = simplifyMoves(three.moves);

  if (centersUser.length) phases.push({ name: 'Centers', moves: centersUser });
  if (edgesUser.length) phases.push({ name: 'Edge pairing', moves: edgesUser });
  if (parityUser.length) phases.push({ name: 'Parity', moves: parityUser });
  if (threeUser.length) phases.push({ name: '3×3 finish', moves: threeUser });

  // final verification on the user-facing move list
  const flat = phases.flatMap((p) => p.moves);
  if (!isSolved(applyMoves(state96, flat))) {
    throw new Error('internal error: user move list failed verification');
  }

  return { phases, totalMoves: flat.length };
}
