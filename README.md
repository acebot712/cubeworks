# CUBEWORKS — 4×4 Rubik's Cube AR Solver

Scan a real 4×4×4 cube with your MacBook camera, confirm the read, and follow
an animated, verified solution move by move.

## Run

```bash
npm install
npm run dev        # http://localhost:5183
npm test           # engine + solver test suite (node --test)
```

The camera needs a secure context — `localhost` works out of the box. No
camera (or permission denied) still lets you walk the whole flow with
**Use sample cube**.

## The flow

1. **01 Scan** — hold the cube up to the camera. A 4×4 grid reads sticker
   colors live (HSV classification with multi-frame voting, Touch-ID style:
   a face only locks after ~1s of consistent, steady frames). Guided mode
   walks a fixed six-face loop (roll, roll, roll, then spin); the app detects
   your turns from frame motion and tells you exactly which way to rotate.
2. **02 Confirm** — only the stickers the classifier was unsure about are
   queued as quick either/or questions; the full 96-sticker net is editable
   underneath. Validation is piece-level (real corners, real edge wings,
   center counts), not just sticker counts.
3. **03 Solve** — a real reduction-method solver runs in a Web Worker:
   centers → edge pairing → parity fixes → Kociemba 3×3 finish. The route is
   verified move-by-move by replay before it's shown. Follow along on the
   animated 3D cube (or camera-overlay AR mode); moves are shown in WCA
   notation plus plain English, and detected turns advance the guide so your
   hands never leave the cube.

## How the solver works (src/solver/)

- **Move engine** (`src/cube/engine.js`) — every move permutation is
  generated from 3D cubie geometry, so tables can't drift out of sync.
- **Centers** (`centers.js`) — solves faces U, D, F, R, B in order via
  iterative-deepening searches over center-only states; every accepted
  sequence is validated by simulation.
- **Edge pairing** (`edges.js`) — slice-flip-slice macros `S1 · s · S2 · s'`
  where S1/S2 are searched outer-move sequences and the goals (fresh pair
  formed, centers restored, net pair gain) are checked on wing/center
  projections. A precomputed last-two-edges case table handles the endgame,
  and a verified odd-wing-parity alg unblocks the parity-locked class.
- **3×3 stage** (`three.js`) — the reduced cube maps to 3×3 facelets; OLL/PLL
  parity are detected mathematically (edge flip sum, permutation signs) and
  fixed with verified algs; the rest is Kociemba two-phase (cubejs).

Every stage re-verifies its work on the full 96-facelet state; `solve4x4`
replays the final user-facing move list before returning it.

## Notes

- Scan orientation math is derived with a label-tracking cube from the
  guided-step rotations (`src/scan/orientations.js`); if you spin the wrong
  way on the last two faces the app auto-detects and corrects it.
- State persists in localStorage; a resume bar offers to continue.
