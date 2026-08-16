// Fixed algorithms used by more than one solver stage. Each is verified by
// tests (and re-verified by simulation at runtime where it is applied), so they
// live in exactly one place — edges.js and three.js both need the OLL parity
// alg and must never drift apart.
import { parseMoves } from '../cube/notation.js';

// Odd wing permutation. Every pairing macro uses inner slice quarters in pairs
// (an even permutation), so a stuck endgame in the wrong parity class needs
// exactly this. Keeps centers solved and every paired edge paired.
export const OLL_PARITY_ALG = parseMoves("r U2 x r U2 r U2 r' U2 l U2 r' U2 r U2 r' U2 r'");

// Toggles 3x3 edge/corner permutation parity of the reduced state.
export const PLL_PARITY_ALG = parseMoves('r2 U2 r2 Uw2 r2 Uw2');

// In-place flip of the FR edge slot: keeps every pair and all centers (it also
// 4-cycles some U-layer slots, which is harmless). Used to change the
// orientation class of a stuck last-two-edges case.
export const FLIP_FR_ALG = parseMoves("R U R' F R' F' R");
