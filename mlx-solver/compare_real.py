"""Apples-to-apples: the learned beam search vs the shipped greedy solver, on
centre states taken from real 40-move cube scrambles."""
import json, time
from pathlib import Path
import numpy as np
from train_centers import SOLVED, TOKENS, N_MOVES, apply_moves, is_solved
from solve_centers import load_net, beam_solve

rows = json.loads(Path("real_centers.json").read_text())
net = load_net("centers_value.npz")

print("  #  | greedy | learned beam | time")
print("-----|--------|--------------|------")
g, b, t0 = [], [], time.time()
for i, row in enumerate(rows):
    st = np.array(row["centers"], dtype=np.int8)
    t = time.time()
    sol = beam_solve(net, st, width=1200)
    ms = time.time() - t
    if sol is None:
        print(f" {i+1:3d} |   {row['greedy']:3d}  |    FAILED    |")
        continue
    chk = st[None, :].copy()
    for mv in sol:
        chk = apply_moves(chk, np.array([TOKENS.index(mv)]))
    assert is_solved(chk)[0], "invalid solution"
    g.append(row["greedy"]); b.append(len(sol))
    print(f" {i+1:3d} |   {row['greedy']:3d}  |     {len(sol):3d}      | {ms:4.1f}s")

if b:
    print(f"\n  greedy mean {np.mean(g):5.1f}   learned mean {np.mean(b):5.1f}"
          f"   -> {(1-np.mean(b)/np.mean(g))*100:.0f}% fewer moves")
    print(f"  total search time {time.time()-t0:.0f}s for {len(b)} states")
