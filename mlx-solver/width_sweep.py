"""How much beam width does the heuristic actually need? Plain JS cannot afford
width 1200, so the question is where quality falls off."""
import json, time
from pathlib import Path
import numpy as np
from train_centers import TOKENS, apply_moves, is_solved
from solve_centers import load_net, beam_solve

rows = json.loads(Path("real_centers.json").read_text())[:8]
import sys
net = load_net(sys.argv[1] if len(sys.argv)>1 else "centers_value.npz")
print("model:", sys.argv[1] if len(sys.argv)>1 else "centers_value.npz")
print("width | mean moves | solved | time/state")
print("------|------------|--------|-----------")
for w in (25, 50, 100, 300):
    lens, fail, t0 = [], 0, time.time()
    for row in rows:
        st = np.array(row["centers"], dtype=np.int8)
        sol = beam_solve(net, st, width=w)
        if sol is None: fail += 1; continue
        chk = st[None, :].copy()
        for mv in sol: chk = apply_moves(chk, np.array([TOKENS.index(mv)]))
        assert is_solved(chk)[0]
        lens.append(len(sol))
    ms = (time.time()-t0)/len(rows)
    print(f"{w:5d} |    {np.mean(lens):5.1f}   |  {len(lens)}/{len(rows)}   | {ms:5.2f}s")
