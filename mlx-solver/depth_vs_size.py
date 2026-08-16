"""Is it the size of the problem that defeats a learned heuristic, or the depth?

On the wing ladder those two cannot be told apart: diameter = k+2 exactly, so
every larger rung is also a deeper one, and a failure at the top could be blamed
on either. This experiment breaks that tie by holding the state space EXACTLY
constant and moving only the depth.

Rung k=4 has 255,024 states. Restricting the generating set leaves that number
untouched — a half turn is two quarter turns, so the group generated is the same
— while stretching the diameter:

    all      63 moves   diameter  6
    oi-q     24 moves   diameter  8
    oi-q4    16 moves   diameter 10
    oi-q3    12 moves   diameter 12

All four verified by exhaustive BFS to reach all 255,024 states. Train the same
network, on the same budget, with the same everything, on each. If performance
falls as the diameter grows, state-space cardinality cannot be the operative
variable — because it never changed.

The honest caveat, stated up front: fewer generators also means a smaller
branching factor, so this design varies depth and branching together and cannot
separate those two. What it CAN do is rule out size, which is the variable the
whole ladder was built around and the one the literature argues about.

    ../.venv-mlx/bin/python depth_vs_size.py --steps 12000 --seeds 3
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = str(Path(sys.executable))
RESULTS = HERE.parent / "eval" / "results"

RUNG = 4
VARIANTS = ["all", "oi-q", "oi-q4", "oi-q3"]      # diameter 6, 8, 10, 12
PROTOCOL = dict(batch=1024, hidden="4096,2048,1024", sync_every=100,
                lr=1e-3, curriculum="on", half="on")


def sh(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--width", type=int, default=100)
    # Scramble depth must exceed the DEEPEST variant's diameter, or the shallow
    # variants would be evaluated on harder states than the deep ones and the
    # comparison would be backwards.
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    diam = {}
    for v in VARIANTS:
        suf = "" if v == "all" else f"-{v}"
        p = RESULTS / f"exact-k{RUNG}{suf}.json"
        diam[v] = json.loads(p.read_text())["diameter"] if p.exists() else None

    print(f"rung k={RUNG}: 255,024 states in every row — only depth changes\n")
    print(f"{'moveset':>8} {'diameter':>9} {'seeds':>6}")
    for v in VARIANTS:
        print(f"{v:>8} {str(diam[v]):>9} {args.seeds:>6}")
    print(f"\nbudget {args.steps:,} steps, scrambles of {args.kmax} moves, "
          f"eval n={args.n} at beam width {args.width}\n")
    if args.dry_run:
        return

    rows, t0 = [], time.time()
    for v in VARIANTS:
        for seed in range(args.seeds):
            tag = f"_dv-{v}_s{seed}"
            task = f"wings-k{RUNG}"
            head = f"{v:>6} (diam {diam[v]}) s{seed}"

            meta = HERE / f"ckpt_{task}{tag}.json"
            done = json.loads(meta.read_text()).get("step", 0) if meta.exists() else 0
            if done < args.steps:
                rc = sh([PY, str(HERE / "davi.py"), "--task", task, "--tag", tag,
                         "--moves", v, "--seed", str(seed), "--steps", str(args.steps),
                         "--kmax", str(args.kmax),
                         "--ckpt-every", str(max(500, args.steps // 8)),
                         "--batch", str(PROTOCOL["batch"]), "--hidden", PROTOCOL["hidden"],
                         "--sync-every", str(PROTOCOL["sync_every"]), "--lr", str(PROTOCOL["lr"]),
                         "--curriculum", PROTOCOL["curriculum"], "--half", PROTOCOL["half"]],
                        HERE / f"log_dv_train_{v}_s{seed}.txt")
                if rc:
                    print(f"{head}  TRAIN FAILED rc={rc}", flush=True)
                    continue

            out = RESULTS / f"eval-{task}{tag}.json"
            if not out.exists():
                sh([PY, str(HERE / "evaluate.py"), "--task", task, "--tag", tag,
                    "--n", str(args.n), "--width", str(args.width),
                    "--scramble", str(args.kmax)],
                   HERE / f"log_dv_eval_{v}_s{seed}.txt")
            if not out.exists():
                print(f"{head}  EVAL FAILED", flush=True)
                continue

            r = json.loads(out.read_text())
            r["moveset"], r["diameter"], r["seed"] = v, diam[v], seed
            # Where the curriculum actually got to. A run that never advances
            # past the first level trained only on states within k_cur moves of
            # the goal, whatever the diameter of the space -- so its failure is
            # a training failure, not evidence about depth. Recording this is
            # what distinguishes the two, and it was missing from the original
            # experiment.
            cm = HERE / f"ckpt_{task}{tag}.json"
            if cm.exists():
                m = json.loads(cm.read_text())
                r["k_cur"], r["kmax_run"] = m.get("k_cur"), m.get("kmax")
                r["deadlocked"] = bool(m.get("k_cur", 0) <= 2)
            rows.append(r)
            o = r.get("optimality")
            print(f"{head}  solved {r['solve_rate']*100:5.1f}% "
                  f"[{r['ci95'][0]*100:4.1f}-{r['ci95'][1]*100:4.1f}]"
                  + (f"  optimal {o['exact_rate']*100:5.1f}%" if o else "")
                  + f"  len {r['mean_len']:.1f}" if r.get("mean_len") else "", flush=True)

            sh([PY, str(HERE / "resolution.py"), "--task", task, "--tag", tag, "--n", "30000"],
               HERE / f"log_dv_res_{v}_s{seed}.txt")

    out = RESULTS / "depth-vs-size.json"
    out.write_text(json.dumps({
        "rung": RUNG, "states": 255024, "steps": args.steps, "seeds": args.seeds,
        "kmax": args.kmax, "protocol": PROTOCOL, "diameters": diam,
        "elapsed_hours": (time.time() - t0) / 3600, "runs": rows,
    }, indent=2) + "\n")
    # The figure's data file used to be assembled by hand, which meant it kept
    # reporting three seeds after this experiment had been re-run at ten. Derive
    # it here so the two cannot disagree again.
    by_ms = {}
    for r in rows:
        by_ms.setdefault(r["moveset"], []).append(r)
    fig = []
    for ms, rs in by_ms.items():
        rates = [x["solve_rate"] for x in rs]
        ok = [x["solve_rate"] for x in rs if not x.get("deadlocked")]
        fig.append({"moveset": ms, "diameter": rs[0]["diameter"],
                    "states": 255024, "n_seeds": len(rs), "rates": rates,
                    "mean_rate": sum(rates) / len(rates),
                    "min_rate": min(rates), "max_rate": max(rates),
                    "n_deadlocked": sum(1 for x in rs if x.get("deadlocked")),
                    "mean_rate_excl_deadlock": (sum(ok) / len(ok)) if ok else None})
    fig.sort(key=lambda r: r["diameter"])

    # Is the deadlock associated with diameter at all? Computed here rather than
    # by hand: the paper previously quoted p = 0.23 for this, which is wrong.
    from scipy.stats import fisher_exact
    deep = max(fig, key=lambda r: r["diameter"])
    a = deep["n_deadlocked"]
    b = deep["n_seeds"] - a
    c = sum(r["n_deadlocked"] for r in fig if r is not deep)
    e = sum(r["n_seeds"] for r in fig if r is not deep) - c
    odds, p = fisher_exact([[a, b], [c, e]])
    deadlock = {"table": [[a, b], [c, e]], "fisher_p": float(p),
                "n_deadlocked_total": sum(r["n_deadlocked"] for r in fig),
                "n_runs_total": sum(r["n_seeds"] for r in fig)}
    print(f"\ndeadlock {deadlock['n_deadlocked_total']}/{deadlock['n_runs_total']}; "
          f"largest diameter {a}/{a+b} vs {c}/{c+e} elsewhere, Fisher p = {p:.3f}",
          flush=True)
    (RESULTS / "fig-depth.json").write_text(json.dumps(
        {"states": 255024, "steps": args.steps, "seeds": args.seeds,
         "kmax": args.kmax, "rows": fig, "deadlock": deadlock}, indent=2) + "\n")

    print(f"\n{len(rows)} runs -> {out.name}  ({(time.time()-t0)/3600:.1f}h)")


if __name__ == "__main__":
    main()
