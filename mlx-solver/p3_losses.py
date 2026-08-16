"""Does targeting the quantity that governs ordering actually produce a better heuristic?

dprime_law.py establishes that per-shell ordering accuracy is
Phi(gap / (sd * sqrt(2))), and tau_theory.py that the decay exists only because
sd grows with distance. Adjacent shells are one move apart, so gap is pinned
near 1 for anything calibrated to distance: within-shell spread is the only free
variable. Squared error does not target it -- it charges bias and variance the
same, while ordering is indifferent to bias and cares only about variance
relative to the gap.

So this compares three objectives, identical in every other respect:

  l2       standard DAVI regression, the baseline
  rank     RankNet-style pairwise logistic -- "optimise to rank, not to
           estimate", the published alternative and the comparator that matters
  dprime   l2 plus an explicit within-shell variance penalty

and measures each on three things the arms cannot game: the per-shell profile
against exact ground truth, the pooled GDRC, and the smallest beam width that
reaches a 90% solve rate. The last is the one that decides it -- an intrinsic
improvement that does not reduce search cost is not worth having.

Every arm uses the scale-invariant curriculum gate. The default gate advances on
squared error, which a scale-free ranking loss can never satisfy, and using it
here would pin that arm at the first curriculum level and fake a result.

Idempotent: re-running skips finished work, so it survives a lid close.

    ../.venv-mlx/bin/python p3_losses.py --steps 12000 --seeds 3
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = str(HERE.parent / ".venv-mlx" / "bin" / "python")
RESULTS = HERE.parent / "eval" / "results"

# Held identical across every arm, so the objective is the only thing that varies.
PROTOCOL = {"batch": 1024, "hidden": "4096,2048,1024", "sync_every": 100,
            "lr": 1e-3, "curriculum": "on", "half": "on", "gate": "order"}

# Chosen so the outcome measure can actually move. On rung k=4 every existing
# budget curve reaches a 90% solve rate at beam width 1, so all three arms would
# sit on the floor and the comparison would measure nothing; the one exception
# is the deepest generating set, kept here because it is cheap and has exact
# ground truth for the profile. Rung k=6 is the workhorse -- it has both exact
# ground truth AND real spread in beam width (18.9 to 89.6 across existing
# runs). Rung k=8 has no enumerable ground truth, so it contributes search cost
# only, and tests whether any effect survives a 300x larger space.
CONFIGS = [("wings-k6", "all", 8),        # profile + search cost
           ("wings-k8", "all", 10),       # search cost only
           ("wings-k4", "oi-q3", 12)]     # profile at the largest diameter
# `rank` and `rank-adj` differ only in which pairs they are shown. Uniformly
# drawn pairs are overwhelmingly far apart and easy to order, so `rank` trains on
# the pooled quantity this work argues is the wrong target, while `rank-adj` sees
# only pairs one shell apart. Running both makes the pair-sampling scheme, rather
# than the loss family, the thing being compared.
ARMS = ["l2", "rank", "rank-adj", "dprime"]


def sh(cmd, log):
    with open(log, "w") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--arms", default="",
                    help="comma list overriding ARMS. An entry may carry its own "
                         "penalty weight as loss:lam, e.g. 'dprime:3,dprime:10' -- "
                         "used to sweep lambda without re-running the other arms")
    ap.add_argument("--configs", default="",
                    help="comma list of tasks or movesets to restrict to, e.g. 'wings-k6'")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--budget-n", type=int, default=120)
    ap.add_argument("--out", default="p3-losses.json",
                    help="result file. A sweep MUST write elsewhere: this file is "
                         "rewritten wholesale each run, so sharing it would silently "
                         "replace the main comparison with the sweep's rows")
    args = ap.parse_args()

    t0 = time.time()
    rows = []
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or ARMS
    configs = CONFIGS
    if args.configs:
        want = {c.strip() for c in args.configs.split(",")}
        # match on task or moveset: two configs share the "all" moveset,
        # so restricting by moveset alone would not isolate a rung
        configs = [c for c in CONFIGS if c[0] in want or c[1] in want]
    total = len(configs) * len(arms) * args.seeds
    i = 0
    print(f"{total} runs: {len(configs)} problems x {len(arms)} objectives x "
          f"{args.seeds} seeds, {args.steps:,} steps each\n", flush=True)

    for task, mv, diam in configs:
        for arm in arms:
            # "dprime:3" means the dprime objective at penalty weight 3. The
            # weight goes in the tag so a sweep does not overwrite the runs it
            # is being compared against.
            loss, _, lam_s = arm.partition(":")
            lam = float(lam_s) if lam_s else args.lam
            for seed in range(args.seeds):
                i += 1
                slug = arm.replace(":", "-lam")
                tag = f"_p3-{mv}-{slug}_s{seed}"
                head = f"[{i:>2}/{total}] {mv:<6} {slug:<12} s{seed}"

                meta = HERE / f"ckpt_{task}{tag}.json"
                done = json.loads(meta.read_text()).get("step", 0) if meta.exists() else 0
                if done < args.steps:
                    cmd = [PY, str(HERE / "davi.py"), "--task", task, "--tag", tag,
                           "--moves", mv, "--seed", str(seed), "--steps", str(args.steps),
                           "--loss", loss, "--gate", PROTOCOL["gate"],
                           "--batch", str(PROTOCOL["batch"]), "--hidden", PROTOCOL["hidden"],
                           "--sync-every", str(PROTOCOL["sync_every"]),
                           "--lr", str(PROTOCOL["lr"]), "--curriculum", PROTOCOL["curriculum"],
                           "--half", PROTOCOL["half"],
                           "--ckpt-every", str(max(500, args.steps // 8))]
                    if loss == "dprime":
                        cmd += ["--lam", str(lam)]
                    if sh(cmd, HERE / f"log_p3_train_{mv}_{slug}_s{seed}.txt"):
                        print(f"{head}  TRAIN FAILED", flush=True)
                        continue

                r = {"task": task, "moves": mv, "diameter": diam, "arm": arm, "lam": lam,
                     "seed": seed, "tag": tag}
                if meta.exists():
                    m = json.loads(meta.read_text())
                    # A run that never left the first curriculum level trained only
                    # on states beside the goal; its numbers say nothing about the
                    # objective, and pooling it in would be the same mistake the
                    # depth experiment originally made.
                    r["k_cur"] = m.get("k_cur")
                    r["deadlocked"] = bool(m.get("k_cur", 0) <= 2)

                ev = RESULTS / f"eval-{task}{tag}.json"
                if not ev.exists():
                    sh([PY, str(HERE / "evaluate.py"), "--task", task, "--tag", tag,
                        "--n", str(args.n), "--width", str(args.width), "--scramble", "30"],
                       HERE / f"log_p3_eval_{mv}_{slug}_s{seed}.txt")
                if ev.exists():
                    e = json.loads(ev.read_text())
                    r["solve_rate"], r["mean_len"] = e.get("solve_rate"), e.get("mean_len")

                pr = RESULTS / f"profile-{task}{tag}.json"
                if not pr.exists():
                    sh([PY, str(HERE / "profiles.py"), "--task", task, "--tag", tag],
                       HERE / f"log_p3_prof_{mv}_{slug}_s{seed}.txt")
                if pr.exists():
                    p = json.loads(pr.read_text())["heuristics"]["learned"]
                    accs = [x["acc"] for x in p["profile"]]
                    r["gdrc"] = p["gdrc"]
                    if accs:
                        r["acc_near"], r["acc_far"] = accs[0], accs[-1]
                        r["decay"] = accs[0] - accs[-1]
                        r["acc_mean"] = sum(accs) / len(accs)
                    sds = [x["sd_lo"] for x in p["profile"]]
                    if sds and sds[0] > 0:
                        r["spread_near"], r["spread_far"] = sds[0], sds[-1]

                bu = RESULTS / f"budget-{task}{tag}.json"
                if not bu.exists():
                    sh([PY, str(HERE / "budget_curve.py"), "--task", task, "--tag", tag,
                        "--n", str(args.budget_n), "--scramble", "30"],
                       HERE / f"log_p3_budget_{mv}_{slug}_s{seed}.txt")
                if bu.exists():
                    b = json.loads(bu.read_text())
                    r["w50"], r["w90"] = b.get("w50"), b.get("w90")

                rows.append(r)
                print(f"{head}  k_cur {str(r.get('k_cur')):>3}  "
                      f"solved {(r.get('solve_rate') or 0)*100:5.1f}%  "
                      f"tau {r.get('gdrc', float('nan')):+.3f}  "
                      f"acc {r.get('acc_mean', float('nan')):.3f}  "
                      f"decay {r.get('decay', float('nan')):+.3f}  "
                      f"w90 {r.get('w90') if r.get('w90') is not None else 'n/a'}",
                      flush=True)

                out = RESULTS / args.out
                out.write_text(json.dumps(
                    {"steps": args.steps, "seeds": args.seeds, "lam": args.lam,
                     "protocol": PROTOCOL, "configs": configs, "arms": arms,
                     "elapsed_hours": (time.time() - t0) / 3600, "runs": rows},
                    indent=2) + "\n")

    print(f"\n{len(rows)} runs -> {args.out}  ({(time.time()-t0)/3600:.1f}h)")


if __name__ == "__main__":
    main()
