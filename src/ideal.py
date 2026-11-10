"""
Task reference levels for the standard probes: chance, ceiling, and the
single-element ceiling — computed exactly on the same stimuli (generators are
pure functions of the seed — no rendering, no feature files).

Protocol matches probe.py: labels = level nearest each display's REALIZED
statistic; the nuisance statistic is held fixed per level and accuracies are
averaged over the nuisance levels.

    chance    majority-class rate after relabeling — what a probe that learned
              nothing scores. Not 1/k: relabeling unbalances the classes at
              small N (e.g. SD at N=2: one class holds ~70% of displays).
    ceiling   an observer that reads every element perfectly and computes the
              display statistic. Equals 1.0 by construction (the label IS the
              realized statistic's nearest level); reported as a check that
              this script's labels match the probe's.
    single    the same observer allowed ONE element (the per_token probe's
              ceiling): mean — the level nearest that element; SD — the MAP
              level argmax_σ N(d; 0, σ) from its deviation d (mean known).
              per_token above it ⇒ tokens carry other elements' information.

Usage:
    python3 ideal.py ori --seed 42 --n_reps 200 --save ideal_ori.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from stimuli.generate import GENERATORS


def wrap(d, period):
    return (d + period / 2) % period - period / 2 if period else d


def reference(gen, n, n_reps):
    """→ {target: {"chance", "ceiling", "single"}} at set size n."""
    period = gen.PERIOD
    mlev = np.array([gen.stat_mean(m) for m in gen.MEANS])
    slev = np.array(gen.SDS, float) / (1000.0 if not period else 1.0)
    targets = ["mean"] + (["variance"] if n >= 2 else [])
    # per nuisance level: class counts, ceiling hits, single-element hits
    acc = {t: {} for t in targets}

    for mi, mean in enumerate(gen.MEANS):
        for si, sd in enumerate(gen.SDS):
            for rep in range(n_reps):
                v = gen.values(gen.trial(n, mean, sd, rep))
                rm, rs = gen.stats(v)
                for t in targets:
                    if t == "mean":
                        nuis, levels = si, mlev
                        lab = np.abs(wrap(mlev - rm, period)).argmin()
                        full = np.abs(wrap(mlev - rm, period)).argmin()
                        one = np.abs(wrap(mlev[None] - v[:, None], period)).argmin(1)
                    else:
                        nuis, levels = mi, slev
                        lab = np.abs(slev - rs).argmin()
                        full = np.abs(slev - rs).argmin()
                        d = wrap(v - gen.stat_mean(mean), period)
                        one = (-0.5 * (d[:, None] / slev[None]) ** 2
                               - np.log(slev)[None]).argmax(1)
                    a = acc[t].setdefault(nuis, {"counts": np.zeros(len(levels)),
                                                 "full": 0, "one": 0, "n": 0, "m": 0})
                    a["counts"][lab] += 1
                    a["full"] += int(full == lab)
                    a["n"] += 1
                    a["one"] += int((one == lab).sum())
                    a["m"] += len(v)

    out = {}
    for t, per in acc.items():
        out[t] = {"chance": float(np.mean([a["counts"].max() / a["n"] for a in per.values()])),
                  "ceiling": float(np.mean([a["full"] / a["n"] for a in per.values()])),
                  "single": float(np.mean([a["one"] / a["m"] for a in per.values()]))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("domain", choices=list(GENERATORS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_reps", type=int, default=200)
    p.add_argument("--set_sizes", default="1,2,3,4,8,16,32")
    p.add_argument("--save", default=None)
    a = p.parse_args()

    gen = GENERATORS[a.domain](a.seed)
    res = {}
    print(f"{a.domain} {gen.VERSION} seed={a.seed} reps={a.n_reps} — task reference levels")
    print(f"{'N':>4s}  {'mean: chance ceiling single':>28s}  {'SD: chance ceiling single':>26s}")
    for n in [int(x) for x in a.set_sizes.split(",")]:
        res[n] = reference(gen, n, a.n_reps)
        row = lambda t: ("{chance:7.3f} {ceiling:7.3f} {single:6.3f}".format(**res[n][t])
                         if t in res[n] else f"{'-':>7s} {'-':>7s} {'-':>6s}")
        print(f"{n:>4d}  {row('mean'):>28s}  {row('variance'):>26s}")
    if a.save:
        Path(a.save).write_text(json.dumps(
            {"domain": a.domain, "version": gen.VERSION, "seed": a.seed,
             "n_reps": a.n_reps, "levels": res}))
        print(f"saved → {a.save}")


if __name__ == "__main__":
    main()
