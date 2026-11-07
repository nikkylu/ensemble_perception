"""
Single-element ideal-observer ceilings for the per_token probe.

per_token classifies the DISPLAY's statistic from ONE element token, so its
ceiling is not 1.0: a single element is a noisy sample of the display's
distribution. This script computes that ceiling exactly on the same stimuli
(generators are pure functions of the seed — no rendering, no feature files):

    mean      read the element's value perfectly, predict the level nearest
              to it; score against the relabeled display label (level nearest
              the realized mean) — matching probe.py's protocol.
    variance  the nuisance mean level is fixed (and known) under the probing
              protocol; from the element's deviation d, take the MAP level
              argmax_σ N(d; 0, σ) — again scored against relabeled labels.

per_token above this ceiling ⇒ the token carries information about OTHER
elements (token mixing); per_token below ⇒ imperfect encoding of its own
element. Ceilings are averaged over elements, trials, and nuisance levels,
exactly like the probe's accuracy.

Usage:
    python3 ideal_single.py ori --seed 42 --n_reps 200 --save ceilings_ori.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from stimuli.generate import GENERATORS


def wrap(d, period):
    return (d + period / 2) % period - period / 2 if period else d


def ceilings(gen, n, n_reps):
    """→ {"mean": acc, "variance": acc} single-element ceilings at set size n."""
    period = gen.PERIOD
    mlev = np.array([gen.stat_mean(m) for m in gen.MEANS])
    slev = np.array(gen.SDS, float) / (1000.0 if not period else 1.0)
    hit_m, tot_m, hit_s, tot_s = 0, 0, 0, 0
    for mean in gen.MEANS:
        for sd in gen.SDS:
            for rep in range(n_reps):
                v = gen.values(gen.trial(n, mean, sd, rep))
                rm, rs = gen.stats(v)
                # mean: nearest level to each element vs nearest to realized mean
                lab = np.abs(wrap(mlev - rm, period)).argmin()
                pred = np.abs(wrap(mlev[None] - v[:, None], period)).argmin(1)
                hit_m += int((pred == lab).sum())
                tot_m += len(v)
                if n < 2:
                    continue
                # variance: MAP over SD levels from one deviation, mean known
                lab = np.abs(slev - rs).argmin()
                d = wrap(v - gen.stat_mean(mean), period)
                ll = (-0.5 * (d[:, None] / slev[None]) ** 2
                      - np.log(slev)[None])                    # N(0, σ) log-lik
                hit_s += int((ll.argmax(1) == lab).sum())
                tot_s += len(v)
    out = {"mean": hit_m / tot_m}
    if tot_s:
        out["variance"] = hit_s / tot_s
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
    print(f"{a.domain} seed={a.seed} reps={a.n_reps} — single-element ideal ceiling")
    print(f"{'N':>4s} {'mean':>7s} {'variance':>9s}")
    for n in [int(x) for x in a.set_sizes.split(",")]:
        res[n] = ceilings(gen, n, a.n_reps)
        print(f"{n:>4d} {res[n]['mean']:>7.3f} "
              f"{res[n].get('variance', float('nan')):>9.3f}")
    if a.save:
        Path(a.save).write_text(json.dumps(
            {"domain": a.domain, "seed": a.seed, "n_reps": a.n_reps,
             "ceiling": res}))
        print(f"saved → {a.save}")


if __name__ == "__main__":
    main()
