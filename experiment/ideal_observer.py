"""
Ideal-observer ceiling: how well can the nominal condition label be
decoded from the stimulus' *realized* statistics alone?

Probes are trained on nominal labels (condition mean / SD), but each
display's realized mean and SD scatter around them — far more at small N.
A model that encoded the realized statistic perfectly would still score
below 1, and its accuracy would rise with N. Compare probe accuracy to
this ceiling before reading a set-size effect as "averaging".

Rule (within each fixed nuisance level, as in probe.py):
  mean: nearest condition mean to the realized mean (circular / log).
  SD:   nearest class-median realized log SD (medians from the first half of
        the simulated reps, scored on the second half: realized SD is biased
        low at small N, so nominal SDs would understate the ceiling).

Usage:
    python3 ideal_observer.py ori               # all set sizes, 200 reps
    python3 ideal_observer.py size --n_reps 500
"""

import argparse

import numpy as np

from stimuli.generate import GENERATORS, SET_SIZES


def nearest(x, centers, period=None):
    d = x[:, None] - np.asarray(centers, float)[None]
    if period:
        d = (d + period / 2) % period - period / 2
    return np.abs(d).argmin(1)


def ceiling(gen, n, n_reps):
    """(mean accuracy, SD accuracy) of the nearest-condition rule at set size n."""
    P = gen.PERIOD
    to_space = (lambda m: m) if P else (lambda m: m / 1000)     # size codes → ln
    means = [to_space(m) for m in gen.MEANS]
    sds = gen.SDS
    stats = {(i, j): np.array([gen.stats(gen.values(gen.trial(n, m, s, r)))
                               for r in range(n_reps)])
             for i, m in enumerate(gen.MEANS) for j, s in enumerate(gen.SDS)}
    half = n_reps // 2
    hit_m, hit_s = [], []
    for (i, j), st in stats.items():
        hit_m += list(nearest(st[:, 0], means, P) == i)
        if n > 1:
            # class medians of realized log SD at this fixed mean (first half)
            med = [np.median(np.log(np.maximum(stats[(i, k)][:half, 1], 1e-9)))
                   for k in range(len(sds))]
            hit_s += list(nearest(np.log(np.maximum(st[half:, 1], 1e-9)), med) == j)
    return np.mean(hit_m), (np.mean(hit_s) if hit_s else np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("domain", choices=GENERATORS)
    p.add_argument("--n_reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345,
                   help="use a seed different from the probed stimuli")
    p.add_argument("--set_sizes", default=None)
    a = p.parse_args()

    gen = GENERATORS[a.domain](a.seed)
    sizes = [int(s) for s in a.set_sizes.split(",")] if a.set_sizes else SET_SIZES
    print(f"ideal observer ({gen.NAME}, {a.n_reps} reps/condition)")
    print(f"{'N':>4}  {'mean':>6}  {'SD':>6}   (chance {1/len(gen.MEANS):.3f} / {1/len(gen.SDS):.3f})")
    for n in sizes:
        m, s = ceiling(gen, n, a.n_reps)
        print(f"{n:>4}  {m:6.3f}  {s:6.3f}")


if __name__ == "__main__":
    main()
