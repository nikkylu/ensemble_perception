"""
Local vs global mean decodability on discount displays.

Complements discount.py's β (what a standard-trained readout spontaneously
reports). Here ridge readouts are trained ON the discount displays themselves,
one per target, and compared:

    local        realized mean of the group (outlier excluded)
    global       arithmetic mean of all elements, outlier unwrapped on its
                 side (the paper's global mean = discount.py's global_mean_linear)
    global_circ  circular mean of all elements (what plain circular averaging
                 of the display would produce; for size: equals global)

Protocol: nested stratified 5-fold CV within the discount set (strata =
condition mean × outlier distinctness). λ is chosen per target by an inner
5-fold CV on each outer training set alone; the outer test folds are predicted
exactly once, so the reported errors are untouched by the λ selection. Errors
are reported separately for less / more distinct outliers, per slice and layer.

Reading the comparison (per slice/layer, "more" distinct is the sensitive case):
    err(global) ≈ err(local)      the outlier's value is linearly available —
                                  a probe can integrate it on demand
    err(global) ≈ err(local) + gap, and the global-trained probe's predictions
    stay nearer the local mean    the representation carries the group mean
                                  only; the outlier is excluded
Δ(g−l) is the paired per-trial |err| difference with a trial-bootstrap 95% CI;
`gap` = mean |global − local| is the ceiling Δ can reach.

Slices: cls, bar, bg, all_patches, and bar_group (mean of the group element
tokens, outlier token removed) — bar_group has no outlier information, so its
global readout is pinned at err ≈ local err + gap: a floor to compare against.

Usage:
    python3 discount_probe.py --features features/ori_discount_spatial_n8_s42.pt \\
                              --save res.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from discount import (encode, decode, diff, standardize, ridge_all, slices,
                      LAMBDAS, N_LAYERS)

TARGETS = ["local", "global", "global_circ"]
N_FOLDS = 5
N_BOOT = 1000


def stratified_folds(strata, n_folds, seed):
    folds = np.empty(len(strata), int)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(strata)), strata)):
        folds[te] = f
    return folds


def cv_readout(X, M, P, strata, seed=0):
    """Nested-CV held-out ridge predictions of every target column of M.
    Outer stratified 5-fold. Per outer fold, λ (per target) is chosen by an
    inner stratified 5-fold CV on the outer TRAINING set alone; the readout is
    refit on the full outer training set at that λ, and the outer test fold is
    predicted exactly once — its error never influences the λ choice. One SVD
    per (outer, inner) fold serves every target and λ.
    Returns (preds (N, T), cv_err (T,), chosen λ (F, T))."""
    N, T = M.shape
    Y = np.concatenate([encode(M[:, t], P) for t in range(T)], 1)
    C = Y.shape[1] // T
    folds = stratified_folds(strata, N_FOLDS, seed)
    preds = np.empty((N, T))
    lams = np.empty((N_FOLDS, T))
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        Xtr, Ytr, Mtr = X[tr], Y[tr], M[tr]
        inner = stratified_folds(strata[tr], N_FOLDS, seed + 1 + f)
        err = np.zeros((len(LAMBDAS), T))
        for g in range(N_FOLDS):
            itr, ite = inner != g, inner == g
            Xi, mu, sd = standardize(Xtr[itr])
            ym = Ytr[itr].mean(0)
            W = ridge_all(Xi, Ytr[itr] - ym, LAMBDAS)       # (Q, D, T·C)
            Z = standardize(Xtr[ite], mu, sd)[0]
            for q in range(len(LAMBDAS)):
                out = Z @ W[q] + ym
                for t in range(T):
                    err[q, t] += np.abs(diff(decode(out[:, t * C:(t + 1) * C], P),
                                             Mtr[ite, t], P)).sum()
        qs = err.argmin(0)                                  # λ per target, this fold
        Xs, mu, sd = standardize(Xtr)
        ym = Ytr.mean(0)
        W = ridge_all(Xs, Ytr - ym, LAMBDAS)
        Z = standardize(X[te], mu, sd)[0]
        for t in range(T):
            out = Z @ W[qs[t]] + ym
            preds[te, t] = decode(out[:, t * C:(t + 1) * C], P)
        lams[f] = LAMBDAS[qs]
    cv_err = np.array([np.abs(diff(preds[:, t], M[:, t], P)).mean()
                       for t in range(T)])
    return preds, cv_err, lams


def bootstrap_mean_ci(d, n_boot=N_BOOT, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    return [float(q) for q in np.percentile(d[idx].mean(1), [2.5, 97.5])]


def analyse(feats, seed=0):
    P = float(feats.get("period", 180))
    n = int(feats["n_bars"][0])
    M = np.stack([feats["local_mean"].numpy(),
                  feats["global_mean_linear"].numpy(),
                  feats["global_mean"].numpy()], 1).astype(float)
    distinct = feats["outlier_distinct"].numpy()
    strata = feats["labels_mean"].numpy() * 2 + distinct
    X = slices(feats, n)
    X.pop("bar_outlier", None)      # single-token slice: not a mean readout

    res = {}
    for st, Xs in X.items():
        rows = []
        for layer in range(N_LAYERS):
            preds, cv_err, lams = cv_readout(Xs[:, layer].astype(np.float64),
                                             M, P, strata, seed=seed)
            row = {"lambda": {t: [float(l) for l in lams[:, i]]
                              for i, t in enumerate(TARGETS)},
                   "cv_err": {t: float(e) for t, e in zip(TARGETS, cv_err)}}
            for k, name in enumerate(["less", "more"]):
                sel = distinct == k
                e = {t: np.abs(diff(preds[sel, i], M[sel, i], P))
                     for i, t in enumerate(TARGETS)}
                d = e["global"] - e["local"]
                row[name] = {
                    "err": {t: float(v.mean()) for t, v in e.items()},
                    "global_minus_local": float(d.mean()),
                    "global_minus_local_ci": bootstrap_mean_ci(d),
                    # where the global-trained probe's predictions actually sit
                    "global_probe_err_vs_local":
                        float(np.abs(diff(preds[sel, 1], M[sel, 0], P)).mean()),
                    "local_probe_err_vs_global":
                        float(np.abs(diff(preds[sel, 0], M[sel, 1], P)).mean()),
                    "gap": float(np.abs(diff(M[sel, 1], M[sel, 0], P)).mean())}
            rows.append(row)
        res[st] = rows
    return res


def print_table(res, unit):
    print(f"\nlocal vs global mean readouts trained on discount displays "
          f"(held-out err in {unit})")
    print(f"{'':12s}{'layer':>6s}{'loc less':>10s}{'loc more':>10s}"
          f"{'glob less':>10s}{'glob more':>10s}{'Δ(g−l) more':>13s}{'gap more':>10s}")
    for st, rows in res.items():
        for layer, r in enumerate(rows):
            print(f"{st:12s}{layer:>6d}"
                  f"{r['less']['err']['local']:>10.3g}{r['more']['err']['local']:>10.3g}"
                  f"{r['less']['err']['global']:>10.3g}{r['more']['err']['global']:>10.3g}"
                  f"{r['more']['global_minus_local']:>13.3g}"
                  f"{r['more']['gap']:>10.3g}")
        print()
    print("at the layer with the lowest local err "
          "(Δ = err(global) − err(local), paired trial-bootstrap 95% CI;\n"
          " Δ ≈ 0: outlier integrable on demand; Δ ≈ gap: group mean only. "
          "glob-probe→loc/glob: where the global-trained probe's predictions sit):")
    for st, rows in res.items():
        b = int(np.argmin([r["cv_err"]["local"] for r in rows]))
        parts = [f"  {st:12s} L{b:02d} "]
        for d in ["less", "more"]:
            r = rows[b][d]
            lo, hi = r["global_minus_local_ci"]
            parts.append(
                f" {d}: Δ {r['global_minus_local']:+.3g} [{lo:+.3g},{hi:+.3g}]"
                f" gap {r['gap']:.3g}"
                f" glob-probe→loc/glob {r['global_probe_err_vs_local']:.3g}"
                f"/{r['err']['global']:.3g} ")
        print("".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="discount-stimuli features (.pt)")
    p.add_argument("--save", default=None, help="JSON output")
    p.add_argument("--seed", type=int, default=0, help="CV fold seed")
    a = p.parse_args()

    feats = torch.load(a.features, map_location="cpu")
    assert "local_mean" in feats, f"{a.features} is not a discount feature file"

    res = analyse(feats, seed=a.seed)
    print_table(res, "ln px" if not feats.get("period", 180) else "deg")
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps({"features": a.features, "results": res}))
        print(f"saved → {a.save}")


if __name__ == "__main__":
    main()
