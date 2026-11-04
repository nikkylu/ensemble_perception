"""
Outlier discounting: is the outlier left out of the represented mean?

After Avci & Boduroglu (2021). A continuous mean readout is trained on the
STANDARD stimuli (same domain and set size) and applied to the DISCOUNT
stimuli, whose displays hold N-1 group elements plus one outlier. Its
prediction is compared with the local mean (group only) and the global mean
(all elements, the outlier unwrapped on its side — the paper's arithmetic mean):

    β = Σ (pred − local)(global − local) / Σ (global − local)²

    β = 0  outlier discounted (prediction tracks the group mean)
    β = 1  outlier weighted like any other element
    β > 1  outlier over-weighted

reported separately for less / more distinct outliers, per slice and layer.

Baseline β_circ: the β a representation would show if it simply averaged
the elements on the circle (circular global mean). For orientation a 90°
outlier is opposite in doubled-angle space, so circular averaging barely
moves the mean (β_circ ≈ 0.1) — β ≈ 0 for "more distinct" is then expected
without any discounting. Compare β with β_circ, not with 0. (Size is linear:
β_circ = 1.)

Readout: ridge regression on features standardized with the training data,
onto the realized mean of each standard display — (cos, sin) of the angle for
orientation / hue, ln size for size. λ is chosen by 5-fold CV on the standard
set (error reported as "cv err": circular degrees, or ln units for size).

Slices: cls, bar (all element tokens incl. the outlier), bg, all_patches, and
bar_group — the mean of the group tokens only (outlier token removed): if
tokens carried only their own element, bar_group should give β ≈ 0.

Usage:
    python3 discount.py --train features/ori_standard_n8_s42.pt \\
                        --test features/ori_discount_spatial_n8_s42.pt --save res.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from loader import token_rows

N_LAYERS = 12
SLICES = ["cls", "bar", "bg", "all_patches"]
LAMBDAS = np.logspace(-2, 4, 13)


# ─────────────────────────────────────────────────────────────────
# statistic space: circular (period P, in degrees) or linear (P = 0)
# ─────────────────────────────────────────────────────────────────

def encode(m, P):
    """Mean value → regression target (cos, sin) on the circle, or itself."""
    if not P:
        return m[:, None]
    a = 2 * np.pi * m / P
    return np.stack([np.cos(a), np.sin(a)], 1)


def decode(z, P):
    return (np.arctan2(z[:, 1], z[:, 0]) * P / (2 * np.pi)) % P if P else z[:, 0]


def diff(a, b, P):
    """Signed a − b (wrapped to ±P/2 on a circle)."""
    d = a - b
    return (d + P / 2) % P - P / 2 if P else d


# ─────────────────────────────────────────────────────────────────
# ridge readout with λ by cross-validation (SVD: all λ at once)
# ─────────────────────────────────────────────────────────────────

def standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def ridge_all(X, Y, lambdas):
    """Ridge weights for every λ (X, Y centered). → (Q, D, T)"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    return np.stack([Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtY) for lam in lambdas])


def fit_readout(X, m, P, n_folds=5, seed=0, groups=None):
    """Ridge from features to the encoded mean; λ by k-fold CV error.
    groups: rows sharing an id (tokens of one display) stay in one fold.
    Returns (predict function, CV error at the chosen λ, chosen λ)."""
    Y = encode(m, P)
    if groups is None:
        groups = np.arange(len(X))
    uniq, inv = np.unique(groups, return_inverse=True)
    folds = (np.random.default_rng(seed).permutation(len(uniq)) % n_folds)[inv]
    err = np.zeros(len(LAMBDAS))
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        Xtr, mu, sd = standardize(X[tr])
        ym = Y[tr].mean(0)
        W = ridge_all(Xtr, Y[tr] - ym, LAMBDAS)
        for q in range(len(LAMBDAS)):
            pred = decode(standardize(X[te], mu, sd)[0] @ W[q] + ym, P)
            err[q] += np.abs(diff(pred, m[te], P)).sum()
    q = int(err.argmin())
    Xs, mu, sd = standardize(X)
    ym = Y.mean(0)
    W = ridge_all(Xs, Y - ym, LAMBDAS[q:q + 1])[0]
    return (lambda Z: decode(standardize(Z, mu, sd)[0] @ W + ym, P),
            err[q] / len(X), float(LAMBDAS[q]))


# ─────────────────────────────────────────────────────────────────
# analysis
# ─────────────────────────────────────────────────────────────────

def slices(feats, n):
    """{slice: (trials, L, D) float array}; bar_group drops the outlier
    token, bar_outlier is the outlier's token alone (readout trained on
    standard single tokens)."""
    out = {st: feats[st].float().numpy() for st in SLICES}
    if "bar_concat" in feats and "outlier_slot" in feats:
        tok = feats["bar_concat"].float().reshape(len(feats["n_bars"]), N_LAYERS, n, -1)
        keep = torch.ones(tok.shape[0], n, dtype=torch.bool)
        keep[torch.arange(tok.shape[0]), feats["outlier_slot"]] = False
        out["bar_group"] = (tok * keep[:, None, :, None]).sum(2).div(n - 1).numpy()
        out["bar_outlier"] = tok[torch.arange(tok.shape[0]), :,
                                 feats["outlier_slot"]].numpy()
    return out


def realized_mean(feats):
    """Realized mean of each standard display (falls back to the nominal mean)."""
    if "realized_mean" in feats:                       # deg, or ln px for size
        return feats["realized_mean"].numpy().astype(float)
    m = np.asarray(feats["mean_values"], float)[feats["labels_mean"].numpy()]
    return m if feats.get("period", 180) else np.log(m)   # size: nominal px → ln px


def analyse(train, test):
    P = float(train.get("period", 180))
    n = int(test["n_bars"][0])
    m_train = realized_mean(train)
    local = test["local_mean"].numpy().astype(float)
    glob = test["global_mean_linear"].numpy().astype(float)
    circ = test["global_mean"].numpy().astype(float)
    distinct = test["outlier_distinct"].numpy()
    X_train = slices(train, int(train["n_bars"][0]))
    X_test = slices(test, n)
    # bar_outlier is a single token: its readout trains on standard single tokens
    X_single = m_single = g_single = None
    if "bar_outlier" in X_test and "bar_concat" in train:
        rows_tok, k = token_rows(train["bar_concat"].float(), int(train["n_bars"][0]))
        X_single = rows_tok.numpy()
        m_single = np.repeat(m_train, k)
        g_single = np.repeat(np.arange(len(m_train)), k)   # fold by display

    res = {}
    for st in X_test:
        if st == "bar_outlier" and X_single is not None:
            X_tr, m_tr, g_tr = X_single, m_single, g_single
        else:
            # bar_group has no standard counterpart: use the readout trained on `bar`
            X_tr, m_tr, g_tr = X_train[st if st in X_train else "bar"], m_train, None
        rows = []
        for layer in range(N_LAYERS):
            predict, cv_err, lam = fit_readout(X_tr[:, layer], m_tr, P, groups=g_tr)
            pred = predict(X_test[st][:, layer])
            row = {"cv_err": cv_err, "lambda": lam}
            for k, name in enumerate(["less", "more"]):
                sel = distinct == k
                p_off = diff(pred[sel], local[sel], P)
                g_off = diff(glob[sel], local[sel], P)
                c_off = diff(circ[sel], local[sel], P)
                row[name] = {
                    "beta": float((p_off * g_off).sum() / (g_off ** 2).sum()),
                    "beta_circ": float((c_off * g_off).sum() / (g_off ** 2).sum()),
                    "err_local": float(np.abs(p_off).mean()),
                    "err_global": float(np.abs(diff(pred[sel], glob[sel], P)).mean()),
                    "local_global_gap": float(np.abs(g_off).mean())}
            rows.append(row)
        res[st] = rows
    return res


def print_table(res, unit):
    print(f"\nβ: 0 = outlier discounted, 1 = equal weight; β_circ = circular averaging"
          f"   (cv err in {unit})")
    print(f"{'':12s}{'layer':>6s}{'cv err':>9s}{'β less':>9s}{'β more':>9s}"
          f"{'|p-loc| more':>14s}{'|p-glob| more':>15s}")
    for st, rows in res.items():
        for layer, r in enumerate(rows):
            print(f"{st:12s}{layer:>6d}{r['cv_err']:>9.3g}{r['less']['beta']:>9.2f}"
                  f"{r['more']['beta']:>9.2f}{r['more']['err_local']:>14.3g}"
                  f"{r['more']['err_global']:>15.3g}")
        print()
    r0 = next(iter(res.values()))[0]
    print(f"circular-averaging baseline: β_circ less {r0['less']['beta_circ']:+.2f}, "
          f"more {r0['more']['beta_circ']:+.2f}")
    # summary at each slice's most accurate layer on the standard stimuli
    print("at the layer with the lowest standard cv err:")
    for st, rows in res.items():
        b = int(np.argmin([r["cv_err"] for r in rows]))
        print(f"  {st:12s} L{b:02d}  cv err {rows[b]['cv_err']:.3g}   "
              f"β less {rows[b]['less']['beta']:+.2f}   β more {rows[b]['more']['beta']:+.2f}   "
              f"(local–global gap: less {rows[b]['less']['local_global_gap']:.3g}, "
              f"more {rows[b]['more']['local_global_gap']:.3g})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="standard-stimuli features (.pt)")
    p.add_argument("--test", required=True, help="discount-stimuli features (.pt)")
    p.add_argument("--save", default=None, help="JSON output")
    a = p.parse_args()

    train = torch.load(a.train, map_location="cpu")
    test = torch.load(a.test, map_location="cpu")
    assert "local_mean" in test, f"{a.test} is not a discount feature file"
    assert int(train["n_bars"][0]) == int(test["n_bars"][0]), "set sizes differ"

    res = analyse(train, test)
    print_table(res, "ln px" if not train.get("period", 180) else "deg")
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps({"train": a.train, "test": a.test, "results": res}))
        print(f"saved → {a.save}")


if __name__ == "__main__":
    main()
