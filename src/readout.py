"""
Shared continuous-readout machinery for the discount analyses.

A "readout" is a closed-form ridge regression from features to a display
statistic that may live on a circle (orientation 180°, hue 360°) or a line
(ln size). Angles are encoded as (cos, sin) so the regression is linear;
predictions are decoded back and compared with wrapped differences.

Used by discount.py (β of the outlier) and discount_probe.py (local vs
global decodability). probe.py's classification probes and per-cell
regression have their own machinery.
"""

import numpy as np

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
