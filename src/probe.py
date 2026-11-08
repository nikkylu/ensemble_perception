"""
Linear probes for ensemble statistics in frozen DINOv2 features.

Decodes the condition mean (6-way) or SD (3-way) of each display from
every representation ("slice") at every layer:

    cls, bar, bg, all_patches   768-d pooled tokens (from loader.py)
    pairwise_diff               2·Var over element tokens (N ≥ 2)
    bar_concat                  element tokens concatenated (N·768-d; multi-patch
                                cells are averaged per element first, so the
                                dimensionality matches across domains)
    bg_concat        (opt-in)   background tokens concatenated (~190k-d)
    per_token / token_vote      (--per_token; pipeline.sh passes it) one 768-d
                                probe shared by all element tokens, scored per
                                token / by averaging an image's token logits —
                                NOTE: token_vote with a linear probe is
                                mathematically ≈ the pooled `bar` slice (mean
                                of logits = logit of the mean); report
                                per_token, don't sell token_vote as evidence
                                beyond pooling

Protocol
- The nuisance statistic is held fixed: probing the mean, each SD level is
  probed separately (and vice versa); accuracies are averaged over levels,
  and the per-level table is printed for the supplement.
- Outer stratified 5-fold CV scores the probes. The L2 strength λ is chosen
  by nested CV inside each outer training set (one λ per slice, from the
  inner accuracy at the middle layer; --lambda_select layer for per-layer),
  then the probe is refit on the full outer training set.
- Features are standardized per fold on training data.
- "best" = max over layers (layer chosen on the test folds: optimistic);
  "cv-best" = layer chosen on the other outer folds (unbiased).

Labels: classification relabels each trial to the level nearest its
REALIZED statistic (needs realized_mean / realized_sd in the features
file); nuisance levels stay defined by the condition labels.

Regression (--target mean_reg / variance_reg): closed-form ridge reads the
realized statistic WITHIN each (mean, SD) cell (λ by LOOCV); predictions
pool across cells; reported per (slice, layer): Pearson r (in the "acc"
slot) and MAE. Runs on REG_SLICES (cls, bar, and — for the supplement's
element-resolved readout, closed-form so free of the optimizer caveats that
retired the logistic concat probe from the main figures — bar_concat).

Efficiency: every (slice, layer, fold, λ) probe is an independent logistic
regression; all of them are solved together in one vectorized full-batch
L-BFGS run (strong-Wolfe line search, zero init). Per-probe losses are
summed; the problems stay independent because the gradient is
block-separable. Adam at a fixed epoch budget was retired after an
L-BFGS comparison on the cluster showed it under-reports pooled slices
by 5-14 points on size - and an under-converged probe is not comparable
across feature scales (pretrained vs @emb / @random controls).

Usage:
    python3 probe.py --features feats.pt --target mean --n_bars 16
    python3 probe.py --features feats.pt --target variance --n_bars 16 --save res.json
    python3 probe.py --features feats.pt --target mean_reg --n_bars 16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from loader import element_tokens

N_LAYERS = 12
POOLED = ["cls", "bar", "bg", "all_patches"]
SLICE_ORDER = ["cls", "bar", "bg", "all_patches", "pairwise_diff", "bar_concat",
               "bg_concat", "per_token", "token_vote"]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────
# VECTORIZED PROBES — independent linear probes per (layer, split, λ)
# ─────────────────────────────────────────────────────────────────

class VectorizedLinearProbes(nn.Module):
    """L×S×Q independent linear probes ŷ = W h + b (layers × splits × λ).

    Per-split standardization is folded into the weights analytically
    ((h-mu)/sd @ W + b == h @ (W/sd) + (b - mu·W/sd)), so standardized
    copies of the features are never materialized.
    """

    def __init__(self, n_layers, n_splits, n_lambdas, dim, n_classes):
        super().__init__()
        shape = (n_layers, n_splits, n_lambdas)
        # zero init: the CE objective is convex, so the solution is unique and
        # the whole fit is deterministic
        self.W = nn.Parameter(torch.zeros(*shape, dim, n_classes))
        self.b = nn.Parameter(torch.zeros(*shape, n_classes))

    def forward(self, H, mu, sd):
        """H (N, L, D), mu/sd (L, S, D) → logits (N, L, S, Q, C)"""
        V = self.W / sd[:, :, None, :, None]
        return (torch.einsum("nld,lsqdc->nlsqc", H, V)
                + self.b - torch.einsum("lsd,lsqdc->lsqc", mu, V))


def split_statistics(H, train):
    """Mean / SD of each (layer, split) over its training rows. → (L, S, D) each."""
    c = train.sum(0).view(1, -1, 1)
    mu = torch.einsum("ns,nld->lsd", train, H) / c
    var = (torch.einsum("ns,nld->lsd", train, H * H) - c * mu ** 2) / (c - 1).clamp(min=1)
    return mu, var.clamp(min=0).sqrt() + 1e-6


def fit_predict(H, y, train, n_classes, lam, max_iter=1500, device="cpu",
                max_params=4e7, tag="fit"):
    """Solve one probe per (layer, split, λ) to convergence; logits for all rows.

    H (N, L, D), y (N,), train (N, S) 0/1 mask, lam broadcastable to (L, S, Q).
    Each probe minimizes mean CE over its training rows + ½λ‖W‖² (bias free),
    by full-batch L-BFGS from zero weights (strong-Wolfe line search;
    tolerance_change=0 so the fit never stops early on the SUMMED loss —
    only on the gradient test, which bounds every probe, or at max_iter;
    cap equivalence 1500 vs 3000 verified on size mean N=4, within 0.004).
    Each fit prints its iteration count and the largest per-probe gradient
    norm, so a full run documents its own convergence. λ is processed in
    chunks so at most `max_params` weights live at once.
    → logits (N, L, S, Q, C) on CPU.
    """
    H, y, train = H.to(device), y.to(device), train.to(device)
    N, L, D = H.shape
    S = train.shape[1]
    lam = torch.as_tensor(lam, dtype=torch.float32)
    lam = lam.expand(L, S, lam.shape[-1]).to(device)
    mu, sd = split_statistics(H, train)
    n_train = train.sum(0).view(1, S, 1)

    chunk = max(1, int(max_params // (L * S * D * n_classes)))
    out = []
    for q in range(0, lam.shape[-1], chunk):
        lq = lam[:, :, q:q + chunk]
        model = VectorizedLinearProbes(L, S, lq.shape[-1], D, n_classes).to(device)
        opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                                history_size=20, tolerance_change=0,
                                tolerance_grad=1e-5, line_search_fn="strong_wolfe")
        target = y.view(N, 1, 1, 1, 1).expand(N, L, S, lq.shape[-1], 1)

        def closure():
            opt.zero_grad()
            ce = -model(H, mu, sd).log_softmax(-1).gather(-1, target).squeeze(-1)
            data = (ce * train.view(N, 1, S, 1)).sum(0) / n_train       # (L, S, Q)
            loss = data.sum() + 0.5 * (lq * model.W.pow(2).sum((-2, -1))).sum()
            loss.backward()
            return loss

        opt.step(closure)
        closure()                                   # grads at the final point
        g = (model.W.grad.pow(2).sum((-2, -1))
             + model.b.grad.pow(2).sum(-1)).sqrt()  # per-probe ‖∇‖ (L, S, Q)
        it = opt.state_dict()["state"][0]["n_iter"]
        # a cap-hit while sweeping the λ grid is expected (small-λ probes on
        # near-separable data diverge); a cap-hit in the final fit at the
        # SELECTED λ is a real convergence failure
        flag = (" — WARNING: selected-λ fit unconverged, raise --max_iter"
                if it >= max_iter and tag == "final" else "")
        print(f"  L-BFGS[{tag}]: {it}/{max_iter} iters, max per-probe ‖∇‖ "
              f"{g.max():.1e} over {g.numel()} probes{flag}")
        with torch.no_grad():
            out.append(model(H, mu, sd).cpu())
    return torch.cat(out, dim=3)


def accuracy(logits, y, mask):
    """logits (N, L, S, ..., C), y (N,), mask (N, S) → accuracy (L, S, ...)."""
    N, S = mask.shape
    correct = (logits.argmax(-1) == y.view(N, *[1] * (logits.dim() - 2))).float()
    m = mask.view(N, 1, S, *[1] * (logits.dim() - 4))
    return ((correct * m).sum(0) / m.sum(0)).numpy()


# ─────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────

def fold_masks(y, n_folds, seed):
    """Stratified K-fold → (N, F) train / test 0/1 masks."""
    tr, te = torch.zeros(len(y), n_folds), torch.zeros(len(y), n_folds)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (a, b) in enumerate(skf.split(np.zeros(len(y)), y.numpy())):
        tr[a, f], te[b, f] = 1.0, 1.0
    return tr, te


def inner_fold_masks(y, outer_train, k, seed):
    """k stratified folds inside each outer training set → (N, F·k) masks
    (column f·k + j = outer fold f, inner fold j)."""
    N, F = outer_train.shape
    tr, va = torch.zeros(N, F * k), torch.zeros(N, F * k)
    for f in range(F):
        idx = outer_train[:, f].nonzero(as_tuple=True)[0]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + 1 + f)
        for j, (a, b) in enumerate(skf.split(np.zeros(len(idx)), y[idx].numpy())):
            tr[idx[a], f * k + j], va[idx[b], f * k + j] = 1.0, 1.0
    return tr, va


def nested_cv(H, y, n_classes, n_slices=1, rows_per_image=1, lambdas=(1e-2,),
              select="slice", ref_layer=N_LAYERS // 2, n_folds=5, inner_folds=4,
              seed=42, **fit):
    """Nested-CV probes for all layers (of possibly several stacked slices).

    H: (N·r, n_slices·12, D) — r rows per image (tokens of one image adjacent),
       slices of equal width stacked on the layer axis.
    y: (N,) image labels. Folds are always split by image.
    Returns (logits (N·r, L, F, C), test mask (N, F), chosen λ (L, F)).
    """
    r, L = rows_per_image, H.shape[1]
    rows = lambda m: m.repeat_interleave(r, dim=0)
    y_rows = y.repeat_interleave(r)
    lambdas = torch.as_tensor(lambdas, dtype=torch.float32)
    tr, te = fold_masks(y, n_folds, seed)

    if len(lambdas) == 1:
        lam = lambdas.expand(L, n_folds)
    else:
        # inner CV on each outer training set, at one layer per slice (or all)
        k = min(inner_folds, int(np.bincount(y.numpy()).min() * (n_folds - 1) // n_folds))
        itr, iva = inner_fold_masks(y, tr, k, seed)
        per = L // n_slices
        ref = min(ref_layer, per - 1)
        layers = (list(range(L)) if select == "layer"
                  else [s * per + ref for s in range(n_slices)])
        lg = fit_predict(H[:, layers], y_rows, rows(itr), n_classes,
                         lambdas.view(1, 1, -1), tag="λ-select", **fit)
        val = accuracy(lg, y_rows, rows(iva))                         # (·, F·k, Q)
        val = val.reshape(len(layers), n_folds, k, -1).mean(2)        # (·, F, Q)
        best = val.shape[-1] - 1 - val[..., ::-1].argmax(-1)          # ties → larger λ
        lam = lambdas[torch.as_tensor(best)]
        if select == "slice":
            lam = lam.repeat_interleave(per, dim=0)                   # (L, F)

    lg = fit_predict(H, y_rows, rows(tr), n_classes, lam.unsqueeze(-1),
                     tag="final", **fit)
    return lg[:, :, :, 0], te, lam.numpy()


def probe_slices(slices, y, n_classes, **kw):
    """{name: (N, 12, D)} of equal width, trained in ONE vectorized fit.
    → {name: {"acc": (12, F), "lam": (12, F)}}"""
    names = list(slices)
    H = torch.cat([slices[n].float() for n in names], dim=1)
    L = H.shape[1] // len(names)
    lg, te, lam = nested_cv(H, y, n_classes, n_slices=len(names), **kw)
    acc = accuracy(lg, y, te)
    return {n: {"acc": acc[i * L:(i + 1) * L], "lam": lam[i * L:(i + 1) * L]}
            for i, n in enumerate(names)}


def probe_per_token(bar_concat, y, n, n_classes, **kw):
    """One 768-d probe shared by all n element tokens (folds split by image).
    → {"per_token": …, "token_vote": …} with accuracy per token / per image
    after averaging the image's token logits."""
    N, L, total = bar_concat.shape
    tokens = bar_concat.float().reshape(N, L, n, total // n).transpose(1, 2)
    tokens = tokens.reshape(N * n, L, -1)                             # image-major rows
    lg, te, lam = nested_cv(tokens, y, n_classes, rows_per_image=n, **kw)
    return {"per_token": {"acc": accuracy(lg, y.repeat_interleave(n),
                                          te.repeat_interleave(n, 0)), "lam": lam},
            "token_vote": {"acc": accuracy(lg.reshape(N, n, *lg.shape[1:]).mean(1), y, te),
                           "lam": lam}}


# ─────────────────────────────────────────────────────────────────
# ONE SWEEP = all slices at one fixed nuisance level
# ─────────────────────────────────────────────────────────────────

def compute_pairwise_diff(bar_concat, n, d):
    """Per-dim mean squared pairwise difference of element vectors,
    mean_{i<j}(h_i - h_j)² = 2·Var_unbiased(h). (N, L, n·block²·d) → (N, L, d)"""
    return 2 * element_tokens(bar_concat.float(), n, d).var(dim=2, unbiased=True)


def select_indices(features, n_bars=None, fix_mean=None, fix_variance=None):
    """Trial indices for one set size and (optionally) one nuisance level."""
    keep = torch.ones(len(features["labels_mean"]), dtype=torch.bool)
    if n_bars is not None:
        keep &= features["n_bars"] == n_bars
    if fix_mean is not None:
        keep &= features["labels_mean"] == fix_mean
    if fix_variance is not None:
        keep &= features["labels_var"] == fix_variance
    return keep.nonzero(as_tuple=True)[0]


def realized_stat(features, target):
    """Each trial's realized statistic, and the condition levels, in matching
    units. Size means: mean_values are px but realized_mean is ln(px) → ln.
    Discount feature files: the GROUP (local) statistic — realized_mean there
    includes the outlier, which would shift labels toward it."""
    key = "realized_mean" if target == "mean" else "realized_sd"
    local = "local_mean" if target == "mean" else "local_sd"
    if local in features:
        key = local
    assert key in features, (
        f"{key} missing from the features file; re-extract with a loader.py "
        "that saves realized statistics (metadata must have actual_* fields)")
    vals = torch.tensor(features["mean_values" if target == "mean" else "var_values"],
                        dtype=torch.float32)
    if target == "mean" and not features.get("period", 0):
        vals = vals.log()
    return features[key].float(), vals


def realized_level(features, target):
    """Index of the condition level nearest each trial's REALIZED statistic.
    mean: circular distance on `period` (ori/hue), linear for size.
    variance: linear distance (circular SD is a non-negative scalar)."""
    real, vals = realized_stat(features, target)
    d = (real[:, None] - vals[None, :]).abs()
    period = features.get("period", 0)
    if target == "mean" and period:
        d = torch.minimum(d, period - d)
    return d.argmin(dim=1)


def realized_deviation(features, target):
    """Signed deviation of each trial's realized statistic from its condition
    value (wrapped to ±period/2 for circular means)."""
    real, vals = realized_stat(features, target)
    labels = features["labels_mean" if target == "mean" else "labels_var"]
    d = real - vals[labels]
    period = features.get("period", 0)
    if target == "mean" and period:
        d = (d + period / 2) % period - period / 2
    return d


# ─────────────────────────────────────────────────────────────────
# REGRESSION — realized statistic within each condition cell
# ─────────────────────────────────────────────────────────────────

REG_LAMBDAS = torch.logspace(-2, 6, 9, dtype=torch.float64)


def ridge_cell(H, y, n_folds, seed, groups=None):
    """Closed-form dual ridge per (layer, fold): per-fold standardization,
    λ chosen by LOOCV over the training rows (exact incl. the intercept's
    1/m leverage; only the per-row μ/σ refit is neglected). No iterations.
    H (n, L, D), y (n,) → (test mask (n, F) bool, predictions (n, L, F)).
    groups: rows with the same group id (e.g. tokens of one image) always
    land in the same fold."""
    n, L, _ = H.shape
    H, y = H.double(), y.double()
    if groups is None:
        groups = torch.arange(n)
    uniq, inv = torch.unique(torch.as_tensor(groups), return_inverse=True)
    assert len(uniq) >= n_folds, (
        f"{len(uniq)} groups cannot fill {n_folds} folds (raise n_reps "
        "or lower --n_folds)")
    gf = torch.as_tensor(np.random.default_rng(seed).permutation(len(uniq)) % n_folds)
    folds = gf[inv]
    te = torch.zeros(n, n_folds, dtype=torch.bool)
    preds = torch.zeros(n, L, n_folds, dtype=torch.float64)
    for f in range(n_folds):
        tr = folds != f
        te[~tr, f] = True
        mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-6
        Z = (H - mu) / sd
        ym = y[tr].mean()
        yc = y[tr] - ym
        Ztr = Z[tr]                                             # (m, L, D)
        K = torch.einsum("mld,nld->lmn", Ztr, Ztr)              # (L, m, m)
        evals, U = torch.linalg.eigh(K)                         # K = U e Uᵀ
        Uy = torch.einsum("lmn,m->ln", U, yc)                   # (L, m)
        # exact LOO of (center y, standardize, ridge): the unpenalized
        # intercept adds 1/m to every leverage, and both residual and
        # 1 − leverage are built from d = λ/(e+λ) so the near-interpolation
        # limit (small λ, D ≫ m) stays numerically stable. Only the per-row
        # refit of μ/σ is still neglected (O(1/m), <0.5% on the LOO curve).
        d = REG_LAMBDAS[:, None, None] / (evals[None] + REG_LAMBDAS[:, None, None])
        resid = torch.einsum("lmn,qln->qlm", U, d * Uy[None])   # yc − fit
        g = (torch.einsum("lmn,qln->qlm", U ** 2, d)
             - 1.0 / int(tr.sum()))                             # 1 − h − 1/m
        loo = (resid / g.clamp(min=1e-10)) ** 2
        best = loo.mean(-1).argmin(0)                           # (L,)
        alpha = torch.einsum("lmn,ln->lm", U,
                             Uy / (evals + REG_LAMBDAS[best][:, None]))
        Kte = torch.einsum("tld,mld->ltm", Z[~tr], Ztr)         # (L, t, m)
        preds[~tr, :, f] = (torch.einsum("ltm,lm->lt", Kte, alpha) + ym).T
    return te, preds.float()


def pooled_metrics(ys, preds, n_folds):
    """ys: per fold, list of (P,); preds: per fold, list of (L, P).
    → r (L, F) Pearson over the fold's pooled cells, mae (L,) over everything."""
    r = np.zeros((preds[0].shape[0], n_folds))
    for f in range(n_folds):
        yt, yp = ys[f] - ys[f].mean(), preds[f] - preds[f].mean(1, keepdims=True)
        r[:, f] = (yp @ yt) / np.maximum(np.linalg.norm(yp, axis=1)
                                         * np.linalg.norm(yt), 1e-12)
    mae = np.concatenate([np.abs(p - y[None]) for y, p in zip(ys, preds)],
                         axis=1).mean(axis=1)
    return r, mae


# cls = the model's own summary, bar = pooled elements; bar_concat = the
# supplement's element-resolved readout — closed-form here, so unlike the
# logistic concat probe it has no optimizer/convergence caveat
REG_SLICES = ("cls", "bar", "bar_concat")


def sweep_reg(features, idx_cells, dev, n_folds=5, seed=42):
    """Ridge-regress the realized deviation within each condition cell; pool
    test predictions over cells. idx_cells: list of trial-index tensors (one
    per target level at a fixed nuisance level).
    → {slice: {"acc": pearson r (12, F), "lam": None, "mae": (12,)}}"""
    out = {}
    for st in REG_SLICES:
        ys = [[] for _ in range(n_folds)]
        preds = [[] for _ in range(n_folds)]
        for idx in idx_cells:
            te, pr = ridge_cell(features[st][idx].float(), dev[idx], n_folds, seed)
            for f in range(n_folds):
                ys[f].append(dev[idx][te[:, f]].numpy())
                preds[f].append(pr[te[:, f], :, f].numpy().T)
        ys = [np.concatenate(v) for v in ys]
        preds = [np.concatenate(v, axis=1) for v in preds]
        r, mae = pooled_metrics(ys, preds, n_folds)
        out[st] = {"acc": r, "lam": None, "mae": mae}
    return out


def sweep(features, target, idx, n=None, per_token=False, bg_concat=False,
          y=None, **kw):
    """Probe every slice on trials `idx`. → {slice: {"acc": (12, F), "lam": (12, F)}}
    `y` overrides the condition labels (e.g. realized-statistic relabeling)."""
    if y is None:
        y = features["labels_mean" if target == "mean" else "labels_var"][idx]
    n_classes = len(features["labels_mean" if target == "mean" else "labels_var"].unique())
    print(f"{len(y)} trials, class counts {np.bincount(y.numpy()).tolist()}")

    # all 768-d slices in one fit; concatenations separately
    pooled = {st: features[st][idx] for st in POOLED}
    if "bar_concat" in features and n and n >= 2:
        pooled["pairwise_diff"] = compute_pairwise_diff(
            features["bar_concat"][idx], n, features["cls"].shape[-1])
    results = probe_slices(pooled, y, n_classes, **kw)
    # bar_concat is probed by the closed-form regression only (REG_SLICES):
    # its logistic version diverges at small λ (p >> n) and is retired
    for st in (["bg_concat"] if bg_concat else []):
        if st in features:
            results.update(probe_slices({st: features[st][idx]}, y, n_classes, **kw))
    if per_token and "bar_concat" in features and n:
        results.update(probe_per_token(features["bar_concat"][idx], y, n, n_classes, **kw))
    return {st: results[st] for st in SLICE_ORDER if st in results}


# ─────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────

def cv_best(acc):
    """Unbiased best-layer accuracy: each outer fold uses the layer that is
    best on the OTHER folds. acc: (L, F)."""
    F = acc.shape[1]
    picks = [int(np.argmax(np.delete(acc, f, axis=1).mean(1))) for f in range(F)]
    return float(np.mean([acc[l, f] for f, l in enumerate(picks)]))


def print_rows(results):
    """One line per (slice, layer); plot_neurips.py parses this format."""
    for st, r in results.items():
        for layer, a in enumerate(r["acc"]):
            print(f"  {st:12s}  layer {layer:2d}  acc = {a.mean():.3f} ± {a.std():.3f}")
        print()


def print_summary(results):
    """Slices × layers, with best (optimistic), cv-best (unbiased) and median λ."""
    print("=" * 80)
    n_l = len(next(iter(results.values()))["acc"])
    print(f"{'':14s}" + "  ".join(f"L{l:02d}" for l in range(n_l)))
    for st, r in results.items():
        m = r["acc"].mean(1)
        lam = f", λ≈{np.median(r['lam']):.0e}" if r.get("lam") is not None else ""
        print(f"{st:14s}" + "  ".join(f"{a:.2f}" for a in m)
              + f"  (best L{int(m.argmax()):02d}={m.max():.3f}, "
              f"cv-best={cv_best(r['acc']):.3f}{lam})")
    print("=" * 80)


def print_level_table(per_level, levels, nuisance):
    """cv-best of each slice at each fixed nuisance level (for the supplement)."""
    print(f"\ncv-best per fixed {nuisance} level")
    print(f"{'':14s}" + "".join(f"{nuisance[:3]}={lv:<6}" for lv in levels) + "  mean")
    for st in per_level[0]:
        v = [cv_best(r[st]["acc"]) for r in per_level]
        print(f"{st:14s}" + "".join(f"{x:<10.3f}" for x in v) + f"  {np.mean(v):.3f}")


def to_json(results):
    return {st: {"folds": r["acc"].tolist(), "best": float(r["acc"].mean(1).max()),
                 "cv_best": cv_best(r["acc"]),
                 "lambda": None if r.get("lam") is None else r["lam"].tolist(),
                 **({"mae": r["mae"].tolist()} if "mae" in r else {})}
            for st, r in results.items()}


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="features .pt from loader.py")
    p.add_argument("--target", required=True,
                   choices=["mean", "variance", "mean_reg", "variance_reg"])
    p.add_argument("--n_bars", type=int, default=None, help="set size to probe")
    p.add_argument("--fix_mean", type=int, default=None,
                   help="probe a single fixed mean level (label index)")
    p.add_argument("--fix_variance", type=int, default=None,
                   help="probe a single fixed SD level (label index)")
    p.add_argument("--n_folds", type=int, default=5, help="outer CV folds")
    p.add_argument("--inner_folds", type=int, default=4, help="inner CV folds (λ)")
    p.add_argument("--lambdas", default="1e-4,5e-4,2e-3,1e-2,5e-2,0.2,1,5",
                   help="L2 grid for the nested search")
    p.add_argument("--weight_decay", type=float, default=None,
                   help="fixed L2 strength instead of the nested search")
    p.add_argument("--lambda_select", choices=["slice", "layer"], default="slice")
    p.add_argument("--lambda_layer", type=int, default=N_LAYERS // 2,
                   help="layer whose inner-CV accuracy picks λ (slice mode)")
    p.add_argument("--per_token", action="store_true", help="add per-token / token-vote")
    p.add_argument("--bg_concat", action="store_true",
                   help="add concatenated background tokens (slow: ~190k-d)")
    p.add_argument("--max_iter", type=int, default=1500,
                   help="L-BFGS iteration cap (1500 vs 3000 verified equivalent)")
    p.add_argument("--device", default=pick_device())
    p.add_argument("--seed", type=int, default=42, help="CV splits and probe init")
    p.add_argument("--save", default=None, help="JSON: fold accuracies + λ per level")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    features = torch.load(a.features, map_location="cpu")
    if "bar_concat" in features:
        # multi-patch cells (size v2): average each element's block² tokens so
        # bar_concat is N·d-dimensional in every domain (ori/hue: no-op)
        d, n = features["cls"].shape[-1], int(features["n_bars"][0])
        if features["bar_concat"].shape[-1] != n * d:
            dtype = features["bar_concat"].dtype
            features["bar_concat"] = element_tokens(
                features["bar_concat"].float(), n, d).reshape(
                len(features["bar_concat"]), -1, n * d).to(dtype)
    kw = dict(n=a.n_bars, per_token=a.per_token, bg_concat=a.bg_concat,
              lambdas=[a.weight_decay] if a.weight_decay is not None
              else [float(x) for x in a.lambdas.split(",")],
              select=a.lambda_select, ref_layer=a.lambda_layer, n_folds=a.n_folds,
              inner_folds=a.inner_folds, seed=a.seed, max_iter=a.max_iter,
              device=a.device)

    # the nuisance statistic is held fixed at each of its levels (or one level)
    base = a.target.replace("_reg", "")
    regress = a.target.endswith("_reg")
    if regress:
        assert (a.fix_mean if base == "mean" else a.fix_variance) is None, (
            f"--fix_{base} conflicts with {a.target}: the regression iterates "
            f"the {base} levels itself; fix only the nuisance")
    nuisance = "variance" if base == "mean" else "mean"
    if a.fix_mean is not None or a.fix_variance is not None:
        nuisance = "mean" if a.fix_mean is not None else "variance"
        levels = [a.fix_mean if nuisance == "mean" else a.fix_variance]
    else:
        levels = features["labels_var" if nuisance == "variance" else "labels_mean"
                          ].unique().tolist()

    labels = features["labels_mean" if base == "mean" else "labels_var"]
    if regress:
        dev = realized_deviation(features, base)
        unit = "°" if features.get("period", 0) else " ln"
    else:
        real = realized_level(features, base)   # label = nearest realized level

    per_level = []
    for level in levels:
        print(f"\n----- fix {nuisance} = {level} -----")
        if regress:
            n_levels = len(features["mean_values" if base == "mean" else "var_values"])
            cells = [select_indices(features, a.n_bars,
                                    **{f"fix_{nuisance}": level,
                                       f"fix_{base}": t}) for t in range(n_levels)]
            d = torch.cat([dev[c] for c in cells]).numpy()
            print(f"regressing realized {base} deviation over {len(cells)} cells, "
                  f"{len(d)} trials (deviation IQR "
                  f"{np.percentile(d, 75) - np.percentile(d, 25):.2f}{unit})")
            per_level.append(sweep_reg(features, cells, dev,
                                       n_folds=a.n_folds, seed=a.seed))
        else:
            idx = select_indices(features, a.n_bars, **{f"fix_{nuisance}": level})
            y = real[idx]
            moved = int((y != labels[idx]).sum())
            print(f"labels = nearest realized level: {moved}/{len(idx)} trials "
                  f"moved from their condition level")
            per_level.append(sweep(features, base, idx, y=y, **kw))
        print_rows(per_level[-1])
        print_summary(per_level[-1])

    averaged = {st: {"acc": np.mean([r[st]["acc"] for r in per_level], axis=0), "lam": None}
                for st in per_level[0]}
    print(f"\n----- averaged over {len(levels)} fixed {nuisance} levels -----")
    print_rows(averaged)
    print_level_table(per_level, levels, nuisance)
    print_summary(averaged)

    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps({
            "args": vars(a), "nuisance": nuisance, "levels": levels,
            "per_level": [to_json(r) for r in per_level], "averaged": to_json(averaged)}))
        print(f"saved → {a.save}")


if __name__ == "__main__":
    main()
