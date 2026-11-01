"""
Linear probes for ensemble statistics in frozen DINOv2 features.

Decodes the condition mean (6-way) or SD (3-way) of each display from
every representation ("slice") at every layer:

    cls, bar, bg, all_patches   768-d pooled tokens (from loader.py)
    pairwise_diff               2·Var over element tokens (N ≥ 2)
    bar_concat                  element tokens concatenated (N·768-d)
    bg_concat        (opt-in)   background tokens concatenated (~190k-d)
    per_token / token_vote      (opt-in) one 768-d probe shared by all element
                                tokens, scored per token / by averaging an
                                image's token logits

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

Efficiency: every (slice, layer, fold, λ) probe is an independent logistic
regression; all of them are trained together in one vectorized full-batch
Adam loop. Per-probe losses are summed, so each probe gets exactly the
gradient it would get alone.

Usage:
    python3 probe.py --features feats.pt --target mean --n_bars 16
    python3 probe.py --features feats.pt --target variance --n_bars 16 --save res.json
    python3 probe.py --features feats.pt --target variance --n_bars 16 --weight_decay 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

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
        k = 1.0 / np.sqrt(dim)  # nn.Linear init range
        shape = (n_layers, n_splits, n_lambdas)
        self.W = nn.Parameter(torch.empty(*shape, dim, n_classes).uniform_(-k, k))
        self.b = nn.Parameter(torch.empty(*shape, n_classes).uniform_(-k, k))

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


def fit_predict(H, y, train, n_classes, lam, lr=1e-3, epochs=200, device="cpu",
                max_params=4e7):
    """Train one probe per (layer, split, λ), full batch; return logits for all rows.

    H (N, L, D), y (N,), train (N, S) 0/1 mask, lam broadcastable to (L, S, Q).
    Each probe minimizes mean CE over its training rows + ½λ‖W‖² (bias free).
    λ is processed in chunks so at most `max_params` weights live at once.
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
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        target = y.view(N, 1, 1, 1, 1).expand(N, L, S, lq.shape[-1], 1)
        for _ in range(epochs):
            ce = -model(H, mu, sd).log_softmax(-1).gather(-1, target).squeeze(-1)
            data = (ce * train.view(N, 1, S, 1)).sum(0) / n_train       # (L, S, Q)
            loss = data.sum() + 0.5 * (lq * model.W.pow(2).sum((-2, -1))).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
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
        layers = (list(range(L)) if select == "layer"
                  else [s * per + ref_layer for s in range(n_slices)])
        lg = fit_predict(H[:, layers], y_rows, rows(itr), n_classes,
                         lambdas.view(1, 1, -1), **fit)
        val = accuracy(lg, y_rows, rows(iva))                         # (·, F·k, Q)
        val = val.reshape(len(layers), n_folds, k, -1).mean(2)        # (·, F, Q)
        best = val.shape[-1] - 1 - val[..., ::-1].argmax(-1)          # ties → larger λ
        lam = lambdas[torch.as_tensor(best)]
        if select == "slice":
            lam = lam.repeat_interleave(per, dim=0)                   # (L, F)

    lg = fit_predict(H, y_rows, rows(tr), n_classes, lam.unsqueeze(-1), **fit)
    return lg[:, :, :, 0], te, lam.numpy()


def probe_slices(slices, y, n_classes, **kw):
    """{name: (N, 12, D)} of equal width, trained in ONE vectorized fit.
    → {name: {"acc": (12, F), "lam": (12, F)}}"""
    names = list(slices)
    H = torch.cat([slices[n].float() for n in names], dim=1)
    lg, te, lam = nested_cv(H, y, n_classes, n_slices=len(names), **kw)
    acc = accuracy(lg, y, te)
    return {n: {"acc": acc[i * N_LAYERS:(i + 1) * N_LAYERS],
                "lam": lam[i * N_LAYERS:(i + 1) * N_LAYERS]}
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

def compute_pairwise_diff(bar_concat, n):
    """Per-dim mean squared pairwise difference of element tokens,
    mean_{i<j}(h_i - h_j)² = 2·Var_unbiased(h). (N, L, n·D) → (N, L, D)"""
    N, L, total = bar_concat.shape
    assert total % n == 0, f"bar_concat dim {total} not divisible by {n}"
    return 2 * bar_concat.float().reshape(N, L, n, -1).var(dim=2, unbiased=True)


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


def sweep(features, target, idx, n=None, per_token=False, bg_concat=False, **kw):
    """Probe every slice on trials `idx`. → {slice: {"acc": (12, F), "lam": (12, F)}}"""
    y = features["labels_mean" if target == "mean" else "labels_var"][idx]
    n_classes = len(features["labels_mean" if target == "mean" else "labels_var"].unique())
    print(f"{len(y)} trials, class counts {np.bincount(y.numpy()).tolist()}")

    # all 768-d slices in one fit; concatenations separately
    pooled = {st: features[st][idx] for st in POOLED}
    if "bar_concat" in features and n and n >= 2:
        pooled["pairwise_diff"] = compute_pairwise_diff(features["bar_concat"][idx], n)
    results = probe_slices(pooled, y, n_classes, **kw)
    for st in ["bar_concat"] + (["bg_concat"] if bg_concat else []):
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
    print(f"{'':14s}" + "  ".join(f"L{l:02d}" for l in range(N_LAYERS)))
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
                 "lambda": None if r.get("lam") is None else r["lam"].tolist()}
            for st, r in results.items()}


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="features .pt from loader.py")
    p.add_argument("--target", required=True, choices=["mean", "variance"])
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--device", default=pick_device())
    p.add_argument("--seed", type=int, default=42, help="CV splits and probe init")
    p.add_argument("--save", default=None, help="JSON: fold accuracies + λ per level")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    features = torch.load(a.features, map_location="cpu")
    kw = dict(n=a.n_bars, per_token=a.per_token, bg_concat=a.bg_concat,
              lambdas=[a.weight_decay] if a.weight_decay is not None
              else [float(x) for x in a.lambdas.split(",")],
              select=a.lambda_select, ref_layer=a.lambda_layer, n_folds=a.n_folds,
              inner_folds=a.inner_folds, seed=a.seed, lr=a.lr, epochs=a.epochs,
              device=a.device)

    # the nuisance statistic is held fixed at each of its levels (or one level)
    nuisance = "variance" if a.target == "mean" else "mean"
    if a.fix_mean is not None or a.fix_variance is not None:
        nuisance = "mean" if a.fix_mean is not None else "variance"
        levels = [a.fix_mean if nuisance == "mean" else a.fix_variance]
    else:
        levels = features["labels_var" if nuisance == "variance" else "labels_mean"
                          ].unique().tolist()

    per_level = []
    for level in levels:
        print(f"\n----- fix {nuisance} = {level} -----")
        idx = select_indices(features, a.n_bars, **{f"fix_{nuisance}": level})
        per_level.append(sweep(features, a.target, idx, **kw))
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
