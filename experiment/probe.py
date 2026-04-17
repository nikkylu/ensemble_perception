"""
Linear and low-rank quadratic probes for ensemble perception.
Trains a classifier on frozen DINOv2 representations
to decode mean orientation (6-way) or variance (3-way).

Sweeps over all slices (cls/bar/bg/all_patches × 12 layers)
and reports accuracy for each.

Usage:
    python3 probe.py --features features.pt --target mean
    python3 probe.py --features features.pt --target variance
    python3 probe.py --features features_n16.pt --target variance --n_bars 16 --probe quadratic --rank 32
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold


# ─────────────────────────────────────────────────────────────────
# PROBE MODEL
# ─────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """ŷ = Wh + b"""

    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, h):
        return self.linear(h)


class QuadraticProbe(nn.Module):
    """ŷ = h^T (U^T U) h + Wh + b
    Low-rank quadratic: project to k dims, compute all pairwise interactions.
    """

    def __init__(self, input_dim, n_classes, rank=32):
        super().__init__()
        self.proj = nn.Linear(input_dim, rank, bias=False)  # U: 768 -> k
        self.quadratic = nn.Linear(
            rank * rank, n_classes)  # pairwise -> classes
        self.linear = nn.Linear(input_dim, n_classes)       # linear term

    def forward(self, h):
        z = self.proj(h)                          # (B, k)
        # outer product: all pairwise interactions
        outer = z.unsqueeze(-1) * z.unsqueeze(-2)  # (B, k, k)
        outer = outer.reshape(h.shape[0], -1)      # (B, k*k)
        return self.quadratic(outer) + self.linear(h)

# ─────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────


def train_probe(h_train, y_train, h_test, y_test,
                n_classes, lr=1e-3, epochs=200, batch_size=64, device="cpu",
                probe_type="linear", rank=32):
    """
    Train a probe on h_train/y_train, evaluate on h_test/y_test.
    probe_type: "linear" or "quadratic"
    Returns test accuracy.
    """
    input_dim = h_train.shape[1]
    if probe_type == "quadratic":
        probe = QuadraticProbe(input_dim, n_classes, rank=rank).to(device)
    else:
        probe = LinearProbe(input_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # dataloader for training
    train_dataset = TensorDataset(h_train.to(device), y_train.to(device))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # train
    probe.train()
    for epoch in range(epochs):
        for h_batch, y_batch in train_loader:
            logits = probe(h_batch)
            loss = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate
    probe.eval()
    with torch.no_grad():
        logits = probe(h_test.to(device))
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y_test.to(device)).float().mean().item()

    return accuracy, probe


# ─────────────────────────────────────────────────────────────────
# CROSS-VALIDATED PROBING
# ─────────────────────────────────────────────────────────────────

def cross_validated_probe(h, y, n_classes, n_folds=5, **train_kwargs):
    """
    Run stratified k-fold cross-validation.
    Returns mean accuracy and std across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, test_idx in skf.split(h, y):
        h_train, h_test = h[train_idx], h[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        accuracy, _ = train_probe(h_train, y_train, h_test, y_test,
                                  n_classes, **train_kwargs)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies), np.std(fold_accuracies)


# ─────────────────────────────────────────────────────────────────
# SWEEP ALL SLICES
# ─────────────────────────────────────────────────────────────────

SLICE_TYPES = ["cls", "bar", "bg", "all_patches"]
N_LAYERS = 12


def compute_pairwise_diff(bar_concat, n_bars):
    """
    Mean squared pairwise difference across bar patch tokens.
    bar_concat: (N, L, n_bars*768)
    Returns:    (N, L, 768)
    """
    N, L, _ = bar_concat.shape
    tokens = bar_concat.reshape(N, L, n_bars, -1)  # (N, L, n_bars, 768)
    pairs = []
    for i in range(n_bars):
        for j in range(i + 1, n_bars):
            diff = (tokens[:, :, i, :] - tokens[:, :, j, :]) ** 2  # (N, L, 768)
            pairs.append(diff)
    return torch.stack(pairs).mean(0)  # (N, L, 768)


def sweep(features, target, n_bars_filter=None, n_folds=5, lr=1e-3, epochs=200, device="cpu",
          probe_type="linear", rank=32, fix_mean=None, fix_variance=None):
    """
    Run probes across all slices (4-5 token groups × 12 layers).
    Returns a dict: results[(slice_type, layer)] = (mean_acc, std_acc)

    fix_mean:     if set, only use trials with this mean orientation label
    fix_variance: if set, only use trials with this variance label
    """
    # select target labels
    if target == "mean":
        y = features["labels_mean"]
        n_classes = len(y.unique())
    elif target == "variance":
        y = features["labels_var"]
        n_classes = len(y.unique())

    # optionally filter by set size
    if n_bars_filter is not None:
        mask = features["n_bars"] == n_bars_filter
        indices = mask.nonzero(as_tuple=True)[0]
    else:
        indices = torch.arange(len(y))

    # fix the nuisance variable to a single level
    if fix_mean is not None:
        keep = features["labels_mean"][indices] == fix_mean
        indices = indices[keep]
    if fix_variance is not None:
        keep = features["labels_var"][indices] == fix_variance
        indices = indices[keep]

    y = y[indices]
    print(f"Target: {target} ({n_classes} classes)")
    print(f"Samples: {len(y)}")
    print(
        f"Class distribution: {[int((y == c).sum()) for c in range(n_classes)]}")
    print()

    # include bar_concat and pairwise_diff if available
    slice_types = list(SLICE_TYPES)
    extra_features = {}
    if "bar_concat" in features:
        slice_types.append("bar_concat")
        if n_bars_filter is not None:
            extra_features["pairwise_diff"] = compute_pairwise_diff(
                features["bar_concat"][indices], n_bars_filter
            )
            slice_types.append("pairwise_diff")

    results = {}

    for slice_type in slice_types:
        # (N, 12, D) — D is 768 for pooled, n_bars*768 for concat
        if slice_type in extra_features:
            all_layers = extra_features[slice_type]
        else:
            all_layers = features[slice_type][indices]

        for layer in range(N_LAYERS):
            # pick one slice: (N, D)
            h = all_layers[:, layer, :]

            mean_acc, std_acc = cross_validated_probe(
                h, y, n_classes, n_folds=n_folds,
                lr=lr, epochs=epochs, device=device,
                probe_type=probe_type, rank=rank,
            )

            results[(slice_type, layer)] = (mean_acc, std_acc)
            print(f"  {slice_type:12s}  layer {layer:2d}  "
                  f"acc = {mean_acc:.3f} ± {std_acc:.3f}")

        print()

    return results


# ─────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────

def print_summary(results):
    """Print results as a table: rows = slice types, columns = layers."""
    # figure out which slice types are in results
    slice_types = sorted(set(st for st, _ in results.keys()),
                         key=lambda s: (s != "cls", s != "bar", s != "bg",
                                        s != "all_patches", s != "bar_concat"))

    print("\n" + "=" * 80)
    print(f"{'':12s}  " + "  ".join(f"L{l:02d}" for l in range(N_LAYERS)))
    print("-" * 80)

    for slice_type in slice_types:
        accs = [results[(slice_type, l)][0] for l in range(N_LAYERS)]
        row = "  ".join(f"{a:.2f}" for a in accs)
        best_layer = np.argmax(accs)
        print(f"{slice_type:12s}  {row}  (best: L{best_layer:02d}={max(accs):.3f})")

    print("=" * 80)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True,
                        help="Path to features.pt from loader.py")
    parser.add_argument("--target", required=True,
                        choices=["mean", "variance"])
    parser.add_argument("--n_bars", type=int, default=None,
                        help="Filter to specific set size")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probe", choices=["linear", "quadratic"], default="linear",
                        help="Probe type: linear or low-rank quadratic")
    parser.add_argument("--rank", type=int, default=32,
                        help="Rank for quadratic probe (ignored for linear)")
    parser.add_argument("--fix_mean", type=int, default=None,
                        help="Fix mean orientation label (0-indexed) while probing variance")
    parser.add_argument("--fix_variance", type=int, default=None,
                        help="Fix variance label (0-indexed) while probing mean")
    args = parser.parse_args()

    features = torch.load(args.features)
    results = sweep(
        features, args.target,
        n_bars_filter=args.n_bars,
        n_folds=args.n_folds,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        probe_type=args.probe,
        rank=args.rank,
        fix_mean=args.fix_mean,
        fix_variance=args.fix_variance,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
