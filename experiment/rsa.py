"""
RSA (Representational Similarity Analysis) for ensemble perception.

For each slice type × layer, computes an RDM from DINOv2 representations
and correlates it (Spearman) with model RDMs based on:
  - mean difference (circular: period 180° for orientation, 360° for hue)
  - variance difference

Usage:
    python3 rsa.py --features features_ori_v5_n16_s42.pt --n_bars 16
    python3 rsa.py --features feats.pt --n_bars 16 --metadata stimuli/metadata.json  # older features

In the fixed-nuisance sweeps only the varying statistic is correlated
(the fixed one's model RDM is constant). The noise ceiling is a split-half
reliability of the *condition-level* RDM; it bounds condition-level RSA,
so treat it as a reliability reference for the trial-level ρ shown here.
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from loader import MEAN_KEYS, VAR_KEYS, get
from probe import compute_pairwise_diff, select_indices

N_LAYERS = 12
SLICE_COLORS = {
    "cls":           "#4C72B0",
    "bar":           "#DD8452",
    "bg":            "#55A868",
    "all_patches":   "#C44E52",
    "bar_concat":    "#8172B2",
    "pairwise_diff": "#937860",
    "bg_concat":     "#64B5CD",
}
LAYERS = list(range(N_LAYERS))


# ─────────────────────────────────────────────────────────────────
# MODEL RDMs
# ─────────────────────────────────────────────────────────────────

def circular_diff(a, b, period=180.0):
    """Circular distance in degrees (180 for orientation, 360 for hue)."""
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def period(features, mean_values):
    """Period of the mean statistic: 360 hue, 180 orientation, 0 size (log-linear)."""
    if "period" in features:
        return float(features["period"])
    return 360.0 if max(mean_values) >= 180 else 180.0


def make_model_rdms(labels_mean, labels_var, mean_values, var_values, period=180.0):
    """
    Build two model RDMs from ground-truth labels.
    mean_values: list of actual orientation degrees, indexed by label int
    var_values:  list of actual variance values, indexed by label int
    Returns:
        rdm_mean: (N, N) pairwise circular orientation difference
        rdm_var:  (N, N) pairwise absolute variance difference
    """
    mean_ori = np.array([mean_values[int(l)] for l in labels_mean])
    var_val  = np.array([var_values[int(l)]  for l in labels_var])

    if period:
        rdm_mean = circular_diff(mean_ori[:, None], mean_ori[None, :], period)
    else:                                   # size: distance in log space
        lm = np.log(mean_ori)
        rdm_mean = np.abs(lm[:, None] - lm[None, :])
    rdm_var  = np.abs(var_val[:, None] - var_val[None, :])

    return rdm_mean, rdm_var


# ─────────────────────────────────────────────────────────────────
# NEURAL RDM
# ─────────────────────────────────────────────────────────────────

def neural_rdm(h):
    """
    Compute pairwise cosine distance RDM from representations.
    h: (N, D) numpy array
    Returns: (N, N)
    """
    return squareform(pdist(h, metric="cosine"))


# ─────────────────────────────────────────────────────────────────
# RSA
# ─────────────────────────────────────────────────────────────────

def get_slice_data(features, slice_types, indices, n_bars_filter):
    """{slice_type: (N, L, D) tensor} for the selected trials."""
    return {st: compute_pairwise_diff(features["bar_concat"][indices], n_bars_filter)
            if st == "pairwise_diff" else features[st][indices].float()
            for st in slice_types}


def noise_ceiling(h, labels_mean, labels_var, n_splits=20, seed=42):
    """
    Split-half noise ceiling for RSA at one layer.
    Splits trials *within each condition* (mean × variance) into two halves,
    computes one representative vector per condition for each half (mean),
    builds condition-level RDMs, then correlates them.
    Applies Spearman-Brown correction: r_full = 2r / (1 + r).
    Returns mean corrected r across splits.
    h: (N, D) numpy array
    """
    rng = np.random.default_rng(seed)
    conditions = list({(int(m), int(v))
                       for m, v in zip(labels_mean, labels_var)})
    conditions.sort()

    # group trial indices by condition
    groups = {c: np.where((labels_mean == c[0]) & (labels_var == c[1]))[0]
              for c in conditions}

    # only keep conditions with ≥2 trials
    groups = {c: idx for c, idx in groups.items() if len(idx) >= 2}
    if len(groups) < 2:
        return None

    rhos = []
    for _ in range(n_splits):
        half1, half2 = [], []
        for idx in groups.values():
            perm = rng.permutation(len(idx))
            mid = len(perm) // 2
            half1.append(h[idx[perm[:mid]]].mean(0))
            half2.append(h[idx[perm[mid:mid*2]]].mean(0))

        half1 = np.stack(half1)  # (n_conditions, D)
        half2 = np.stack(half2)

        n_cond = half1.shape[0]
        tri = np.triu_indices(n_cond, k=1)
        rdm1 = neural_rdm(half1)[tri]
        rdm2 = neural_rdm(half2)[tri]
        r, _ = spearmanr(rdm1, rdm2)
        # Spearman-Brown; undefined at r = -1 (or NaN with < 3 conditions)
        rhos.append(2 * r / (1 + r) if r > -1 else np.nan)

    return float(np.nanmean(rhos)) if np.isfinite(rhos).any() else None


def rsa_sweep(features, mean_values, var_values, n_bars_filter=None,
              fix_mean=None, fix_variance=None, compute_nc=True,
              targets=("mean", "variance")):
    """
    For each slice × layer, compute Spearman correlation between
    neural RDM and each model RDM.
    Optionally fix mean or variance to control for nuisance variable.

    Returns:
        results[(slice_type, layer)] = {"mean": rho, "variance": rho, "nc": rho}
    """
    indices = select_indices(features, n_bars_filter, fix_mean, fix_variance)
    labels_mean = features["labels_mean"][indices].numpy()
    labels_var  = features["labels_var"][indices].numpy()

    rdm_mean, rdm_var = make_model_rdms(
        labels_mean, labels_var, mean_values, var_values, period(features, mean_values)
    )
    tri = np.triu_indices(len(indices), k=1)
    model_vecs = {"mean": rdm_mean[tri], "variance": rdm_var[tri]}

    slice_types = ["cls", "bar", "bg", "all_patches"]
    for extra in ["bar_concat", "bg_concat"]:
        if extra in features:
            slice_types.append(extra)
    if "bar_concat" in features and n_bars_filter is not None and n_bars_filter >= 2:
        slice_types.append("pairwise_diff")

    slice_data = get_slice_data(features, slice_types, indices, n_bars_filter)

    results = {}
    for st in slice_types:
        for layer in range(N_LAYERS):
            h = slice_data[st][:, layer, :].numpy()
            nrdm = neural_rdm(h)[tri]
            r = {t: float(spearmanr(nrdm, model_vecs[t])[0]) for t in targets}
            r["nc"] = noise_ceiling(h, labels_mean, labels_var) if compute_nc else None
            results[(st, layer)] = r
            print(f"  {st:12s}  L{layer:02d}  "
                  + "  ".join(f"ρ_{t}={r[t]:+.3f}" for t in targets)
                  + (f"  nc={r['nc']:.3f}" if r["nc"] is not None else ""))
        print()

    return results


def avg_rsa_over_levels(results_list):
    """Average RSA results across multiple fixed-level sweeps."""
    return {k: {stat: float(np.nanmean([r[k][stat] for r in results_list]))
                for stat in results_list[0][k]
                if all(r[k][stat] is not None for r in results_list)}
            for k in results_list[0]}


# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────

def plot_rsa_curves(results, n_bars, out_path, targets=("mean", "variance"), title_suffix=""):
    """
    RSA correlation vs layer for each slice type.
    targets: which model RDMs to plot (one panel each).
    Noise ceiling shown as shaded band.
    """
    slice_types = list(dict.fromkeys(st for st, _ in results.keys()))
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5), sharey=False)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)

        # noise ceiling band
        nc_vals = [results[(slice_types[0], l)].get("nc") for l in LAYERS]
        nc_vals = [v for v in nc_vals if v is not None]
        if nc_vals:
            nc_arr = np.array([results[(slice_types[0], l)].get("nc", 0) for l in LAYERS])
            ax.fill_between(LAYERS, 0, nc_arr, color="gray", alpha=0.1,
                            label=f"noise ceiling ({slice_types[0]})")

        for st in slice_types:
            rhos = [results[(st, l)][target] for l in LAYERS]
            color = SLICE_COLORS.get(st, "black")
            ax.plot(LAYERS, rhos, color=color, marker="o",
                    markersize=4, label=st)
        ax.set_title(f"RSA: neural RDM × {target} RDM{title_suffix}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Spearman ρ")
        ax.set_xticks(LAYERS)
        ax.legend(fontsize=8)

    fig.suptitle(f"n_bars={n_bars}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_rsa_heatmap(results, target, n_bars, out_path):
    """Heatmap of RSA ρ: slice × layer for one model RDM."""
    slice_types = list(dict.fromkeys(st for st, _ in results.keys()))
    data = np.array([[results[(st, l)][target] for l in LAYERS]
                     for st in slice_types])
    vmax = max(abs(data).max(), 0.1)

    fig, ax = plt.subplots(figsize=(12, len(slice_types) * 0.8 + 1))
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_yticks(range(len(slice_types)))
    ax.set_yticklabels(slice_types)
    ax.set_xlabel("Layer")
    ax.set_title(f"RSA ρ — {target} RDM  (n_bars={n_bars})")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    for i, st in enumerate(slice_types):
        for j in LAYERS:
            ax.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(data[i, j]) > vmax * 0.6 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_rdms(features, n_bars, indices, mean_values, var_values, out_path):
    """Visualize the two model RDMs side by side."""
    labels_mean = features["labels_mean"][indices].numpy()
    labels_var  = features["labels_var"][indices].numpy()
    rdm_mean, rdm_var = make_model_rdms(
        labels_mean, labels_var, mean_values, var_values, period(features, mean_values)
    )
    # sort by mean then variance for cleaner visualization
    order = np.lexsort((labels_var, labels_mean))
    rdm_mean = rdm_mean[np.ix_(order, order)]
    rdm_var  = rdm_var[np.ix_(order, order)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, rdm, title in zip(axes,
                               [rdm_mean, rdm_var],
                               ["Mean diff (°, circular)", "SD diff (°)"]):
        im = ax.imshow(rdm, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Trial (sorted by mean, var)")
        ax.set_ylabel("Trial")
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"Model RDMs  (n_bars={n_bars})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", default=None,
                        help="metadata.json (only for features saved without "
                             "mean_values / var_values)")
    parser.add_argument("--n_bars", type=int, default=None)
    parser.add_argument("--output_dir", default="figures/rsa")
    args = parser.parse_args()

    features = torch.load(args.features, map_location="cpu")
    if "mean_values" in features:
        mean_values, var_values = features["mean_values"], features["var_values"]
    else:
        # sorted unique condition values, matching loader.py's label maps
        assert args.metadata, "these features need --metadata"
        trials = json.load(open(args.metadata))["trials"]
        mean_values = sorted({get(t, MEAN_KEYS) for t in trials})
        var_values = sorted({get(t, VAR_KEYS) for t in trials})
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = select_indices(features, args.n_bars)

    print(f"\nRSA  n_bars={args.n_bars}  N={len(indices)}\n")
    print(f"Mean values: {mean_values}")
    print(f"Var values:  {var_values}\n")

    n_mean_levels = len(mean_values)
    n_var_levels  = len(var_values)

    # model RDMs visualization
    plot_rdms(features, args.n_bars, indices, mean_values, var_values,
              out_dir / "model_rdms.png")

    # ── standard RSA sweep ───────────────────────────────────────
    print("\n=== Standard RSA ===")
    results = rsa_sweep(features, mean_values, var_values,
                        n_bars_filter=args.n_bars)
    plot_rsa_curves(results, args.n_bars, out_dir / "rsa_curves.png")
    plot_rsa_heatmap(results, "mean", args.n_bars,
                     out_dir / "rsa_heatmap_mean.png")
    plot_rsa_heatmap(results, "variance", args.n_bars,
                     out_dir / "rsa_heatmap_variance.png")

    # ── fix mean → probe variance RSA ───────────────────────────
    print("\n=== RSA: fix mean, probe variance ===")
    fixed_mean_results = []
    for level in range(n_mean_levels):
        print(f"  fix mean={level}")
        r = rsa_sweep(features, mean_values, var_values,
                      n_bars_filter=args.n_bars, fix_mean=level,
                      targets=("variance",))
        fixed_mean_results.append(r)
    avg_fix_mean = avg_rsa_over_levels(fixed_mean_results)
    plot_rsa_curves(avg_fix_mean, args.n_bars,
                    out_dir / "rsa_curves_fixmean.png",
                    targets=("variance",),
                    title_suffix=" (fix mean, avg)")
    plot_rsa_heatmap(avg_fix_mean, "variance", args.n_bars,
                     out_dir / "rsa_heatmap_variance_fixmean.png")

    # ── fix variance → probe mean RSA ───────────────────────────
    print("\n=== RSA: fix variance, probe mean ===")
    fixed_var_results = []
    for level in range(n_var_levels):
        print(f"  fix variance={level}")
        r = rsa_sweep(features, mean_values, var_values,
                      n_bars_filter=args.n_bars, fix_variance=level,
                      targets=("mean",))
        fixed_var_results.append(r)
    avg_fix_var = avg_rsa_over_levels(fixed_var_results)
    plot_rsa_curves(avg_fix_var, args.n_bars,
                    out_dir / "rsa_curves_fixvar.png",
                    targets=("mean",),
                    title_suffix=" (fix variance, avg)")
    plot_rsa_heatmap(avg_fix_var, "mean", args.n_bars,
                     out_dir / "rsa_heatmap_mean_fixvar.png")

    print(f"\nDone. Figures in {out_dir}/")


if __name__ == "__main__":
    main()
