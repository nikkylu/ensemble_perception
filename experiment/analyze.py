"""
Analysis script for ensemble perception probing.

Configure the RUN_CONFIG dict at the top, then run:
    python3 analyze.py

Figures generated per n_bars:
  1. layer_curves_{target}.png      — all slices × layers, standard probe
  2. heatmap_{target}.png           — slice × layer accuracy grid
  3. fixed_avg_{target}.png         — accuracy averaged across all fixed-nuisance runs vs standard
  4. fixed_per_slice_{target}.png   — per-slice accuracy (averaged over fixed levels) vs standard
  5. layer_comparison.png           — best-layer accuracy across conditions and slices
  6. pairwise_vs_bar.png            — pairwise_diff vs bar for variance (if available)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from probe import sweep, N_LAYERS

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — edit this
# ─────────────────────────────────────────────────────────────────

RUN_CONFIG = dict(
    features_path = "featuresv3/features_n16.pt",
    output_dir    = "figures",

    # which set sizes to probe ([] = all available)
    n_bars_list   = [16],

    # probe training
    n_folds  = 5,
    lr       = 1e-3,
    epochs   = 200,
    device   = "cpu",
)

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
# HELPERS
# ─────────────────────────────────────────────────────────────────

def get_slice_types(results):
    return list(dict.fromkeys(st for st, _ in results.keys()))


def accs(results, st):
    return np.array([results[(st, l)][0] for l in LAYERS])


def errs(results, st):
    return np.array([results[(st, l)][1] for l in LAYERS])


def run(features, target, n_bars, fix_mean=None, fix_variance=None, **kwargs):
    return sweep(features, target, n_bars_filter=n_bars,
                 fix_mean=fix_mean, fix_variance=fix_variance, **kwargs)


def avg_results_over_levels(results_list):
    """Average accuracy (and std) across multiple results dicts (same keys)."""
    slice_types = get_slice_types(results_list[0])
    averaged = {}
    for st in slice_types:
        all_accs = np.stack([accs(r, st) for r in results_list])  # (n_levels, 12)
        all_errs = np.stack([errs(r, st) for r in results_list])
        averaged[(st, "mu")] = all_accs.mean(0)   # (12,)
        averaged[(st, "sd")] = all_errs.mean(0)   # propagate mean std
    return averaged, slice_types


# ─────────────────────────────────────────────────────────────────
# PLOT 1: Layer accuracy curves (all slices)
# ─────────────────────────────────────────────────────────────────

def plot_layer_curves(results, title, chance, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label="chance")
    for st in get_slice_types(results):
        mu, sd = accs(results, st), errs(results, st)
        color = SLICE_COLORS.get(st, "black")
        ax.plot(LAYERS, mu, color=color, marker="o", markersize=4, label=st)
        ax.fill_between(LAYERS, mu - sd, mu + sd, color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────
# PLOT 2: Heatmap (slice × layer)
# ─────────────────────────────────────────────────────────────────

def plot_heatmap(results, title, out_path):
    slice_types = get_slice_types(results)
    data = np.array([accs(results, st) for st in slice_types])
    fig, ax = plt.subplots(figsize=(12, len(slice_types) * 0.8 + 1))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_yticks(range(len(slice_types)))
    ax.set_yticklabels(slice_types)
    ax.set_xlabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Accuracy")
    for i, st in enumerate(slice_types):
        for j in LAYERS:
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if data[i, j] < 0.6 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────
# PLOT 3: Fixed nuisance — averaged across levels, all slices
# ─────────────────────────────────────────────────────────────────

def plot_fixed_avg(std_results, fixed_results_list, target, nuisance, chance, out_path):
    """
    For each slice: plot standard accuracy vs average-over-fixed-levels accuracy.
    Shows whether fixing the nuisance variable raises/lowers decoding.
    """
    avg, slice_types = avg_results_over_levels(fixed_results_list)
    n_slices = len(slice_types)
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4), sharey=True)
    if n_slices == 1:
        axes = [axes]

    for ax, st in zip(axes, slice_types):
        color = SLICE_COLORS.get(st, "black")
        ax.axhline(chance, color="gray", linestyle="--", linewidth=1)

        # standard
        mu_std = accs(std_results, st)
        ax.plot(LAYERS, mu_std, color=color, marker="o", markersize=4,
                linewidth=2, label="standard", alpha=0.9)

        # averaged fixed
        mu_fix = avg[(st, "mu")]
        sd_fix = avg[(st, "sd")]
        ax.plot(LAYERS, mu_fix, color=color, marker="s", markersize=4,
                linewidth=2, linestyle="--", label=f"fix {nuisance} (avg)", alpha=0.9)
        ax.fill_between(LAYERS, mu_fix - sd_fix, mu_fix + sd_fix,
                        color=color, alpha=0.12)

        ax.set_title(st, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_xticks(LAYERS[::2])
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)

    fig.suptitle(f"Decode {target} — standard vs fix {nuisance} (averaged over levels)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────
# PLOT 4: Best-layer accuracy — conditions × slices bar chart
# ─────────────────────────────────────────────────────────────────

def plot_best_layer_bars(std_results, fixed_results_list, target, nuisance, chance, out_path):
    """
    Bar chart: best-layer accuracy per slice, comparing standard vs fixed-nuisance-averaged.
    """
    avg, slice_types = avg_results_over_levels(fixed_results_list)

    x = np.arange(len(slice_types))
    width = 0.35

    std_best  = [accs(std_results, st).max() for st in slice_types]
    fixed_best = [avg[(st, "mu")].max() for st in slice_types]

    fig, ax = plt.subplots(figsize=(max(6, len(slice_types) * 1.5), 5))
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label="chance")
    bars1 = ax.bar(x - width/2, std_best, width, label="standard",
                   color=[SLICE_COLORS.get(st, "gray") for st in slice_types], alpha=0.9)
    bars2 = ax.bar(x + width/2, fixed_best, width, label=f"fix {nuisance} (avg)",
                   color=[SLICE_COLORS.get(st, "gray") for st in slice_types],
                   alpha=0.5, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(slice_types, rotation=20, ha="right")
    ax.set_ylabel("Best-layer accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Decode {target} — best layer per slice (standard vs fix {nuisance})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────
# PLOT 5: pairwise_diff vs bar, variance only
# ─────────────────────────────────────────────────────────────────

def plot_pairwise_vs_bar(results, n_bars, chance, out_path):
    slice_types = get_slice_types(results)
    if "pairwise_diff" not in slice_types:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label="chance")
    for st in ["bar", "pairwise_diff", "bg", "bg_concat"]:
        if st not in slice_types:
            continue
        mu, sd = accs(results, st), errs(results, st)
        color = SLICE_COLORS[st]
        ax.plot(LAYERS, mu, color=color, marker="o", markersize=4, label=st)
        ax.fill_between(LAYERS, mu - sd, mu + sd, color=color, alpha=0.15)
    ax.set_title(f"n_bars={n_bars} | variance decoding: bar slices comparison")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    cfg = RUN_CONFIG
    features = torch.load(cfg["features_path"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_classes_mean = int(features["labels_mean"].unique().numel())
    n_classes_var  = int(features["labels_var"].unique().numel())
    chance = {"mean": 1/n_classes_mean, "variance": 1/n_classes_var}

    sweep_kwargs = dict(
        n_folds=cfg["n_folds"],
        lr=cfg["lr"],
        epochs=cfg["epochs"],
        device=cfg["device"],
    )

    n_bars_list = cfg["n_bars_list"] if cfg["n_bars_list"] else [
        int(n) for n in features["n_bars"].unique().tolist()
    ]

    for n_bars in n_bars_list:
        tag = f"n{n_bars}" if n_bars else "all"
        nb_dir = out_dir / tag
        nb_dir.mkdir(exist_ok=True)
        print(f"\n{'='*60}\nn_bars = {n_bars}\n{'='*60}")

        for target in ["mean", "variance"]:
            t_dir = nb_dir / target
            t_dir.mkdir(exist_ok=True)
            nuisance     = "variance" if target == "mean" else "mean"
            fix_key_arg  = "fix_variance" if target == "mean" else "fix_mean"
            n_fix_levels = n_classes_var if target == "mean" else n_classes_mean
            print(f"\n── target: {target} ──")

            # standard sweep
            print("  standard")
            std = run(features, target, n_bars, **sweep_kwargs)
            plot_layer_curves(std,
                              f"n_bars={n_bars} | decode {target} (standard)",
                              chance[target],
                              t_dir / "layer_curves.png")
            plot_heatmap(std,
                         f"n_bars={n_bars} | decode {target}",
                         t_dir / "heatmap.png")

            # pairwise_diff vs bar (variance only)
            if target == "variance":
                plot_pairwise_vs_bar(std, n_bars, chance[target],
                                     t_dir / "pairwise_vs_bar.png")

            # fixed nuisance sweeps — collect all levels
            fixed_results = []
            for level in range(n_fix_levels):
                print(f"  fix {nuisance}={level}")
                r = run(features, target, n_bars,
                        **{fix_key_arg: level}, **sweep_kwargs)
                fixed_results.append(r)

            # plot 3: layer curves standard vs averaged fixed
            plot_fixed_avg(std, fixed_results, target, nuisance,
                           chance[target],
                           t_dir / f"fixed_avg_{nuisance}.png")

            # plot 4: best-layer bar chart
            plot_best_layer_bars(std, fixed_results, target, nuisance,
                                 chance[target],
                                 t_dir / f"best_layer_{nuisance}.png")

    print(f"\nDone. All figures in {out_dir}/")


if __name__ == "__main__":
    main()
