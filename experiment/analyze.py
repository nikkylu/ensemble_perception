"""
Analysis script for ensemble perception probing.

Configure the RUN_CONFIG dict at the top, then run:
    python3 analyze.py

All probe combinations are run automatically and figures saved to output_dir.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

from probe import sweep, N_LAYERS

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — edit this
# ─────────────────────────────────────────────────────────────────

RUN_CONFIG = dict(
    features_path="featuresv3/features_n16.pt",
    output_dir="figures",

    # which set sizes to probe (None = all)
    n_bars_list=[16],

    # probe training
    n_folds=5,
    lr=1e-3,
    epochs=200,
    device="cpu",
)

SLICE_COLORS = {
    "cls":           "#4C72B0",
    "bar":           "#DD8452",
    "bg":            "#55A868",
    "all_patches":   "#C44E52",
    "bar_concat":    "#8172B2",
    "pairwise_diff": "#937860",
}

LAYERS = list(range(N_LAYERS))


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def get_slice_types(results):
    return list(dict.fromkeys(st for st, _ in results.keys()))


def accs(results, st):
    return [results[(st, l)][0] for l in LAYERS]


def errs(results, st):
    return [results[(st, l)][1] for l in LAYERS]


def run(features, target, n_bars, fix_mean=None, fix_variance=None, **kwargs):
    return sweep(features, target, n_bars_filter=n_bars,
                 fix_mean=fix_mean, fix_variance=fix_variance, **kwargs)


# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────

def plot_layer_curves(results, title, chance, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(chance, color="gray", linestyle="--",
               linewidth=1, label="chance")
    for st in get_slice_types(results):
        mu, sd = accs(results, st), errs(results, st)
        color = SLICE_COLORS.get(st, "black")
        ax.plot(LAYERS, mu, color=color, marker="o", markersize=4, label=st)
        ax.fill_between(LAYERS,
                        [m - s for m, s in zip(mu, sd)],
                        [m + s for m, s in zip(mu, sd)],
                        color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


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


def plot_fixed_comparison(all_results, target, n_bars, chance, out_path):
    """
    Overlay standard probe with fixed-nuisance probes, using bar slice.
    all_results: dict keyed by (fix_key, fix_level) or None for standard.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(chance, color="gray", linestyle="--",
               linewidth=1, label="chance")

    std_results = all_results.get(None)
    if std_results and "bar" in get_slice_types(std_results):
        mu = accs(std_results, "bar")
        ax.plot(LAYERS, mu, color="black", linewidth=2,
                marker="o", markersize=4, label="all levels (bar)")

    nuisance = "variance" if target == "mean" else "mean"
    fixed_keys = [k for k in all_results if k is not None]
    colors = plt.cm.tab10(np.linspace(0, 0.7, max(len(fixed_keys), 1)))

    for (fix_key, fix_level), color in zip(fixed_keys, colors):
        res = all_results[(fix_key, fix_level)]
        if "bar" not in get_slice_types(res):
            continue
        mu = accs(res, "bar")
        ax.plot(LAYERS, mu, color=color, linestyle="--",
                marker="o", markersize=3, label=f"{nuisance}={fix_level}")

    ax.set_title(
        f"n_bars={n_bars} | decode {target}, fix {nuisance} (bar slice)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_pairwise_vs_bar(results, n_bars, chance, out_path):
    slice_types = get_slice_types(results)
    if "pairwise_diff" not in slice_types:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(chance, color="gray", linestyle="--",
               linewidth=1, label="chance")
    for st in ["bar", "pairwise_diff"]:
        if st not in slice_types:
            continue
        mu, sd = accs(results, st), errs(results, st)
        color = SLICE_COLORS[st]
        ax.plot(LAYERS, mu, color=color, marker="o", markersize=4, label=st)
        ax.fill_between(LAYERS,
                        [m - s for m, s in zip(mu, sd)],
                        [m + s for m, s in zip(mu, sd)],
                        color=color, alpha=0.15)
    ax.set_title(f"n_bars={n_bars} | variance: pooled bar vs pairwise diff")
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
    n_classes_var = int(features["labels_var"].unique().numel())
    n_classes = {"mean": n_classes_mean, "variance": n_classes_var}
    chance = {"mean": 1/n_classes_mean, "variance": 1/n_classes_var}

    sweep_kwargs = dict(
        n_folds=cfg["n_folds"],
        lr=cfg["lr"],
        epochs=cfg["epochs"],
        device=cfg["device"],
    )

    # resolve n_bars list — if None, use all available
    available_n_bars = features["n_bars"].unique().tolist()
    n_bars_list = cfg["n_bars_list"] if cfg["n_bars_list"] else [None] + [
        int(n) for n in available_n_bars
    ]

    for n_bars in n_bars_list:
        tag = f"n{n_bars}" if n_bars else "all"
        nb_dir = out_dir / tag
        nb_dir.mkdir(exist_ok=True)
        print(f"\n{'='*60}")
        print(f"n_bars = {n_bars}")
        print(f"{'='*60}")

        for target in ["mean", "variance"]:
            t_dir = nb_dir / target
            t_dir.mkdir(exist_ok=True)
            print(f"\n── target: {target} ──")

            # 1. standard sweep (no fixing)
            print("  standard")
            std = run(features, target, n_bars, **sweep_kwargs)
            plot_layer_curves(std,
                              f"n_bars={n_bars} | decode {target}",
                              chance[target],
                              t_dir / "layer_curves.png")
            plot_heatmap(std,
                         f"n_bars={n_bars} | decode {target}",
                         t_dir / "heatmap.png")

            # 2. pairwise_diff vs bar (variance only)
            if target == "variance":
                plot_pairwise_vs_bar(std, n_bars, chance[target],
                                     t_dir / "pairwise_vs_bar.png")

            # 3. fixed nuisance sweeps
            nuisance = "variance" if target == "mean" else "mean"
            fix_key_arg = "fix_variance" if target == "mean" else "fix_mean"
            n_fixed_levels = n_classes_var if target == "mean" else n_classes_mean

            all_results = {None: std}
            for level in range(n_fixed_levels):
                print(f"  fix {nuisance}={level}")
                fix_kwargs = {fix_key_arg: level}
                all_results[(fix_key_arg, level)] = run(
                    features, target, n_bars, **fix_kwargs, **sweep_kwargs
                )

            plot_fixed_comparison(all_results, target, n_bars,
                                  chance[target],
                                  t_dir / f"fixed_{nuisance}.png")

    print(f"\nDone. All figures in {out_dir}/")


if __name__ == "__main__":
    main()
