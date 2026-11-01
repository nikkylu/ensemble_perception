"""
NeurIPS-style probe_results figures for the hue / orientation results.

Reads results_hue_ori/{ori,hue}/n{N}_{target}_{probe}.txt and parses only
the final "averaged over K fixed <nuisance> levels" section of each file,
so every accuracy shown is nuisance-controlled (mean probed with variance
fixed, variance probed with mean fixed).

Produces one composite figure per domain x probe (skipped when no
result files exist for it):
    figures/{domain}/probe_results_neurips_{domain}_{probe}.{pdf,png}

Note: "quadratic" results come from the earlier low-rank quadratic probe,
which probe.py no longer implements; pipeline.sh writes "linear" files.

Usage (from experiment/):
    python3 plot_neurips.py
    python3 plot_neurips.py --results_dir results_hue_ori --output_dir figures
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

N_BARS_LIST = [1, 2, 3, 4, 8, 16, 32]
N_LAYERS = 12
TARGETS = ["mean", "variance"]
PROBES = ["linear", "quadratic"]
DOMAINS = ["ori", "hue", "size"]
CHANCE = {"mean": 1 / 6, "variance": 1 / 3}
HEATMAP_NS = [4, 8, 16, 32]

SLICE_TYPES = ["cls", "bar", "bg", "all_patches",
               "bar_concat", "bg_concat", "pairwise_diff"]
SLICE_COLORS = {
    "cls":           "#4C72B0",
    "bar":           "#DD8452",
    "bg":            "#55A868",
    "all_patches":   "#C44E52",
    "bar_concat":    "#8172B2",
    "bg_concat":     "#64B5CD",
    "pairwise_diff": "#937860",
}
SLICE_LABELS = {
    "ori": {
        "cls":           "CLS",
        "bar":           "bar",
        "bg":            "background",
        "all_patches":   "all patches",
        "bar_concat":    "bar concat",
        "bg_concat":     "bg concat",
        "pairwise_diff": "pairwise diff",
    },
    "hue": {
        "cls":           "CLS",
        "bar":           "shape",
        "bg":            "background",
        "all_patches":   "all patches",
        "bar_concat":    "shape concat",
        "bg_concat":     "bg concat",
        "pairwise_diff": "pairwise diff",
    },
}
SLICE_LABELS["size"] = SLICE_LABELS["hue"]
ELEMENT = {"ori": "bar", "hue": "shape", "size": "shape"}

FIG_WIDTH  = 5.5    # in — NeurIPS \textwidth on letter paper
FIG_HEIGHT = 6.6
FONT_SIZE  = 10

mpl.rcParams.update({
    "font.family":       "Arial",
    "font.size":         FONT_SIZE,
    "axes.titlesize":    FONT_SIZE,
    "axes.labelsize":    FONT_SIZE,
    "xtick.labelsize":   FONT_SIZE,
    "ytick.labelsize":   FONT_SIZE,
    "legend.fontsize":   FONT_SIZE,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "axes.titlepad":     4,
    "axes.labelpad":     2,
    "legend.frameon":    False,
    "legend.handlelength": 1.6,
    "legend.columnspacing": 1.0,
    "pdf.fonttype":      42,   # embed TrueType so text stays editable
    "ps.fonttype":       42,
    "savefig.dpi":       300,
})


# ─────────────────────────────────────────────────────────────────
# RESULT PARSING
# ─────────────────────────────────────────────────────────────────

_LINE = re.compile(r"^\s+(\w+)\s+layer\s+(\d+)\s+acc = ([\d.]+) ± ([\d.]+)")
_AVG = re.compile(r"^-+ averaged over \d+ fixed \w+ levels -+")


def parse_averaged(path):
    """Parse the final averaged section of one result file.

    Returns {slice: (accs[N_LAYERS], stds[N_LAYERS])}.
    """
    accs, stds = {}, {}
    in_avg = False
    for line in Path(path).read_text().splitlines():
        if _AVG.match(line.strip()):
            in_avg = True
            continue
        if not in_avg:
            continue
        m = _LINE.match(line)
        if not m:
            continue
        st, layer = m.group(1), int(m.group(2))
        accs.setdefault(st, np.full(N_LAYERS, np.nan))[layer] = float(m.group(3))
        stds.setdefault(st, np.full(N_LAYERS, np.nan))[layer] = float(m.group(4))
    return {st: (accs[st], stds[st]) for st in accs}


def load_results(results_dir, domain, probe):
    """results[target][n_bars][slice] = (accs, stds); missing files skipped."""
    results_dir = Path(results_dir) / domain
    results = {t: {} for t in TARGETS}
    for t in TARGETS:
        for n in N_BARS_LIST:
            path = results_dir / f"n{n}_{t}_{probe}.txt"
            if path.exists():
                results[t][n] = parse_averaged(path)
    return results


def layer_accs(results, target, n, slice_type):
    entry = results[target].get(n, {}).get(slice_type)
    return entry[0] if entry is not None else np.full(N_LAYERS, np.nan)


def best_acc(results, target, n, slice_type):
    accs = layer_accs(results, target, n, slice_type)
    return np.nan if np.all(np.isnan(accs)) else float(np.nanmax(accs))


# ─────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────

def panel_best_acc(ax, results, target, title, labels):
    ns = [n for n in N_BARS_LIST if n in results[target]]
    for st in SLICE_TYPES:
        accs = [best_acc(results, target, n, st) for n in ns]
        if all(np.isnan(a) for a in accs):
            continue
        concat = st in ("bar_concat", "bg_concat")
        ax.plot(ns, accs, color=SLICE_COLORS[st], label=labels[st],
                marker="s" if concat else "o", markersize=3,
                linewidth=1.8 if st == "bar" else 1.0,
                linestyle="--" if concat else "-",
                zorder=3 if st == "bar" else 2)
    ax.axhline(CHANCE[target], color="0.5", linestyle=":", linewidth=1,
               label="Chance", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(N_BARS_LIST)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("set size (N elements)")
    ax.set_title(title)


def panel_layer_profile(ax, results, target, title):
    cmap = plt.cm.Blues
    for i, n in enumerate(N_BARS_LIST):
        if n not in results[target]:
            continue
        ax.plot(range(N_LAYERS), layer_accs(results, target, n, "bar"),
                color=cmap(0.3 + 0.7 * i / (len(N_BARS_LIST) - 1)),
                marker="o", markersize=2.5, linewidth=1.2, label=f"{n}")
    ax.axhline(CHANCE[target], color="0.5", linestyle=":", linewidth=1)
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("layer")
    ax.set_title(title)


def make_figure(results, domain, out_stem):
    labels = SLICE_LABELS[domain]
    element = ELEMENT[domain]
    feature = {"ori": "orientation", "hue": "hue", "size": "size"}[domain]

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 0.82, 1], hspace=0.6,
                             left=0.19, right=0.84, top=0.875, bottom=0.07)

    # ── row 1: best accuracy vs set size ─────────────────────────────
    row1 = outer[0].subgridspec(1, 2, wspace=0.12)
    ax_a = fig.add_subplot(row1[0])
    ax_b = fig.add_subplot(row1[1], sharey=ax_a)
    panel_best_acc(ax_a, results, "mean",     f"(a) mean {feature}", labels)
    panel_best_acc(ax_b, results, "variance", f"(b) {feature} variance", labels)
    ax_a.set_ylabel("best accuracy")
    ax_b.tick_params(labelleft=False)
    handles, leg_labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.52, 1.0), handletextpad=0.4)

    # ── row 2: variance heatmaps (slice type × layer) ────────────────
    row2 = outer[1].subgridspec(1, len(HEATMAP_NS), wspace=0.08)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=CHANCE["variance"], vmax=1.0)
    hm_axes = []
    for j, n in enumerate(HEATMAP_NS):
        ax = fig.add_subplot(row2[j])
        mat = np.array([layer_accs(results, "variance", n, st) for st in SLICE_TYPES])
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks([0, 4, 8])
        ax.set_yticks(range(len(SLICE_TYPES)))
        ax.set_yticklabels([labels[s] for s in SLICE_TYPES] if j == 0 else [])
        ax.set_title(f"N = {n}")
        hm_axes.append(ax)
    hm_axes[0].text(0, 1.2, "(c) variance probe accuracy by layer and token slice",
                    transform=hm_axes[0].transAxes, ha="left", va="bottom")
    fig.text(0.52, hm_axes[0].get_position().y0 - 0.032, "layer",
             ha="center", va="top")
    pos = hm_axes[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.015, pos.height])
    cbar = fig.colorbar(im, cax=cax, ticks=[0, CHANCE["variance"], 1.0])
    cbar.ax.set_yticklabels(["0", f"{CHANCE['variance']:.2f}", "1"])
    cbar.set_label("accuracy", labelpad=6)
    cbar.outline.set_linewidth(0.8)

    # ── row 3: layer profile, element token ──────────────────────────
    row3 = outer[2].subgridspec(1, 2, wspace=0.12)
    ax_d = fig.add_subplot(row3[0])
    ax_e = fig.add_subplot(row3[1], sharey=ax_d)
    panel_layer_profile(ax_d, results, "mean",     f"(d) {element} token: mean")
    panel_layer_profile(ax_e, results, "variance", f"(e) {element} token: variance")
    ax_d.set_ylabel("accuracy (5-fold CV)")
    ax_e.tick_params(labelleft=False)
    ax_e.legend(title="N", loc="center left", bbox_to_anchor=(1.01, 0.5),
                handlelength=1.2, labelspacing=0.25, borderaxespad=0)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
        print(f"Figure saved → {out_stem.with_suffix(f'.{ext}')}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_hue_ori",
                        help="Directory with {ori,hue}/n{N}_{target}_{probe}.txt")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()

    for domain in DOMAINS:
        for probe in PROBES:
            results = load_results(args.results_dir, domain, probe)
            if not any(results.values()):
                print(f"skip {domain}/{probe}: no result files")
                continue
            out = Path(args.output_dir) / domain / f"probe_results_neurips_{domain}_{probe}"
            make_figure(results, domain, out)


if __name__ == "__main__":
    main()
