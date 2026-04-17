"""
Visualize patch masks overlaid on stimulus images.
Picks a few stimuli and draws the 16x16 patch grid,
highlighting bar patches in red.

Usage:
    python3 verify_mask.py --metadata stimuli_v2/metadata.json
    python3 verify_mask.py --metadata stimuli_v2/metadata.json --n_bars 4 --n_examples 3
"""

from loader import make_patch_mask, PATCH_PX, GRID
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.append(".")


def visualize_mask(trial, meta_root, ax):
    """Draw one stimulus with its patch mask overlaid."""
    # load image
    img_path = meta_root.parent.parent / trial["filename"]
    img = plt.imread(str(img_path))
    ax.imshow(img)

    # compute mask
    mask = make_patch_mask(
        trial["positions_x"], trial["positions_y"],
        bar_len=trial.get("bar_length", 12.0)
    ).reshape(GRID, GRID)

    # draw grid and highlight bar patches
    for row in range(GRID):
        for col in range(GRID):
            x = col * PATCH_PX - 0.5
            y = row * PATCH_PX - 0.5

            if mask[row, col]:
                # bar patch: red overlay
                rect = mpatches.Rectangle(
                    (x, y), PATCH_PX, PATCH_PX,
                    linewidth=0.5, edgecolor="red",
                    facecolor="red", alpha=0.3
                )
            else:
                # background patch: just grid lines
                rect = mpatches.Rectangle(
                    (x, y), PATCH_PX, PATCH_PX,
                    linewidth=0.3, edgecolor="gray",
                    facecolor="none", alpha=0.3
                )
            ax.add_patch(rect)

    # mark bar centers
    for bx, by in zip(trial["positions_x"], trial["positions_y"]):
        ax.plot(bx, 224-by, "w+", markersize=6, markeredgewidth=1.5)

    ax.set_title(
        f"n={trial['n_bars']}  μ={trial['mean_orientation']}°  σ={trial['variance']}°",
        fontsize=9
    )
    ax.set_xlim(-0.5, 223.5)
    ax.set_ylim(223.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--n_bars", type=int, default=4)
    parser.add_argument("--n_examples", type=int, default=4)
    parser.add_argument("--output", default="verify_mask.png")
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    with open(meta_path) as f:
        meta = json.load(f)

    # filter trials by n_bars
    trials = [t for t in meta["trials"] if t["n_bars"] == args.n_bars]
    if not trials:
        print(f"No trials with n_bars={args.n_bars}")
        return

    # pick a few examples spread across conditions
    step = max(1, len(trials) // args.n_examples)
    examples = trials[::step][:args.n_examples]

    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4))
    if len(examples) == 1:
        axes = [axes]

    for ax, trial in zip(axes, examples):
        visualize_mask(trial, meta_path, ax)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
