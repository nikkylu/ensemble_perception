"""
Verify the loader's patch masks against the rendered pixels.

--check (default): for every trial, the non-background pixels must lie
only inside masked patches, every masked patch must contain ink, and the
number of masked patches must equal the set size — i.e. one element per
patch and no element crossing a patch boundary.

--plot: overlay the 16x16 grid and masked patches on a few stimuli.

Usage:
    python3 verify_mask.py --metadata stimuli_ori_s42/metadata.json
    python3 verify_mask.py --metadata stimuli_ori_s42/metadata.json --plot --n_bars 16
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from loader import GRID, N_KEYS, PATCH_PX, StimuliDataset, get, make_patch_mask


def load(ds, t):
    img = np.asarray(Image.open(ds.root / t["filename"]).convert("RGB")).astype(int)
    mask = make_patch_mask(t["positions_x"], t["positions_y"],
                           ds.extent, ds.center_only, ds.block).reshape(GRID, GRID)
    return img, mask


def check(ds):
    bad = 0
    B, G = ds.block, GRID // ds.block
    for t in ds.trials:
        img, mask = load(ds, t)
        colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
        bg = colors[counts.argmax()]                              # most common color
        ink = np.abs(img - bg).sum(-1) > 0
        ink_patches = ink.reshape(GRID, PATCH_PX, GRID, PATCH_PX).any((1, 3))
        # cell granularity: with BLOCK×BLOCK cells a small element need not
        # ink every patch of its cell, but every masked CELL must contain ink
        ink_cells = ink_patches.reshape(G, B, G, B).any((1, 3))
        mask_cells = mask.reshape(G, B, G, B).any((1, 3))
        n = get(t, N_KEYS)
        errors = []
        if mask.sum() != n * B * B:
            errors.append(f"{mask.sum()} masked patches for {n} elements × {B * B}")
        if (ink_patches & ~mask).any():
            errors.append(f"ink outside mask in {int((ink_patches & ~mask).sum())} patches")
        if (mask_cells & ~ink_cells).any():
            errors.append(f"{int((mask_cells & ~ink_cells).sum())} masked cells without ink")
        if errors:
            bad += 1
            if bad <= 10:
                print(f"  FAIL {t['filename']}: {'; '.join(errors)}")
    print(f"{len(ds.trials) - bad}/{len(ds.trials)} trials pass "
          f"(one element per {B}×{B}-patch cell, no element crosses its cell)")
    return bad == 0


def plot(ds, n_examples, out):
    step = max(1, len(ds.trials) // n_examples)
    trials = ds.trials[::step][:n_examples]
    fig, axes = plt.subplots(1, len(trials), figsize=(4 * len(trials), 4), squeeze=False)
    for ax, t in zip(axes[0], trials):
        img, mask = load(ds, t)
        ax.imshow(img.astype(np.uint8))
        for r in range(GRID):
            for c in range(GRID):
                ax.add_patch(mpatches.Rectangle(
                    (c * PATCH_PX - 0.5, r * PATCH_PX - 0.5), PATCH_PX, PATCH_PX,
                    linewidth=0.4, edgecolor="red" if mask[r, c] else "gray",
                    facecolor="red" if mask[r, c] else "none", alpha=0.3))
        ax.set_title(f"N={get(t, N_KEYS)}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--n_bars", type=int, default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--n_examples", type=int, default=4)
    p.add_argument("--output", default="verify_mask.png")
    a = p.parse_args()

    ds = StimuliDataset(a.metadata, a.n_bars)
    if a.plot:
        plot(ds, a.n_examples, a.output)
    else:
        raise SystemExit(0 if check(ds) else 1)


if __name__ == "__main__":
    main()
