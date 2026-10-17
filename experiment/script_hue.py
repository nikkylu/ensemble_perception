"""
Hue ensemble stimuli generator.

Each stimulus contains N circles placed one-per-patch on a 224px canvas.
Circle hues are sampled from a von Mises distribution (circular) in HSV
space around a mean hue, with fixed saturation and value.
Background is picked at random from black / gray / white (achromatic).

Ground truth stored per trial:
  - mean_hue, variance_hue       : condition labels (degrees, 0-360)
  - actual_circular_mean_hue     : realized circular mean of drawn hues (degrees)
  - actual_circular_sd_hue       : realized circular SD of drawn hues (degrees)
  - hues                         : individual hue values per circle (degrees)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
import itertools
import json
from datetime import datetime
from scipy import stats

# ─────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────

# Hue conditions (degrees, 0–360, full circle)
HUE_MEANS = [0, 60, 120, 180, 240, 300]   # mean hues (degrees)
# circular SD in hue space (degrees)
HUE_VARIANCES = [10, 20, 30]
N_CIRCLES_LIST = [1, 2, 3, 4, 8, 16, 32]      # set sizes
N_REPETITIONS = 10                            # stimuli per condition

# Circle appearance — fixed S and V, only H varies
CIRCLE_SATURATION = 0.85
CIRCLE_VALUE = 0.90
# fits within 14px patch (diameter=10, margin=2px each side)
CIRCLE_RADIUS_PX = 5

# Background colors — achromatic (S=0), so the background carries no hue
BG_NAMES = ["black", "gray", "white"]
BG_COLORS = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)]

# Display geometry
CANVAS_SIZE = 224
PATCH_PX = 14
JITTER_REGION = 0.85

# Appearance
FIXATION = False           # no fixation cross — keeps center patches clean
FIX_SIZE = 4
DPI = 72

# Output
OUTPUT_DIR = Path("stimuli_hue_v2")
SAVE_METADATA = True
SEED = 42


# ─────────────────────────────────────────────────────────────────
# COLOR UTILITIES
# ─────────────────────────────────────────────────────────────────

def hue_to_rgb(hue_deg: float, saturation: float, value: float) -> tuple:
    """Convert HSV (hue in degrees 0-360) to RGB tuple."""
    h = (hue_deg % 360) / 360.0
    return tuple(mcolors.hsv_to_rgb([h, saturation, value]))


def pick_bg_color(rng: np.random.Generator) -> tuple:
    """
    Pick a random achromatic background independently of circle hues.
    No conditioning on mean hue — avoids systematic color-condition correlation.
    """
    idx = int(rng.integers(len(BG_COLORS)))
    return BG_COLORS[idx], idx


# ─────────────────────────────────────────────────────────────────
# HUE SAMPLING
# ─────────────────────────────────────────────────────────────────

def sample_hues(mean_hue_deg: float, sd_deg: float, n: int,
                rng: np.random.Generator) -> np.ndarray:
    """
    Sample hues from von Mises distribution in HSV hue space (0–360°, full circle).
    Returns hue values in degrees [0, 360).
    """
    mu_rad = np.deg2rad(mean_hue_deg)
    sd_rad = np.deg2rad(sd_deg)

    if sd_rad < 1e-6:
        return np.full(n, mean_hue_deg % 360)

    kappa = 1.0 / (sd_rad ** 2)
    kappa = min(kappa, 1e4)

    samples_rad = rng.vonmises(mu_rad, kappa, size=n)
    samples_deg = np.rad2deg(samples_rad) % 360
    return samples_deg


def circular_mean_sd_hue(hues_deg: np.ndarray):
    """
    Circular mean and circular SD of hues (degrees, full 360° circle).
    Returns (mean_deg, sd_deg).
    """
    radians = np.deg2rad(hues_deg)
    resultant = np.mean(np.exp(1j * radians))
    mean_deg = np.rad2deg(np.angle(resultant)) % 360
    r = np.abs(resultant)
    sd_deg = np.rad2deg(np.sqrt(-2 * np.log(np.clip(r, 1e-10, 1.0))))
    return float(mean_deg), float(sd_deg)


# ─────────────────────────────────────────────────────────────────
# PLACEMENT
# ─────────────────────────────────────────────────────────────────

def place_circles(n: int, canvas_size: int, jitter_frac: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Place circle centers at patch centers, one per patch, no adjacent patches.
    Returns (n, 2) array of (x, y) pixel coordinates.
    """
    grid = canvas_size // PATCH_PX
    margin_patches = int(np.ceil(grid * (1 - jitter_frac) / 2))
    lo, hi = margin_patches, grid - margin_patches

    available = [(r, c) for r in range(lo, hi) for c in range(lo, hi)]
    rng.shuffle(available)

    chosen = []
    for r, c in available:
        if len(chosen) >= n:
            break
        if all(max(abs(r - cr), abs(c - cc)) >= 1 for cr, cc in chosen):
            chosen.append((r, c))

    if len(chosen) < n:
        rng.shuffle(available)
        chosen = list(available[:n])

    positions = np.array([
        [(c + 0.5) * PATCH_PX, (r + 0.5) * PATCH_PX]
        for r, c in chosen
    ])
    return positions


# ─────────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────────

def render_stimulus(positions: np.ndarray, hues_deg: np.ndarray,
                    bg_color: tuple, canvas_size: int,
                    save_path: Path = None) -> plt.Figure:
    """Render one hue stimulus: circles with individual hue colors on bg."""
    fig_inches = canvas_size / DPI
    fig, ax = plt.subplots(1, 1, figsize=(fig_inches, fig_inches), dpi=DPI)
    ax.set_xlim(0, canvas_size)
    ax.set_ylim(canvas_size, 0)   # top-down (image coords)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    for (x, y), hue in zip(positions, hues_deg):
        # y is in matplotlib coords (bottom-up); convert to image coords (top-down)
        img_y = canvas_size - y
        color = hue_to_rgb(hue, CIRCLE_SATURATION, CIRCLE_VALUE)
        circle = mpatches.Circle((x, img_y), radius=CIRCLE_RADIUS_PX,
                                 color=color, antialiased=True)
        ax.add_patch(circle)

    if FIXATION:
        # keep the cross visible on any achromatic background
        fix_color = (0.75,) * 3 if np.mean(bg_color) < 0.25 else (0.25,) * 3
        cx, cy = canvas_size / 2, canvas_size / 2
        ax.plot([cx - FIX_SIZE, cx + FIX_SIZE], [cy, cy],
                color=fix_color, linewidth=1)
        ax.plot([cx, cx], [cy - FIX_SIZE, cy + FIX_SIZE],
                color=fix_color, linewidth=1)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, facecolor=fig.get_facecolor(),
                    pad_inches=0)

    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────
# EXAMPLE GRID
# ─────────────────────────────────────────────────────────────────

def render_example_grid(n_circles: int, rng: np.random.Generator):
    n_means = len(HUE_MEANS)
    n_vars = len(HUE_VARIANCES)
    fig, axes = plt.subplots(n_vars, n_means,
                             figsize=(2.2 * n_means, 2.2 * n_vars), dpi=150)
    if n_vars == 1:
        axes = axes[np.newaxis, :]
    if n_means == 1:
        axes = axes[:, np.newaxis]

    for i, var_hue in enumerate(HUE_VARIANCES):
        for j, mean_hue in enumerate(HUE_MEANS):
            ax = axes[i, j]
            hues = sample_hues(mean_hue, var_hue, n_circles, rng)
            positions = place_circles(
                n_circles, CANVAS_SIZE, JITTER_REGION, rng)
            bg_color, _ = pick_bg_color(rng)

            ax.set_xlim(0, CANVAS_SIZE)
            ax.set_ylim(CANVAS_SIZE, 0)
            ax.set_aspect("equal")
            ax.set_facecolor(bg_color)
            ax.set_xticks([])
            ax.set_yticks([])

            for (x, y), hue in zip(positions, hues):
                img_y = CANVAS_SIZE - y
                color = hue_to_rgb(hue, CIRCLE_SATURATION, CIRCLE_VALUE)
                circle = mpatches.Circle((x, img_y), radius=CIRCLE_RADIUS_PX,
                                         color=color, antialiased=True)
                ax.add_patch(circle)

            if i == 0:
                ax.set_title(f"μ={mean_hue}°", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"σ={var_hue}°", fontsize=8)

    fig.suptitle(f"N={n_circles} circles — hue stimuli", fontsize=10, y=1.01)
    fig.supxlabel("Mean hue μ (°)", fontsize=9)
    fig.supylabel("Hue circular SD σ (°)", fontsize=9)
    plt.tight_layout()
    grid_path = OUTPUT_DIR / f"example_grid_n{n_circles:02d}.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Example grid (N={n_circles}) → {grid_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def generate_all():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Hue stimuli generator")
    print(f"  HUE_MEANS:      {HUE_MEANS}")
    print(f"  HUE_VARIANCES:  {HUE_VARIANCES}")
    print(f"  N_CIRCLES_LIST: {N_CIRCLES_LIST}")
    print(f"  N_REPETITIONS:  {N_REPETITIONS}")
    print(
        f"  Circle S={CIRCLE_SATURATION}, V={CIRCLE_VALUE}, r={CIRCLE_RADIUS_PX}px")
    print(f"  Canvas: {CANVAS_SIZE}px, patch: {PATCH_PX}px")
    print(f"  Output: {OUTPUT_DIR.resolve()}\n")

    metadata = []
    conditions = list(itertools.product(
        N_CIRCLES_LIST, HUE_MEANS, HUE_VARIANCES))
    total = len(conditions) * N_REPETITIONS
    print(f"Generating {total} stimuli across {len(conditions)} conditions\n")

    count = 0
    for n_circles, mean_hue, var_hue in conditions:
        cond_dir = OUTPUT_DIR / f"ncircles{n_circles:02d}" / \
            f"mean{mean_hue:03d}_var{var_hue:03d}"

        for rep in range(N_REPETITIONS):
            hues = sample_hues(mean_hue, var_hue, n_circles, rng)
            positions = place_circles(
                n_circles, CANVAS_SIZE, JITTER_REGION, rng)

            bg_color, bg_idx = pick_bg_color(rng)

            fname = (f"stim_n{n_circles:02d}_mean{mean_hue:03d}"
                     f"_var{var_hue:03d}_rep{rep:02d}.png")
            save_path = cond_dir / fname

            render_stimulus(positions, hues, bg_color, CANVAS_SIZE,
                            save_path=save_path)

            # ground truth circular statistics
            actual_mean, actual_sd = circular_mean_sd_hue(hues)

            trial_meta = {
                "filename":              str(save_path),
                "n_circles":             n_circles,
                "mean_hue":              float(mean_hue),
                "variance_hue":          float(var_hue),
                "repetition":            rep,
                "bg_color_idx":          int(bg_idx),
                "bg_color_name":         BG_NAMES[bg_idx],
                "bg_color_rgb":          list(bg_color),
                "hues":                  hues.tolist(),
                "positions_x":           positions[:, 0].tolist(),
                "positions_y":           positions[:, 1].tolist(),
                "actual_circular_mean_hue": actual_mean,
                "actual_circular_sd_hue":   actual_sd,
            }
            metadata.append(trial_meta)

            count += 1
            if count % 50 == 0 or count == total:
                print(f"  [{count}/{total}]")

    if SAVE_METADATA:
        bg_bank_serialized = [
            {"index": i, "name": BG_NAMES[i], "rgb": list(c),
             "hex": mcolors.to_hex(c)}
            for i, c in enumerate(BG_COLORS)
        ]
        meta_path = OUTPUT_DIR / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "generated": datetime.now().isoformat(),
                "params": {
                    "hue_means":       HUE_MEANS,
                    "hue_variances":   HUE_VARIANCES,
                    "n_circles_list":  N_CIRCLES_LIST,
                    "n_repetitions":   N_REPETITIONS,
                    "canvas_size":     CANVAS_SIZE,
                    "patch_px":        PATCH_PX,
                    "circle_radius_px": CIRCLE_RADIUS_PX,
                    "circle_saturation": CIRCLE_SATURATION,
                    "circle_value":    CIRCLE_VALUE,
                    "jitter_region":   JITTER_REGION,
                    "bg_colors":       BG_NAMES,
                    "seed":            SEED,
                },
                "bg_bank": bg_bank_serialized,
                "trials":  metadata,
            }, f, indent=2)
        print(f"\nMetadata → {meta_path}")

    # Example grids
    print("\nRendering example grids...")
    for n_circles in N_CIRCLES_LIST:
        render_example_grid(n_circles,
                            np.random.default_rng(SEED + 999 + n_circles))

    print(f"\nDone. {count} stimuli in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    generate_all()
