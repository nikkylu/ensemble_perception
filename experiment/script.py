"""
- Bar size tuned for DINO ViT 14x14 patch grid (224px input)
- Sampling from 15 colors 
- Taking circular mean and variance of realized orientations for metadata
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import itertools
import json
from datetime import datetime
from scipy import stats

# ─────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────

# Orientation conditions
MEANS = [0, 30, 60, 90, 120, 150]       # mean orientations (degrees)
VARIANCES = [5, 10, 15]
# add 1, 2, 3 for comparison                    # circular SD (degrees)
N_BARS_LIST = [1, 2, 3, 4, 8, 16, 32]                     # set sizes to test
N_REPETITIONS = 10                     # stimuli per condition

# Color bank
N_COLORS = 15                                  # colors spanning the hue wheel
COLOR_SATURATION = 0.6                         # HSV saturation
COLOR_VALUE = 0.85                             # HSV value (brightness)
# WCAG contrast ratio threshold (bar vs background)
MIN_CONTRAST_RATIO = 3.0

# Display geometry — tuned for DINO ViT-B/14 (224px, 14x14 patches = 16px/patch)
CANVAS_SIZE = 224          # DINO input resolution
PATCH_SIZE = 14            # 224 / 14 = 16 patches per side
BAR_LENGTH = 11            # pixels — fits within one patch, < 14
BAR_WIDTH = 2            # pixels
MIN_SPACING = 14           # Chebyshev distance: max(|Δx|,|Δy|) >= 1 patch
JITTER_REGION = 0.85       # fraction of canvas used for placement

# Appearance
FIXATION = False           # no fixation cross — keeps center patches clean
FIX_SIZE = 4
FIX_COLOR = (0.3, 0.3, 0.3)
DPI = 72                   # at 72 DPI, 224px canvas = ~3.1 inches

# Output
OUTPUT_DIR = Path("stimuli_v4")
SAVE_METADATA = True

# Random seed
SEED = 42


# ─────────────────────────────────────────────────────────────────
# COLOR BANK
# ─────────────────────────────────────────────────────────────────

def build_color_bank(n: int, saturation: float = 0.8, value: float = 0.85) -> list[tuple]:
    """
    Generate N maximally-spaced colors around the HSV hue wheel.
    Returns list of RGB tuples (0-1 range).
    """
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = []
    for h in hues:
        rgb = mcolors.hsv_to_rgb([h, saturation, value])
        colors.append(tuple(rgb))
    return colors


def relative_luminance(rgb: tuple) -> float:
    """
    Compute WCAG relative luminance from an sRGB tuple (values in [0, 1]).
    https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
    """
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = [linearize(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(color_a: tuple, color_b: tuple) -> float:
    """WCAG contrast ratio between two sRGB colors. Range: [1, 21]."""
    la = relative_luminance(color_a)
    lb = relative_luminance(color_b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def pick_bar_bg_colors(color_bank: list, rng: np.random.Generator,
                       min_contrast: float = MIN_CONTRAST_RATIO) -> tuple:
    """
    Randomly pick bar color and background color from the bank.
    Guarantees they are different colors and meet the minimum contrast ratio.
    Raises ValueError if no valid pair exists in the bank.
    """
    n = len(color_bank)
    # Build list of all valid pairs, then pick one at random
    valid_pairs = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and contrast_ratio(color_bank[i], color_bank[j]) >= min_contrast
    ]
    if not valid_pairs:
        raise ValueError(
            f"No color pair in the bank meets contrast ratio >= {min_contrast}. "
            "Try increasing N_COLORS, adjusting saturation/value, or lowering MIN_CONTRAST_RATIO."
        )
    idx = rng.integers(len(valid_pairs))
    bar_idx, bg_idx = valid_pairs[idx]
    return color_bank[bar_idx], color_bank[bg_idx], bar_idx, bg_idx


# ─────────────────────────────────────────────────────────────────
# ORIENTATION SAMPLING
# ─────────────────────────────────────────────────────────────────

def sample_orientations(mean_deg: float, sd_deg: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from von Mises in doubled-angle orientation space.
    Returns degrees in [0, 180).
    """
    mu_rad = np.deg2rad(2 * mean_deg)
    sd_rad = np.deg2rad(2 * sd_deg)

    if sd_rad < 1e-6:
        return np.full(n, mean_deg % 180)

    kappa = 1.0 / (sd_rad ** 2)
    kappa = min(kappa, 1e4)

    samples_rad = rng.vonmises(mu_rad, kappa, size=n)
    samples_deg = np.rad2deg(samples_rad) / 2.0
    samples_deg = samples_deg % 180
    return samples_deg


def place_bars(n: int, canvas_size: int, min_spacing: float,
               jitter_frac: float, rng: np.random.Generator,
               max_attempts: int = 5000) -> np.ndarray:
    """
    Place bar centers on patch centers, ensuring no two share a patch.
    Each bar lands exactly at the center of a patch cell.
    """
    patch_px = 14
    grid = canvas_size // patch_px  # 16

    # margin in patch units — skip edge patches
    margin_patches = int(np.ceil(grid * (1 - jitter_frac) / 2))
    lo = margin_patches
    hi = grid - margin_patches  # exclusive

    # available patch indices
    available = [(r, c) for r in range(lo, hi) for c in range(lo, hi)]
    rng.shuffle(available)

    chosen = []
    for r, c in available:
        if len(chosen) >= n:
            break
        # ensure no adjacent patch is already taken (Chebyshev >= 1 patch)
        if all(max(abs(r - cr), abs(c - cc)) >= 1 for cr, cc in chosen):
            chosen.append((r, c))

    if len(chosen) < n:
        # fallback: just take random non-overlapping patches
        rng.shuffle(available)
        chosen = list(available[:n])

    # convert patch indices to pixel coordinates (center of each patch)
    positions = np.array([
        [(c + 0.5) * patch_px, (r + 0.5) * patch_px]
        for r, c in chosen
    ])

    return positions


# ─────────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────────

def render_stimulus(positions: np.ndarray, orientations: np.ndarray,
                    canvas_size: int, bar_length: float, bar_width: float,
                    bar_color: tuple, bg_color: tuple,
                    save_path: Path = None, show: bool = False) -> plt.Figure:
    """Render a single stimulus display."""
    fig_inches = canvas_size / DPI
    fig, ax = plt.subplots(1, 1, figsize=(fig_inches, fig_inches), dpi=DPI)
    ax.set_xlim(0, canvas_size)
    ax.set_ylim(0, canvas_size)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    half_len = bar_length / 2.0

    for (x, y), theta_deg in zip(positions, orientations):
        theta_rad = np.deg2rad(theta_deg)
        dx = half_len * np.cos(theta_rad)
        dy = half_len * np.sin(theta_rad)
        ax.plot([x - dx, x + dx], [y - dy, y + dy],
                color=bar_color, linewidth=bar_width, solid_capstyle="butt")

    if FIXATION:
        cx, cy = canvas_size / 2, canvas_size / 2
        ax.plot([cx - FIX_SIZE, cx + FIX_SIZE], [cy, cy],
                color=FIX_COLOR, linewidth=1)
        ax.plot([cx, cx], [cy - FIX_SIZE, cy + FIX_SIZE],
                color=FIX_COLOR, linewidth=1)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, facecolor=fig.get_facecolor(),
                    pad_inches=0)

    if not show:
        plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────
# DECORRELATION DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────

def run_decorrelation_check(metadata: list, color_bank: list):
    """
    Check that color assignments are decorrelated from orientation conditions.

    Tests:
    1. Chi-squared test: bar_color_idx x condition (joint independence)
    2. Chi-squared test: bg_color_idx x condition
    3. Point-biserial / eta-squared: mean hue vs mean_orientation
    4. Point-biserial / eta-squared: mean hue vs variance
    5. Summary stats: color usage frequency per condition
    """
    print("\n" + "=" * 60)
    print("DECORRELATION DIAGNOSTICS")
    print("=" * 60)

    n_trials = len(metadata)
    bar_indices = np.array([t["bar_color_idx"] for t in metadata])
    bg_indices = np.array([t["bg_color_idx"] for t in metadata])
    mean_oris = np.array([t["mean_orientation"] for t in metadata])
    variances = np.array([t["variance"] for t in metadata])

    # Encode condition as a single label
    conditions = [
        f"m{t['mean_orientation']}_v{t['variance']}" for t in metadata]
    cond_labels, cond_codes = np.unique(conditions, return_inverse=True)
    n_conditions = len(cond_labels)

    # ── 1. Chi-squared: bar color x condition ──
    print(f"\n1. Bar color index × condition (chi-squared)")
    contingency_bar = np.zeros((N_COLORS, n_conditions), dtype=int)
    for ci, cond in zip(cond_codes, bar_indices):
        contingency_bar[cond, ci] += 1
    # Remove empty rows/cols for chi2
    mask_rows = contingency_bar.sum(axis=1) > 0
    mask_cols = contingency_bar.sum(axis=0) > 0
    table = contingency_bar[mask_rows][:, mask_cols]
    if table.shape[0] > 1 and table.shape[1] > 1:
        chi2, p, dof, _ = stats.chi2_contingency(table)
        print(f"   chi2 = {chi2:.2f}, dof = {dof}, p = {p:.4f}")
        print(f"   {'✓ NOT significant (good — no association)' if p > 0.05 else '⚠ SIGNIFICANT — color and condition are associated!'}")
    else:
        print("   (table too sparse for chi2)")

    # ── 2. Chi-squared: bg color x condition ──
    print(f"\n2. Background color index × condition (chi-squared)")
    contingency_bg = np.zeros((N_COLORS, n_conditions), dtype=int)
    for ci, cond in zip(cond_codes, bg_indices):
        contingency_bg[cond, ci] += 1
    mask_rows = contingency_bg.sum(axis=1) > 0
    mask_cols = contingency_bg.sum(axis=0) > 0
    table = contingency_bg[mask_rows][:, mask_cols]
    if table.shape[0] > 1 and table.shape[1] > 1:
        chi2, p, dof, _ = stats.chi2_contingency(table)
        print(f"   chi2 = {chi2:.2f}, dof = {dof}, p = {p:.4f}")
        print(f"   {'✓ NOT significant (good)' if p > 0.05 else '⚠ SIGNIFICANT'}")
    else:
        print("   (table too sparse for chi2)")

    # ── 3. Kruskal-Wallis: bar hue across mean orientations ──
    print(f"\n3. Bar color hue × mean orientation (Kruskal-Wallis)")
    # hue ∈ [0,1)
    bar_hues = np.array([bar_indices[i] / N_COLORS for i in range(n_trials)])
    groups_by_mean = [bar_hues[mean_oris == m] for m in MEANS]
    if all(len(g) > 0 for g in groups_by_mean):
        h_stat, p = stats.kruskal(*groups_by_mean)
        eta2 = (h_stat - len(MEANS) + 1) / (n_trials - len(MEANS))
        print(f"   H = {h_stat:.2f}, p = {p:.4f}, η² ≈ {eta2:.4f}")
        print(f"   {'✓ NOT significant (good)' if p > 0.05 else '⚠ SIGNIFICANT'}")

    # ── 4. Kruskal-Wallis: bar hue across variances ──
    print(f"\n4. Bar color hue × variance (Kruskal-Wallis)")
    groups_by_var = [bar_hues[variances == v] for v in VARIANCES]
    if all(len(g) > 0 for g in groups_by_var):
        h_stat, p = stats.kruskal(*groups_by_var)
        eta2 = (h_stat - len(VARIANCES) + 1) / (n_trials - len(VARIANCES))
        print(f"   H = {h_stat:.2f}, p = {p:.4f}, η² ≈ {eta2:.4f}")
        print(f"   {'✓ NOT significant (good)' if p > 0.05 else '⚠ SIGNIFICANT'}")

    # ── 5. Color usage frequencies ──
    print(f"\n5. Color usage summary")
    bar_counts = np.bincount(bar_indices, minlength=N_COLORS)
    bg_counts = np.bincount(bg_indices, minlength=N_COLORS)
    expected = n_trials / N_COLORS

    print(f"   Expected per color (uniform): {expected:.1f}")
    print(f"   Bar color counts:  min={bar_counts.min()}, max={bar_counts.max()}, "
          f"std={bar_counts.std():.1f}")
    print(f"   BG  color counts:  min={bg_counts.min()}, max={bg_counts.max()}, "
          f"std={bg_counts.std():.1f}")

    # ── 6. Pair reuse check ──
    pairs = [(t["bar_color_idx"], t["bg_color_idx"]) for t in metadata]
    unique_pairs = set(pairs)
    max_possible = N_COLORS * (N_COLORS - 1)
    print(f"\n6. Color pair diversity")
    print(
        f"   Unique (bar, bg) pairs used: {len(unique_pairs)} / {max_possible} possible")
    pair_counts = {}
    for p in pairs:
        pair_counts[p] = pair_counts.get(p, 0) + 1
    most_common = max(pair_counts.values())
    print(
        f"   Most-reused pair appears {most_common}x across {n_trials} trials")

    print("\n" + "=" * 60)

    return {
        "unique_pairs": len(unique_pairs),
        "max_possible_pairs": max_possible,
    }


# ─────────────────────────────────────────────────────────────────
# EXAMPLE GRID
# ─────────────────────────────────────────────────────────────────

def render_example_grid(n_bars: int, color_bank: list, rng: np.random.Generator):
    """Render grid showing one sample per condition with random colors for a given set size."""
    n_means = len(MEANS)
    n_vars = len(VARIANCES)

    fig, axes = plt.subplots(n_vars, n_means, figsize=(
        2.2 * n_means, 2.2 * n_vars), dpi=150)
    if n_vars == 1:
        axes = axes[np.newaxis, :]
    if n_means == 1:
        axes = axes[:, np.newaxis]

    for i, var_ori in enumerate(VARIANCES):
        for j, mean_ori in enumerate(MEANS):
            ax = axes[i, j]
            bar_color, bg_color, _, _ = pick_bar_bg_colors(color_bank, rng)
            orientations = sample_orientations(mean_ori, var_ori, n_bars, rng)
            positions = place_bars(n_bars, CANVAS_SIZE,
                                   MIN_SPACING, JITTER_REGION, rng)

            ax.set_xlim(0, CANVAS_SIZE)
            ax.set_ylim(0, CANVAS_SIZE)
            ax.set_aspect("equal")
            ax.set_facecolor(bg_color)
            ax.set_xticks([])
            ax.set_yticks([])

            half_len = BAR_LENGTH / 2.0
            for (x, y), theta_deg in zip(positions, orientations):
                theta_rad = np.deg2rad(theta_deg)
                dx = half_len * np.cos(theta_rad)
                dy = half_len * np.sin(theta_rad)
                ax.plot([x - dx, x + dx], [y - dy, y + dy],
                        color=bar_color, linewidth=BAR_WIDTH * 0.8,
                        solid_capstyle="round")

            if i == 0:
                ax.set_title(f"μ={mean_ori}°", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"σ={var_ori}°", fontsize=8)

    fig.suptitle(f"N={n_bars} bars — {N_COLORS} colors, 224px",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    grid_path = OUTPUT_DIR / f"example_grid_n{n_bars:02d}.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Example grid (N={n_bars}) → {grid_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def generate_all():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build color bank
    color_bank = build_color_bank(N_COLORS, COLOR_SATURATION, COLOR_VALUE)

    print(f"Color bank ({N_COLORS} colors):")
    for i, c in enumerate(color_bank):
        hex_color = mcolors.to_hex(c)
        print(f"  [{i:2d}] {hex_color}  RGB=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")

    metadata = []
    conditions = list(itertools.product(N_BARS_LIST, MEANS, VARIANCES))
    total = len(conditions) * N_REPETITIONS
    print(f"\nGenerating {total} stimuli across {len(conditions)} conditions")
    print(f"  N_bars:     {N_BARS_LIST}")
    print(f"  Means:      {MEANS}")
    print(f"  Variances:  {VARIANCES}")
    print(f"  Reps/cond:  {N_REPETITIONS}")
    print(f"  Canvas:     {CANVAS_SIZE}px (DINO patch: {PATCH_SIZE}px)")
    print(f"  Bar size:   {BAR_LENGTH}px length, {BAR_WIDTH}px width")
    print(f"  Output:     {OUTPUT_DIR.resolve()}\n")

    count = 0
    for n_bars, mean_ori, var_ori in conditions:
        cond_dir = OUTPUT_DIR / \
            f"nbars{n_bars:02d}" / f"mean{mean_ori:03d}_var{var_ori:02d}"

        for rep in range(N_REPETITIONS):
            # Random color pair for this image
            bar_color, bg_color, bar_idx, bg_idx = pick_bar_bg_colors(
                color_bank, rng)

            orientations = sample_orientations(mean_ori, var_ori, n_bars, rng)
            positions = place_bars(n_bars, CANVAS_SIZE,
                                   MIN_SPACING, JITTER_REGION, rng)

            fname = f"stim_n{n_bars:02d}_mean{mean_ori:03d}_var{var_ori:02d}_rep{rep:02d}.png"
            save_path = cond_dir / fname

            render_stimulus(positions, orientations, CANVAS_SIZE,
                            BAR_LENGTH, BAR_WIDTH,
                            bar_color=bar_color, bg_color=bg_color,
                            save_path=save_path)

            # Compute realized circular statistics
            doubled = np.deg2rad(2 * orientations)
            resultant = np.mean(np.exp(1j * doubled))
            actual_mean = (np.rad2deg(np.angle(resultant)) / 2) % 180
            actual_csd = np.rad2deg(
                np.sqrt(-2 * np.log(np.abs(resultant)))) / 2

            trial_meta = {
                "filename": str(save_path),
                "n_bars": n_bars,
                "mean_orientation": float(mean_ori),
                "variance": float(var_ori),
                "repetition": rep,
                "bar_color_idx": bar_idx,
                "bg_color_idx": bg_idx,
                "bar_color_rgb": list(bar_color),
                "bg_color_rgb": list(bg_color),
                "orientations": orientations.tolist(),
                "positions_x": positions[:, 0].tolist(),
                "positions_y": positions[:, 1].tolist(),
                "actual_circular_mean": float(actual_mean),
                "actual_circular_sd": float(actual_csd),
            }
            metadata.append(trial_meta)

            count += 1
            if count % 50 == 0 or count == total:
                print(f"  [{count}/{total}]")

    # Save metadata
    if SAVE_METADATA:
        color_bank_serialized = [
            {"index": i, "rgb": list(c), "hex": mcolors.to_hex(c)}
            for i, c in enumerate(color_bank)
        ]
        meta_path = OUTPUT_DIR / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "generated": datetime.now().isoformat(),
                "params": {
                    "means": MEANS,
                    "variances": VARIANCES,
                    "n_bars_list": N_BARS_LIST,
                    "n_repetitions": N_REPETITIONS,
                    "canvas_size": CANVAS_SIZE,
                    "patch_size": PATCH_SIZE,
                    "bar_length": BAR_LENGTH,
                    "bar_width": BAR_WIDTH,
                    "min_spacing": MIN_SPACING,
                    "n_colors": N_COLORS,
                    "seed": SEED,
                },
                "color_bank": color_bank_serialized,
                "trials": metadata,
            }, f, indent=2)
        print(f"\nMetadata → {meta_path}")

    # Decorrelation check (per set size)
    for n_bars in N_BARS_LIST:
        subset = [t for t in metadata if t["n_bars"] == n_bars]
        print(f"\n>>> N_BARS = {n_bars}")
        run_decorrelation_check(subset, color_bank)

    # Example grid (one per set size)
    print("\nRendering example grids...")
    for n_bars in N_BARS_LIST:
        render_example_grid(n_bars, color_bank,
                            np.random.default_rng(SEED + 999 + n_bars))

    print(f"\nDone. {count} stimuli in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    generate_all()
