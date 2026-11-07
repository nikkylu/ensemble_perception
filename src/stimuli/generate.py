"""
Seed-based ensemble stimulus generators: orientation (bars), hue (colored
shapes) and size (shapes of varying size).

Each generator is an object holding a seed; every trial is a pure function
of (seed, domain, set size, mean, sd, rep), so a generator is an infinite,
reproducible stimulus stream:

    gen = OrientationGenerator(seed=42)
    trial = gen.trial(n=16, mean=30, sd=10, rep=0)      # one stimulus
    gen.generate("stimuli_ori_s42", n_reps=50)          # write a dataset

Rerunning with a larger n_reps reproduces the first reps byte-identically.

Discount mode (Avci & Boduroglu, 2021, APP 83:1141): each display has N-1
group elements plus one outlier whose value lies beyond the group's range
edge by a "less" (ori 15±3°) or "more" (ori 90±3°) distinct amount. As in the
paper, the group's range is capped (ori 30°), so the outlier is always outside
the group; discount mode therefore uses the smallest SD level only. By default
the outlier is spatially separated (group in one quadrant, outlier in the
diagonally opposite one). Metadata keeps local (group) and global (all)
realized statistics — global both circular and linear (the outlier unwrapped
on its side, as the paper's arithmetic mean) — to test discounting:

    trial = gen.trial(8, 30, 5, 0, outlier=("more", "spatial"))
    gen.generate("stimuli_ori_discount_s42", n_reps=50, mode="discount")

Layout (shared by all domains, tuned for DINOv2 ViT-B/14 at 224px):
- elements sit on a grid of BLOCK×BLOCK-patch cells, at most one element
  per cell (ori/hue: BLOCK=1 → 16×16 patches; size v2: BLOCK=2 → 8×8
  cells of 28px, so elements are large enough to carry size information);
- task-irrelevant jitter: element size per element, and center jitter
  bounded so each element stays inside its own cell (the loader's mask is
  then exact: one BLOCK×BLOCK cell per element).

CLI (from experiment/):
    python3 stimuli/generate.py ori --seed 42 --n_reps 10     # → stimuli_ori_s42/
    python3 stimuli/generate.py hue --seed 42 --set_sizes 16 --out my_dir
    python3 stimuli/generate.py size --seed 42
    python3 stimuli/generate.py ori --mode discount --set_sizes 8
"""

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy import stats

CANVAS = 224               # DINOv2 input resolution
PATCH = 14                 # ViT-B/14 patch → 16x16 grid
GRID = CANVAS // PATCH
POSITION_MARGIN = 1.0      # px kept between element and patch edge (antialiasing
                           # of sharp polygon corners can bleed ~0.5 px)
DPI = 72                   # 224px canvas = 224/72 in
SET_SIZES = [1, 2, 3, 4, 8, 16, 32]


def circular_stats(angles_deg, period):
    """Circular mean and SD of angles on a circle of the given period."""
    scale = 360.0 / period
    R = np.mean(np.exp(1j * np.deg2rad(np.asarray(angles_deg) * scale)))
    mean = (np.rad2deg(np.angle(R)) / scale) % period
    sd = np.rad2deg(np.sqrt(-2 * np.log(np.clip(abs(R), 1e-10, 1.0)))) / scale
    return float(mean), float(sd)


def sample_von_mises(mean_deg, sd_deg, n, rng, period):
    """Von Mises samples on a circle of `period` degrees, sd in that space."""
    scale = 360.0 / period
    kappa = min(1.0 / np.deg2rad(scale * sd_deg) ** 2, 1e4)
    s = rng.vonmises(np.deg2rad(scale * mean_deg), kappa, size=n)
    return (np.rad2deg(s) / scale) % period


# ── shapes: unit-area polygons, so "size" = sqrt(area) is shape-independent ──

def _unit_area(v):
    x, y = v.T
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return v / np.sqrt(area)


def _leaf(aspect=2.0, n=33):
    u = np.linspace(-1, 1, n)
    top = np.c_[u, (1 - u ** 2) / aspect]            # pointed lens
    return np.r_[top, top[-2:0:-1] * [1, -1]]


_t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
_k = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
SHAPES = {
    "circle": _unit_area(np.c_[np.cos(_t), np.sin(_t)]),
    "square": _unit_area(np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)),
    "triangle": _unit_area(np.c_[np.cos(_k), np.sin(_k)]),
    "leaf": _unit_area(_leaf()),
}


def shape_vertices(shape, size, rotation_deg):
    """Polygon of area size**2 rotated by rotation_deg, centered at 0."""
    t = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return size * SHAPES[shape] @ R.T


def max_size(shapes):
    """Largest sqrt(area) at which every shape fits a patch at any rotation."""
    diam = max(np.linalg.norm(SHAPES[s][:, None] - SHAPES[s][None], axis=-1).max()
               for s in shapes)
    return (PATCH - 2 * POSITION_MARGIN) / diam


def _luminance(rgb):
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def color_pairs(n_colors=15, saturation=0.6, value=0.85, min_contrast=3.0):
    """HSV color bank + all ordered (fg, bg) index pairs with WCAG contrast."""
    hues = np.linspace(0, 1, n_colors, endpoint=False)
    colors = [tuple(mcolors.hsv_to_rgb([h, saturation, value])) for h in hues]
    pairs = [(i, j) for i in range(n_colors) for j in range(n_colors)
             if i != j and _contrast(colors[i], colors[j]) >= min_contrast]
    assert pairs, "no color pair meets min_contrast"
    return colors, pairs


def draw_shapes(ax, positions, shapes, sizes, rotations, colors, flip_y=False):
    for (x, y), sh, sz, rot, c in zip(positions, shapes, sizes, rotations, colors):
        v = shape_vertices(sh, sz, rot) + [x, y]
        if flip_y:                                   # axes drawn top-down
            v[:, 1] = CANVAS - v[:, 1]
        ax.add_patch(mpatches.Polygon(v, closed=True, facecolor=c, linewidth=0))


def shape_half_extent(shapes, sizes, rotations):
    """(n, 2) half bounding box of each rotated shape (for patch-bounded jitter)."""
    return np.array([np.abs(shape_vertices(sh, sz, r)).max(0)
                     for sh, sz, r in zip(shapes, sizes, rotations)])


def chi2_independence(codes_a, codes_b, name):
    """Print a chi-squared independence test between two label arrays."""
    _, a = np.unique(codes_a, return_inverse=True)
    _, b = np.unique(codes_b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=int)
    np.add.at(table, (a, b), 1)
    if min(table.shape) < 2:
        print(f"  {name}: (table too small)")
        return
    chi2, p, dof, _ = stats.chi2_contingency(table)
    flag = "ok" if p > 0.05 else "SIGNIFICANT — check confound"
    print(f"  {name} × condition: chi2={chi2:.1f} dof={dof} p={p:.3f}  {flag}")


class EnsembleGenerator:
    """Shared machinery. Subclasses define the domain-specific parts:
    MEANS / SDS / DOMAIN_TAG / NAME / element-specific sampling & drawing."""

    NAME = ""             # "ori" | "hue" | "size"
    ELEMENT = ""          # metadata key of the set size: n_bars / n_circles / n_items
    DOMAIN_TAG = 0        # keeps the domains' RNG streams disjoint
    VERSION = ""
    MEANS: list = []
    SDS: list = []
    YLIM = (0, CANVAS)    # matplotlib y-axis orientation used for drawing
    BORDER = 0            # empty cell rings at the image edge (2 reproduces ori v5)
    BLOCK = 1             # element cell = BLOCK×BLOCK patches (loader masks the cell)
    PERIOD = None         # 180 / 360 for circular statistics; None = linear (log size)
    # discount mode: outlier distance beyond the group's range edge (+ uniform jitter),
    # in the statistic's own units
    OUTLIER_DELTA: dict = {}
    OUTLIER_JITTER = 0.0
    MAX_GROUP_RANGE = np.inf   # discount mode: cap on the group's value range
    DISTINCT = ("less", "more")
    LAYOUTS = ("spatial", "mixed")

    def __init__(self, seed=42):
        self.seed = int(seed)

    # ── randomness ─────────────────────────────────────────────────
    def rng(self, *key):
        """RNG fully determined by (seed, domain, key...)."""
        return np.random.default_rng(
            [self.seed, self.DOMAIN_TAG] + [int(k) for k in key])

    def sds(self, mode="standard"):
        """SD levels; discount mode uses the smallest (group range is capped)."""
        return self.SDS if mode == "standard" else self.SDS[:1]

    def okey(self, outlier):
        """Extra RNG key for discount trials (standard trials keep their stream)."""
        if outlier is None:
            return ()
        distinct, layout = outlier
        return (100 + 10 * self.LAYOUTS.index(layout) + self.DISTINCT.index(distinct),)

    def place(self, n, rng, outlier=None):
        """n distinct cell centers (at most one element per BLOCK×BLOCK cell).
        Spatial-outlier trials: the first n-1 (group) in a random quadrant,
        the last (outlier) in the diagonally opposite quadrant."""
        G, C = GRID // self.BLOCK, PATCH * self.BLOCK
        if outlier is None or outlier[1] == "mixed":
            b = self.BORDER
            cells = [(r, c) for r in range(b, G - b) for c in range(b, G - b)]
            assert n <= len(cells), f"N={n} does not fit the {G}×{G} cell grid"
            rng.shuffle(cells)
            cells = cells[:n]
        else:
            h = G // 2
            qr, qc = rng.integers(2, size=2)
            group = [(qr * h + r, qc * h + c) for r in range(h) for c in range(h)]
            far = [((1 - qr) * h + r, (1 - qc) * h + c) for r in range(h) for c in range(h)]
            assert n - 1 <= len(group), f"N={n} does not fit a quadrant"
            rng.shuffle(group)
            cells = group[:n - 1] + [far[rng.integers(len(far))]]
        return np.array([[(c + 0.5) * C, (r + 0.5) * C]
                         for r, c in cells], dtype=float)

    # ── statistic values (ori/hue: degrees, size: ln px) ───────────
    def sample(self, mean, sd, n, rng):
        raise NotImplementedError

    def values(self, trial):
        raise NotImplementedError

    def stats(self, v):
        """(mean, sd) of values in the statistic's own space."""
        if self.PERIOD:
            return circular_stats(v, self.PERIOD)
        return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    def bounds(self):
        """Allowed value range for linear statistics (outlier must stay inside)."""
        return -np.inf, np.inf

    def stat_mean(self, mean):
        """Condition mean in statistic space (deg; ln px for size)."""
        return float(mean)

    def both_sides_fit(self, mean, distinct):
        """Discount mode: can the outlier go to either side of the group at
        this mean? One-sided cells tie the outlier's direction to the mean —
        a readout-shrinkage bias then mimics outlier weighting — so generate()
        drops them. Circular statistics have no walls: always True."""
        lo, hi = self.bounds()
        d = self.OUTLIER_DELTA[distinct] + self.OUTLIER_JITTER
        return lo + d <= self.stat_mean(mean) <= hi - d

    def deviations(self, v, m):
        """Signed deviations of v from m (wrapped to ±PERIOD/2 if circular)."""
        P = self.PERIOD
        return (v - m + P / 2) % P - P / 2 if P else v - m

    def sample_values(self, mean, sd, n, rng, outlier=None):
        """(values, sign). With an outlier: n-1 group values (range capped at
        MAX_GROUP_RANGE, resampled otherwise) + the outlier last, placed
        OUTLIER_DELTA ± OUTLIER_JITTER beyond the group's edge on side `sign`."""
        if outlier is None:
            return self.sample(mean, sd, n, rng), 0
        assert n >= 3, "discount mode needs a group of at least 2"
        for _ in range(1000):
            group = self.sample(mean, sd, n - 1, rng)
            m = self.stats(group)[0]
            dev = self.deviations(group, m)
            if dev.max() - dev.min() <= self.MAX_GROUP_RANGE:
                break
        else:
            raise RuntimeError(f"group range > {self.MAX_GROUP_RANGE} (sd={sd}, N={n})")
        d = self.OUTLIER_DELTA[outlier[0]] + rng.uniform(-self.OUTLIER_JITTER,
                                                         self.OUTLIER_JITTER)
        lo, hi = self.bounds()
        sides = [sg for sg in (-1, 1)
                 if lo <= m + (dev.max() if sg > 0 else dev.min()) + sg * d <= hi]
        assert sides, "outlier does not fit on either side"
        sign = int(sides[rng.integers(len(sides))])
        out = m + (dev.max() if sign > 0 else dev.min()) + sign * d
        return np.append(group, out % self.PERIOD if self.PERIOD else out), sign

    def outlier_meta(self, t):
        v = self.values(t)
        lm, ls = self.stats(v[:-1])
        gm, gs = self.stats(v)
        # outlier offset from the local mean, unwrapped on the side it was placed
        off = self.deviations(v[-1], lm)
        if self.PERIOD and np.sign(off) != t["outlier_sign"]:
            off += t["outlier_sign"] * self.PERIOD
        lin = lm + (self.deviations(v[:-1], lm).sum() + off) / len(v)
        return {"outlier": {"distinct": t["outlier"][0], "layout": t["outlier"][1],
                            "index": len(v) - 1, "value": float(v[-1]),
                            "offset": float(off)},
                "local_mean": lm, "local_sd": ls, "global_mean": gm, "global_sd": gs,
                "global_mean_linear": float(lin % self.PERIOD if self.PERIOD else lin)}

    def jitter(self, positions, half_extent_xy, rng):
        """Uniform center jitter keeping each element inside its cell.
        half_extent_xy: (n, 2) half bounding-box size of each element."""
        out = positions.copy()
        room = np.maximum(0.0, PATCH * self.BLOCK / 2 - half_extent_xy
                          - POSITION_MARGIN)
        for i, (jx, jy) in enumerate(room):
            out[i, 0] += rng.uniform(-jx, jx)
            out[i, 1] += rng.uniform(-jy, jy)
        return out

    # ── per-domain hooks ───────────────────────────────────────────
    def trial(self, n, mean, sd, rep, outlier=None):
        raise NotImplementedError

    def draw(self, ax, trial, preview=False):
        raise NotImplementedError

    def trial_meta(self, trial):
        raise NotImplementedError

    def params(self):
        return {}

    def fmt_mean(self, m):
        return f"{m}°"

    def fmt_sd(self, s):
        return f"{s}°"

    def check_confounds(self, metadata):
        pass

    # ── rendering ──────────────────────────────────────────────────
    def _setup_ax(self, ax, bg):
        ax.set_xlim(0, CANVAS)
        ax.set_ylim(*self.YLIM)
        ax.set_aspect("equal")
        ax.set_facecolor(bg)

    def render(self, trial, path):
        fig = plt.figure(figsize=(CANVAS / DPI, CANVAS / DPI), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.patch.set_facecolor(trial["bg_color"])
        self._setup_ax(ax, trial["bg_color"])
        self.draw(ax, trial)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor(), pad_inches=0)
        plt.close(fig)

    def example_grid(self, n, out_dir, outlier=None):
        """One example per (mean, sd) condition at rep 0."""
        sds = self.sds("standard" if outlier is None else "discount")
        fig, axes = plt.subplots(len(sds), len(self.MEANS), squeeze=False,
                                 figsize=(2.2 * len(self.MEANS), 2.2 * len(sds) + 0.6),
                                 dpi=150)
        for i, sd in enumerate(sds):
            for j, mean in enumerate(self.MEANS):
                ax = axes[i, j]
                trial = self.trial(n, mean, sd, 0, outlier)
                self._setup_ax(ax, trial["bg_color"])
                ax.set_xticks([])
                ax.set_yticks([])
                self.draw(ax, trial, preview=True)
                if i == 0:
                    ax.set_title(f"μ={self.fmt_mean(mean)}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"σ={self.fmt_sd(sd)}", fontsize=8)
        tag = "" if outlier is None else f"  outlier: {outlier[0]} distinct, {outlier[1]}"
        fig.suptitle(f"{self.NAME}  N={n}  seed={self.seed}{tag}", fontsize=10)
        fig.supxlabel("mean μ", fontsize=9)
        fig.supylabel("SD σ", fontsize=9)
        fig.tight_layout()
        suffix = "" if outlier is None else f"_{outlier[0]}_{outlier[1]}"
        path = Path(out_dir) / f"example_grid_n{n:02d}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  example grid → {path}")

    # ── dataset ────────────────────────────────────────────────────
    def up_to_date(self, meta_path, n_reps, set_sizes, mode, layout):
        """True if existing metadata already covers this request (same
        generator version / seed / mode, at least as many reps and sizes)."""
        if not meta_path.exists():
            return False
        p = json.loads(meta_path.read_text())["params"]
        return (p.get("generator_version") == self.VERSION
                and p.get("seed") == self.seed and p.get("mode") == mode
                and p.get("n_repetitions", 0) >= n_reps
                and set(set_sizes) <= set(p.get("set_sizes", []))
                and (mode == "standard" or p.get("outlier_layout") == layout))

    def generate(self, out_dir, n_reps, set_sizes=SET_SIZES, checks=True,
                 mode="standard", layout="spatial"):
        out_dir = Path(out_dir)
        if mode == "discount" and layout == "spatial":
            # filter BEFORE the up-to-date check, which compares set_sizes
            quad = (GRID // self.BLOCK // 2) ** 2          # cells per quadrant
            fits = [n for n in set_sizes if n - 1 <= quad]
            if fits != list(set_sizes):
                print(f"{self.NAME}: spatial layout skips N="
                      f"{sorted(set(set_sizes) - set(fits))} "
                      f"(group of N-1 must fit a {quad}-cell quadrant)")
                set_sizes = fits
        if self.up_to_date(out_dir / "metadata.json", n_reps, set_sizes, mode, layout):
            print(f"{self.NAME} {self.VERSION} [{mode}] seed={self.seed}: "
                  f"{out_dir} is up to date, skipping (delete it to regenerate)")
            return json.loads((out_dir / "metadata.json").read_text())
        # reuse files only when the old metadata matches (a version change
        # must overwrite same-named files)
        reuse = self.up_to_date(out_dir / "metadata.json", 0, [], mode, layout)
        outliers = [None] if mode == "standard" else [(d, layout) for d in self.DISTINCT]
        sds = self.sds(mode)
        conditions = [c for c in itertools.product(set_sizes, self.MEANS, sds, outliers)
                      if c[3] is None or self.both_sides_fit(c[1], c[3][0])]
        dropped = sorted({(m, o[0]) for _, m, _, o in
                          itertools.product(set_sizes, self.MEANS, sds, outliers)
                          if o is not None and not self.both_sides_fit(m, o[0])})
        for m, d in dropped:
            print(f"  dropping mean={m} '{d}': outlier fits on one side only "
                  f"(direction would be tied to the mean)")
        total = len(conditions) * n_reps
        print(f"{self.NAME} {self.VERSION} [{mode}]  seed={self.seed}  N={list(set_sizes)}  "
              f"means={self.MEANS}  sds={sds}  reps={n_reps}  "
              f"→ {total} stimuli in {out_dir}")

        trials = []
        for n, mean, sd, outlier in conditions:
            tag = "" if outlier is None else f"_{outlier[0]}"
            for rep in range(n_reps):
                t = self.trial(n, mean, sd, rep, outlier)
                # filename relative to out_dir (loader resolves against it)
                fname = (f"n{n:02d}/mean{mean:03d}_sd{sd:03d}{tag}/"
                         f"stim_n{n:02d}_mean{mean:03d}_sd{sd:03d}{tag}_rep{rep:04d}.png")
                if not (reuse and (out_dir / fname).exists()):   # additive-reps rerun
                    self.render(t, out_dir / fname)              # renders only new files
                trials.append({"filename": fname, "repetition": rep, self.ELEMENT: n,
                               **self.trial_meta(t),
                               **(self.outlier_meta(t) if outlier else {})})
                if len(trials) % 500 == 0 or len(trials) == total:
                    print(f"  [{len(trials)}/{total}]")

        meta = {
            "generated": datetime.now().isoformat(),
            "params": {"domain": self.NAME, "generator_version": self.VERSION,
                       "seed": self.seed, "means": self.MEANS, "sds": sds,
                       "set_sizes": list(set_sizes), "n_repetitions": n_reps,
                       "canvas_size": CANVAS, "patch_size": PATCH,
                       "border_patches": self.BORDER, "position_margin": POSITION_MARGIN,
                       "elements_within_patch": True, "patch_block": self.BLOCK,
                       "paths_relative_to": "metadata",
                       "mode": mode,
                       **({"outlier_layout": layout, "outlier_delta": self.OUTLIER_DELTA,
                           "outlier_jitter": self.OUTLIER_JITTER,
                           "max_group_range": self.MAX_GROUP_RANGE,
                           "outlier_reference": "Avci & Boduroglu 2021"}
                          if mode == "discount" else {}),
                       **self.params()},
            "trials": trials,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=1)
        print(f"metadata → {out_dir / 'metadata.json'}")

        if checks:
            self.check_confounds(trials)
            for n in set_sizes:
                for o in outliers:
                    self.example_grid(n, out_dir, o)
        return meta


# ─────────────────────────────────────────────────────────────────
# ORIENTATION
# ─────────────────────────────────────────────────────────────────

class OrientationGenerator(EnsembleGenerator):
    """Bars of one color on a colored background; orientation on [0, 180)."""

    NAME, ELEMENT, DOMAIN_TAG, VERSION = "ori", "n_bars", 1, "v6"   # v6: no border
    MEANS = [0, 30, 60, 90, 120, 150]
    SDS = [5, 10, 15]
    N_COLORS, SATURATION, VALUE = 15, 0.6, 0.85
    MIN_CONTRAST = 3.0                 # WCAG ratio, bar vs background
    LENGTH_RANGE = (8.0, 13.0)         # px
    WIDTH_RANGE = (1.5, 2.5)           # px
    PERIOD = 180
    OUTLIER_DELTA = {"less": 15.0, "more": 90.0}   # Avci & Boduroglu 2021: 15±3°, 90±3°
    OUTLIER_JITTER = 3.0
    MAX_GROUP_RANGE = 30.0                         # paper: ≤ 30° among group lines

    def __init__(self, seed=42):
        super().__init__(seed)
        self.colors, self.pairs = color_pairs(self.N_COLORS, self.SATURATION,
                                              self.VALUE, self.MIN_CONTRAST)

    def sample(self, mean, sd, n, rng):
        return sample_von_mises(mean, sd, n, rng, period=180)

    def values(self, t):
        return t["orientations"]

    def trial(self, n, mean, sd, rep, outlier=None):
        rng = self.rng(n, mean, sd, rep, *self.okey(outlier))   # draw order fixed
        bar_idx, bg_idx = self.pairs[rng.integers(len(self.pairs))]
        ori, sign = self.sample_values(mean, sd, n, rng, outlier)
        lengths = rng.uniform(*self.LENGTH_RANGE, size=n)
        widths = rng.uniform(*self.WIDTH_RANGE, size=n)
        pos = self.place(n, rng, outlier)
        t = np.deg2rad(ori)
        c, s = np.abs(np.cos(t)), np.abs(np.sin(t))
        half = np.stack([lengths / 2 * c + widths / 2 * s,
                         lengths / 2 * s + widths / 2 * c], axis=1)
        pos = self.jitter(pos, half, rng)
        return dict(orientations=ori, lengths=lengths, widths=widths,
                    positions=pos, bar_idx=int(bar_idx), bg_idx=int(bg_idx),
                    bar_color=self.colors[bar_idx], bg_color=self.colors[bg_idx],
                    mean=mean, sd=sd, outlier=outlier, outlier_sign=sign)

    def draw(self, ax, trial, preview=False):
        cap = "round" if preview else "butt"
        for (x, y), th, L, w in zip(trial["positions"], trial["orientations"],
                                    trial["lengths"], trial["widths"]):
            dx, dy = L / 2 * np.cos(np.deg2rad(th)), L / 2 * np.sin(np.deg2rad(th))
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=trial["bar_color"],
                    linewidth=w, solid_capstyle=cap)

    def trial_meta(self, t):
        m, s = circular_stats(t["orientations"], period=180)
        return {"mean_orientation": float(t["mean"]), "variance": float(t["sd"]),
                "bar_color_idx": t["bar_idx"], "bg_color_idx": t["bg_idx"],
                "orientations": t["orientations"].tolist(),
                "bar_lengths": t["lengths"].tolist(), "bar_widths": t["widths"].tolist(),
                "positions_x": t["positions"][:, 0].tolist(),
                "positions_y": t["positions"][:, 1].tolist(),
                "actual_circular_mean": m, "actual_circular_sd": s}

    def params(self):
        return {"bar_length_range": list(self.LENGTH_RANGE),
                "bar_width_range": list(self.WIDTH_RANGE),
                "color_bank": [mcolors.to_hex(c) for c in self.colors],
                "min_contrast_ratio": self.MIN_CONTRAST}

    def check_confounds(self, trials):
        cond = [f"{t['mean_orientation']}_{t['variance']}" for t in trials]
        print("Confound checks (color should be independent of condition):")
        chi2_independence([t["bar_color_idx"] for t in trials], cond, "bar color")
        chi2_independence([t["bg_color_idx"] for t in trials], cond, "bg color")


# ─────────────────────────────────────────────────────────────────
# HUE
# ─────────────────────────────────────────────────────────────────

class HueGenerator(EnsembleGenerator):
    """Colored shapes (fixed S, V) on an achromatic background; hue on [0, 360).
    Task-irrelevant: shape per TRIAL (one shape per image, v5); size (sqrt
    area) and rotation per element."""

    NAME, ELEMENT, DOMAIN_TAG, VERSION = "hue", "n_circles", 2, "v5"
    MEANS = [0, 60, 120, 180, 240, 300]
    SDS = [10, 20, 30]
    SATURATION, VALUE = 0.85, 0.90
    SHAPE_SET = ("circle", "square", "triangle")
    SIZE_RANGE = (5.0, 7.8)            # px, sqrt(area); max fitting size is 7.9
    BG_NAMES = ["black", "gray", "white"]
    BG_COLORS = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)]
    YLIM = (CANVAS, 0)                 # drawn in image coords (top-down)
    PERIOD = 360
    OUTLIER_DELTA = {"less": 30.0, "more": 180.0}  # ori's 15° / 90° on a 360° circle
    OUTLIER_JITTER = 6.0
    MAX_GROUP_RANGE = 60.0

    def sample(self, mean, sd, n, rng):
        return sample_von_mises(mean, sd, n, rng, period=360)

    def values(self, t):
        return t["hues"]

    def trial(self, n, mean, sd, rep, outlier=None):
        assert self.SIZE_RANGE[1] <= max_size(self.SHAPE_SET)
        rng = self.rng(n, mean, sd, rep, *self.okey(outlier))   # draw order fixed
        hues, sign = self.sample_values(mean, sd, n, rng, outlier)
        sizes = rng.uniform(*self.SIZE_RANGE, size=n)
        shapes = [self.SHAPE_SET[rng.integers(len(self.SHAPE_SET))]] * n  # one per image
        rotations = rng.uniform(0, 360, size=n)
        pos = self.place(n, rng, outlier)
        pos = self.jitter(pos, shape_half_extent(shapes, sizes, rotations), rng)
        bg_idx = int(rng.integers(len(self.BG_COLORS)))
        return dict(hues=hues, sizes=sizes, shapes=shapes, rotations=rotations,
                    positions=pos, bg_idx=bg_idx, bg_color=self.BG_COLORS[bg_idx],
                    mean=mean, sd=sd, outlier=outlier, outlier_sign=sign)

    def draw(self, ax, trial, preview=False):
        colors = [mcolors.hsv_to_rgb([(h % 360) / 360, self.SATURATION, self.VALUE])
                  for h in trial["hues"]]
        draw_shapes(ax, trial["positions"], trial["shapes"], trial["sizes"],
                    trial["rotations"], colors, flip_y=True)

    def trial_meta(self, t):
        m, s = circular_stats(t["hues"], period=360)
        return {"mean_hue": float(t["mean"]), "variance_hue": float(t["sd"]),
                "bg_color_idx": t["bg_idx"], "bg_color_name": self.BG_NAMES[t["bg_idx"]],
                "hues": t["hues"].tolist(), "sizes": t["sizes"].tolist(),
                "shapes": t["shapes"], "rotations": t["rotations"].tolist(),
                "positions_x": t["positions"][:, 0].tolist(),
                "positions_y": t["positions"][:, 1].tolist(),
                "actual_circular_mean_hue": m, "actual_circular_sd_hue": s}

    def params(self):
        return {"size_range": list(self.SIZE_RANGE), "shapes": list(self.SHAPE_SET),
                "circle_saturation": self.SATURATION, "circle_value": self.VALUE,
                "bg_colors": self.BG_NAMES}

    def check_confounds(self, trials):
        cond = [f"{t['mean_hue']}_{t['variance_hue']}" for t in trials]
        print("Confound checks (nuisance should be independent of condition):")
        chi2_independence([t["bg_color_idx"] for t in trials], cond, "bg color")
        chi2_independence([t["shapes"][0] for t in trials], cond, "shape (per trial)")


# ─────────────────────────────────────────────────────────────────
# SIZE
# ─────────────────────────────────────────────────────────────────

class SizeGenerator(EnsembleGenerator):
    """Shapes of one color on a colored background; statistic = size.

    size = sqrt(area) (px), so a circle, square, triangle and leaf of the same
    size have the same area. Size is sampled in log space — ln(size) ~ Normal,
    the line analogue of the von Mises used on the orientation / hue circles;
    ratio spacing follows the mean-size literature (Ariely 2001; Chong &
    Treisman 2003; Whitney & Yamanashi Leib 2018 AnnuRev).

    v2: elements on an 8×8 grid of 2×2-patch cells (BLOCK = 2), ~2x every
    size of v1 (whose adjacent mean levels differed by sub-pixel amounts);
    one shape per image. Up to N = 32 of the 64 cells.

    Matched difficulty across domains: in every domain the mean step is
    6x / 3x / 2x the three SD levels (ori 30° vs 5/10/15°, hue 60° vs
    10/20/30°, size 0.15 vs 0.025/0.05/0.075 ln units), so the 6-way mean
    and 3-way SD tasks are equally hard in each domain's own statistic space.
    mean ± 3 SD stays inside [SIZE_MIN, max fitting size] for every condition.

    Condition codes (integers, used in RNG keys and filenames) are in
    milli-ln units: mean code 1640 → exp(1.64) = 5.16 px; SD code 25 → 0.025.
    Task-irrelevant per element: rotation; per trial: shape, color pair.
    """

    # v3: SD levels ×2.4 and larger shapes (circle/square only, means shifted up)
    # so the variance signal clears DINOv2's ≈0.5 px per-element size noise
    # floor, which left v2's variance probes at chance (sub-pixel deviations)
    NAME, ELEMENT, DOMAIN_TAG, VERSION = "size", "n_items", 3, "v3"
    BLOCK = 2                                     # 2×2-patch cells, 8×8 grid
    MEANS = [1790, 1940, 2090, 2240, 2390, 2540]  # ln(px)×1000: 5.99 … 12.68 px
    SDS = [60, 120, 180]                          # SD of ln(size) ×1000
    SHAPE_SET = ("circle", "square")
    SIZE_MIN = 4.0                    # px; max is set by the cell (17.0 px with square)
    N_COLORS, SATURATION, VALUE, MIN_CONTRAST = 15, 0.6, 0.85, 3.0
    # discount: ori's 15° / 90° under the 30° ↔ 0.15 ln mapping. Means whose
    # bounds admit only one outlier side are dropped per distinctness
    # (both_sides_fit): "more" runs on the interior means only
    OUTLIER_DELTA = {"less": 0.075, "more": 0.45}
    OUTLIER_JITTER = 0.015
    MAX_GROUP_RANGE = 0.15                         # ori's 30° under 30° ↔ 0.15 ln

    def __init__(self, seed=42):
        super().__init__(seed)
        self.colors, self.pairs = color_pairs(self.N_COLORS, self.SATURATION,
                                              self.VALUE, self.MIN_CONTRAST)
        self.size_max = max_size(self.SHAPE_SET) * self.BLOCK   # cell, not patch

    def fmt_mean(self, m):
        return f"{np.exp(m / 1000):.2f}px"

    def fmt_sd(self, s):
        return f"{s / 1000:.3f} ln"

    def sample_sizes(self, mean, sd, n, rng):
        """sizes (px): ln(size) ~ N(mean, sd) (milli-ln codes), redrawn outside
        [SIZE_MIN, size_max] (< 0.3% of draws: mean ± 3 SD fits)."""
        mu, sigma = mean / 1000, sd / 1000
        lo, hi = np.log(self.SIZE_MIN), np.log(self.size_max)
        x = rng.normal(mu, sigma, size=n)
        while (bad := (x < lo) | (x > hi)).any():
            x[bad] = rng.normal(mu, sigma, size=int(bad.sum()))
        return np.exp(x)

    def sample(self, mean, sd, n, rng):
        return np.log(self.sample_sizes(mean, sd, n, rng))

    def values(self, t):
        return np.log(t["sizes"])

    def bounds(self):
        return np.log(self.SIZE_MIN), np.log(self.size_max)

    def stat_mean(self, mean):
        return mean / 1000                        # milli-ln codes → ln px

    def trial(self, n, mean, sd, rep, outlier=None):
        rng = self.rng(n, mean, sd, rep, *self.okey(outlier))   # draw order fixed
        fg_idx, bg_idx = self.pairs[rng.integers(len(self.pairs))]
        ln_sizes, sign = self.sample_values(mean, sd, n, rng, outlier)
        sizes = np.exp(ln_sizes)
        shapes = [self.SHAPE_SET[rng.integers(len(self.SHAPE_SET))]] * n  # one per image
        rotations = rng.uniform(0, 360, size=n)
        pos = self.place(n, rng, outlier)
        pos = self.jitter(pos, shape_half_extent(shapes, sizes, rotations), rng)
        return dict(sizes=sizes, shapes=shapes, rotations=rotations, positions=pos,
                    fg_idx=int(fg_idx), bg_idx=int(bg_idx),
                    fg_color=self.colors[fg_idx], bg_color=self.colors[bg_idx],
                    mean=mean, sd=sd, outlier=outlier, outlier_sign=sign)

    def draw(self, ax, trial, preview=False):
        draw_shapes(ax, trial["positions"], trial["shapes"], trial["sizes"],
                    trial["rotations"], [trial["fg_color"]] * len(trial["sizes"]))

    def trial_meta(self, t):
        ln = np.log(t["sizes"])
        return {"mean_size": float(np.exp(t["mean"] / 1000)),   # geometric mean, px
                "sd_size": t["sd"] / 1000,                        # SD of ln(size)
                "fg_color_idx": t["fg_idx"], "bg_color_idx": t["bg_idx"],
                "sizes": t["sizes"].tolist(), "shapes": t["shapes"],
                "rotations": t["rotations"].tolist(),
                "positions_x": t["positions"][:, 0].tolist(),
                "positions_y": t["positions"][:, 1].tolist(),
                # realized statistics in the candidate spaces observers might average
                "actual_mean_log_size": float(ln.mean()),
                "actual_sd_log_size": float(ln.std(ddof=1)) if len(ln) > 1 else 0.0,
                "actual_mean_size": float(t["sizes"].mean()),
                "actual_mean_area": float((t["sizes"] ** 2).mean())}

    def params(self):
        return {"size_definition": "sqrt(area) px", "size_space": "log",
                "size_min": self.SIZE_MIN, "size_max": self.size_max,
                "condition_codes": "milli-ln: mean=ln(px)*1000, sd=ln-units*1000",
                "shapes": list(self.SHAPE_SET),
                "color_bank": [mcolors.to_hex(c) for c in self.colors],
                "min_contrast_ratio": self.MIN_CONTRAST}

    def check_confounds(self, trials):
        cond = [f"{t['mean_size']}_{t['sd_size']}" for t in trials]
        print("Confound checks (nuisance should be independent of condition):")
        chi2_independence([t["fg_color_idx"] for t in trials], cond, "fg color")
        chi2_independence([t["bg_color_idx"] for t in trials], cond, "bg color")
        chi2_independence([t["shapes"][0] for t in trials], cond, "shape (per trial)")


GENERATORS = {"ori": OrientationGenerator, "orientation": OrientationGenerator,
              "hue": HueGenerator, "size": SizeGenerator}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("domain", choices=GENERATORS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_reps", type=int, default=50,
                   help="reps per condition; a larger value reproduces the "
                        "same first reps and adds new ones")
    p.add_argument("--out", default=None,
                   help="output dir (default: stimuli_<domain>_s<seed>)")
    p.add_argument("--set_sizes", default=None, help="comma-separated, default all")
    p.add_argument("--mode", choices=["standard", "discount"], default="standard",
                   help="discount: N-1 group elements + 1 outlier (less/more distinct)")
    p.add_argument("--layout", choices=["spatial", "mixed"], default="spatial",
                   help="discount mode: outlier in the opposite quadrant (spatial) "
                        "or among the group (mixed)")
    p.add_argument("--skip_checks", action="store_true",
                   help="skip confound checks and example grids")
    a = p.parse_args()

    gen = GENERATORS[a.domain](a.seed)
    sizes = [int(s) for s in a.set_sizes.split(",")] if a.set_sizes else (
        SET_SIZES if a.mode == "standard" else [n for n in SET_SIZES if n >= 3])
    tag = "" if a.mode == "standard" else f"_discount_{a.layout}"
    gen.generate(a.out or f"stimuli_{gen.NAME}{tag}_s{a.seed}", a.n_reps, sizes,
                 checks=not a.skip_checks, mode=a.mode, layout=a.layout)


if __name__ == "__main__":
    main()
