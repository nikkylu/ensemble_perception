# Ensemble Perception in DINOv2

Do vision transformers encode ensemble statistics? We probe frozen DINOv2
(ViT-B/14) for the **mean** and **SD** of a set of elements' orientation, hue
or size, across layers, token types and set sizes.

```
experiment/
  stimuli/generate.py  stimulus generators (ori | hue | size), standard + discount mode
  verify_mask.py       pixel check: one element per patch, none crossing a patch
  loader.py            DINOv2 features per layer: CLS, element / background / all patches, concat
  probe.py             linear probes, nested CV, all layers × slices in one vectorized fit
  discount.py          outlier discounting: mean readout from standard → outlier displays
  rsa.py               representational similarity analysis
  plot_neurips.py      paper figures from probe logs
  pipeline.sh          generate → verify → extract → probe
```

## Quick start

```bash
pip install -r Requirements.txt
cd experiment
./pipeline.sh                              # all domains × modes, seed 42, 50 reps
./pipeline.sh "ori hue" 7 50 "8 16"        # domains, seed, reps, set sizes
MODES=standard ./pipeline.sh size          # only the standard stimuli
python3 plot_neurips.py --results_dir results_s42
```

For every domain (ori, hue, size) and mode — `standard`, `discount_spatial`,
`discount_mixed` — `pipeline.sh` generates the stimuli once, then for each set
size N verifies the patch masks, extracts features and probes. Standard sets
probe mean and SD for all N; discount sets (one SD level, N ≥ 3) probe the
group mean and run the discount analysis. Outputs are keyed by seed:

| | path |
|---|---|
| stimuli | `stimuli_<domain>[_discount_<layout>]_s<seed>/` |
| features | `features/<domain>_<mode>_n<N>_s<seed>.pt` |
| standard results | `results_s<seed>/<domain>/n<N>_<target>_linear.{txt,json}` (read by `plot_neurips.py`) |
| discount results | `results_s<seed>/<domain>_<mode>/n<N>_{mean_linear,discount}.{txt,json}` |

## Stimuli

Commands below run from `experiment/`.

```bash
python3 stimuli/generate.py ori --seed 42 --n_reps 50     # → stimuli_ori_s42/
python3 stimuli/generate.py size --set_sizes 8,16
```
```python
from stimuli.generate import OrientationGenerator
gen = OrientationGenerator(seed=42)
trial = gen.trial(n=16, mean=30, sd=10, rep=0)   # any rep: an infinite, reproducible stream
```

Each image is a pure function of `(seed, domain, N, mean, SD, rep)`: the same
seed always gives the same stimuli, and a larger `--n_reps` adds new reps
without changing existing ones.

| domain | elements | statistic | means | SDs | task-irrelevant |
|---|---|---|---|---|---|
| ori  | bars | orientation, von Mises (doubled angle) | 0…150°, step 30° | 5 / 10 / 15° | length, width, color pair |
| hue  | circle, square, triangle | hue, von Mises | 0…300°, step 60° | 10 / 20 / 30° | shape, size, rotation, gray-level bg |
| size | circle, square, triangle, leaf | ln √area, Gaussian | 2.5…5.4 px, step 0.15 ln | 0.025 / 0.05 / 0.075 ln | shape, rotation, color pair |

- **Comparable difficulty:** in every domain the mean step is 6× / 3× / 2× the
  SD levels, so the 6-way mean and 3-way SD tasks are equally hard in each
  domain's own statistic space. Size is linear (the end levels have one
  neighbour), and at fixed N mean size co-varies with total area; metadata
  stores realized mean log size, mean size and mean area.
- **Layout:** elements sit anywhere on the 16×16 patch grid, at most one per
  patch, with jitter bounded so each element stays inside its patch.

**Discount mode** (outliers, after Avci & Boduroglu, 2021, *APP* 83:1141):

```bash
python3 stimuli/generate.py ori --mode discount                  # outlier in the opposite quadrant
python3 stimuli/generate.py hue --mode discount --layout mixed   # outlier among the group
```

N−1 group elements (smallest SD; range capped at ori 30°, hue 60°, size
0.15 ln) plus one outlier beyond the group's edge by a "less" (ori 15±3°) or
"more" (ori 90±3°) distinct amount. Metadata and features keep local (group)
and global means — global both circular and linear (the paper's arithmetic
mean) — plus the outlier's value and its token slot in `bar_concat`.

## Features

```bash
python3 loader.py --metadata stimuli_ori_s42/metadata.json --n_bars 16 --output feats.pt
```

Per layer (fp16): `cls`, `bar` (mean of element patches), `bg` (mean of
background patches), `all_patches`, and — when one set size is extracted —
`bar_concat` / `bg_concat` (tokens concatenated in raster order).

## Probes

```bash
python3 probe.py --features feats.pt --target mean --n_bars 16
python3 probe.py --features feats.pt --target variance --n_bars 16 --save res.json
```

- **Slices:** `cls`, `bar`, `bg`, `all_patches`, `pairwise_diff`
  (per-dim mean squared difference between element tokens = 2·Var), `bar_concat`;
  opt-in `--bg_concat`, `--per_token` (one probe shared by all element tokens,
  scored per token and by averaging an image's token logits).
- **Nuisance fixed:** the mean is probed separately at each SD level (and vice
  versa); accuracies are averaged over levels, and the per-level table is
  printed too (`--save` keeps every level's fold accuracies and λ).
- **Nested CV:** outer 5 folds score; inside each outer training set an inner
  CV picks the L2 strength λ (one per slice, at the middle layer;
  `--lambda_select layer` for per-layer, `--weight_decay x` to fix it).
  Features are standardized per fold.
- **Two summaries:** `best` = max over layers (layer picked on the test folds,
  optimistic) and `cv-best` = layer picked on the other folds (unbiased).
- **Fast:** all layers × folds × λ × 768-d slices are independent logistic
  regressions trained in one vectorized full-batch loop.

## Discount analysis

```bash
python3 discount.py --train features/ori_standard_n8_s42.pt \
                    --test features/ori_discount_spatial_n8_s42.pt
```

A ridge readout of the realized mean ((cos, sin) for ori / hue, ln size for
size; λ by CV) is trained on the standard displays and applied to the outlier
displays. Its outlier weight

β = Σ (pred − local)(global − local) / Σ (global − local)²

is 0 if the outlier is discounted and 1 if it is averaged like any element
(global = the paper's arithmetic mean). **Compare β with β_circ**, the value
pure circular averaging would give: a 90° orientation outlier is opposite in
doubled-angle space, so circular averaging alone gives β_circ ≈ 0 for "more
distinct" outliers. `bar_group` (group tokens only, outlier token removed)
should give β ≈ 0 if tokens only encode their own element.

