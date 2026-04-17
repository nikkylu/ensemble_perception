# Ensemble Perception

Probing DINOv2 (ViT-B/14) for ensemble perception — testing whether the model encodes the **mean** and **variance** of bar orientations in a stimulus.

## Structure

```
experiment/     stimulis, feature extraction and probing
  script.py       generate stimuli images + metadata
  loader.py       DINOv2 feature extraction (per-layer CLS, bar, bg, all_patches)
  verify_mask.py  visualize patch masks overlaid on stimuli
  probe.py        linear / quadratic probes over extracted features
Requirements.txt
```

## Pipeline

### 1. Generate stimuli

```bash
python3 experiment/script.py
```

Produces `stimuli_v*/` with images and `metadata.json` (bar positions, orientations, mean/variance labels).

### 2. Extract features

```bash
python3 experiment/loader.py --metadata stimuli_v2/metadata.json --output featuresv3/features_n4.pt --n_bars 4
```

Runs all images through frozen DINOv2 and saves per-layer representations:

- `cls` — CLS token
- `bar` — mean-pooled bar patch tokens
- `bg` — mean-pooled background patch tokens
- `all_patches` — mean-pooled all patch tokens
- `bar_concat` — concatenated bar patch tokens (fixed set size only)

### 3. Verify patch masks (optional)

```bash
python3 experiment/verify_mask.py --metadata stimuli_v2/metadata.json --n_bars 4
```

Saves `verify_mask.png` showing the 16×16 patch grid aligned to the image with bar patches highlighted.

### 4. Probe

```bash
# single run
python3 probe.py --features featuresv3/features_n16.pt --target variance --n_bars 16 --probe linear
python3 probe.py --features featuresv3/features_n16.pt --target variance --n_bars 16 --probe quadratic

# fix nuisance variable
python3 experiment/probe.py --features featuresv3/features_n16.pt --target mean --n_bars 4 --probe linear --fix_variance 0
python3 experiment/probe.py --features featuresv3/features_n16.pt --target variance --n_bars 4 --probe linear --fix_mean 2
```

## Key probing design

- **Mean**: linear probe on mean-pooled bar tokens — first-order statistic
- **Variance**: linear probe on `pairwise_diff` — mean squared pairwise difference across bar patch tokens, directly measures representational spread:
  `E[(hᵢ - hⱼ)²] = 2 · Var(h)`
- **Fix nuisance**: probe mean while fixing variance to one level (and vice versa) to test whether the two statistics are independently represented

## Requirements

```bash
pip install -r Requirements.txt
# DINOv2
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
```
