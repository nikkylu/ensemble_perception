# Ensemble Perception in DINOv2

Do vision transformers encode ensemble statistics? We probe frozen DINOv2
(ViT-B/14) for the **mean** and **SD** of a set of elements' orientation, hue
or size, across layers, token slices and set sizes — and test whether an
**outlier** is averaged in or discounted, mirroring Avci & Boduroglu (2021).

## Repository map

```
src/
  stimuli/generate.py  seed-deterministic stimulus generators (ori | hue | size),
                       standard + discount mode; every display is a pure function
                       of (seed, domain, N, μ, σ, rep)
  verify_mask.py       pixel check: each element inside its own patch cell
  loader.py            DINOv2 feature extraction per layer: CLS, element /
                       background / all-patch pools, element-token concat
  probe.py             classification probes (vectorized logistic, nested CV)
                       and within-cell regression (closed-form ridge, exact LOOCV)
  readout.py           shared continuous ridge readout (circular / linear targets)
  discount.py          β: the outlier's weight in a standard-trained mean readout
  discount_probe.py    local vs global mean decodability on discount displays
  ideal_single.py      single-element ideal-observer ceilings for per_token
  pipeline.sh          generate → verify → extract → probe → discount, all domains
```

## Quick start

```bash
pip install -r requirements.txt
cd src
./pipeline.sh                              # all domains × modes, seed 42, 200 reps
./pipeline.sh "ori hue" 7 50 "8 16"        # domains, seed, reps, set sizes
MODES=standard ./pipeline.sh size          # restrict modes
```

Stimuli are reused when up to date; features are re-extracted only when the
stimuli are newer; probes rerun from cached features. Everything is keyed by
seed and reruns byte-identically.

| output | path (under `src/`) |
|---|---|
| stimuli | `stimuli_<domain>[_discount_<layout>]_s<seed>/` |
| features | `features/<domain>_<mode>_n<N>_s<seed>.pt` |
| standard probes | `results_s<seed>/<domain>/n<N>_<target>_linear.{txt,json}` |
| discount | `results_s<seed>/<domain>_<mode>/n<N>_{mean_linear,discount}.{txt,json}` |

## Probing protocol (probe.py)

- **Targets** `mean` (6-way) and `variance` (3-way): labels are relabeled to
  the level nearest each display's *realized* statistic, so the ceiling is
  1.0 by construction; the nuisance statistic is held fixed per level.
  `mean_reg` / `variance_reg` regress the realized statistic within each
  (μ, σ) cell — the precision analysis below the 30° level spacing.
- **Slices**: CLS, element tokens (pooled), background, all patches,
  pairwise_diff (token variance), and `per_token` — one shared probe scored
  on single element tokens. Its ceiling is not 1.0: compare against
  `ideal_single.py` (above the ceiling ⇒ token mixing). The element-token
  *concat* probe is reported only as a closed-form regression (supplement):
  its logistic version confounds representation with p ≫ n estimation.
- λ by nested CV (classification) or exact LOOCV (regression); outer test
  folds never influence model selection.

### Aggregation-vs-computation controls

The pooled slices average tokens as part of the *analysis*, so decoding from
them shows that the information survives in token space — not that the model
computed the statistic (CLS, the model's own summary, and per_token above its
single-element ceiling are the model-side evidence). Two controls separate
the contributions; run them with the existing tools:

```bash
python3 loader.py --metadata <stimuli>/metadata.json --n_bars <N> \
    --model dinov2_vitb14@emb    --output features/<...>_emb.pt      # pre-block embeddings
python3 loader.py --metadata <stimuli>/metadata.json --n_bars <N> \
    --model dinov2_vitb14@random --output features/<...>_rand.pt     # untrained weights
python3 probe.py --features features/<...>_emb.pt --target mean --n_bars <N>
```

`@emb` stores the patch embedding + positions (what enters block 1) as a
single "layer"; `@random` keeps the architecture with untrained weights.
probe.py handles any number of layers.

## Outlier discounting (discount mode)

Displays hold N−1 group elements plus one outlier beyond the group's range
edge ("less" / "more" distinct; spatial or mixed layout). Two analyses:

- **`discount.py` — β**: a mean readout trained on standard displays is
  applied to discount displays; β is the outlier's effective weight
  (0 = discounted, 1 = averaged like any element), with trial-bootstrap CIs.
  Compare β against β_circ, the value plain circular averaging would give —
  for a 90° orientation outlier, β_circ ≈ −0.1, not 0.
- **`discount_probe.py` — local vs global**: ridge readouts trained on the
  discount displays themselves (nested CV), one per target (group-only mean,
  arithmetic global mean, circular mean). Whether the global target is
  decodable at all — and whether a global-trained probe's predictions stay
  near the local mean — separates "the pooled vector is just an average"
  from "element-resolved information survives pooling". `bar_group`
  (outlier token removed by ground truth) is the no-information floor.

Size is the decisive domain: its statistic is linear, so "average everything"
and "exclude the outlier" give different answers (circular domains let a 90° /
180° outlier cancel itself, which mimics discounting). Discount cells whose
size bounds admit only one outlier side are dropped at generation time —
otherwise the outlier's direction is tied to the mean and readout shrinkage
mimics outlier weighting.

## Requirements

Python ≥ 3.10, PyTorch (+ torchvision, downloads DINOv2 via torch.hub on
first run), numpy, scipy, scikit-learn, matplotlib, pillow — see
`requirements.txt`.
