"""
DINOv2 feature extraction for orientation / hue / size ensemble stimuli.

Output .pt dict:
    cls / bar / bg / all_patches : (N, L, D)  CLS token and mean-pooled patch groups
    bar_concat                   : (N, L, k*D) element tokens in block-major
                                   raster order — the block² tokens of one
                                   element are adjacent (block=1: plain
                                   raster); only when all trials share a set size
    bg_concat        (opt-in)    : background tokens likewise (--bg_concat;
                                   ~190k-d, dominates the file size)
    labels_mean / labels_var     : (N,) int class labels (sorted condition values)
    mean_values / var_values     : condition value (deg) of each label
    period                       : circle period of the mean (180 ori, 360 hue,
                                   0 = linear in log space for size)
    realized_mean / realized_sd  : (N,) each display's realized statistics
                                   (deg, or ln px for size)
  discount-mode stimuli additionally (statistic space: deg, or ln px for size):
    local_mean / global_mean     : (N,) realized mean without / with the outlier
    global_mean_linear           : (N,) with the outlier unwrapped on its side
                                   (the paper's arithmetic global mean)
    local_sd / global_sd         : (N,)
    outlier_value                : (N,) the outlier's value
    outlier_distinct             : (N,) 0 = less, 1 = more distinct
    outlier_slot                 : (N,) position of the outlier token among the
                                   element tokens in bar_concat (raster order)
    n_bars                       : (N,) set size
    filenames                    : list[str]

Features are stored in fp16 by default (probe.py casts back to fp32).

Usage:
    python3 loader.py --metadata stimuli_ori_s42/metadata.json --output feats.pt --n_bars 16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

PATCH_PX = 14
GRID = 16          # 224 / 14
N_LAYERS = 12

# metadata key per field: orientation domain first, then hue domain
MEAN_KEYS = ("mean_orientation", "mean_hue", "mean_size")
VAR_KEYS = ("variance", "variance_hue", "sd_size")
N_KEYS = ("n_bars", "n_circles", "n_items")
REALIZED_MEAN_KEYS = ("actual_circular_mean", "actual_circular_mean_hue", "actual_mean_log_size")
REALIZED_SD_KEYS = ("actual_circular_sd", "actual_circular_sd_hue", "actual_sd_log_size")


def get(trial, keys):
    for k in keys:
        if k in trial:
            return trial[k]
    raise KeyError(f"none of {keys} in trial metadata")


def element_tokens(bar_concat, n, d):
    """One d-dim vector per element: the block² patch tokens of each element
    (adjacent in this module's block-major concat order) are averaged.
    (T, L, n·block²·d) → (T, L, n, d); block=1 is a no-op reshape."""
    T, L, tot = bar_concat.shape
    assert tot % (n * d) == 0, f"concat dim {tot} is not n={n} × block² × d={d}"
    return bar_concat.reshape(T, L, n, tot // (n * d), d).mean(3)


def token_rows(bar_concat, n, d, max_slots=8, seed=0):
    """Single element vectors as separate rows: ≤ max_slots random element
    slots of each display, trial-major. → ((T·k, L, d), k)"""
    toks = element_tokens(bar_concat, n, d)
    k = min(n, max_slots)
    slots = torch.as_tensor(np.random.default_rng(seed).choice(n, k, replace=False))
    return toks[:, :, slots].permute(0, 2, 1, 3).reshape(-1, toks.shape[1], d), k


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────
# PATCH MASK
# ─────────────────────────────────────────────────────────────────

def make_patch_mask(xs, ys, extent=12.0, center_only=True, block=1):
    """(256,) bool mask of patches containing an element.

    xs, ys: element centers in px, y bottom-up (generator convention).
    center_only: generators v5+/hue v3 keep each element inside its own
    BLOCK×BLOCK patch cell (block=1: single patch), so the cell containing
    the center is exact. Older stimuli fall back to the element's
    axis-aligned box of size `extent`.
    """
    mask = np.zeros((GRID, GRID), dtype=bool)
    if center_only:
        cell, g = PATCH_PX * block, GRID // block
        for x, y in zip(xs, ys):
            rb = min(int((224 - y) // cell), g - 1)           # image rows top-down,
            cb = min(int(x // cell), g - 1)                   # clamped at the edge
            mask[rb * block:(rb + 1) * block, cb * block:(cb + 1) * block] = True
        return mask.ravel()
    half = extent / 2
    for x, y in zip(xs, ys):
        y = 224 - y                                    # → image rows (top-down)
        r0, r1 = int(max(0, y - half) // PATCH_PX), int(min(223, y + half) // PATCH_PX)
        c0, c1 = int(max(0, x - half) // PATCH_PX), int(min(223, x + half) // PATCH_PX)
        mask[r0:r1 + 1, c0:c1 + 1] = True
    return mask.ravel()


# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────

class StimuliDataset(Dataset):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    def __init__(self, meta_path, n_bars=None):
        meta_path = Path(meta_path)
        meta = json.loads(meta_path.read_text())
        params = meta["params"]
        self.trials = [t for t in meta["trials"]
                       if n_bars is None or get(t, N_KEYS) == n_bars]
        assert self.trials, f"no trials with set size {n_bars}"
        # new generators store paths relative to the metadata's folder;
        # older ones relative to the folder the generator ran in
        self.root = (meta_path.parent if params.get("paths_relative_to") == "metadata"
                     else meta_path.parent.parent)
        self.center_only = bool(params.get("elements_within_patch", False))
        self.block = int(params.get("patch_block", 1))
        if "bar_length_range" in params:
            self.extent = float(max(params["bar_length_range"]))
        elif "size_range" in params:
            self.extent = 1.6 * float(max(params["size_range"]))
        elif "circle_radius_range" in params:
            self.extent = 2 * float(max(params["circle_radius_range"]))
        else:
            self.extent = float(params.get("bar_length", 12.0))

        # labels: index of the condition value in the sorted unique values
        self.mean_values = sorted({get(t, MEAN_KEYS) for t in meta["trials"]})
        self.var_values = sorted({get(t, VAR_KEYS) for t in meta["trials"]})
        self.set_sizes = sorted({get(t, N_KEYS) for t in self.trials})
        first = self.trials[0]
        self.period = 360 if "mean_hue" in first else 0 if "mean_size" in first else 180

    def realized_info(self):
        """Realized per-display mean / SD (statistic space), if in the metadata."""
        try:
            return {"realized_mean": torch.tensor([float(get(t, REALIZED_MEAN_KEYS))
                                                   for t in self.trials]),
                    "realized_sd": torch.tensor([float(get(t, REALIZED_SD_KEYS))
                                                 for t in self.trials])}
        except KeyError:
            return {}

    def discount_info(self):
        """Per-trial outlier fields (discount-mode stimuli), else {}."""
        if "outlier" not in self.trials[0]:
            return {}
        info = {k: torch.tensor([t[k] for t in self.trials], dtype=torch.float32)
                for k in ("local_mean", "global_mean", "global_mean_linear",
                          "local_sd", "global_sd")}
        info["outlier_value"] = torch.tensor(
            [t["outlier"]["value"] for t in self.trials], dtype=torch.float32)
        info["outlier_distinct"] = torch.tensor(
            [t["outlier"]["distinct"] == "more" for t in self.trials], dtype=torch.long)
        slots = []
        cell, g = PATCH_PX * self.block, GRID // self.block
        for t in self.trials:
            i = t["outlier"]["index"]
            # cell index of every element; the outlier's rank in cell raster
            # order is its slot among the element token groups (one element
            # per BLOCK×BLOCK cell; extract() emits tokens in block-major order)
            cells = [int((224 - y) // cell) * g + int(x // cell)
                     for x, y in zip(t["positions_x"], t["positions_y"])]
            slots.append(sorted(cells).index(cells[i]))
        info["outlier_slot"] = torch.tensor(slots)
        return info

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, i):
        t = self.trials[i]
        image = self.normalize(Image.open(self.root / t["filename"]).convert("RGB"))
        mask = torch.from_numpy(make_patch_mask(
            t["positions_x"], t["positions_y"], self.extent, self.center_only,
            self.block))
        return (image, mask,
                self.mean_values.index(get(t, MEAN_KEYS)),
                self.var_values.index(get(t, VAR_KEYS)),
                get(t, N_KEYS), t["filename"])


def collate(batch):
    images, masks, lm, lv, n, names = zip(*batch)
    return (torch.stack(images), torch.stack(masks), torch.tensor(lm),
            torch.tensor(lv), torch.tensor(n), list(names))


# ─────────────────────────────────────────────────────────────────
# POOLING
# ─────────────────────────────────────────────────────────────────

def masked_mean(tokens, mask):
    """tokens (B, P, D), mask (B, P) → (B, D) mean over masked tokens (fp32 accumulate)."""
    w = mask.unsqueeze(-1).float()
    return ((tokens.float() * w).sum(1) / w.sum(1).clamp(min=1)).to(tokens.dtype)


def masked_concat(tokens, mask):
    """tokens (B, P, D), mask (B, P) with equal counts per row → (B, k*D),
    masked tokens concatenated in raster order."""
    return tokens[mask].reshape(tokens.shape[0], -1)


# ─────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(model, dataset, device, batch_size=32, out_dtype=torch.float16,
            bg_concat=False, embedding=False):
    # concatenated slices are only well-defined when every trial has the
    # same number of element patches; bg_concat (~190k-d) dominates the
    # file size, so it is opt-in (as in probe.py)
    concat = len(dataset.set_sizes) == 1
    # block-major patch order: the block² tokens of one element stay adjacent
    # in the *_concat slices (block=1: identity / plain raster order)
    B = dataset.block
    idx = torch.arange(GRID * GRID).reshape(GRID, GRID)
    perm = torch.cat([idx[r:r + B, c:c + B].reshape(-1)
                      for r in range(0, GRID, B) for c in range(0, GRID, B)])
    loader = DataLoader(dataset, batch_size, num_workers=4, collate_fn=collate)
    keys = (["cls", "bar", "bg", "all_patches"]
            + (["bar_concat"] if concat else [])
            + (["bg_concat"] if concat and bg_concat else []))
    out = {k: [] for k in keys + ["labels_mean", "labels_var", "n_bars"]}
    filenames = []

    for images, masks, lm, lv, n, names in loader:
        if embedding:   # pre-block control: patch embed + positions, one "layer"
            tok = model.prepare_tokens_with_masks(images.to(device))
            layers = [(tok[:, 1:], tok[:, 0])]
        else:
            layers = model.get_intermediate_layers(
                images.to(device), n=N_LAYERS, return_class_token=True)
        masks = masks[:, perm]
        per_layer = {k: [] for k in keys}
        for patches, cls in layers:
            patches = patches.float().cpu().to(out_dtype)[:, perm]    # (B, 256, D)
            per_layer["cls"].append(cls.float().cpu().to(out_dtype))
            per_layer["all_patches"].append(patches.float().mean(1).to(out_dtype))
            per_layer["bar"].append(masked_mean(patches, masks))
            per_layer["bg"].append(masked_mean(patches, ~masks))
            if concat:
                per_layer["bar_concat"].append(masked_concat(patches, masks))
            if "bg_concat" in keys:
                per_layer["bg_concat"].append(masked_concat(patches, ~masks))
        for k in keys:
            out[k].append(torch.stack(per_layer[k], dim=1))          # (B, L, ·)
        out["labels_mean"].append(lm)
        out["labels_var"].append(lv)
        out["n_bars"].append(n)
        filenames += names

    out = {k: torch.cat(v) for k, v in out.items()}
    out.update(filenames=filenames, mean_values=dataset.mean_values,
               var_values=dataset.var_values, period=dataset.period,
               **dataset.realized_info(), **dataset.discount_info())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", default="features.pt")
    p.add_argument("--model", default="dinov2_vitb14",
                   help="hub model; append @random (untrained weights) or @emb "
                        "(pre-block patch embeddings, a single 'layer') for the "
                        "aggregation-vs-computation controls")
    p.add_argument("--model_seed", type=int, default=0,
                   help="seeds the @random initialization, so every domain and "
                        "set size probes the SAME untrained network")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=pick_device())
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--n_bars", type=int, default=None,
                   help="only extract this set size (needed for *_concat slices)")
    p.add_argument("--bg_concat", action="store_true",
                   help="also save concatenated background tokens "
                        "(~190k-d, dominates the file size)")
    a = p.parse_args()

    dataset = StimuliDataset(a.metadata, a.n_bars)
    name, _, variant = a.model.partition("@")
    torch.manual_seed(a.model_seed)   # one fixed @random network across runs
    model = torch.hub.load("facebookresearch/dinov2", name,
                           pretrained=variant != "random").to(a.device).eval()
    dtype = torch.float16 if a.precision == "fp16" else torch.float32
    feats = extract(model, dataset, a.device, a.batch_size, dtype, a.bg_concat,
                    embedding=variant == "emb")

    for k, v in feats.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s} {tuple(v.shape)}")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, a.output)
    print(f"Saved → {a.output}")


if __name__ == "__main__":
    main()
