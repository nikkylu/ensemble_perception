"""
DINOv2 feature extraction for orientation / hue / size ensemble stimuli.

Output .pt dict:
    cls / bar / bg / all_patches : (N, L, D)  CLS token and mean-pooled patch groups
    bar_concat / bg_concat       : (N, L, k*D) element / background tokens in
                                   raster order (only when all trials share a set size)
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


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────
# PATCH MASK
# ─────────────────────────────────────────────────────────────────

def make_patch_mask(xs, ys, extent=12.0, center_only=True):
    """(256,) bool mask of patches containing an element.

    xs, ys: element centers in px, y bottom-up (generator convention).
    center_only: generators v5+/hue v3 keep each element inside its own
    patch, so the patch containing the center is exact. Older stimuli
    fall back to the element's axis-aligned box of size `extent`.
    """
    mask = np.zeros((GRID, GRID), dtype=bool)
    half = 0.0 if center_only else extent / 2
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
        for t in self.trials:
            i = t["outlier"]["index"]
            # patch index of every element; the outlier's rank in raster order
            # is its slot among the masked tokens (one element per patch)
            patch = [int((224 - y) // PATCH_PX) * GRID + int(x // PATCH_PX)
                     for x, y in zip(t["positions_x"], t["positions_y"])]
            slots.append(sorted(patch).index(patch[i]))
        info["outlier_slot"] = torch.tensor(slots)
        return info

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, i):
        t = self.trials[i]
        image = self.normalize(Image.open(self.root / t["filename"]).convert("RGB"))
        mask = torch.from_numpy(make_patch_mask(
            t["positions_x"], t["positions_y"], self.extent, self.center_only))
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
def extract(model, dataset, device, batch_size=32, out_dtype=torch.float16):
    # concatenated slices are only well-defined when every trial has the
    # same number of element patches
    concat = len(dataset.set_sizes) == 1
    loader = DataLoader(dataset, batch_size, num_workers=4, collate_fn=collate)
    keys = ["cls", "bar", "bg", "all_patches"] + (["bar_concat", "bg_concat"] if concat else [])
    out = {k: [] for k in keys + ["labels_mean", "labels_var", "n_bars"]}
    filenames = []

    for images, masks, lm, lv, n, names in loader:
        layers = model.get_intermediate_layers(
            images.to(device), n=N_LAYERS, return_class_token=True)
        per_layer = {k: [] for k in keys}
        for patches, cls in layers:
            patches = patches.float().cpu().to(out_dtype)             # (B, 256, D)
            per_layer["cls"].append(cls.float().cpu().to(out_dtype))
            per_layer["all_patches"].append(patches.float().mean(1).to(out_dtype))
            per_layer["bar"].append(masked_mean(patches, masks))
            per_layer["bg"].append(masked_mean(patches, ~masks))
            if concat:
                per_layer["bar_concat"].append(masked_concat(patches, masks))
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
    p.add_argument("--model", default="dinov2_vitb14")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=pick_device())
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--n_bars", type=int, default=None,
                   help="only extract this set size (needed for *_concat slices)")
    a = p.parse_args()

    dataset = StimuliDataset(a.metadata, a.n_bars)
    model = torch.hub.load("facebookresearch/dinov2", a.model).to(a.device).eval()
    dtype = torch.float16 if a.precision == "fp16" else torch.float32
    feats = extract(model, dataset, a.device, a.batch_size, dtype)

    for k, v in feats.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s} {tuple(v.shape)}")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, a.output)
    print(f"Saved → {a.output}")


if __name__ == "__main__":
    main()
