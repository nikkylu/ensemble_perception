"""
DINOv2 feature extraction for orientation stimuli.
Extracts per-layer CLS, bar-patch, bg-patch, and all-patch representations.

Output: .pt dict with keys
    cls / bar / bg / all_patches : (N, L, D) tensors
    labels_mean / labels_var     : (N,) int tensors (classification bins)
    n_bars                       : (N,) int tensor
    filenames                    : list[str]

Usage:
    python loader.py --metadata stimuli_v2/metadata.json --output features.pt
"""

import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# DINOv2 ViT-B/14 on 224px input
PATCH_PX = 14
GRID = 16       # 224 / 14 = 16 patches per side
N_PATCHES = 256  # 16 * 16
N_LAYERS = 12
EMBED_DIM = 768


# ─────────────────────────────────────────────────────────────────
# PATCH MASK
# ─────────────────────────────────────────────────────────────────

def make_patch_mask(bar_xs, bar_ys, bar_len=12.0):
    """
    Given bar center positions (in pixels), return a (256,) bool array
    where True means that patch contains part of a bar.
    """
    mask = np.zeros((GRID, GRID), dtype=bool)
    half = bar_len / 2

    for x, y in zip(bar_xs, bar_ys):
        y = 224 - y  # convert to image coords (origin top-left)
        # bounding box of this bar in pixel coords
        left = max(0, x - half)
        right = min(223, x + half)
        top = max(0, y - half)
        bottom = min(223, y + half)

        # convert to patch grid indices
        col_start = int(left // PATCH_PX)
        col_end = min(int(right // PATCH_PX), GRID - 1)
        row_start = int(top // PATCH_PX)
        row_end = min(int(bottom // PATCH_PX), GRID - 1)

        mask[row_start:row_end+1, col_start:col_end+1] = True

    flat = mask.ravel()
    print(
        f"make_patch_mask: {flat.sum()} patches contain bars (n_bars={len(bar_xs)})")
    return flat


# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────

class StimuliDataset(Dataset):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    def __init__(self, meta_path):
        meta_path = Path(meta_path)
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.trials = self.meta["trials"]
        self.root = meta_path.parent.parent
        self.bar_len = self.meta["params"].get("bar_length", 12.0)

        # build integer label maps:
        #   mean orientation (e.g. 0,30,60,90,120,150) -> (0,1,2,3,4,5)
        #   variance         (e.g. 5,10,15)            -> (0,1,2)
        means = sorted({t["mean_orientation"] for t in self.trials})
        varis = sorted({t["variance"] for t in self.trials})
        self.mean_to_label = {m: i for i, m in enumerate(means)}
        self.var_to_label = {v: i for i, v in enumerate(varis)}

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]

        # load and normalize image
        image = Image.open(self.root / trial["filename"]).convert("RGB")
        image = self.normalize(image)

        # compute which patches contain bars
        patch_mask = make_patch_mask(
            trial["positions_x"], trial["positions_y"], self.bar_len
        )
        patch_mask = torch.from_numpy(patch_mask)

        # integer classification labels
        label_mean = self.mean_to_label[trial["mean_orientation"]]
        label_var = self.var_to_label[trial["variance"]]

        return (image, patch_mask,
                label_mean, label_var,
                trial["n_bars"], trial["filename"])


def collate(batch):
    """
    DataLoader can't auto-collate mixed types (tensors + strings).
    This manually stacks each field across the batch.
    """
    images, masks, label_means, label_vars, n_bars, filenames = zip(*batch)

    return (
        torch.stack(images),            # (B, 3, 224, 224)
        torch.stack(masks),             # (B, 256) bool
        torch.tensor(label_means),      # (B,) int
        torch.tensor(label_vars),       # (B,) int
        torch.tensor(n_bars),           # (B,) int
        list(filenames),                # list of B strings
    )


# ─────────────────────────────────────────────────────────────────
# MASKED MEAN-POOL (vectorized)
# ─────────────────────────────────────────────────────────────────

def masked_mean(tokens, mask):
    """
    Mean-pool only the tokens where mask is True.

    tokens: (B, 256, 768) — all patch tokens for this layer
    mask:   (B, 256)       — True = bar patch, False = background

    Returns: (B, 768) — one pooled vector per image
    """
    # expand mask to match token dimensions: (B, 256) -> (B, 256, 1)
    weights = mask.unsqueeze(-1).float()

    # zero out tokens where mask is False, then sum
    weighted_sum = (tokens * weights).sum(dim=1)    # (B, 768)

    # count how many True patches per image (avoid /0)
    count = weights.sum(dim=1).clamp(min=1)         # (B, 1)

    return weighted_sum / count


def masked_concat(tokens, mask):
    """
    Concatenate only the tokens where mask is True.
    Requires all images in the batch to have the same number of True patches.

    tokens: (B, 256, 768)
    mask:   (B, 256) bool

    Returns: (B, n_bar_patches * 768) — concatenated bar patch tokens
             or None if patch counts are inconsistent
    """
    counts = mask.sum(dim=1)
    if counts.min() != counts.max():
        return None  # inconsistent patch counts across batch

    n_bar = int(counts[0].item())
    if n_bar == 0:
        return None

    result = []
    for i in range(tokens.shape[0]):
        bar_tokens = tokens[i, mask[i]]    # (n_bar, 768)
        result.append(bar_tokens.reshape(-1))  # (n_bar * 768,)

    return torch.stack(result)  # (B, n_bar * 768)


# ─────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(model, dataloader, device, n_layers=N_LAYERS):
    """
    Run all images through DINOv2 and collect four types of
    representations at every layer:
        cls         — CLS token
        bar         — mean-pooled bar patches
        bg          — mean-pooled background patches
        all_patches — mean-pooled all patches (naive baseline)
    """
    # accumulators — one list per slice type
    all_cls, all_bar, all_bg, all_avg, all_bar_concat = [], [], [], [], []
    all_label_mean, all_label_var, all_n_bars, all_filenames = [], [], [], []

    for images, masks, label_means, label_vars, n_bars, filenames in dataloader:

        # get representations from every transformer layer
        # returns list of (patch_tokens, cls_token) tuples, one per layer
        layer_outputs = model.get_intermediate_layers(
            images.to(device), n=n_layers, return_class_token=True
        )

        # process each layer
        cls_per_layer = []
        bar_per_layer = []
        bg_per_layer = []
        avg_per_layer = []
        bar_concat_per_layer = []

        for patch_tokens, cls_token in layer_outputs:
            patch_tokens = patch_tokens.cpu()   # (B, 256, 768)
            cls_token = cls_token.cpu()         # (B, 768)

            cls_per_layer.append(cls_token)
            # mean over ALL patches
            avg_per_layer.append(patch_tokens.mean(dim=1))
            # mean over bar patches
            bar_per_layer.append(masked_mean(patch_tokens, masks))
            # mean over background patches
            bg_per_layer.append(masked_mean(patch_tokens, ~masks))

            # concatenated bar patches (only works when all images have same n_bars)
            concat = masked_concat(patch_tokens, masks)
            bar_concat_per_layer.append(concat)

        # stack layers: list of 12 x (B, 768) -> (12, B, 768) -> (B, 12, 768)
        all_cls.append(torch.stack(cls_per_layer).permute(1, 0, 2))
        all_bar.append(torch.stack(bar_per_layer).permute(1, 0, 2))
        all_bg.append(torch.stack(bg_per_layer).permute(1, 0, 2))
        all_avg.append(torch.stack(avg_per_layer).permute(1, 0, 2))

        # bar_concat: (12, B, n_bar*768) -> (B, 12, n_bar*768), only if all layers succeeded
        if all(c is not None for c in bar_concat_per_layer):
            all_bar_concat.append(torch.stack(
                bar_concat_per_layer).permute(1, 0, 2))

        # accumulate labels
        all_label_mean.append(label_means)
        all_label_var.append(label_vars)
        all_n_bars.append(n_bars)
        all_filenames.extend(filenames)

    # concatenate all batches: list of (B, 12, 768) -> (N, 12, 768)
    output = {
        "cls":          torch.cat(all_cls),
        "bar":          torch.cat(all_bar),
        "bg":           torch.cat(all_bg),
        "all_patches":  torch.cat(all_avg),
        "labels_mean":  torch.cat(all_label_mean),
        "labels_var":   torch.cat(all_label_var),
        "n_bars":       torch.cat(all_n_bars),
        "filenames":    all_filenames,
    }

    # bar_concat only available when all images have same n_bars
    if all_bar_concat:
        dims = [t.shape[-1] for t in all_bar_concat]
    if len(set(dims)) == 1:
        output["bar_concat"] = torch.cat(all_bar_concat)
    else:
        print("  bar_concat skipped (inconsistent patch counts)")

    return output


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="features.pt")
    parser.add_argument("--model", default="dinov2_vitb14")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_bars", type=int, default=None,
                        help="Only extract features for this set size")
    args = parser.parse_args()

    # load dataset
    dataset = StimuliDataset(args.metadata)
    if args.n_bars:
        indices = [i for i, t in enumerate(
            dataset.trials) if t["n_bars"] == args.n_bars]
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(
        dataset, args.batch_size, num_workers=4, pin_memory=True, collate_fn=collate
    )

    # load model
    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model = model.to(args.device).eval()

    # extract features
    features = extract(model, dataloader, args.device)

    # print summary
    for key, val in features.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:15s} {str(val.shape):>20s}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
