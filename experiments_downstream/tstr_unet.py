"""Exp1 (downstream utility): train a U-Net on one data arm, test on real.

Train-synthetic-test-real (TSTR). For each training arm we train a small U-Net
for binary microtubule foreground segmentation and evaluate it on a held-out set
of *real* images, scoring with the same skeleton-IoU metric the paper uses
(``calculate_segmentation_metrics`` from ``helper_scripts/eval/mask``).

Arms (``--arm``):
    real     - real images + their binary GT masks
    v1       - parametric SynthMT v1 synthetic images + masks
    v2       - diffusion-model (SynthMT v2) synthetic images + masks
    real_v2  - union of real + v2

Leakage control: the real images are split into train/test groups *by source
recording* (the filename prefix before ``_frame``). The held-out test groups are
fixed for every arm. For the ``real`` and ``real_v2`` arms the real-train groups
are used; v2 images whose source mask belongs to a test group are dropped (v2 is
mask-paired to real, so this prevents test-geometry leakage).

One run = one (arm, fold, seed). Writes a one-row JSON result so an array job can
fan out over arms x folds and a gather step can aggregate.

Run on the cluster (``mt`` env: torch + skimage + scipy).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import glob
import random
from pathlib import Path

import numpy as np
try:
    import imageio.v2 as imageio          # present in the cluster `mt` env
except ModuleNotFoundError:
    imageio = None                          # allow a model-only local smoke test
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- skeleton-IoU scorer (reused, not reimplemented) ----------------------
# Imported lazily inside evaluate() so the training loop runs in environments
# without skimage/scipy (e.g. a quick local smoke test of the model + data).
def _load_scorer():
    helpers = Path(__file__).resolve().parents[1] / "helper_scripts" / "eval" / "mask"
    sys.path.insert(0, str(helpers))
    from mask_metrics import calculate_segmentation_metrics
    return calculate_segmentation_metrics


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def group_of(fname: str) -> str:
    """Source-recording key: everything before the first '_frame'/'_tile' token."""
    base = os.path.basename(fname)
    for sep in ("_frame", "_tile"):
        if sep in base:
            return base.split(sep)[0]
    return os.path.splitext(base)[0]


def _read_gray(path: str) -> np.ndarray:
    a = np.asarray(imageio.imread(path))
    if a.ndim == 3:
        a = a[..., 0]
    return a


def _read_binary(path: str) -> np.ndarray:
    """Binary foreground (any microtubule pixel), robust to the mask formats:
    2D label image, RGB PNG, or multi-page instance-stack TIFF."""
    a = np.asarray(imageio.imread(path))
    if a.ndim == 3 and a.shape[-1] in (3, 4):      # RGB/RGBA -> one channel
        a = a[..., 0]
    elif a.ndim == 3:                               # instance stack -> OR frames
        a = np.any(a > 0, axis=0)
    elif a.ndim > 3:
        a = np.any(a.reshape(-1, *a.shape[-2:]) > 0, axis=0)
    return (a > 0).astype(np.float32)


def list_pairs(img_dir: str, mask_dir: str) -> list[tuple[str, str]]:
    """Pair images to masks by basename (try same ext, then .tif/.png)."""
    pairs = []
    for img in sorted(glob.glob(os.path.join(img_dir, "*.png"))
                      + glob.glob(os.path.join(img_dir, "*.tif"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        for ext in (".png", ".tif", ".tiff"):
            m = os.path.join(mask_dir, stem + ext)
            if os.path.exists(m):
                pairs.append((img, m))
                break
    return pairs


class SegSet(torch.utils.data.Dataset):
    def __init__(self, pairs, augment: bool, size: int = 256):
        self.pairs = pairs
        self.augment = augment
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_p, mask_p = self.pairs[i]
        img = _read_gray(img_p).astype(np.float32) / 255.0
        mask = _read_binary(mask_p)
        if img.shape != mask.shape:
            mask = mask[: img.shape[0], : img.shape[1]]
        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                img, mask = img[::-1, :].copy(), mask[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()
            # mild intensity jitter (synthetic-to-real robustness)
            img = np.clip(img * random.uniform(0.85, 1.15)
                          + random.uniform(-0.05, 0.05), 0, 1)
        x = torch.from_numpy(img)[None]          # (1,H,W)
        y = torch.from_numpy(mask)[None]         # (1,H,W)
        return x.float(), y.float()


# ---------------------------------------------------------------------------
# Compact U-Net
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.d1 = DoubleConv(1, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        x = self.u3(torch.cat([self.up3(c4), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.out(x)


def dice_bce_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def build_arms(roots: dict, test_groups: set, train_groups: set):
    """Return {arm: list[(img,mask)]} respecting the group split."""
    real = [p for p in list_pairs(roots["real_img"], roots["real_mask"])
            if group_of(p[0]) in train_groups]
    v2 = [p for p in list_pairs(roots["v2_img"], roots["v2_mask"])
          if group_of(p[0]) not in test_groups]          # drop test-geometry leakage
    v2last = [p for p in list_pairs(roots["v2last_img"], roots["v2last_mask"])
              if group_of(p[0]) not in test_groups]       # Last-epoch (no-diagnostic) arm
    v1 = list_pairs(roots["v1_img"], roots["v1_mask"])
    if roots.get("v1_limit"):
        random.shuffle(v1)
        v1 = v1[: roots["v1_limit"]]
    return {"real": real, "v1": v1, "v2": v2, "real_v2": real + v2,
            "v2last": v2last, "real_v2last": real + v2last}


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, pairs, dev) -> dict:
    calculate_segmentation_metrics = _load_scorer()
    model.eval()
    gts, preds = [], []
    for img_p, mask_p in pairs:
        img = _read_gray(img_p).astype(np.float32) / 255.0
        x = torch.from_numpy(img)[None, None].float().to(dev)
        logit = model(x)
        pred = (torch.sigmoid(logit)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        gts.append(_read_binary(mask_p).astype(np.uint8))
        preds.append(pred)
    skel, _ = calculate_segmentation_metrics(gt_masks=gts, pred_masks=preds,
                                             use_skeletonized_version=True,
                                             thresholds=[0.5, 0.75])
    classic, _ = calculate_segmentation_metrics(gt_masks=gts, pred_masks=preds,
                                               use_skeletonized_version=False,
                                               thresholds=[0.5, 0.75])
    return {
        "SKIoU": float(skel.get("SKIoU_mean", 0.0)),
        "F1@0.50": float(skel.get("F1@0.50", 0.0)),
        "F1@0.75": float(skel.get("F1@0.75", 0.0)),
        "IoU": float(classic.get("IoU/T_mean", 0.0)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True,
                    choices=["real", "v1", "v2", "real_v2", "v2last", "real_v2last"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--v1-limit", type=int, default=2000)
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    ap.add_argument("--v2-img-dir", default=None,
                    help="Override the v2 images directory (default: <data-root>/SynMT_DiffMT/v2/images)")
    ap.add_argument("--v2-mask-dir", default=None,
                    help="Override the v2 masks directory (default: <data-root>/SynMT_DiffMT/v2/masks_binary)")
    ap.add_argument("--out", default="results/downstream")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate wiring")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dr = args.data_root
    roots = {
        "real_img":  f"{dr}/SynMT_DiffMT/real/images",
        "real_mask": f"{dr}/SynMT_DiffMT/real/masks_binary",
        "v2_img":    args.v2_img_dir if args.v2_img_dir else f"{dr}/SynMT_DiffMT/v2/images",
        "v2_mask":   args.v2_mask_dir if args.v2_mask_dir else f"{dr}/SynMT_DiffMT/v2/masks_binary",
        "v2last_img":  f"{dr}/SynMT_DiffMT/v2last/images",
        "v2last_mask": f"{dr}/SynMT_DiffMT/v2last/masks_binary",
        "v1_img":    f"{dr}/SynMT/synthetic/full/images",
        "v1_mask":   f"{dr}/SynMT/synthetic/full/image_masks",
        "v1_limit":  args.v1_limit,
    }

    # group-split the real set: fixed test groups per fold
    real_pairs = list_pairs(roots["real_img"], roots["real_mask"])
    groups = sorted({group_of(p[0]) for p in real_pairs})
    rng = random.Random(12345)            # fixed split seed (shared across arms)
    rng.shuffle(groups)
    folds = [groups[i::args.n_folds] for i in range(args.n_folds)]
    test_groups = set(folds[args.fold % args.n_folds])
    train_groups = set(g for g in groups if g not in test_groups)
    test_pairs = [p for p in real_pairs if group_of(p[0]) in test_groups]

    arms = build_arms(roots, test_groups, train_groups)
    train_pairs = arms[args.arm]
    if args.smoke:
        train_pairs = train_pairs[:8]
        args.epochs = 1

    print(f"[arm={args.arm} fold={args.fold} seed={args.seed}] "
          f"train={len(train_pairs)} test={len(test_pairs)} "
          f"(test groups={len(test_groups)})")
    if not train_pairs or not test_pairs:
        sys.exit("empty train or test split -- check data roots")

    dev = device()
    model = UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        SegSet(train_pairs, augment=True), batch_size=args.batch_size,
        shuffle=True, num_workers=4, drop_last=False)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = dice_bce_loss(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss={tot/max(1,len(train_pairs)):.4f}")

    metrics = evaluate(model, test_pairs, dev)
    result = {
        "model": "unet", "arm": args.arm, "fold": args.fold, "seed": args.seed,
        "n_train": len(train_pairs), "n_test": len(test_pairs),
        "epochs": args.epochs, **metrics,
    }
    print("[result]", json.dumps(result))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm_tag = args.arm
    if args.v2_img_dir and args.arm in ("v2", "real_v2"):
        arm_tag = args.arm + "_" + os.path.basename(args.v2_img_dir.rstrip("/"))
    fn = out / f"unet_{arm_tag}_fold{args.fold}_seed{args.seed}.json"
    fn.write_text(json.dumps(result, indent=2))
    print(f"[done] {fn}")


if __name__ == "__main__":
    main()
