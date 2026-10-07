"""Upload the DiffuMT dataset to the Hugging Face Hub.

Target repo: https://huggingface.co/datasets/HTW-KI-Werkstatt/DiffuMT

DiffuMT is a *triplet* dataset — (binary mask, real IRM microtubule image,
mask-conditioned synthetic counterpart) — over the three splits used in the paper:

    test       200 annotated held-out patches   (mask + real + synthetic)
    train     2961 training tiles               (mask + real + synthetic)
    unlabeled 1336 reference patches            (real only)

Matches the SynthMT-paper convention (`scripts/synthetic_data/upload_real.py`):
build a `datasets.Dataset` from a generator and `push_to_hub`.

ANONYMISATION (required): the original filenames encode experiment metadata AND
the recording-group structure, so they are DISCARDED. Each item gets a plain
counter id `diffumt_{idx:05d}`, and images are stored as pixel data (PIL objects,
not file paths) — no original filename ever reaches the Hub. Triplet members
share the same id, so mask/real/synthetic stay aligned.

--------------------------------------------------------------------------------
POSSIBLE?  Yes. Only prerequisite: the data exists locally. real+mask exist for
all splits; the SYNTHETIC counterpart currently exists only for the 200 test
patches (SynMT_DiffMT/v2). Generate one synthetic per train mask and set the
train `synth` path for a full triplet release. A split with no `synth` is pushed
without the synthetic column (a warning is printed).
--------------------------------------------------------------------------------

Requires:  pip install datasets huggingface_hub ; HF_TOKEN with write access.

Usage:
    python upload_diffumt.py --dry-run        # count/validate pairs, don't push
    python upload_diffumt.py                  # build + push_to_hub
    python upload_diffumt.py --src train:synth=/path/to/train_synth   # override a source
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "HTW-KI-Werkstatt/DiffuMT"
ID_PREFIX = "diffumt"      # anonymised id -> f"{ID_PREFIX}_{idx:05d}"

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
SPLITS: dict[str, dict[str, str | None]] = {
    "test": {
        "real":  f"{DATA_ROOT}/SynMT_DiffMT/real/images",
        "mask":  f"{DATA_ROOT}/SynMT_DiffMT/real/masks_binary",
        "synth": f"{DATA_ROOT}/SynMT_DiffMT/v2/images",      # "ours" synthetic
    },
    "train": {
        "real":  f"{DATA_ROOT}/SynMT_DiffMT/train/images",
        "mask":  f"{DATA_ROOT}/SynMT_DiffMT/train/masks_binary",
        "synth": None,   # <- set once train synthetics are generated
    },
    "unlabeled": {
        "real":  f"{DATA_ROOT}/SynMT_DiffMT/unlabeled/images",
        "mask":  None,
        "synth": None,
    },
}

IMG_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

CARD = """\
---
license: cc-by-4.0
pretty_name: DiffuMT
task_categories: [image-segmentation, image-to-image]
tags: [microscopy, microtubules, diffusion, synthetic-data, mask-conditioned]
size_categories: [1K<n<10K]
---

# DiffuMT

Mask-conditioned diffusion dataset of in-vitro IRM **microtubule** microscopy.
Each labelled item is a **triplet** — a pixel-precise **binary mask**, the **real**
recording, and a **mask-conditioned synthetic** counterpart — with images that
domain experts cannot reliably distinguish from real ones in a 2AFC study.

Splits: `test` (200) is live now. `train` (2961) and `unlabeled` (1336, real only)
are planned additions, not yet released.
Filenames are anonymised to `diffumt_{counter}`; mask/real/synthetic of one item
share the same id.

2AFC challenge (integrated into the project page): https://huggingface.co/spaces/HTW-KI-Werkstatt/DiffuMT
"""


def _stems(d: str | None) -> dict[str, Path]:
    if not d or not os.path.isdir(d):
        return {}
    return {f.stem: f for f in sorted(Path(d).iterdir()) if f.suffix.lower() in IMG_EXTS}


def build() -> tuple["DatasetDict", int]:
    from datasets import Dataset, DatasetDict, Features, Image, Value
    from PIL import Image as PILImage

    counter = {"i": 0}
    dd = {}
    for split, srcs in SPLITS.items():
        reals = _stems(srcs.get("real"))
        masks = _stems(srcs.get("mask"))
        synth = _stems(srcs.get("synth"))
        if not reals:
            print(f"[skip] {split}: real dir missing ({srcs.get('real')})")
            continue
        has_mask = bool(masks)
        has_synth = bool(synth)
        if split != "unlabeled" and not has_synth:
            print(f"[warn] {split}: no synthetic dir -> column omitted "
                  f"(generate synthetics, set SPLITS['{split}']['synth'])")
        # item list is defined by the real set; require a mask when masks exist
        stems = [s for s in reals if (not has_mask or s in masks)]
        base = counter["i"]

        def gen(stems=stems, base=base, reals=reals, masks=masks, synth=synth,
                has_mask=has_mask, has_synth=has_synth):
            for k, stem in enumerate(stems):
                rec = {"id": f"{ID_PREFIX}_{base + k:05d}",
                       "image": PILImage.open(reals[stem]).convert("RGB")}
                if has_mask:
                    rec["mask"] = PILImage.open(masks[stem]).convert("L")
                if has_synth:
                    rec["synthetic"] = PILImage.open(synth[stem]).convert("RGB")
                yield rec

        feats = {"id": Value("string"), "image": Image()}
        if has_mask:
            feats["mask"] = Image()
        if has_synth:
            feats["synthetic"] = Image()
        dd[split] = Dataset.from_generator(gen, features=Features(feats))
        counter["i"] = base + len(stems)
        print(f"[build] {split}: {len(stems)} items "
              f"(mask={has_mask} synth={has_synth}) ids {base}..{counter['i']-1}")
    return DatasetDict(dd), counter["i"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="validate + count, don't push")
    ap.add_argument("--src", action="append", default=[],
                    help="override a source, e.g. --src train:synth=/path")
    args = ap.parse_args()
    for ov in args.src:
        loc, path = ov.split("=", 1)
        split, key = loc.split(":", 1)
        SPLITS.setdefault(split, {})[key] = path

    dd, total = build()
    if not dd:
        raise SystemExit("no splits staged (check source paths / DATA_ROOT)")
    print(f"\ntotal anonymised items: {total}")
    if args.dry_run:
        print("[dry-run] not pushing. Re-run without --dry-run to push_to_hub.")
        return
    if not os.environ.get("HF_TOKEN"):
        print("[warn] HF_TOKEN not set; relying on cached login (hf auth login).")
    dd.push_to_hub(REPO_ID, commit_message="Upload DiffuMT (anonymised mask/real/synthetic triplets)")
    # dataset card
    from huggingface_hub import upload_file
    import io
    upload_file(path_or_fileobj=io.BytesIO(CARD.encode()), path_in_repo="README.md",
                repo_id=REPO_ID, repo_type="dataset", commit_message="Add dataset card")
    print(f"[done] https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
