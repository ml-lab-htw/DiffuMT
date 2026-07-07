"""Build the DiffuMT triplet dataset from the 309 labeled real crops.

The labeled real crops are 512x512 with pixel-precise binary masks. Instead of a
fixed tiling we sample *arbitrarily many* random 256x256 views, which lets us grow
a large, diverse triplet set from a small annotated pool. For every 256 view we
(a) keep the real crop and its binary mask and (b) generate a mask-conditioned
synthetic counterpart with the diagnostic-selected diffusion checkpoint (epoch
290). Splits are drawn by *recording group* (no source recording appears in more
than one split), so train/val/test are leakage-free.

    train  2000 triplets      val  400 triplets      test  400 triplets

Pipeline (all local, Apple-MPS):
    --stage      crop + recording-grouped split   (no GPU)  -> staging + manifest
    --generate   sample synthetic per staged mask (MPS)     -> synthetic/  (resumable)
    --push       assemble anonymised triplets -> push_to_hub

Sampling matches training/SynthMT-Studio exactly: images in [-1,1]
(Normalize(0.5,0.5)); the seg is scaled to {0, 1/255~=0.0039} (NOT {0,1} --
that is 255x too strong and pushes the model out of distribution) and
concatenated as a 4th channel to the 3-channel noisy image at every DDIM step;
deterministic DDIM (eta=0) from the checkpoint's saved scheduler.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "diffumt_src"                 # real/ + mask/  (309 pairs, 512x512)
STAGE = REPO / "data" / "diffumt_build"             # staging + synthetic + manifest
CKPT = REPO / "checkpoint-0290"                     # diagnostic-selected model
REPO_ID = "HTW-KI-Werkstatt/DiffuMT"

CROP = 256
TARGETS = {"train": 2000, "val": 400, "test": 400}
SPLIT_FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_FG = 0.002          # >=0.2% mask foreground -> the view actually contains MTs
MAX_TRIES = 40          # random-window attempts per accepted crop
SEED = 1234


def group_of(stem: str) -> str:
    """Recording group = filename prefix before the first crop/frame/tile token.
    (No \\b after the token: 'crop_7' has no word boundary before '_'.)"""
    return re.split(r"_(?:crop|frame|tile)", stem, maxsplit=1)[0]


# --------------------------------------------------------------------------- stage
def stage() -> None:
    reals = {p.stem: p for p in sorted((SRC / "real").glob("*.png"))}
    masks = {p.stem: p for p in sorted((SRC / "mask").glob("*.png"))}
    stems = [s for s in reals if s in masks]
    if not stems:
        sys.exit(f"no real/mask pairs under {SRC}")

    groups: dict[str, list[str]] = {}
    for s in stems:
        groups.setdefault(group_of(s), []).append(s)
    gnames = sorted(groups)
    rng = np.random.RandomState(SEED)
    rng.shuffle(gnames)
    n = len(gnames)
    n_tr = int(round(SPLIT_FRAC["train"] * n))
    n_va = int(round(SPLIT_FRAC["val"] * n))
    split_groups = {"train": gnames[:n_tr], "val": gnames[n_tr:n_tr + n_va],
                    "test": gnames[n_tr + n_va:]}
    print(f"{n} recording groups -> train {len(split_groups['train'])} / "
          f"val {len(split_groups['val'])} / test {len(split_groups['test'])}")

    manifest = {"crop": CROP, "min_fg": MIN_FG, "seed": SEED, "splits": {}}
    for split, gl in split_groups.items():
        srcs = [s for g in gl for s in groups[g]]
        target = TARGETS[split]
        out_real = STAGE / split / "real"; out_mask = STAGE / split / "mask"
        out_real.mkdir(parents=True, exist_ok=True); out_mask.mkdir(parents=True, exist_ok=True)
        # even quota per source crop, remainder spread over the first sources
        base, rem = divmod(target, len(srcs))
        quota = {s: base + (1 if i < rem else 0) for i, s in enumerate(srcs)}
        items = []
        k = 0
        for s in srcs:
            rimg = np.asarray(Image.open(reals[s]).convert("RGB"))
            mimg = np.asarray(Image.open(masks[s]).convert("L"))
            H, W = mimg.shape
            srng = np.random.RandomState(abs(hash((SEED, s))) % (2**32))
            made = 0
            attempts = 0
            while made < quota[s] and attempts < quota[s] * MAX_TRIES + 50:
                attempts += 1
                y = srng.randint(0, H - CROP + 1); x = srng.randint(0, W - CROP + 1)
                mc = mimg[y:y + CROP, x:x + CROP]
                if (mc > 0).mean() < MIN_FG:
                    continue
                cid = f"diffumt_{split}_{k:05d}"
                Image.fromarray(rimg[y:y + CROP, x:x + CROP], "RGB").save(out_real / f"{cid}.png")
                Image.fromarray((mc > 0).astype("uint8") * 255, "L").save(out_mask / f"{cid}.png")
                items.append({"id": cid, "group": group_of(s)})
                made += 1; k += 1
        manifest["splits"][split] = {"n": len(items), "items": items}
        print(f"[stage] {split}: {len(items)} crops from {len(srcs)} sources -> {out_real.parent}")
    (STAGE).mkdir(parents=True, exist_ok=True)
    (STAGE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[stage] manifest -> {STAGE/'manifest.json'}")


# ------------------------------------------------------------------------ generate
def _load_pipe(device):
    # Deterministic DDIM (eta=0) from the checkpoint's saved scheduler config --
    # exactly the sampler the SynthMT-Studio app / the released "ours" images use.
    from diffusers import UNet2DModel, DDIMScheduler
    unet = UNet2DModel.from_pretrained(str(CKPT / "unet"), use_safetensors=True).to(device).eval()
    sched = DDIMScheduler.from_pretrained(str(CKPT / "scheduler"))
    return unet, sched


def generate(steps: int, batch: int, device_str: str) -> None:
    import torch
    device = torch.device(device_str)
    unet, sched = _load_pipe(device)
    sched.set_timesteps(steps)
    manifest = json.loads((STAGE / "manifest.json").read_text())
    for split in ("train", "val", "test"):
        mask_dir = STAGE / split / "mask"
        syn_dir = STAGE / split / "synthetic"; syn_dir.mkdir(parents=True, exist_ok=True)
        ids = [it["id"] for it in manifest["splits"][split]["items"]]
        todo = [i for i in ids if not (syn_dir / f"{i}.png").exists()]
        print(f"[gen] {split}: {len(todo)}/{len(ids)} to sample (DDIM {len(sched.timesteps)} steps, bs={batch}, {device_str})")
        for b0 in range(0, len(todo), batch):
            chunk = todo[b0:b0 + batch]
            # CRITICAL: the model was trained on {0,1} label PNGs run through
            # ToTensor (/255), so the conditioning channel is {0, 1/255~=0.0039},
            # NOT {0,1}. Feeding {0,1} is 255x too strong and pushes the model out
            # of distribution (dark, grainy, thick filaments). Match training.
            seg = torch.stack([
                torch.from_numpy((np.asarray(Image.open(mask_dir / f"{i}.png").convert("L")) > 0)
                                 .astype("float32") / 255.0)[None] for i in chunk]).to(device)
            # per-crop reproducible seed; cpu generator (MPS randn not reproducible)
            g = torch.Generator(device="cpu").manual_seed(SEED + b0)
            img = torch.randn(len(chunk), 3, CROP, CROP, generator=g).to(device)
            with torch.no_grad():
                for t in sched.timesteps:
                    eps = unet(torch.cat([img, seg], dim=1), t).sample
                    img = sched.step(eps, t, img).prev_sample  # deterministic DDIM (eta=0)
            out = ((img / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
            for i, cid in enumerate(chunk):
                Image.fromarray(out[i], "RGB").save(syn_dir / f"{cid}.png")
            print(f"    {split}: {min(b0+batch,len(todo))}/{len(todo)}", flush=True)
    print("[gen] done")


# ---------------------------------------------------------------------------- push
def push(dry: bool, private: bool) -> None:
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
    manifest = json.loads((STAGE / "manifest.json").read_text())
    dd = {}
    for split in ("train", "val", "test"):
        ids = [it["id"] for it in manifest["splits"][split]["items"]]
        rd, md, sd = (STAGE / split / k for k in ("real", "mask", "synthetic"))
        ids = [i for i in ids if (rd / f"{i}.png").exists() and (md / f"{i}.png").exists()
               and (sd / f"{i}.png").exists()]

        def gen(ids=ids, rd=rd, md=md, sd=sd):
            for cid in ids:
                yield {"id": cid,
                       "image": Image.open(rd / f"{cid}.png").convert("RGB"),
                       "mask": Image.open(md / f"{cid}.png").convert("L"),
                       "synthetic": Image.open(sd / f"{cid}.png").convert("RGB")}
        feats = Features({"id": Value("string"), "image": HFImage(),
                          "mask": HFImage(), "synthetic": HFImage()})
        dd[split] = Dataset.from_generator(gen, features=feats)
        print(f"[push] {split}: {len(ids)} complete triplets")
    dsd = DatasetDict(dd)
    if dry:
        print("[dry-run] not pushing.")
        return
    if not os.environ.get("HF_TOKEN"):
        print("[warn] HF_TOKEN not set; relying on cached login (hf auth login).")
    dsd.push_to_hub(REPO_ID, private=private,
                    commit_message="DiffuMT: anonymised mask/real/synthetic triplets (256px, recording-grouped)")
    print(f"[done] https://huggingface.co/datasets/{REPO_ID}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="with --push: build but don't upload")
    ap.add_argument("--private", action="store_true", help="with --push: create the repo private")
    ap.add_argument("--steps", type=int, default=50)    # DDIM eta=0 (SynthMT-Studio default)
    ap.add_argument("--batch", type=int, default=4)     # MPS compute-bound: batch ~irrelevant
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    if not (args.stage or args.generate or args.push):
        ap.error("pick at least one of --stage / --generate / --push")
    if args.stage:
        stage()
    if args.generate:
        generate(args.steps, args.batch, args.device)
    if args.push:
        push(args.dry_run, args.private)


if __name__ == "__main__":
    main()
