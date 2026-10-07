# DiffuMT — Mask-Conditioned Diffusion for Synthetic Microtubule Images

[![CI](https://github.com/HTW-KI-Werkstatt/DiffuMT/actions/workflows/ci.yml/badge.svg)](https://github.com/HTW-KI-Werkstatt/DiffuMT/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/HTW-KI-Werkstatt/DiffuMT)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange)](https://huggingface.co/datasets/HTW-KI-Werkstatt/DiffuMT)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/HTW-KI-Werkstatt/DiffuMT)

Source code for the paper **"Diagnosing Diversity Collapse and Validating Mask-Conditioned Diffusion for Labeled Microtubule Microscopy"** *(under review)*.

We present a three-axis diagnostic — DINOv2 inter/intra-similarity, CIELAB color distribution, and baseline-aware mask fidelity — for evaluating and selecting checkpoints from a mask-conditioned diffusion model trained on IRM microtubule images.

---

## Quickstart

```python
# Generate an image from a mask — no local data needed
from diffusers import UNet2DModel, DDIMScheduler
import torch, numpy as np
from PIL import Image
from torchvision import transforms

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

unet = UNet2DModel.from_pretrained("HTW-KI-Werkstatt/DiffuMT", subfolder="unet").to(device).eval()
scheduler = DDIMScheduler.from_pretrained("HTW-KI-Werkstatt/DiffuMT", subfolder="scheduler")
scheduler.set_timesteps(50)

# Load your binary mask (or generate a synthetic one)
mask_pil = Image.open("mask.png").convert("L").resize((256, 256))
seg = (transforms.ToTensor()(mask_pil) / 255.0).unsqueeze(0).to(device)  # {0, 1/255}

gen = torch.Generator(device=device).manual_seed(42)
x = torch.randn((1, 3, 256, 256), generator=gen, device=device)
with torch.no_grad():
    for t in scheduler.timesteps:
        x = scheduler.step(unet(torch.cat([x, seg], dim=1), t).sample, t, x).prev_sample

image = Image.fromarray((((x / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()) * 255).astype(np.uint8))
```

See [`notebooks/01_sampling_demo.ipynb`](notebooks/01_sampling_demo.ipynb) for a full walkthrough.

---

## Repository structure

```
DiffuMT/
├── main.py                        # Training / sampling entry point
├── training.py                    # Training loop (DDPM/DDIM, offset noise)
├── eval.py                        # SegGuidedDDIMPipeline
├── utils.py                       # ScaleSeg (mask → {0, 1/255}), AddGaussianNoise
│
├── eval_pipeline/                 # Three-axis diagnostic suite
│   ├── eval_config.py             # Metric weights and paths
│   ├── 01_sample_checkpoints.py   # Generate images for each checkpoint
│   ├── 02a_evaluate_masks.py      # Mask fidelity (SAM3Text; needs sam3 env)
│   ├── 02b_evaluate_sim.py        # DINOv2 inter/intra similarity + FID/KID
│   ├── 02c_evaluate_color.py      # CIELAB foreground/background EMD
│   ├── 03_select_best_epoch.py    # Select checkpoint by geometric-mean score
│   ├── scoring.py                 # Composite score (geometric mean, paper eq. 1)
│   └── sam3.py                    # SAM3 wrapper
│
├── experiments_downstream/        # TSTR downstream evaluation
│   ├── build_diffumt.py           # Build the DiffuMT dataset from masks + images
│   ├── tstr_unet.py               # Train-on-Synthetic Test-on-Real (U-Net)
│   ├── gather_results.py          # Aggregate result JSONs → summary + LaTeX macros
│   └── upload_diffumt.py          # Upload dataset to Hugging Face
│
├── helper_scripts/
│   ├── data_prep/                 # Data preparation utilities
│   └── eval/                     # Eval internals (color, similarity, mask metrics)
│                                  # — imported by eval_pipeline and tstr_unet
│
├── notebooks/
│   └── 01_sampling_demo.ipynb    # End-to-end sampling demo (HF model, no local data)
│
├── tests/                         # CPU-only unit tests (CI)
└── requirements.txt
```

---

## Installation

```bash
# Main environment — training, sampling, evaluation
conda create -n diffumt python=3.11
conda activate diffumt
pip install torch torchvision          # add --index-url for CUDA if needed
pip install -r requirements.txt

# SAM3 environment — mask fidelity evaluation only (02a)
conda create -n sam3 python=3.11
conda activate sam3
pip install git+https://github.com/JoaoMarcosCSilva/SAM3.git
```

---

## Training

```bash
python main.py --mode train \
    --dataset microtubule --img_size 256 --num_img_channels 3 \
    --seg_dir data/masks --img_dir data/imgs \
    --segmentation_guided --num_segmentation_classes 2 \
    --offset_noise --num_epochs 600 --save_model_every 20
```

Offset noise (`--offset_noise`) is important for matching the narrow brightness distribution of IRM acquisitions. Save every 20 epochs so the diagnostic has enough checkpoints to rank.

---

## Diagnostic evaluation

```bash
# 1. Sample images from each checkpoint
python eval_pipeline/01_sample_checkpoints.py

# 2. Score the three axes
python eval_pipeline/02b_evaluate_sim.py --runs_dir results/my_run
python eval_pipeline/02c_evaluate_color.py --run_dir results/my_run
# optional: python eval_pipeline/02a_evaluate_masks.py  (needs sam3 env)

# 3. Select the best checkpoint
python eval_pipeline/03_select_best_epoch.py --run_dir results/my_run
```

The composite score is the geometric mean of the three axes — if any axis collapses (diversity, realism, or color), the global score drops to zero.

---

## Downstream evaluation (TSTR)

```bash
export DATA_ROOT=/path/to/data

# Train a U-Net on synthetic arm v2 (fold 0 of 4)
python experiments_downstream/tstr_unet.py --arm v2 --fold 0 --n-folds 4

# Aggregate all result JSONs from results/downstream/
python experiments_downstream/gather_results.py
```

Arms: `real` (oracle), `v1` (SynthMT parametric), `v2` (DiffuMT diagnostic-selected), `real_v2` (combined).

---

## Checkpoints

| Checkpoint | Epoch | Use |
|---|---|---|
| `checkpoint-0290` | 290 | Diagnostic-selected — use for paper results |
| `checkpoint-0400` | 400 | Last epoch (no diagnostic) — ablation only |

Hosted at [huggingface.co/HTW-KI-Werkstatt/DiffuMT](https://huggingface.co/HTW-KI-Werkstatt/DiffuMT).

```python
from diffusers import UNet2DModel, DDIMScheduler
unet      = UNet2DModel.from_pretrained("HTW-KI-Werkstatt/DiffuMT", subfolder="unet")
scheduler = DDIMScheduler.from_pretrained("HTW-KI-Werkstatt/DiffuMT", subfolder="scheduler")
```

### Critical: mask conditioning scale

The model was trained with masks scaled to `{0, 1/255}` — **not** `{0, 1}`. A wrong scale (255× stronger signal) produces dark, grainy output. Always apply:

```python
seg = (transforms.ToTensor()(mask_pil) / 255.0).unsqueeze(0)  # {0, 1/255}
```

`ScaleSeg` in `utils.py` does the same thing.

---

## Dataset

**DiffuMT** — 2800 mask/real/synthetic triplets (2000 train / 400 val / 400 test):

```python
from datasets import load_dataset
ds = load_dataset("HTW-KI-Werkstatt/DiffuMT")
```

Splits are recording-group stratified to prevent geometry leakage.

---

## Demo

Draw a binary mask and watch DDIM sampling in real time:
[huggingface.co/spaces/HTW-KI-Werkstatt/DiffuMT](https://huggingface.co/spaces/HTW-KI-Werkstatt/DiffuMT)

---

## Citation

```bibtex
@inproceedings{koddenbrock2027diffumt,
  title     = {Diagnosing Diversity Collapse and Validating Mask-Conditioned
               Diffusion for Labeled Microtubule Microscopy},
  author    = {Koddenbrock, Mario and Rapp, Frederic and Reber, Simone and
               Rodner, Erik},
  booktitle = {Proceedings of Machine Learning Research (PMLR)},
  year      = {2027},
  note      = {Northern Lights Deep Learning Conference (NLDL), Spotlight}
}

@inproceedings{konz2024segguideddiffusion,
  title     = {Anatomically-Controllable Medical Image Generation with
               Segmentation-Guided Diffusion Models},
  author    = {Nicholas Konz and Yuwen Chen and Haoyu Dong and Maciej A. Mazurowski},
  booktitle = {MICCAI},
  year      = {2024}
}
```
