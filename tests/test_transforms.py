"""Tests for image transforms in utils.py — runs CPU-only, no data required."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ScaleSeg, AddGaussianNoise


def test_scale_seg_foreground():
    t = ScaleSeg()
    inp = torch.ones(1, 256, 256)
    out = t(inp)
    assert abs(out.max().item() - 1 / 255.0) < 1e-7, f"Expected 1/255, got {out.max().item()}"


def test_scale_seg_background():
    t = ScaleSeg()
    inp = torch.zeros(1, 256, 256)
    out = t(inp)
    assert out.max().item() == 0.0


def test_scale_seg_mixed():
    """Only non-zero pixels should map to 1/255."""
    t = ScaleSeg()
    inp = torch.zeros(1, 8, 8)
    inp[0, 2, 3] = 0.5   # any non-zero value → 1/255
    inp[0, 4, 4] = 255.0
    out = t(inp)
    assert abs(out[0, 2, 3].item() - 1 / 255.0) < 1e-7
    assert abs(out[0, 4, 4].item() - 1 / 255.0) < 1e-7
    assert out[0, 0, 0].item() == 0.0


def test_scale_seg_dtype():
    """Output should be float32."""
    t = ScaleSeg()
    out = t(torch.ones(1, 4, 4, dtype=torch.uint8))
    assert out.dtype == torch.float32


def test_add_gaussian_noise_shape():
    t = AddGaussianNoise(mean=0.0, std=0.1)
    inp = torch.zeros(3, 64, 64)
    out = t(inp)
    assert out.shape == inp.shape


def test_add_gaussian_noise_nonzero():
    """With std > 0 there should be some deviation from the input."""
    torch.manual_seed(0)
    t = AddGaussianNoise(mean=0.0, std=0.1)
    inp = torch.ones(3, 32, 32)
    out = t(inp)
    assert not torch.allclose(out, inp)
