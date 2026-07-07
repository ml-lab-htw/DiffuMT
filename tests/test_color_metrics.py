"""Tests for ColorExtractor in helper_scripts/eval/color/ — CPU-only, no data."""
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helper_scripts" / "eval" / "color"))
from extract_color_stats import ColorExtractor


def _make_rgb(r, g, b, size=64):
    """Return an (H, W, 3) uint8 RGB array filled with a constant color."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return img


def _solid_mask(size=64, fg=True):
    """Return an (H, W) uint8 mask: all foreground or all background."""
    return np.full((size, size), 255 if fg else 0, dtype=np.uint8)


def test_process_returns_dict():
    extractor = ColorExtractor()
    img = _make_rgb(128, 64, 32)
    result = extractor.process_image(img)
    assert isinstance(result, dict)


def test_fg_stats_populated():
    extractor = ColorExtractor()
    img = _make_rgb(200, 100, 50)
    mask = _solid_mask(fg=True)
    result = extractor.process_image(img, mask=mask)
    assert "FG_L" in result or any("FG" in k for k in result)


def test_bg_stats_populated():
    extractor = ColorExtractor()
    img = _make_rgb(50, 50, 200)
    mask = _solid_mask(fg=False)
    result = extractor.process_image(img, mask=mask)
    assert any("BG" in k for k in result)


def test_histogram_bins():
    extractor = ColorExtractor()
    img = _make_rgb(100, 150, 200)
    result = extractor.process_image(img, return_histograms=True)
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            assert len(val) == 256, f"Expected 256 bins for {key}, got {len(val)}"


def test_empty_mask_no_crash():
    """All-background mask should not raise; FG stats should be nan/zero."""
    extractor = ColorExtractor()
    img = _make_rgb(100, 100, 100)
    mask = _solid_mask(fg=False)  # no foreground pixels
    # Should complete without exception
    result = extractor.process_image(img, mask=mask)
    assert result is not None
