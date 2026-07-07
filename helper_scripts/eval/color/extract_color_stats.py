import os
import glob
import argparse
import logging
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ColorExtractor:
    def _load_image(self, path):
        img = cv2.imread(path)
        if img is not None:
            # OpenCV loads BGR, convert to RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def _load_mask(self, path):
        # 0=BG, 1=FG (or 255)
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def _get_channel_stats(self, pixels, prefix):
        """Calculates mean stats for a flat array of pixels."""
        if len(pixels) == 0:
            return {f"{prefix}_L": np.nan, f"{prefix}_A": np.nan, f"{prefix}_B": np.nan}

        # L: 0..255, A: 0..255, B: 0..255 in OpenCV LAB
        return {
            f"{prefix}_L": np.mean(pixels[:, 0]),
            f"{prefix}_A": np.mean(pixels[:, 1]),
            f"{prefix}_B": np.mean(pixels[:, 2])
        }

    def _get_channel_hists(self, pixels, prefix):
        """Calculates 256-bin histograms for a flat array of pixels."""
        if len(pixels) == 0:
            return {
                f"{prefix}_L": np.zeros(256, dtype=np.int64),
                f"{prefix}_A": np.zeros(256, dtype=np.int64),
                f"{prefix}_B": np.zeros(256, dtype=np.int64)
            }
        
        # Calculate histograms for 0-255 range
        hist_l, _ = np.histogram(pixels[:, 0], bins=256, range=(0, 256))
        hist_a, _ = np.histogram(pixels[:, 1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(pixels[:, 2], bins=256, range=(0, 256))
        
        return {
            f"{prefix}_L": hist_l,
            f"{prefix}_A": hist_a,
            f"{prefix}_B": hist_b
        }

    def process_image(self, image, mask=None, return_histograms=False):
        """
        Extracts L*a*b* stats or histograms.
        - Always calculates 'Global' (Whole Image).
        - Calculates 'FG'/'BG' only if mask is provided.
        """
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # --- 1. Global Stats (Always) ---
        global_pixels = lab_image.reshape(-1, 3)
        if return_histograms:
            stats_global = self._get_channel_hists(global_pixels, "Global")
        else:
            stats_global = self._get_channel_stats(global_pixels, "Global")

        # --- 2. Segmented Stats (Optional) ---
        if mask is not None:
            fg_mask = (mask > 0)
            bg_mask = (mask == 0)

            fg_pixels = lab_image[fg_mask]
            bg_pixels = lab_image[bg_mask]

            if return_histograms:
                stats_fg = self._get_channel_hists(fg_pixels, "FG")
                stats_bg = self._get_channel_hists(bg_pixels, "BG")
            else:
                stats_fg = self._get_channel_stats(fg_pixels, "FG")
                stats_bg = self._get_channel_stats(bg_pixels, "BG")
        else:
            # Fill FG/BG with NaNs or Zeros
            if return_histograms:
                nan_dict = {k: np.zeros(256, dtype=np.int64) for k in ["_L", "_A", "_B"]}
            else:
                nan_dict = {k: np.nan for k in ["_L", "_A", "_B"]}
                
            stats_fg = {f"FG{k}": v for k, v in nan_dict.items()}
            stats_bg = {f"BG{k}": v for k, v in nan_dict.items()}

        return {**stats_global, **stats_bg, **stats_fg}


    def scan_dataset(self, img_dir, mask_dir, output_csv):
        dataset_stats = []
        
        # Detect Mode
        has_masks = (mask_dir is not None) and os.path.exists(mask_dir)
        
        if has_masks:
            # Scan masks (to match with images)
            files = sorted(glob.glob(os.path.join(mask_dir, "*.*")))
            logger.info(f"Mode: MASKED. Scanning {len(files)} files in mask dir...")
        else:
            # Scan images directly
            files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
            logger.info(f"Mode: GLOBAL (Unlabeled). Scanning {len(files)} files in img dir...")

        if not files:
            logger.error("No files found! Check your paths.")
            return

        for path in tqdm(files):
            basename = os.path.basename(path)
            name_no_ext = os.path.splitext(basename)[0]
            
            img_path = None
            mask = None

            if has_masks:
                mask_path = path
                # Find matching image
                # Try png, jpg, tif patterns
                img_candidates = glob.glob(os.path.join(img_dir, f"{name_no_ext}.*"))
                if img_candidates:
                    img_path = img_candidates[0]
                    mask = self._load_mask(mask_path)
            else:
                img_path = path
                mask = None # No mask

            if img_path is None:
                continue

            img = self._load_image(img_path)
            if img is None: 
                continue

            # Resize mask if dimensions mismatch (e.g. slight rounding errors)
            if mask is not None and img.shape[:2] != mask.shape:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            stats = self.process_image(img, mask)
            stats["Image"] = basename
            dataset_stats.append(stats)
            
        if not dataset_stats:
            logger.error("No images processed successfully.")
            return

        df = pd.DataFrame(dataset_stats)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved stats to {output_csv} ({len(df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="Path to images")
    parser.add_argument("--mask_dir", default=None, help="Path to masks (Optional)")
    parser.add_argument("--out_csv", required=True, help="Output CSV file")
    
    args = parser.parse_args()
    
    extractor = ColorExtractor()
    extractor.scan_dataset(args.img_dir, args.mask_dir, args.out_csv)