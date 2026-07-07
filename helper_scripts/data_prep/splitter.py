import shutil
import random
import json
import sys
from pathlib import Path
from PIL import Image

def tile_image(stem: str, src_img: Path, src_mask: Path, 
               out_data_dir: Path, out_mask_dir: Path, 
               crop_size: int, stride: int, debug: bool) -> list:
    """
    Handles cropping using a Sliding Window approach.
    """
    created_files = []
    
    try:
        im = Image.open(src_img)
        mask = Image.open(src_mask)
        w, h = im.size

        # Fallback if image is smaller than crop
        if crop_size and (w < crop_size or h < crop_size):
            if debug:
                print(f"[DEBUG] {stem} ({w}x{h}) smaller than {crop_size}. Copying full.")
            shutil.copy2(src_img, out_data_dir / f"{stem}.png")
            shutil.copy2(src_mask, out_mask_dir / f"{stem}.png")
            created_files.append(f"{stem}.png")
            return created_files

        # --- Sliding Window Logic ---
        if crop_size:
            # Iterate through y and x with the defined stride
            # Range ensures we don't go out of bounds
            y_starts = range(0, h - crop_size + 1, stride)
            x_starts = range(0, w - crop_size + 1, stride)
            
            count = 0
            for r, y in enumerate(y_starts):
                for c, x in enumerate(x_starts):
                    left = x
                    upper = y
                    right = x + crop_size
                    lower = y + crop_size
                    
                    tile_name = f"{stem}_tile{r}_{c}.png"
                    
                    im.crop((left, upper, right, lower)).save(out_data_dir / tile_name)
                    mask.crop((left, upper, right, lower)).save(out_mask_dir / tile_name)
                    
                    created_files.append(tile_name)
                    count += 1
            
            if debug and count > 0:
                # Only verbose debug if unusual count or specifically asked
                pass 
                
        else:
            # No crop defined
            shutil.copy2(src_img, out_data_dir / f"{stem}.png")
            shutil.copy2(src_mask, out_mask_dir / f"{stem}.png")
            created_files.append(f"{stem}.png")

    except Exception as e:
        print(f"Error processing {stem}: {e}", file=sys.stderr)
    
    return created_files

def run_split(in_dir: Path, out_dir: Path, 
              train_n: int, val_n: int, test_n: int, 
              seed: int = 42, crop_size: int = None, 
              stride: int = None, debug: bool = False):
    
    print("Starting dataset split...")
    random.seed(seed)
    
    in_images = in_dir / "images"
    in_masks = in_dir / "masks"

    # 1. Gather stems
    all_imgs = sorted(list(in_images.glob("*.png")))
    stems = [p.stem for p in all_imgs]
    
    # 2. Validation
    total_needed = train_n + val_n + test_n
    if total_needed > len(stems):
        raise ValueError(f"Not enough images! Have {len(stems)}, need {total_needed}.")

    # 3. Random Split
    chosen = random.sample(stems, total_needed)
    splits = {
        "train": chosen[:train_n],
        "val":   chosen[train_n : train_n + val_n],
        "test":  chosen[train_n + val_n :]
    }

    # 4. Process
    stats = {"seed": seed, "params": {"crop_size": crop_size, "train_stride": stride}, "counts": {}}
    
    for split_name, split_stems in splits.items():
        print(f"Processing split: {split_name} ({len(split_stems)} source images)...")
        
        data_out = out_dir / "DATA_FOLDER" / split_name
        mask_out = out_dir / "MASK_FOLDER" / "all" / split_name
        
        data_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        
        file_list = []
        
        # Apply overlapping stride ONLY to training set.
        # For val/test, use stride = crop_size (grid/no overlap) to avoid leakage.
        if split_name == "train":
            # If no stride arg, default to crop_size (no overlap)
            current_stride = stride if stride is not None else crop_size
        else:
            current_stride = crop_size
        
        # Safety check if crop_size is None
        if crop_size is None:
            current_stride = None

        if debug:
            print(f"  > Mode: {split_name}, Stride: {current_stride} (Crop: {crop_size})")

        for stem in split_stems:
            src_img = in_images / f"{stem}.png"
            src_mask = in_masks / f"{stem}.png"
            
            created = tile_image(stem, src_img, src_mask, 
                                 data_out, mask_out, 
                                 crop_size, current_stride, debug)
            file_list.extend(created)
            
        stats["counts"][split_name] = len(file_list)

    with open(out_dir / "split_log.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Split complete. Results in {out_dir}")