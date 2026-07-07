import sys
import re
import json
import csv
import tifffile
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional

# --- Helper Functions: TIF ---

def convert_2D_binary(mask_path: Path, debug: bool = False) -> np.ndarray:
    """Converts TIF mask (2D or 3D stack) to binary 2D uint8 array."""
    mask_stack = tifffile.imread(str(mask_path))
    
    if debug:
        print(f"[DEBUG] Loaded TIF {mask_path.name}, shape={mask_stack.shape}")

    if mask_stack.ndim == 3:
        binmask = (mask_stack > 0).any(axis=0) # Collapse channels
    elif mask_stack.ndim == 2:
        binmask = (mask_stack > 0)
    else:
        raise ValueError(f"Unexpected dimensions {mask_stack.ndim} for {mask_path}")

    return binmask.astype(np.uint8)

def collect_pairs_tif(raw_images: Path, raw_masks: Path, img_ext: str = ".png") -> List[Dict]:
    """Matches TIF masks with source images."""
    items = []
    imgs = sorted(raw_images.glob(f"*{img_ext}"))
    for p in imgs:
        mask_path = raw_masks / f"{p.stem}.tif"
        if mask_path.exists():
            items.append({
                "image_path": p,
                "mask_source": "tif",
                "mask_path": mask_path,
                "stem": p.stem
            })
        else:
            print(f"Warning: Mask missing for {p.stem}", file=sys.stderr)
    return items

def stack_pngs_to_array(png_paths: List[Path]) -> np.ndarray:
    """
    Stacks multiple LabelStudio instance masks into a 3D array.
    Output Shape: (N_instances, Height, Width)
    Values: 0 or 255 (Requested for visibility in TIF viewers)
    """
    if not png_paths:
        raise ValueError("No PNG paths provided for stacking.")
    
    # Load all images as arrays
    arrays = [np.array(Image.open(p)) for p in png_paths]
    
    # Convert to binary (0 or 255)
    # HIER IST DIE ÄNDERUNG: * 255
    arrays = [(a > 0).astype(np.uint8) * 255 for a in arrays]
    
    # Stack along the first dimension
    return np.stack(arrays, axis=0)

# --- Helper Functions: LabelStudio (PNG) ---

def get_task_id_from_path(task_path: Path) -> Optional[int]:
    m = re.search(r"task[-_](\d+)", Path(task_path).stem)
    return int(m.group(1)) if m else None

def merge_pngs_to_mask(png_paths: List[Path]) -> np.ndarray:
    """Merges multiple LabelStudio instance masks into one binary mask."""
    if not png_paths:
        raise ValueError("No PNG paths provided for merging.")
    arrays = [np.array(Image.open(p)) for p in png_paths]
    h, w = arrays[0].shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    for a in arrays:
        combined = np.maximum(combined, (a > 0).astype(np.uint8))
    return combined

def find_filename_in_json(data: Any, task_id: int) -> Optional[str]:
    """Finds original filename for a task_id in LabelStudio JSON."""
    items = data if isinstance(data, list) else data.get("tasks", data.get("data", []))
    if not isinstance(items, list): items = [data]

    def _recursive_find(obj):
        if isinstance(obj, str) and obj.lower().endswith(('.png', '.jpg', '.tif')):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                res = _recursive_find(v)
                if res: return res
        if isinstance(obj, list):
            for v in obj:
                res = _recursive_find(v)
                if res: return res
        return None

    for item in items:
        # Resolve ID
        current_id = item.get("id")
        if "task" in item and isinstance(item["task"], dict):
            current_id = item["task"].get("id")
        
        if current_id == task_id:
            return _recursive_find(item)
    return None

def load_mapping_csv(mapping_path: Path) -> Dict[str, str]:
    mapping = {}
    if mapping_path and mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c = row.get("candidate_basename") or row.get("candidate_path")
                o = row.get("orig_basename") or row.get("orig_path")
                if c and o: mapping[c] = o
    return mapping

def collect_pairs_png(raw_masks: Path, raw_images: Path, json_path: Path, 
                      mask_only: bool, mapping_csv: Path = None, 
                      stack_instances: bool = False, 
                      debug: bool = False) -> List[Dict]:
    """Matches LabelStudio PNG instances with images using JSON/CSV."""
    
    # Load Metadata
    mapping = load_mapping_csv(mapping_csv)
    json_data = None
    if json_path and json_path.exists():
        with json_path.open("r") as f: json_data = json.load(f)

    # Group PNGs by Task ID
    png_files = list(raw_masks.glob("*.png"))
    task_map = {}
    for p in png_files:
        tid = get_task_id_from_path(p)
        if tid is not None:
            task_map.setdefault(tid, []).append(p)

    items = []
    for tid, pngs in sorted(task_map.items()):
        try:
            if stack_instances:
                mask_arr = stack_pngs_to_array(pngs)
            else:
                mask_arr = merge_pngs_to_mask(pngs)
        except Exception as e:
            print(f"Error processing task {tid}: {e}", file=sys.stderr)
            continue

        # Determine Stem / Image Path
        stem = f"task-{tid}" # Fallback
        image_candidate = None
        
        json_filename = find_filename_in_json(json_data, tid) if json_data else None
        
        if json_filename:
            ls_basename = Path(json_filename).name # e.g. "uuid-image.png"

            if ls_basename in mapping:
                # if: Mapping exists -> use original name from column
                real_name = mapping[ls_basename]
                stem = Path(real_name).stem
                
                if not mask_only:
                    # search with real name
                    candidate = raw_images / real_name
                    if candidate.exists(): 
                        image_candidate = candidate
            else:
                # if no mapping take current FN
                clean_name = json_filename.split("-", 1)[1] if "-" in json_filename else json_filename
                stem = Path(clean_name).stem
                
                print(f"[WARNING] Task {tid}: No mapping found for '{ls_basename}'. Trying cleaned name '{clean_name}'.", file=sys.stderr)
                
                if not mask_only:
                    candidate_clean = raw_images / clean_name
                    if candidate_clean.exists():
                        image_candidate = candidate_clean
                    else:
                        candidate_raw = raw_images / ls_basename
                        if candidate_raw.exists():
                             image_candidate = candidate_raw
                        

        if not mask_only and not image_candidate:
            if debug: print(f"[DEBUG] Skip task {tid}: Image not found.", file=sys.stderr)
            continue

        items.append({
            "image_path": image_candidate,
            "mask_source": "ndarray",
            "mask_array": mask_arr,
            "stem": stem
        })
    
    return items

# --- Main Conversion Logic ---

def run_conversion(mode: str, mask_dir: Path, images_dir: Path, out_dir: Path, 
                   json_path: Path = None, mask_only: bool = False, 
                   mapping_csv: Path = None, stack_instances: bool = False,
                   debug: bool = False):
    
    print(f"Starting conversion (Mode: {mode})...")
    
    # Prepare Output Dirs
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks_viz").mkdir(parents=True, exist_ok=True)

    # Collect Data
    if mode == "tif":
        # TIF mode usually assumes inputs are already correct, 
        # but we use existing logic here.
        items = collect_pairs_tif(images_dir, mask_dir)
    else:
        if not json_path:
             raise ValueError("JSON path required for PNG mode.")
        # Pass the stack_instances flag to the collector
        items = collect_pairs_png(mask_dir, images_dir, json_path, mask_only, 
                                  mapping_csv, stack_instances, debug)

    if not items:
        print("No pairs found.")
        return

    # Process
    count = 0
    for item in items:
        stem = item["stem"]
        
        # 1. Get Mask
        try:
            if item["mask_source"] == "tif":
                mask = convert_2D_binary(item["mask_path"], debug)
            else:
                mask = item["mask_array"]
        except Exception as e:
            print(f"Failed to process mask {stem}: {e}")
            continue

# 2. Save Mask & Viz
        if mask.ndim == 3:
            # set minisblack to avoid RGBA / RGB convert if there are 4/3 Layers
            tifffile.imwrite(
                out_dir / "masks" / f"{stem}.tif", 
                mask, 
                photometric='minisblack',
                metadata={'axes': 'ZYX'} # Z=Layer, Y=Height, X=Width
            )
            
            # Create preview
            preview = np.max(mask, axis=0) 
            Image.fromarray(preview.astype(np.uint8)).save(out_dir / "masks_viz" / f"{stem}.png")
        else:
            # Save 2D Mask as PNG (Values are 1)
            Image.fromarray(mask).save(out_dir / "masks" / f"{stem}.png")
            
            Image.fromarray(mask * 255).save(out_dir / "masks_viz" / f"{stem}.png")

        # 3. Save Image (if exists)
        if item["image_path"]:
            try:
                img = Image.open(item["image_path"])
                img.save(out_dir / "images" / f"{stem}.png")
            except Exception as e:
                print(f"Failed to save image {stem}: {e}")
        
        count += 1

    print(f"Conversion complete. Processed {count} items.")