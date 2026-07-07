import argparse
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm  # pip install tqdm (optional, for progress bar)

def get_used_stems(train_dir: Path) -> set:
    """
    Parses the training directory to find which original images were used.
    Assumes filenames are formatted like: {stem}_tile{r}_{c}.png
    """
    used_stems = set()
    files = list(train_dir.glob("*.png"))
    
    print(f"Scanning {len(files)} files in training directory...")
    
    for p in files:
        # Example: "myimage_tile0_0.png" -> split by "_tile" -> "myimage"
        # We use rsplit to ensure we only split at the LAST occurrence of _tile if multiple exist
        if "_tile" in p.stem:
            stem = p.stem.rsplit("_tile", 1)[0]
            used_stems.add(stem)
        else:
            # Fallback if naming convention is different, though unlikely based on your description
            used_stems.add(p.stem)
            
    return used_stems

def crop_and_save(src_path: Path, out_dir: Path, crop_size: int):
    """
    Crops the image into a non-overlapping grid (standard 2x2 for 512->256).
    """
    try:
        im = Image.open(src_path)
        w, h = im.size
        stem = src_path.stem

        # Calculate grid (Stride = Crop Size -> No Overlap)
        y_starts = range(0, h - crop_size + 1, crop_size)
        x_starts = range(0, w - crop_size + 1, crop_size)

        for r, y in enumerate(y_starts):
            for c, x in enumerate(x_starts):
                left = x
                upper = y
                right = x + crop_size
                lower = y + crop_size

                # Create filename consistent with splitter.py
                tile_name = f"{stem}_tile{r}_{c}.png"
                out_path = out_dir / tile_name

                # Crop and Save
                crop = im.crop((left, upper, right, lower))
                crop.save(out_path)

    except Exception as e:
        print(f"Error processing {src_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Create Evaluation Set from Unused Images")
    
    parser.add_argument("--all_images_dir", type=str, required=True, 
                        help="Path to the folder containing ALL original 512x512 images (before cropping)")
    parser.add_argument("--train_dir", type=str, required=True, 
                        help="Path to the current training images folder (containing the 256x256 crops)")
    parser.add_argument("--out_dir", type=str, required=True, 
                        help="Where to save the new evaluation crops")
    parser.add_argument("--crop_size", type=int, default=256, 
                        help="Size of crops (default: 256)")

    args = parser.parse_args()

    # Paths
    all_images_path = Path(args.all_images_dir)
    train_path = Path(args.train_dir)
    out_path = Path(args.out_dir)

    # Validation
    if not all_images_path.exists():
        sys.exit(f"Error: All images directory not found: {all_images_path}")
    if not train_path.exists():
        sys.exit(f"Error: Train directory not found: {train_path}")

    # 1. Identify used images
    used_stems = get_used_stems(train_path)
    print(f"-> Found {len(used_stems)} unique original images used in training.")

    # 2. Identify unused images
    all_files = sorted(list(all_images_path.glob("*.png")))
    unused_files = [f for f in all_files if f.stem not in used_stems]
    
    print(f"-> Found {len(all_files)} total images.")
    print(f"-> {len(unused_files)} images are UNUSED and will be processed for evaluation.")

    if len(unused_files) == 0:
        print("Warning: No unused images found. Check your directories or file naming.")
        return

    # 3. Process and Crop
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Processing images into {out_path}...")

    for f in tqdm(unused_files, desc="Cropping"):
        crop_and_save(f, out_path, args.crop_size)

    print("Done!")

if __name__ == "__main__":
    main()