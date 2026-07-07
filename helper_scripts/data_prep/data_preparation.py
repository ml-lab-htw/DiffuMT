import argparse
from pathlib import Path

# Import our new modules
import converter
import splitter

"""Usage examples:
  - Convert TIF mode:
    python data_preparation.py convert --mode tif --mask_dir ./raw_masks \
      --images_dir ./raw_images --out_dir ./converted

  - Convert PNG mode (LabelStudio):
    python data_preparation.py convert --mode png --mask_dir ./ls_masks \
      --images_dir ./raw_images --json_dir ./batch1.json --out_dir ./converted"""

def main():
    ap = argparse.ArgumentParser(description="Data Preparation Pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- Command: Convert ---
    p_conv = sub.add_parser("convert", help="Raw Data -> Clean PNGs or TIFs")
    p_conv.add_argument("--mode", choices=["tif", "png"], default="tif")
    p_conv.add_argument("--mask_dir", type=str, required=True)
    p_conv.add_argument("--images_dir", type=str, required=True)
    p_conv.add_argument("--out_dir", type=str, required=True)
    p_conv.add_argument("--json_dir", type=str, help="LabelStudio JSON")
    p_conv.add_argument("--mapping_csv", type=str)
    p_conv.add_argument("--mask_only", action="store_true")
    p_conv.add_argument("--stack_instances", action="store_true", 
                        help="If set, saves LabelStudio instances as a 3D TIF stack instead of a flat PNG.")
    p_conv.add_argument("--debug", action="store_true")

    # --- Command: Split ---
    p_split = sub.add_parser("split", help="Clean PNGs -> Train/Val/Test + Tiling")
    p_split.add_argument("--in_dir", type=str, required=True, help="Converted folder")
    p_split.add_argument("--out_dir", type=str, required=True, help="Final dataset folder")
    p_split.add_argument("--train_n", type=int, required=True)
    p_split.add_argument("--validation_n", type=int, required=True)
    p_split.add_argument("--test_n", type=int, required=True)
    
    # New Arguments for Sliding Window
    p_split.add_argument("--crop_size", type=int, default=None, help="Size of square crop (e.g. 256)")
    p_split.add_argument("--stride", type=int, default=None, help="Step size. If < crop_size, crops will overlap.")
    
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    # Dispatcher
    if args.cmd == "convert":
        converter.run_conversion(
            mode=args.mode,
            mask_dir=Path(args.mask_dir),
            images_dir=Path(args.images_dir),
            out_dir=Path(args.out_dir),
            json_path=Path(args.json_dir) if args.json_dir else None,
            mask_only=args.mask_only,
            mapping_csv=Path(args.mapping_csv) if args.mapping_csv else None,
            stack_instances=args.stack_instances,
            debug=args.debug
        )

    elif args.cmd == "split":
        splitter.run_split(
            in_dir=Path(args.in_dir),
            out_dir=Path(args.out_dir),
            train_n=args.train_n,
            val_n=args.validation_n,
            test_n=args.test_n,
            seed=args.seed,
            crop_size=args.crop_size,
            stride=args.stride,
            debug=args.debug
        )

if __name__ == "__main__":
    main()