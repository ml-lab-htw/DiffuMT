"""
Functionality of sam3.py adapted for Microtubule Tracking.
Generates both Binary Masks (collapsed) and Instance Masks (TIFF Stacks).
Refined with robust logic from sam3_temp.py.
"""
import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from cachetools import cached, LRUCache

# transformers provides Sam3Processor/Sam3Model in newer versions
try:
    from transformers import Sam3Processor, Sam3Model, pipeline
except Exception as _e:
    Sam3Processor = None
    Sam3Model = None

from sam_utils.base import BaseModel

# Module logger
logger = logging.getLogger(__name__)
# Ensure logging is visible during execution
logging.basicConfig(level=logging.INFO)

# Cache for the model and processor
# We can use a simple LRUCache here. maxsize=2 is a reasonable default if you expect
# to work with models on both CPU and GPU.
model_cache = LRUCache(maxsize=2)

@cached(model_cache)
def _load_sam3_model_and_processor(aig: bool = False, device: str = "cuda"):
    """
    Loads and caches the SAM3 model and processor.
    The 'device' argument is used as the cache key.
    """
    model_name = "facebook/sam3"

    if aig:
        model = pipeline("mask-generation", model=model_name, device=device)
        return model, None
    else:
        if Sam3Model is None or Sam3Processor is None:
            logger.error(
                "Sam3Model/Sam3Processor not available. Ensure 'transformers' is installed with SAM3 support."
            )
            raise ImportError("transformers.Sam3Model or Sam3Processor not available")

        logger.debug("Loading Sam3Model.from_pretrained('facebook/sam3') on device %s", device)
        start = time.time()
        model = Sam3Model.from_pretrained(model_name).to(device)
        processor = Sam3Processor.from_pretrained(model_name)
        elapsed = time.time() - start
        logger.debug("Loaded SAM3 model and processor in %.3f seconds", elapsed)
        return model, processor


class SAM3Text(BaseModel):
    """
    Wrapper for SAM3 model using text prompts for segmentation.
    """
    # Predefined text prompts for microtubule segmentation
    TEXT_PROMPTS = {
        0: "thin line",
        1: "elongated structure",
        2: "straight black line",
        3: "linear biological filament",
        4: "narrow elongated line",
        5: "thin red-highlighted line",
        6: "thin bright-on-dark microstructure",
        7: "thin structure in noisy background",
        8: "small linear object",
        9: "thin linear structure among noise",
    }

    def __init__(
        self,
        model_name: str = "SAM3Text",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        text_prompt_option: int = 0,
        **kwargs,
    ):
        logger.debug(
            "Initializing SAM3 model wrapper: model_name=%s, threshold=%s, mask_threshold=%s, text_prompt_option=%s",
            model_name, threshold, mask_threshold, text_prompt_option
        )
        super().__init__(model_name, **kwargs)

        self.text_prompt_option = text_prompt_option
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self._model = None
        self._processor = None

    def load_model(self):
        """Load the model for text-prompted segmentation."""
        logger.debug("Called load_model()")
        if self._model is not None:
            logger.debug("Model already loaded; skipping load_model")
            return
        try:
            # This will use the cached version if available for the current device
            self._model, self._processor = _load_sam3_model_and_processor(False, self._device)
        except Exception as e:
            logger.exception("Failed to load SAM3 model/processor: %s", e)
            raise

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Runs inference on the image.
        Returns:
            np.ndarray: Stack of boolean instance masks (N, H, W).
                        N is the number of instances.
                        Returns (0, H, W) if no masks found.
        """
        # 1. Validation & Load
        logger.debug(
            "predict() called with image shape=%s, dtype=%s",
            getattr(image, "shape", None),
            getattr(image, "dtype", None),
        )

        if image.ndim not in (2, 3):
            logger.error("Invalid image dimensionality: %s", image.shape)
            raise ValueError(f"image must be 2D or 3D (H,W[,C]); got {image.shape}")

        if self._model is None:
            logger.debug("Model not loaded in predict(); calling load_model()")
            self.load_model()

        assert self._model is not None
        assert self._processor is not None

        # Ensure the image is uint8
        if image.dtype != np.uint8:
            try:
                img_min = float(image.min())
                img_max = float(image.max())
                logger.debug("Normalizing image with min=%s and max=%s", img_min, img_max)
                if img_max == img_min:
                    # Avoid division by zero; produce a zero image
                    logger.warning(
                        "Image has zero dynamic range (max == min == %s). Producing zeros before casting to uint8.",
                        img_max,
                    )
                    image = np.zeros_like(image, dtype=np.uint8)
                else:
                    image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            except Exception as e:
                logger.exception("Error normalizing image to uint8: %s", e)
                raise

        pil_image = Image.fromarray(image)
        logger.debug("Converted image to PIL with size=%s (width,height)", pil_image.size)
        
        # 3. Prompting
        text_prompt = self.TEXT_PROMPTS.get(self.text_prompt_option, self.TEXT_PROMPTS[0])
        logger.debug("Using text prompt option %s: '%s'", self.text_prompt_option, text_prompt)

        # 4. Inference
        try:
            inputs = self._processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self._device)
            
            with torch.no_grad():
                t0 = time.time()
                outputs = self._model(**inputs)
                t_infer = time.time() - t0
                logger.debug("Model forward pass completed in %.3f seconds", t_infer)

            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

        # 5. Extract Masks
        masks_output = results.get("masks", [])
        logger.debug("Post-processed results: found masks count=%d", len(masks_output))
        
        if len(masks_output) == 0:
            h, w = pil_image.size[1], pil_image.size[0]
            logger.debug("No masks found; returning empty array with shape (0,%d,%d)", h, w)
            return np.empty((0, h, w), dtype=bool)

        # Convert tensors to numpy boolean arrays
        masks_list = []
        for i, mask_data in enumerate(masks_output):
            try:
                # mask_data may be a torch tensor or numpy array
                if hasattr(mask_data, "cpu"):
                    m = mask_data.detach().cpu().numpy()
                else:
                    m = np.array(mask_data)
                
                # Ensure mask is boolean
                mask_bool = m > 0 if m.dtype != bool else m
                masks_list.append(mask_bool)
            except Exception as e:
                logger.exception("Failed to convert mask #%d to numpy array: %s", i, e)
                raise

        # Stack into (N, H, W) boolean array
        if not masks_list:
             h, w = pil_image.size[1], pil_image.size[0]
             return np.empty((0, h, w), dtype=bool)
             
        masks_stack = np.array(masks_list, dtype=bool)
        logger.debug("Returning masks_stack with shape %s", masks_stack.shape)
        
        return masks_stack


def save_tiff_stack(masks: np.ndarray, path: str):
    """
    Saves a (N, H, W) boolean or uint8 array as a multi-page TIFF.
    """
    if masks.ndim != 3:
        return
    
    # Convert to list of PIL Images
    images = []
    for i in range(masks.shape[0]):
        # Convert boolean to 0-255 uint8
        img = Image.fromarray((masks[i] * 255).astype(np.uint8), mode='L')
        images.append(img)
    
    if images:
        images[0].save(path, save_all=True, append_images=images[1:], compression="tiff_deflate")


if __name__ == "__main__":
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="SAM3 Text Prompt Segmentation")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, default="sam3_predictions", help="Root output directory")
    parser.add_argument("--prompt_id", type=int, default=0, help="ID of the text prompt to use")

    args = parser.parse_args()

    # Preprocessing check
    try:
        import sam_utils.preprocessing as pre
    except ImportError:
        print("ERROR: Could not find 'sam_utils.preprocessing'.")
        sys.exit(1)

    # 2. Config
    preprocess_params = {
        "grayscale": True,
        "clip_to_percentiles": True,
        "rescale_using_percentiles": True,
        "invert": False,
        "histogram_normalization": False,
        "sharpen_radius": 0.38717865927737405,
        "smooth_radius": 3.410555347410367,
        "percentile_min": 3.506993161284297,
        "percentile_max": 99.46537030219284,
    }

    threshold = 0.4018846604226014
    mask_threshold = 0.44189937830348586

    # 3. Init Model
    print(f"--- Initializing SAM3Text (Prompt {args.prompt_id}) ---")
    sam3text = SAM3Text(
        threshold=threshold,
        mask_threshold=mask_threshold,
        text_prompt_option=args.prompt_id
    )
    sam3text.load_model()

    # 4. Setup Output Dirs
    out_binary = os.path.join(args.output_dir, "binary")
    out_instances = os.path.join(args.output_dir, "instances")
    
    os.makedirs(out_binary, exist_ok=True)
    os.makedirs(out_instances, exist_ok=True)
    
    valid_ext = ('.png')
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_ext)])
    
    print(f"Processing {len(files)} images from '{args.input_dir}'...")

    # 5. Process Loop
    total_files = len(files)
    print(f"Processing {total_files} images from '{args.input_dir}'...")
    
    for i, fname in enumerate(files, 1):
        print(f"   [{i}/{total_files}] Segmenting: {fname}")
        
        fpath = os.path.join(args.input_dir, fname)
        base_name = os.path.splitext(fname)[0]
        
        try:
            # Load & Preprocess
            image_pil = Image.open(fpath).convert("RGB") 
            image_np = np.array(image_pil, dtype=np.uint8)
            processed_image = pre.process_image(image_np, **preprocess_params)

            # Predict (Returns Boolean Stack N, H, W)
            masks = sam3text.predict(processed_image)

            # --- A. Save Binary Mask (Collapsed) ---
            if masks.shape[0] > 0:
                # Collapse all instances into one binary layer
                binary_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            else:
                binary_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            
            save_name_bin = os.path.join(out_binary, f"{base_name}.png")            
            Image.fromarray(binary_mask, mode='L').save(save_name_bin)

            # --- B. Save Instance Mask (TIFF Stack) ---
            save_name_inst = os.path.join(out_instances, f"{base_name}.tiff")
            if masks.shape[0] > 0:
                save_tiff_stack(masks, save_name_inst)
            else:
                # Create an empty mask stack: 1 layer (N=1), all zeros (False)
                # This maintains the 1:1 file mapping for the evaluation script
                empty_stack = np.zeros((1, image_np.shape[0], image_np.shape[1]), dtype=bool)
                save_tiff_stack(empty_stack, save_name_inst)

        except Exception as e:
            print(f"\nError processing {fname}: {e}")
            continue

    print("\nFinished!")