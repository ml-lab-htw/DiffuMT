import logging
import os
from glob import glob
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, CLIPModel, CLIPImageProcessor, AutoImageProcessor
from .config import EvalConfig

""" Copied and adapted from /mario-koddenbrock/microtubule_tracking/mt/data_generation/optimization/embeddings.py"""


logger = logging.getLogger(__name__)

class ImageEmbeddingExtractor:
    """A class to extract and optionally reduce image embeddings using transformer models and PCA."""
    def __init__(self, cfg: EvalConfig):
        self.config = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model, self.processor = self._load_model()

    def _load_model(self):
        """Loads the model and processor from Hugging Face based on the config."""
        logger.info(f"Loading model: {self.config.model_name}")
        try:
            if "clip" in self.config.model_name.lower():
                model = CLIPModel.from_pretrained(self.config.model_name)
                processor = CLIPImageProcessor.from_pretrained(self.config.model_name)
            else:
                model = AutoModel.from_pretrained(self.config.model_name)
                processor = AutoImageProcessor.from_pretrained(self.config.model_name)
            
            model.to(self.device)
            model.eval()
            return model, processor
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _compute_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Computes the raw (pre-PCA) embedding for a single RGB image.

        Args:
            image (np.ndarray): An image in RGB format (H, W, C).

        Returns:
            np.ndarray: A 1D numpy array representing the raw image embedding.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if isinstance(self.model, CLIPModel):
                # For CLIP, we call the vision_model specifically to get hidden states
                outputs = self.model.vision_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                layer_idx = self.config.embedding_layer
                if layer_idx < 0:
                    layer_idx += len(hidden_states)
                embedding = hidden_states[layer_idx]
                embedding = embedding[:, 0, :]
                
            else:
                # For other models like DINOv2, the main model forward pass is used
                out = self.model(**inputs, output_hidden_states=True)

                if not hasattr(out, "hidden_states") or not out.hidden_states:
                    msg = "Model output does not contain 'hidden_states'. Cannot select a specific layer."
                    logger.error(msg)
                    raise ValueError(msg)

                # hidden_states is a tuple of (batch_size, sequence_length, hidden_size)
                # The first element is the input embeddings, subsequent are layer outputs.
                hidden_states = out.hidden_states
                layer_idx = self.config.embedding_layer

                if layer_idx >= len(hidden_states):
                    logger.warning(
                        f"Configured embedding_layer {layer_idx} is out of bounds for model with "
                        f"{len(hidden_states)} layers. Falling back to the last layer ({len(hidden_states) - 1})."                        
                        )
                    layer_idx = -1  # Use the last layer

                logger.debug(f"Extracting embedding from layer {layer_idx}.")
                # Select the specified layer's hidden state.
                embedding_tensor = hidden_states[layer_idx]
                # Take the mean over the sequence dimension (patch tokens) to get a single vector.
                embedding = embedding_tensor.mean(dim=1)
            return embedding.squeeze().cpu().numpy()

    def load_images_and_extract_embeddings(self, directory: str) -> Tuple[np.ndarray, List[str]]:
        """Load Images, Calc embeddings and return (Matrix, Filename)"""
        exts = ('*.png')
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(directory, ext)))
        files.sort()
        
        if not files:
            raise ValueError(f"No images found in {directory}")

        embeddings = []
        valid_files = []

        for image_path in tqdm(files, desc=f"Processing {os.path.basename(directory)}"):
            try:
                img = Image.open(image_path).convert("RGB")
                emb = self._compute_embedding(np.array(img))
                embeddings.append(emb)
                valid_files.append(image_path)
            except Exception as e:
                logger.warning(f"Error reading {image_path}: {e}")

        return np.stack(embeddings), valid_files