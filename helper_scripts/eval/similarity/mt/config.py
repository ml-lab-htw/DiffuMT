from dataclasses import dataclass

@dataclass
class EvalConfig:
    
    # Paths
    reference_images_dir: str
    synthetic_images_dir: str
    output_dir: str = "evaluation_results"
    
    # Model
    model_name: str = "facebook/dinov2-base"  # oder "openai/clip-vit-base-patch32"
    embedding_layer: int = -1
    device: str = "cuda"
    
    #  Settings
    id: str = "eval_run" 
    similarity_metric: str = "cosine"