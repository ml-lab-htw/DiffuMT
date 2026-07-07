"""
training utils
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import time
import numpy as np
import os
from diffusers import DDPMPipeline, DDIMPipeline

from eval import evaluate, add_segmentations_to_noise, SegGuidedDDPMPipeline, SegGuidedDDIMPipeline

@dataclass
class TrainingConfig:
    model_type: str = "DDPM"
    image_size: int = 256  # the generated image resolution
    train_batch_size: int = 32
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_epochs: int = 200
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    save_model_epochs: int = 30
    #NEW list for saving on specific epochs
    save_model_specific_epochs: list = None
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = None

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0

    # custom options
    segmentation_guided: bool = False
    segmentation_channel_mode: str = "single"
    num_segmentation_classes: int = None # INCLUDING background
    use_ablated_segmentations: bool = False
    dataset: str = "breast_mri"
    resume_epoch: int = None

    # --- NEW: Offset Noise Configuration ---
    offset_noise: bool = False
    offset_noise_strength: float = 0.1
    
    # --- NEW: Scheduler Configuration ---
    beta_schedule: str = "linear" # default for DDPM, but we want to change it to "squaredcos_cap_v2"
    loss_type: str = "mse" # Options: 'mse', 'l1', 'huber'

    # EXPERIMENTAL/UNTESTED: classifier-free class guidance and image translation
    class_conditional: bool = False
    cfg_p_uncond: float = 0.2 # p_uncond in classifier-free guidance paper
    cfg_weight: float = 0.3 # w in the paper
    trans_noise_level: float = 0.5 # ratio of time step t to noise trans_start_images to total T before denoising in translation. e.g. value of 0.5 means t = 500 for default T = 1000.
    use_cfg_for_eval_conditioning: bool = True  # whether to use classifier-free guidance for or just naive class conditioning for main sampling loop
    cfg_maskguidance_condmodel_only: bool = True  # if using mask guidance AND cfg, only give mask to conditional network
    # ^ this is because giving mask to both uncond and cond model make class guidance not work 
    # (see "Classifier-free guidance resolution weighting." in ControlNet paper)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, lr_scheduler, device='cuda'):
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    global_step = 0

    print(f"\n{'='*40}")
    print(f"STARTING TRAINING: {config.model_type}")
    print(f"Dataset: {config.dataset} | Image Size: {config.image_size}")
    print(f"Epochs: {config.num_epochs} | Batch Size: {config.train_batch_size}")
    print(f"{'='*40}\n")

    # for loading segs to condition on:
    eval_iterator = iter(eval_dataloader)

    # Now you train the model
    start_epoch = 0
    if config.resume_epoch is not None:
        start_epoch = config.resume_epoch
    
    if config.loss_type == "l1":
        loss_fn = F.l1_loss
    elif config.loss_type == "huber":
        loss_fn = F.huber_loss
    else:
        loss_fn = F.mse_loss

    for epoch in range(start_epoch, config.num_epochs):

        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            clean_images = clean_images.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)

            # --- NEW: Apply Offset Noise if enabled ---
            if config.offset_noise:
                # Generate offset noise: random value per image, broadcasted to all pixels
                noise_offset = torch.randn(clean_images.shape[0], clean_images.shape[1], 1, 1).to(device)
                noise = noise + config.offset_noise_strength * noise_offset
            # ------------------------------------------

            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if config.segmentation_guided:
                noisy_images = add_segmentations_to_noise(noisy_images, batch, config, device)

            # Predict the noise residual
            if config.class_conditional:
                class_labels = torch.ones(noisy_images.size(0)).long().to(device)
                # classifier-free guidance
                a = np.random.uniform()
                if a <= config.cfg_p_uncond:
                    class_labels = torch.zeros_like(class_labels).long()
                noise_pred = model(noisy_images, timesteps, class_labels=class_labels, return_dict=False)[0]
            else:
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = loss_fn(noise_pred, noise)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Logging accumulation
            loss_val = loss.detach().item()
            if not  torch.isnan(loss).any():
                epoch_loss += loss_val
                num_batches += 1
            
            global_step += 1

        # --- END OF EPOCH SUMMARY ---
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        current_lr = lr_scheduler.get_last_lr()[0]
        duration = time.time() - epoch_start_time
        
        # 3. Clean Print Statement (One line per epoch)
        print(f"Epoch {epoch+1:03d}/{config.num_epochs} | "
              f"Loss: {avg_loss:.5f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {duration:.1f}s")

        # --- EVALUATION & SAVING LOGIC ---
        # 4. We check if we need to act BEFORE initializing the pipeline
        is_specific_save_time = (config.save_model_specific_epochs is not None) and ((epoch + 1) in config.save_model_specific_epochs)
        
        is_save_img_time = (epoch + 1) % config.save_image_epochs == 0
        
        # Save if interval matches OR specific epoch matches OR it's the last epoch
        is_save_model_time = ((epoch + 1) % config.save_model_epochs == 0) or is_specific_save_time
        
        is_last_epoch = (epoch == config.num_epochs - 1)

        if is_save_img_time or is_save_model_time or is_last_epoch:
            # ONLY NOW we initialize the pipeline to save memory/compute
            # Note: We pass model.module if using DataParallel, usually model is enough
            unet_to_save = model.module if hasattr(model, "module") else model
            
            if config.model_type == "DDPM":
                PipelineClass = SegGuidedDDPMPipeline if config.segmentation_guided else DDPMPipeline
            else: # DDIM
                PipelineClass = SegGuidedDDIMPipeline if config.segmentation_guided else DDIMPipeline
            
            # Initialize Pipeline with correct arguments
            pipeline = PipelineClass(
                unet=unet_to_save,
                scheduler=noise_scheduler,
                eval_dataloader=eval_dataloader,
                external_config=config           
            )

            # A) Evaluation (Generate Images)
            if is_save_img_time or is_last_epoch:
                print(f"--> Generating evaluation samples for Epoch {epoch+1}...")
                model.eval()
                if config.segmentation_guided:
                    # Get a batch of masks safely
                    try:
                        seg_batch = next(eval_iterator)
                    except StopIteration:
                        eval_iterator = iter(eval_dataloader)
                        seg_batch = next(eval_iterator)
                        
                    evaluate(config, epoch, pipeline, seg_batch)
                else:
                    evaluate(config, epoch, pipeline)
                model.train() # Switch back to train mode

            # B) Save Model
            if is_save_model_time or is_last_epoch:
                checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{epoch+1:04d}")
                
                print(f"--> Saving model checkpoint to {checkpoint_dir}...")
                
               
                os.makedirs(checkpoint_dir, exist_ok=True)
                
              
                pipeline.save_pretrained(checkpoint_dir)