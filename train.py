import os
import math
import imageio
import numpy as np
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import multiprocessing as mp


from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.image_processor import VaeImageProcessor

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.data.dataset import WebVid10M
from animatediff.modelshigh.unet import UNet3DConditionModel
from animatediff.modelshigh.controlnet import ControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print


def prepare_image(
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):  
        control_image_processor = VaeImageProcessor()
        image = control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image





def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    # local_rank      = init_dist(launcher=launcher)
    # local_rank = 1
    # global_rank     = dist.get_rank()
    # num_processes   = dist.get_world_size()
    # is_main_process = global_rank == 0
    is_main_process =  True 

    # seed = global_seed + global_rank
    seed = 42
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 需要设置wandb账号
    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff_pics_controlnetonly", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    #-----------------------------------------------------------------------------------------------
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet2d = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    # controlnet = ControlNetModel.from_unet(unet2d)
    controlnet = ControlNetModel()
    # unet = UNet3DConditionModel()
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        
    # Load pretrained unet weights
    # if unet_checkpoint_path != "":
    #     zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
    #     unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
    #     if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
    #     state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

    #     m, u = unet.load_state_dict(state_dict, strict=False)
    #     zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    #     assert len(u) == 0
    motion_module_state_dict = torch.load("models/Motion_Module/mm_sd_v15_v2.ckpt", map_location="cpu")
    # # print(motion_module_state_dict)
    # # if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
    missing, unexpected = unet.load_state_dict(motion_module_state_dict, strict=False)
    # print(f"### missing keys: {len(missing)}; \n### unexpected keys: {len(unexpected)};")
    # print(f"### missing keys:\n{missing}\n### unexpected keys:\n{unexpected}\n")
    assert len(unexpected) == 0

    controlnet_state_dict = torch.load("/root/lh/AnimateDiffcontrolnet-main/outputs/1/checkpoints/controlnet_checkpoint-epoch-30.ckpt", map_location="cpu")
    missing, unexpected = controlnet.load_state_dict(controlnet_state_dict["state_dict"], strict=False)
    assert len(unexpected) == 0
    #-----------------------------------------------------------------------------------------------
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # controlnet.requires_grad_(False)
    # for name, param in controlnet.named_parameters():
    #     print(name, ": ", param.requires_grad )
    # print("---------------------------------------")

    
    # 把这里打上断点 看一下unet的结构
    # Set unet trainable parameters
    # print(unet)
    unet.requires_grad_(False)
    # unet.requires_grad_(True)
    # for name, param in unet.named_parameters():
    #     for trainable_module_name in trainable_modules:
    #         if trainable_module_name in name:
    #             param.requires_grad = True
    #             break
    # trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    trainable_params = []
    for name, param in controlnet.named_parameters():
        if (param.requires_grad):
            trainable_params.append(param)
        # print(name, ": ", param.requires_grad )
    # trainable_params.append(list(filter(lambda p: p.requires_grad, controlnet.parameters())))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    #-----------------------------------------------------------------------------------------------

    if is_main_process:
        # zero_rank_print(f"trainable params number: {len(trainable_params)}")
        # zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
        print(f"trainable params number: {len(trainable_params)}")
        print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
    #-----------------------------------------------------------------------------------------------

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    # vae.to(local_rank)
    # text_encoder.to(local_rank)
    vae.to("cuda")
    controlnet.to("cuda")
    text_encoder.to("cuda")
    unet.to("cuda")
    #-----------------------------------------------------------------------------------------------

    # Get the training dataset
    train_dataset = WebVid10M(**train_data, is_image=image_finetune)
    # distributed_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=1,
    #     rank=0,
    #     shuffle=True,
    #     seed=global_seed,
    # )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        # sampler=distributed_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    #-----------------------------------------------------------------------------------------------

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )
    #-----------------------------------------------------------------------------------------------

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, controlnet=controlnet,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()
    #-----------------------------------------------------------------------------------------------

    # DDP warpper
    # unet = DDP(unet, device_ids=["cuda:0"], output_device="cuda:0")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    # mp.set_start_method('spawn')
    for epoch in range(first_epoch, num_train_epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        # mp.set_start_method('spawn')
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        # save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_seed}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_seed}-{idx}'}.png")
                    
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space
            orig_img = batch['image'].squeeze(1)
            orig_img = orig_img / 2. + 0.5       
            pixel_values = batch["pixel_values"].to("cuda")
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215


            #--------------------------------------
            image = prepare_image(
                image=orig_img,
                width=orig_img.shape[-1],
                height=orig_img.shape[-2],
                batch_size=orig_img.shape[0],
                num_images_per_prompt=1,
                device="cuda",
                dtype=controlnet.dtype,
            )

            #--------------------------------------


            # Sample noise that we'll add to the latents 如果要加原图信号的话 就是在这里加
            noise = torch.randn_like(latents)  # latents shape为 [4, 4, 16, 32, 32]
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device) # shape [4]
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) # shape[4, 4, 16, 32, 32]
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device) # shape [4, 77]
                encoder_hidden_states = text_encoder(prompt_ids)[0] # shape [4, 77, 768]
                
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            

            #------------------------------------------------
            down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=noisy_latents[:,:,0,:,:], 
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states, # [4,77,768]
                    controlnet_cond=image,
                    return_dict=False,
                )
            
            # down_block_additional_residuals
            # mid_block_additional_residual
            #------------------------------------------------

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                # noisy_latents shape [4, 4, 16, 32, 32]
                # encoder_hidden_states [4, 77, 768]
                model_pred = unet(sample=noisy_latents, 
                                  timestep=timesteps, 
                                  encoder_hidden_states=encoder_hidden_states,
                                  down_block_additional_residuals=down_block_res_samples,
                                  mid_block_additional_residual=mid_block_res_sample,
                                  is_opticalflow=False).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                # state_dict = {
                #     "epoch": epoch,
                #     "global_step": global_step,
                #     "state_dict": unet.state_dict(),
                # }
                controlnet_state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": controlnet.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    # torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                    torch.save(controlnet_state_dict, os.path.join(save_path, f"controlnet_checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    # torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                    torch.save(controlnet_state_dict, os.path.join(save_path, f"controlnet_checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                # prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts
                prompts = batch['text']
                
                init_images = batch['image'].squeeze(1) #[b, 1, c, h, w]
                controlnet_images = init_images / 2. + 0.5
                init_images = init_images.permute(0,2,3,1) # [b, 1, h, w, c]
                init_images = np.array(init_images.cpu())
                for idx, prompt in enumerate(prompts):
                    if not image_finetune:
                        controlnet_image = controlnet_images[idx, :, :, :]
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = train_data.sample_n_frames,
                            height       = height,
                            width        = width,
                            controlnet_image=controlnet_image,
                            **validation_data,
                        ).videos
                        init_image = init_images[idx, :, :, :]
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        imageio.imsave(f"{output_dir}/samples/sample-{global_step}/{idx}.jpg", init_image)
                        samples.append(sample)
                        
                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = validation_data.get("num_inference_steps", 25),
                            guidance_scale      = validation_data.get("guidance_scale", 8.),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)
                
                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    # dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="./configs/training/training.yaml")
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    # parser.add_argument("--wandb",    action="store_true")
    parser.add_argument("--wandb",    type=bool, default=True)

    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
