from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch.nn.functional as F
import torch
from diffusers import DDPMPipeline
from feature_extractor import get_features
# from feature_extractor import get_features_e2d

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images, angle_context, cot_data = batch["reflectance"],batch["angles"],batch["cot"]

            cot_data = cot_data.to(dtype=torch.float32)

            # get features using pretrained resnet18 model. 
            cot_context = get_features(cot_data,accelerator.device)
            del cot_data
            cot_context = cot_context.squeeze()
            # print(cot_context.dtype)
            # print(angle_context.dtype)
            # print(clean_images.dtype)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            # timesteps = torch.randint(
            #     0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
            #     dtype=torch.int64
            # )
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (1,), device=clean_images.device,
                dtype=torch.int64
            )
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            time_embedding = get_time_embedding(timesteps,accelerator.device)
            # print(time_embedding.device)
            # time_embedding = accelerator.prepare(time_embedding)

            with accelerator.accumulate(model):
                # Predict the noise residual
                # noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                noise_pred = model(noisy_images,cot_context,angle_context,time_embedding)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            
            # torch.save(model.state_dict(), model_saved_path)

            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     evaluate(config, epoch, pipeline)



            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    # save model
                    model_saved_path = os.path.join(config.output_dir,"model.pth")
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), model_saved_path)
            # unwrapped_model.save_pretrained(
            #     model_saved_path,
            #     is_main_process=accelerator.is_main_process,
            #     save_function=accelerator.save,
            # )
# def get_time_embedding(timestep,device="cpu"):
#     # Shape: (160,)
#     freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160).to(device) 
#     # Shape: (1, 160)
#     x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
#     # Shape: (1, 160 * 2)
#     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def get_time_embedding(timesteps,device="cpu"):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160).to(device) 
    # print(freqs.shape)
    # Shape: (batch_size, 1, 160)
    # timesteps = timesteps.unsqueeze(-1)
    # timesteps = timesteps.unsqueeze(-1)
    # print(timesteps.shape)
    # x = timesteps * freqs[None, None, :]
    x = timesteps*freqs[None,:]
    # print(x.shape)
    # Shape: (batch_size, 1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
