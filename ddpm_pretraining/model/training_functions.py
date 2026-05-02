import copy
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from torchmetrics import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid
from tqdm import tqdm

from diffusers_xray import XrayDDPMPipeline, build_scheduler, build_unet, prepare_model_input
from utils import (
    check_pixels_range_of_image,
    compute_diff,
    count_parameters,
    generate_path,
    save_images,
)


def _prepare_optimizer(optimizer: str, parameters, lr: float):
    if optimizer == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    if optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    raise NotImplementedError(f"Unsupported optimizer: {optimizer}")


def _save_checkpoint(
    pipeline: XrayDDPMPipeline,
    epoch: int,
    n_iter: int,
    loss: torch.Tensor,
    save_path: str,
    name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    checkpoint_dir = os.path.join(save_path, name)
    pipeline.save_pretrained(checkpoint_dir)
    training_state = {
        "epoch": epoch,
        "n_iter": n_iter,
        "loss": loss.item() if torch.is_tensor(loss) else loss,
    }
    if optimizer is not None:
        training_state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))


def _load_checkpoint(
    checkpoint_dir: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist")

    pipeline = XrayDDPMPipeline.from_pretrained(checkpoint_dir)
    pipeline.to(device)

    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.exists(state_path):
        return {"pipeline": pipeline, "epoch": 0, "n_iter": 0, "loss": np.inf}

    state = torch.load(state_path, map_location=device)
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return {
        "pipeline": pipeline,
        "epoch": state.get("epoch", 0),
        "n_iter": state.get("n_iter", 0),
        "loss": state.get("loss", np.inf),
    }


def _build_pipeline(config: Dict[str, Any], device: torch.device) -> XrayDDPMPipeline:
    unet_config = config["model"]["unet"]
    unet = build_unet(
        image_size=config["dataset"]["image_size"],
        channels=config["dataset"]["image_channels"],
        channel_mults=unet_config["channel_mults"],
        attention_head_dim=unet_config["attn_res"],
        norm_num_groups=unet_config["res_blocks"],
        self_condition=unet_config["self_condition"],
    )
    scheduler = build_scheduler(config["model"]["beta_schedule"]["train"])
    pipeline = XrayDDPMPipeline(unet=unet, scheduler=scheduler, self_condition=unet_config["self_condition"])
    pipeline.to(device)
    return pipeline


def _save_noising_process_image(pipeline: XrayDDPMPipeline, x_start: torch.Tensor, filename: str) -> None:
    images = [x_start[0:1]]
    timesteps = pipeline.scheduler.config.num_train_timesteps
    step = max(timesteps // 10, 1)
    timesteps_to_visualize = [i for i in range(0, timesteps, step)]

    for t in timesteps_to_visualize:
        t_tensor = torch.tensor([t], device=x_start.device)
        noise = torch.randn_like(x_start)
        noised_image = pipeline.scheduler.add_noise(x_start, noise, t_tensor)
        images.append(noised_image[0:1])

    image_grid = make_grid(torch.cat(images), nrow=len(timesteps_to_visualize) + 1, normalize=False)
    image_grid = image_grid.clamp(0, 1)
    np_image_grid = image_grid.cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(len(images) * 2, 2))
    plt.imshow(np_image_grid)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def train_diffusion_model(config, train_dataloader, save_model_path, root_path, device, continue_training=False):
    image_size = config["dataset"]["image_size"]
    channels = config["dataset"]["image_channels"]
    grad_accumulation = config["dataset"]["grad_accumulation"]
    iterations = config["model"]["iterations"]
    epochs = int(iterations / (len(train_dataloader) / grad_accumulation)) + 1

    loss_type = config["model"]["loss_type"]
    timesteps = config["model"]["beta_schedule"]["train"]["n_timestep"]
    freq_metrics = config["model"]["freq_metrics"]
    freq_checkpoint = config["model"]["freq_checkpoint"]
    use_ema = config["model"]["use_ema"]

    pipeline = _build_pipeline(config, device)
    optimizer = _prepare_optimizer(config["model"]["optimizer"], pipeline.unet.parameters(), config["model"]["lr"])

    ema_model = EMAModel(pipeline.unet.parameters(), decay=0.995) if use_ema else None
    ema_unet = None
    ema_pipeline = None
    if use_ema:
        ema_unet = copy.deepcopy(pipeline.unet).eval().requires_grad_(False).to(device)
        ema_pipeline = XrayDDPMPipeline(
            unet=ema_unet, scheduler=pipeline.scheduler, self_condition=pipeline.config.self_condition
        )

    table, total_params = count_parameters(pipeline.unet)
    logging.info(f"Total Trainable Params: {total_params}")

    if continue_training:
        checkpoint_path = os.path.join(save_model_path, "last_model")
        checkpoint = _load_checkpoint(checkpoint_path, device, optimizer)
        pipeline = checkpoint["pipeline"]
        start_epoch = checkpoint["epoch"]
        n_iter = checkpoint["n_iter"]
        best_loss = checkpoint["loss"]
    else:
        start_epoch, n_iter, best_loss = 0, 0, np.inf

    ssim = StructuralSimilarityIndexMeasure(data_range=None, reduction="elementwise_mean")
    mse = MeanSquaredError()
    fid = FrechetInceptionDistance(normalize=True)

    start_time = time.time()

    try:
        for epoch in tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs, desc="Epoch"):
            epoch_loss = 0.0
            torch.cuda.empty_cache()

            for batch_idx, data in enumerate(tqdm(train_dataloader, desc="Batch", leave=False)):
                x = data["image"].to(device)
                x_names = data["name"]
                batch_size = x.shape[0]

                if batch_idx == 0 and epoch == 0:
                    _save_noising_process_image(pipeline, x, f"{root_path}/noising_process.png")

                noise = torch.randn_like(x)
                t = torch.randint(1, pipeline.scheduler.config.num_train_timesteps, (batch_size,), device=device)
                noisy_images = pipeline.scheduler.add_noise(x, noise, t)

                model_input = prepare_model_input(
                    noisy_images, pipeline.config.self_condition, pipeline.unet.config.in_channels
                )
                predicted_noise = pipeline.unet(model_input, t).sample

                if loss_type == "l1":
                    loss = F.l1_loss(noise, predicted_noise)
                elif loss_type == "l2":
                    loss = F.mse_loss(noise, predicted_noise)
                elif loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, predicted_noise)
                else:
                    raise NotImplementedError()

                raw_loss = loss.detach()
                loss = loss / grad_accumulation
                loss.backward()
                epoch_loss += loss.item()

                if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(train_dataloader)):
                    torch.nn.utils.clip_grad_value_(pipeline.unet.parameters(), clip_value=1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                    if use_ema:
                        ema_model.step(pipeline.unet.parameters())

                    n_iter += 1

                    _save_checkpoint(
                        pipeline, epoch, n_iter, raw_loss, save_model_path, name="last_model", optimizer=optimizer
                    )
                    if use_ema:
                        ema_model.copy_to(ema_unet.parameters())
                        _save_checkpoint(ema_pipeline, epoch, n_iter, raw_loss, save_model_path, name="last_ema_model")

                if n_iter % freq_metrics == 0 and batch_idx % grad_accumulation == 0 and n_iter != 0:
                    pipeline.unet.eval()

                    x_hat = pipeline(
                        batch_size=batch_size, num_inference_steps=timesteps, x_cond=x, return_dict=False
                    )[0]

                    if use_ema:
                        ema_model.copy_to(ema_unet.parameters())
                        ema_x_hat = ema_pipeline(
                            batch_size=batch_size, num_inference_steps=timesteps, x_cond=x, return_dict=False
                        )[0]

                    x_cpu = x.detach().cpu()
                    x_hat_cpu = x_hat.detach().cpu()

                    x_hat_min, x_hat_max = check_pixels_range_of_image(x_hat_cpu)

                    ssim_metric = ssim(x_hat_cpu, x_cpu)
                    mse_metric = mse(x_hat_cpu, x_cpu)

                    real_images = x_cpu if channels == 3 else x_cpu.repeat(1, 3, 1, 1)
                    fake_images = x_hat_cpu if channels == 3 else x_hat_cpu.repeat(1, 3, 1, 1)

                    fid.update(real_images, real=True)
                    fid.update(fake_images.clamp(0, 1), real=False)
                    fid_score = fid.compute()

                    diff = compute_diff(real_images, fake_images)

                    train_imgs_path = generate_path(f"{root_path}/images/train")
                    image_titles = [f"{x_names[i]}" for i in range(batch_size)]
                    save_images(
                        [real_images, fake_images, diff],
                        f"{train_imgs_path}/train_epoch{epoch}_iteration{n_iter}_batch{batch_idx}.jpg",
                        f"Epoch {epoch} - Iteration {n_iter} - Batch {batch_idx} - Timesteps {timesteps}",
                        image_titles,
                    )

                    logging.info(
                        f"\nEpoch/Iteration {epoch}/{n_iter} \t Batch {batch_idx} \t Loss: {raw_loss.item():.6f}"
                    )
                    logging.info(
                        f"\t\t SSIM: {ssim_metric.item():.4f} \t MSE: {mse_metric.item():.6f} \t FID: {fid_score:.2f} \t Pixel range: [{x_hat_min:.2f}, {x_hat_max:.2f}]"
                    )

                    message = (
                        f"<b>Epoch/Iteration {epoch}/{n_iter}</b> --> [{x_hat_min:.2f}, {x_hat_max:.2f}] \n"
                        f"  • <b>Loss:</b> {raw_loss.item():.4f} \n"
                        f"  • <b>SSIM:</b> {ssim_metric.item():.4f} \n"
                        f"  • <b>MSE:</b> {mse_metric.item():.4f} \n"
                        f"  • <b>FID:</b> {fid_score:.4f}"
                    )

                    if use_ema:
                        ema_x_hat_cpu = ema_x_hat.detach().cpu()
                        ema_ssim_metric = ssim(ema_x_hat_cpu, x_cpu)
                        ema_mse_metric = mse(ema_x_hat_cpu, x_cpu)

                        ema_x_hat_min, ema_x_hat_max = check_pixels_range_of_image(ema_x_hat_cpu)

                        ema_fake_images = ema_x_hat_cpu if channels == 3 else ema_x_hat_cpu.repeat(1, 3, 1, 1)

                        fid.update(ema_fake_images.clamp(0, 1), real=False)
                        ema_fid_score = fid.compute()

                        ema_diff = compute_diff(real_images, ema_fake_images)

                        ema_image_titles = [f"{x_names[i]}" for i in range(batch_size)]
                        save_images(
                            [real_images, ema_fake_images, ema_diff],
                            f"{train_imgs_path}/train_epoch{epoch}_iteration{n_iter}_batch{batch_idx}_ema.jpg",
                            f"Epoch {epoch} - Iteration {n_iter} - Batch {batch_idx} - Timesteps {timesteps}",
                            ema_image_titles,
                        )

                        logging.info(
                            f"\t\t SSIM: {ema_ssim_metric.item():.4f} \t MSE: {ema_mse_metric.item():.6f} \t FID: {ema_fid_score:.2f}\t Pixel range: [{ema_x_hat_min:.2f}, {ema_x_hat_max:.2f}]"
                        )

                        message += (
                            f"\n\n<b>EMA Epoch/Iteration {epoch}/{n_iter}</b> --> [{ema_x_hat_min:.2f}, {ema_x_hat_max:.2f}] \n"
                            f"  • <b>Loss:</b> {raw_loss.item():.4f} \n"
                            f"  • <b>SSIM:</b> {ema_ssim_metric.item():.4f} \n"
                            f"  • <b>MSE:</b> {ema_mse_metric.item():.4f} \n"
                            f"  • <b>FID:</b> {ema_fid_score:.4f}"
                        )

                    del x_cpu, x_hat_cpu, real_images, fake_images, diff
                    if use_ema:
                        del ema_x_hat_cpu, ema_fake_images, ema_diff

                    print(message)
                    logging.info(message)

                    pipeline.unet.train()

                if n_iter % freq_checkpoint == 0 and n_iter != 0 and batch_idx % grad_accumulation == 0:
                    print(
                        f"Saving model checkpoint at epoch {epoch} and iteration {n_iter} with loss: {raw_loss.item():.4f}"
                    )
                    _save_checkpoint(
                        pipeline, epoch, n_iter, raw_loss, save_model_path, name=f"model_epoch{epoch}_step{n_iter}"
                    )
                    if use_ema:
                        ema_model.copy_to(ema_unet.parameters())
                        _save_checkpoint(
                            ema_pipeline,
                            epoch,
                            n_iter,
                            raw_loss,
                            save_model_path,
                            name=f"ema_model_epoch{epoch}_step{n_iter}",
                        )

                    if n_iter % iterations == 0:
                        print(f"Reaching {iterations} iterations. Exiting training...")
                        exit()

            epoch_loss /= len(train_dataloader)

    except (KeyboardInterrupt, SystemExit, Exception) as e:
        if isinstance(e, Exception):
            print(f"Exception: {e}")
        print("\nTraining interrupted. Saving final state...")
        _save_checkpoint(pipeline, epoch, n_iter, loss, save_model_path, name="last_model", optimizer=optimizer)
        if use_ema:
            ema_model.copy_to(ema_unet.parameters())
            _save_checkpoint(ema_pipeline, epoch, n_iter, loss, save_model_path, name="last_ema_model")

    finally:
        print(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        logging.info(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        del pipeline
        torch.cuda.empty_cache()
