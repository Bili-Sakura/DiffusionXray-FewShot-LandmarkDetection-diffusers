from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from diffusers import DDPMScheduler, DiffusionPipeline, UNet2DModel
from diffusers.utils import BaseOutput


@dataclass
class XrayPipelineOutput(BaseOutput):
    images: torch.Tensor


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def _linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def _quadratic_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def _sigmoid_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def build_betas(
    schedule: str,
    timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    if schedule == "linear":
        return _linear_beta_schedule(timesteps, beta_start, beta_end)
    if schedule == "cosine":
        return _cosine_beta_schedule(timesteps)
    if schedule == "quadratic":
        return _quadratic_beta_schedule(timesteps, beta_start, beta_end)
    if schedule == "sigmoid":
        return _sigmoid_beta_schedule(timesteps, beta_start, beta_end)
    raise NotImplementedError(f"Unsupported beta schedule: {schedule}")


def build_scheduler(schedule_config: dict) -> DDPMScheduler:
    betas = build_betas(
        schedule=schedule_config["schedule"],
        timesteps=schedule_config["n_timestep"],
        beta_start=schedule_config["linear_start"],
        beta_end=schedule_config["linear_end"],
    )
    return DDPMScheduler(
        num_train_timesteps=schedule_config["n_timestep"],
        beta_schedule="linear",
        trained_betas=betas.cpu().numpy(),
        clip_sample=False,
        prediction_type="epsilon",
    )


def build_unet(
    image_size: int,
    channels: int,
    channel_mults: Sequence[int],
    attention_head_dim: int,
    norm_num_groups: int,
    self_condition: bool = False,
    base_channels: Optional[int] = None,
) -> UNet2DModel:
    base_channels = image_size if base_channels is None else base_channels
    block_out_channels = tuple(base_channels * mult for mult in channel_mults)
    in_channels = channels + (1 if self_condition else 0)
    return UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=channels,
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=["AttnDownBlock2D"] * len(block_out_channels),
        up_block_types=["AttnUpBlock2D"] * len(block_out_channels),
        attention_head_dim=attention_head_dim,
        norm_num_groups=norm_num_groups,
    )


def prepare_model_input(sample: torch.Tensor, self_condition: bool, in_channels: int) -> torch.Tensor:
    if self_condition:
        if sample.shape[1] == in_channels:
            return sample
        if sample.shape[1] == in_channels - 1:
            zeros = torch.zeros(
                (sample.shape[0], 1, sample.shape[2], sample.shape[3]),
                device=sample.device,
                dtype=sample.dtype,
            )
            return torch.cat([zeros, sample], dim=1)
        raise ValueError(
            f"Self-conditioned input should have {in_channels - 1} or {in_channels} channels, got {sample.shape[1]}"
        )

    if sample.shape[1] != in_channels:
        raise ValueError(f"Input should have {in_channels} channels, got {sample.shape[1]}")
    return sample


class XrayDDPMPipeline(DiffusionPipeline):
    def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler, self_condition: bool = False) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.register_to_config(self_condition=self_condition)

    def _prepare_model_input(self, sample: torch.Tensor) -> torch.Tensor:
        return prepare_model_input(sample, self.config.self_condition, self.unet.config.in_channels)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        x_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> XrayPipelineOutput:
        """Run sampling for downstream inference or reconstruction.

        Example:
            >>> pipeline = XrayDDPMPipeline.from_pretrained(".../models/last_model").to("cuda")
            >>> x_cond = torch.randn(1, 1, 256, 256, device="cuda")
            >>> images = pipeline(x_cond=x_cond, num_inference_steps=500).images
        """
        device = self.device
        if x_cond is not None:
            x_cond = x_cond.to(device)
            batch_size = x_cond.shape[0]
        if batch_size is None:
            raise ValueError("batch_size must be provided when x_cond is None")

        num_inference_steps = num_inference_steps or self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        if x_cond is None:
            sample = torch.randn(
                (batch_size, self.unet.config.out_channels, self.unet.config.sample_size, self.unet.config.sample_size),
                device=device,
                generator=generator,
            )
        else:
            noise = torch.randn_like(x_cond, generator=generator)
            t_start = self.scheduler.timesteps[0]
            sample = self.scheduler.add_noise(x_cond, noise, t_start)

        for t in self.scheduler.timesteps:
            model_input = self._prepare_model_input(sample)
            model_output = self.unet(model_input, t).sample
            sample = self.scheduler.step(model_output, t, sample, generator=generator).prev_sample

        if not return_dict:
            return (sample,)
        return XrayPipelineOutput(images=sample)
