import warnings

import torch
from torch import nn
from segmentation_models_pytorch import Unet as smpUnet

from diffusers_xray import build_unet, prepare_model_input


class Unet(nn.Module):
    def __init__(
        self,
        dim=None,
        image_size=None,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
        att_res=32,
        att_heads=4,
        base_channels=None,
    ):
        """
        Args:
            image_size: Spatial size of the input images (legacy alias: dim).
            base_channels: Base channel width used to derive UNet block_out_channels.
        """
        super().__init__()
        if image_size is None and dim is None:
            raise ValueError("image_size must be provided for the diffusers UNet")
        if image_size is None:
            warnings.warn("dim is deprecated; use image_size instead", DeprecationWarning)
            image_size = dim
        elif dim is not None and dim != image_size:
            raise ValueError("Provide only one of image_size or dim (legacy alias)")

        if base_channels is None:
            base_channels = image_size

        self.self_condition = self_condition
        self.unet = build_unet(
            image_size=image_size,
            channels=channels,
            channel_mults=dim_mults,
            attention_head_dim=att_res,
            norm_num_groups=resnet_block_groups,
            self_condition=self_condition,
            base_channels=base_channels,
        )

    @property
    def final_conv(self) -> nn.Module:
        return self.unet.conv_out

    @final_conv.setter
    def final_conv(self, module: nn.Module) -> None:
        self.unet.conv_out = module

    def forward(self, x, time=None):
        if time is None:
            # Downstream fine-tuning treats the UNet as a deterministic backbone at t=0.
            time = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        model_input = prepare_model_input(x, self.self_condition, self.unet.config.in_channels)
        return self.unet(model_input, time).sample


class smpUnet(smpUnet):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes):
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
