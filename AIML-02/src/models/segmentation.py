"""3D U-Net builder (MONAI) for volumetric segmentation.

Requires monai + torch. Provided as the brief's segmentation model; trained on
the Medical Segmentation Decathlon (see train_segmentation.py / the Colab
notebook).
"""
from __future__ import annotations

from ..common import config as C


def build_unet():
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=3,
        in_channels=C.SEG_IN_CHANNELS,
        out_channels=C.SEG_OUT_CHANNELS,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
