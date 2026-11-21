"""DenseNet121 transfer-learning classifier (chest X-ray, normal vs abnormal).

Requires torch + torchvision (install on Colab/GPU; see requirements-dl.txt).
Transfer learning: ImageNet-pretrained DenseNet121 with the final classifier
swapped for a 2-class head. EfficientNet is a drop-in alternative the brief
mentions - see build_model(arch=...).
"""
from __future__ import annotations

from ..common import config as C


def build_model(arch: str = "densenet121", pretrained: bool = True):
    import torch.nn as nn
    import torchvision.models as models

    if arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_feats = model.classifier.in_features
        model.classifier = nn.Linear(in_feats, len(C.CLASSES))
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, len(C.CLASSES))
    else:
        raise ValueError(f"unknown arch: {arch}")
    return model


def load_model(checkpoint_path=C.CLASSIFIER_PATH, arch: str = "densenet121"):
    import torch

    model = build_model(arch, pretrained=False)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model
