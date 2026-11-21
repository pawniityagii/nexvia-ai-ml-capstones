"""Grad-CAM explanation heatmaps for the DenseNet classifier.

Grad-CAM weights each feature map of the last convolutional block by the
gradient of the target class score, giving a coarse localisation map of the
pixels that drove the prediction. Requires torch.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""
from __future__ import annotations

import numpy as np


class GradCAM:
    def __init__(self, model, target_layer=None):
        import torch  # noqa: F401

        self.model = model.eval()
        # default: DenseNet feature extractor output (last conv block)
        self.target_layer = target_layer or self.model.features
        self._activations = None
        self._gradients = None
        self.target_layer.register_forward_hook(self._fwd_hook)
        self.target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx: int | None = None) -> np.ndarray:
        """input_tensor: [1,3,H,W]. Returns a HxW heatmap in [0,1]."""
        import torch
        import torch.nn.functional as F

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads = self._gradients            # [1,C,h,w]
        acts = self._activations           # [1,C,h,w]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
