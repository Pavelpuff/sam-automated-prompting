from collections.abc import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GradCAM:
    """
    Pure PyTorch implementation of Grad-CAM for extracting saliency maps.
    """

    def __init__(self, model: nn.Module, target_layers: Iterable[nn.Module]) -> None:
        self.model = model
        self.target_layers = list(target_layers)
        self.activations: dict[nn.Module, Tensor] = {}
        self.gradients: dict[nn.Module, Tensor] = {}

        for layer in self.target_layers:
            layer.register_forward_hook(self._get_activation_hook(layer))
            layer.register_full_backward_hook(self._get_gradient_hook(layer))

    def _get_activation_hook(self, layer: nn.Module) -> Callable:
        def hook(module: nn.Module, input: Tensor, output: Tensor) -> None:
            self.activations[layer] = output

        return hook

    def _get_gradient_hook(self, layer: nn.Module) -> Callable:
        def hook(
            module: nn.Module, grad_input: tuple[Tensor, ...], grad_output: tuple[Tensor, ...]
        ) -> None:
            self.gradients[layer] = grad_output[0]

        return hook

    def __call__(self, input_tensor: Tensor, targets: Iterable[int] | None = None) -> np.ndarray:
        self.model.zero_grad()
        self.model.eval()

        B, C, H, W = input_tensor.shape
        output = self.model(input_tensor)

        if targets is None:
            targets = output.argmax(dim=1)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=input_tensor.device)

        target_scores = output.gather(dim=1, index=targets.view(-1, 1)).squeeze()
        target_scores.sum().backward(retain_graph=True)

        cams_per_layer: list[Tensor] = []
        for layer in self.target_layers:
            grads = self.gradients[layer]
            acts = self.activations[layer]

            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

            cam_min = cam.flatten(1).min(1).values.view(B, 1, 1, 1)
            cam_max = cam.flatten(1).max(1).values.view(B, 1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-9)
            cams_per_layer.append(cam)

        final_cam = (
            torch.stack(cams_per_layer).mean(dim=0)
            if len(cams_per_layer) > 1
            else cams_per_layer[0]
        )
        return final_cam.squeeze(1).detach().cpu().numpy()
