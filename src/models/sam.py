from collections.abc import Callable

import numpy as np
import torch
from segment_anything import SamPredictor
from torch import Tensor

from src.models.gradcam import GradCAM


class BasicSamPipeline:
    def __init__(self, sam_predictor: SamPredictor, inverse_transform: Callable) -> None:
        self.sam_predictor = sam_predictor
        self.inverse_transform = inverse_transform

    def segment(self, images: Tensor, point_coords: np.ndarray, point_labels: np.ndarray) -> Tensor:
        results = []
        for image, pt_coords, pt_labels in zip(images, point_coords, point_labels, strict=True):
            img_np = np.array(self.inverse_transform(image))
            self.sam_predictor.set_image(img_np)
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=pt_coords, point_labels=pt_labels, multimask_output=True
            )
            best_mask = masks[np.argmax(scores)]
            results.append(torch.tensor(best_mask, dtype=torch.bool))
        return torch.stack(results)


class SamPipelineForeground(BasicSamPipeline):
    """Approach 1: Use only one foreground point (max Grad-CAM activation)."""

    def __init__(
        self, sam_predictor: SamPredictor, inverse_transform: Callable, grad_cam: GradCAM
    ) -> None:
        super().__init__(sam_predictor, inverse_transform)
        self.grad_cam = grad_cam

    def __call__(self, images: Tensor) -> Tensor:
        heatmaps = self.grad_cam(images, targets=None)
        B = heatmaps.shape[0]
        point_coords, point_labels = [], []

        for i in range(B):
            y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
            point_coords.append([(x, y)])
            point_labels.append([1])

        return self.segment(images, np.array(point_coords), np.array(point_labels))


class SamPipelineWithBackground(BasicSamPipeline):
    """Approach 2: Use foreground (max activation) and background (min activation) points."""

    def __init__(
        self, sam_predictor: SamPredictor, inverse_transform: Callable, grad_cam: GradCAM
    ) -> None:
        super().__init__(sam_predictor, inverse_transform)
        self.grad_cam = grad_cam

    def __call__(self, images: Tensor) -> Tensor:
        heatmaps = self.grad_cam(images, targets=None)
        B = heatmaps.shape[0]
        point_coords, point_labels = [], []

        for i in range(B):
            y_fg, x_fg = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
            y_bg, x_bg = np.unravel_index(np.argmin(heatmaps[i]), heatmaps[i].shape)
            point_coords.append([(x_fg, y_fg), (x_bg, y_bg)])
            point_labels.append([1, 0])

        return self.segment(images, np.array(point_coords), np.array(point_labels))
