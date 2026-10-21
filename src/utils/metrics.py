import numpy as np
from torch import Tensor


def compute_iou(pred_mask: Tensor, gt_mask: Tensor) -> float:
    intersection = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()


def get_center_of_mass(mask: np.ndarray) -> tuple[float, float] | None:
    if not mask.any():
        return None
    y_indices, x_indices = np.where(mask)
    return np.mean(x_indices), np.mean(y_indices)
