import os
import numpy as np
import torch
import PIL.Image
from pathlib import Path
from typing import Callable, Literal, TypeVar, Tuple
from torch.utils.data import Dataset

T = TypeVar('T')


class SyntheticData(Dataset):
    """
    A small synthetic segmentation dataset based on CIFAR-10.
    Returns transformed image, class label, and boolean ground-truth mask.
    """
    CLASSES: tuple[str, ...] = ("circle", "square", "triangle", "star", "diamond")

    def __init__(
            self,
            path: Path | str,
            split: Literal["train", "val", "test"],
            transform: Callable[[PIL.Image.Image], T],
    ) -> None:
        self.path = Path(path)
        self.transform = transform
        self.split = split
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.CLASSES)}

        images_dir = self.path / "images" / self.split
        masks_dir = self.path / "masks" / self.split
        labels_dir = self.path / "labels" / self.split

        self.images = sorted(images_dir.glob("*.png"))
        self.masks = sorted(masks_dir.glob("*.png"))
        self.labels = sorted(labels_dir.glob("*.txt"))

        assert self.images, f"No images found in {images_dir}"
        assert len(self.images) == len(self.masks) == len(self.labels), "Mismatch in dataset files count."

        self.image_names = [p.stem for p in self.images]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[T, int, np.ndarray]:
        name = self.image_names[idx]
        img_path = self.path / "images" / self.split / f"{name}.png"
        mask_path = self.path / "masks" / self.split / f"{name}.png"
        label_path = self.path / "labels" / self.split / f"{name}.txt"

        img = PIL.Image.open(img_path).convert("RGB")
        mask = np.array(PIL.Image.open(mask_path), dtype=bool)
        label = self.class_to_idx[label_path.read_text().strip()]

        img_transformed: T = self.transform(img)

        return img_transformed, label, mask