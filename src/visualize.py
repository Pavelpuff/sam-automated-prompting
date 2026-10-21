import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import torchvision
from torchvision.transforms import v2

from src.data.loader import SyntheticData
from src.models.gradcam import GradCAM


def heatmap_to_rgb_image(heatmap: np.ndarray) -> PIL.Image.Image:
    heatmap = heatmap.astype(np.float32)
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
    heatmap_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(heatmap_color)


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    inverse_transform = v2.Compose(
        [
            v2.Normalize(
                [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)], [1 / s for s in IMAGENET_STD]
            ),
            v2.ToPILImage(),
        ]
    )

    test_dataset = SyntheticData(args.data_dir, split="test", transform=transform)

    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(
        torch.load(args.classifier_weights, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    grad_cam = GradCAM(model, target_layers=[model.layer2[-1]])

    test_indices = [3, 10, 42]
    fig, axs = plt.subplots(len(test_indices), 4, figsize=(12, 3 * len(test_indices)))
    col_titles = ["Input image", "Grad-CAM heatmap", "Overlay", "Ground-truth mask"]

    for row_idx, test_idx in enumerate(test_indices):
        img, label, mask = test_dataset[test_idx]
        cam = grad_cam(img.unsqueeze(0).to(device), targets=[label])
        heatmap_img = heatmap_to_rgb_image(cam.squeeze(0))
        inv_img = inverse_transform(img)
        overlay = PIL.Image.blend(inv_img, heatmap_img, alpha=0.3)

        images = [inv_img, heatmap_img, overlay, mask]
        for col_idx, (title, display_img) in enumerate(zip(col_titles, images)):
            ax = axs[row_idx, col_idx]
            ax.imshow(display_img)
            if row_idx == 0:
                ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    save_path = Path("assets") / "gradcam_visualization.png"
    plt.savefig(save_path, dpi=300)
    print(f"[*] Visualization saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic_shapes")
    parser.add_argument("--classifier_weights", type=str, default="model_checkpoint.pth")
    main(parser.parse_args())
