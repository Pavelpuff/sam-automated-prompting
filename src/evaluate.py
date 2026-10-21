import argparse

import numpy as np
import torch
import torchvision
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

import wandb
from src.data.loader import SyntheticData
from src.models.gradcam import GradCAM
from src.models.sam import SamPipelineForeground, SamPipelineWithBackground
from src.utils.metrics import compute_iou, get_center_of_mass
from src.utils.seed import set_seed


def main(args: argparse.Namespace) -> None:
    set_seed(42)
    wandb.init(project="sam-zero-shot-segmentation", config=vars(args))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[*] Using device: {device}")

    # Transforms
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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)

    # Load Models
    resnet = torchvision.models.resnet18(weights=None)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 5)
    resnet.load_state_dict(
        torch.load(args.classifier_weights, map_location=device, weights_only=True)
    )
    resnet.to(device).eval()

    target_layers = [resnet.layer2[-1]]
    grad_cam = GradCAM(resnet, target_layers)

    sam_model = sam_model_registry["vit_b"](checkpoint=args.sam_weights).to(device)
    sam_predictor = SamPredictor(sam_model)

    pipeline = (
        SamPipelineForeground(sam_predictor, inverse_transform, grad_cam)
        if args.approach == 1
        else SamPipelineWithBackground(sam_predictor, inverse_transform, grad_cam)
    )

    # Evaluation Loop
    ious, distances, hits, total_points = [], [], 0, 0

    for images, labels, gt_masks in tqdm(test_loader, desc=f"Evaluating Approach {args.approach}"):
        images, gt_masks = images.to(device), gt_masks.to(device)
        pred_masks = pipeline(images).to(device)

        for pred, gt in zip(pred_masks, gt_masks):
            ious.append(compute_iou(pred, gt))

        heatmaps = grad_cam(images, targets=None)
        gt_masks_np = gt_masks.cpu().numpy()

        for i, heatmap in enumerate(heatmaps):
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            if gt_masks_np[i][y, x]:
                hits += 1
            total_points += 1

            center = get_center_of_mass(gt_masks_np[i])
            if center:
                distances.append(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2))

    metrics = {
        "mean_iou": np.mean(ious),
        "hit_rate": hits / total_points,
        "mean_distance_px": np.mean(distances),
    }

    wandb.log(metrics)
    print(
        f"[*] Evaluation completed: IoU: {metrics['mean_iou']:.2%}, "
        f"Hit Rate: {metrics['hit_rate']:.2%}, "
        f"Dist: {metrics['mean_distance_px']:.2f}px"
    )
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic_shapes")
    parser.add_argument("--classifier_weights", type=str, default="model_checkpoint.pth")
    parser.add_argument("--sam_weights", type=str, default="sam_vit_b_01ec64.pth")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument(
        "--approach", type=int, choices=[1, 2], default=1, help="1: FG only, 2: FG+BG"
    )
    main(parser.parse_args())
