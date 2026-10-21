import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from src.data.loader import SyntheticData
from src.utils.seed import set_seed


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    desc: str,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    progress_bar = tqdm(dataloader, desc=desc)

    for imgs, labels, _ in progress_bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        total_samples += labels.shape[0]
        total_correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            train_loss=f"{total_loss / (total_samples / labels.shape[0]):.3f}",
            train_acc=f"{total_correct / total_samples:.1%}",
        )
    return {"loss": total_loss / len(dataloader), "accuracy": total_correct / total_samples}


def evaluate(
    model: nn.Module, dataloader: DataLoader, desc: str, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels, _ in tqdm(dataloader, desc=desc):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total_samples += labels.shape[0]
            total_correct += (predicted == labels).sum().item()

    return {"loss": total_loss / len(dataloader), "accuracy": total_correct / total_samples}


def main(args: argparse.Namespace) -> None:
    set_seed(42)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[*] Using device: {device}")

    IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = SyntheticData(args.data_dir, split="train", transform=transform)
    val_dataset = SyntheticData(args.data_dir, split="val", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, num_workers=2, shuffle=False
    )

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"🚀 Training CNN for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        _train_epoch(
            model, train_loader, optimizer, f"Epoch {epoch + 1}/{args.epochs} training", device
        )
        val_metrics = evaluate(
            model, val_loader, f"Epoch {epoch + 1}/{args.epochs} validation", device
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} — "
            f"val loss: {val_metrics['loss']:.3f}, "
            f"val acc: {val_metrics['accuracy']:.1%}"
        )

    torch.save(model.state_dict(), args.save_path)
    print(f"✅ Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic_shapes")
    parser.add_argument("--save_path", type=str, default="model_checkpoint.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    main(parser.parse_args())
