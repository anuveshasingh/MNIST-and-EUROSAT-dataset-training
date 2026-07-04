"""Evaluation script for A02."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dataset import EuroSATDataset
from model import FineTuner
from utils import get_device, get_transforms


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate A02 checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Evaluate accuracy, macro F1, and per-class accuracy."""
    device = get_device()

    test_ds = EuroSATDataset(
        args.data_dir,
        "test",
        get_transforms("test", args.image_size),
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    with Path(args.data_dir, "splits.json").open("r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    model = FineTuner(num_classes=len(classes), freeze=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())

    print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"Macro F1: {f1_score(labels, preds, average='macro'):.4f}")
    for i, c in enumerate(classes):
        idx = [j for j, y in enumerate(labels) if y == i]
        acc = sum(int(preds[j] == labels[j]) for j in idx) / max(len(idx), 1)
        print(f"{c:>20s}: {acc:.4f}")


if __name__ == "__main__":
    main(parse_args())
