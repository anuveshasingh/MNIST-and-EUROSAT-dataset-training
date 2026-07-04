"""Training script for A02 — runs Experiment A (frozen) and B (full) sequentially."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EuroSATDataset
from model import FineTuner
from utils import compute_metrics, get_device, get_transforms, set_seed

# ──────────────────────────────────────────────────────────────
# Experiment configurations (matches assignment spec)
# ──────────────────────────────────────────────────────────────
EXPERIMENTS: dict[str, dict] = {
    "frozen":        {"freeze": True,  "lr": 3e-4},
    "full_finetune": {"freeze": False, "lr": 1e-4},
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train A02 FineTuner — runs both experiments back-to-back"
    )
    parser.add_argument("--data_dir",    type=str,   default="./data")
    parser.add_argument("--save_dir",    type=str,   default="./artifacts")
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--image_size",  type=int,   default=224)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────

def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, float]:
    """Compute validation accuracy and macro F1."""
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.extend(model(x).argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())
    return compute_metrics(preds, labels, class_names)


# ──────────────────────────────────────────────────────────────
# Single-experiment trainer
# ──────────────────────────────────────────────────────────────

def train_experiment(
    name: str,
    freeze: bool,
    lr: float,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> Path:
    """Train one experiment; return path of best checkpoint."""
    print(f"\n{'=' * 62}")
    print(f"  Experiment : {name}")
    print(f"  freeze     : {freeze}")
    print(f"  lr         : {lr}")
    print(f"{'=' * 62}")

    model = FineTuner(num_classes=len(class_names), freeze=freeze).to(device)
    trainable, total = model.trainable_parameter_count()
    print(f"  Trainable params: {trainable:,} / {total:,}\n")

    # Only pass trainable parameters to the optimiser — matters when frozen.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(args.save_dir) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_ckpt = save_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        # ── Training step ──────────────────────────────────────
        # set_train_mode() keeps frozen BatchNorm layers in eval() to prevent
        # their running statistics from being corrupted by EuroSAT batches.
        model.set_train_mode()
        run_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        # ── Validation step ────────────────────────────────────
        val_metrics = run_eval(model, val_loader, device, class_names)
        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["macro_f1"]

        print(
            f"  Epoch {epoch:>2}/{args.epochs} | "
            f"Loss: {run_loss / max(seen, 1):.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "experiment": name,
                    "freeze": freeze,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                },
                best_ckpt,
            )

    print(f"\n  Best val accuracy [{name}]: {best_val:.4f}")
    print(f"  Checkpoint saved  : {best_ckpt}")
    return best_ckpt


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Run both experiments sequentially."""
    set_seed(args.seed)
    device = get_device()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Build shared data loaders (same splits for both experiments).
    train_ds = EuroSATDataset(
        args.data_dir, "train", get_transforms("train", args.image_size)
    )
    val_ds = EuroSATDataset(
        args.data_dir, "val", get_transforms("val", args.image_size)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # EuroSAT class names come from the split file written by download_data.py.
    import json
    with (Path(args.data_dir) / "splits.json").open() as f:
        class_names: list[str] = json.load(f)["classes"]

    checkpoints: dict[str, Path] = {}
    for name, cfg in EXPERIMENTS.items():
        checkpoints[name] = train_experiment(
            name=name,
            freeze=cfg["freeze"],
            lr=cfg["lr"],
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
        )

    print("\n\nAll experiments complete.")
    print("Run evaluate.py on each checkpoint to generate results_comparison.txt:")
    for name, ckpt in checkpoints.items():
        print(f"  python evaluate.py --checkpoint {ckpt}  # {name}")


if __name__ == "__main__":
    main(parse_args())
