"""Model definition for A02 transfer learning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class FineTuner(nn.Module):
    """ResNet-18 fine-tuning wrapper for EuroSAT.

    Input shape:
        x: (B, 3, H, W)
    Output shape:
        logits: (B, num_classes)

    Args:
        num_classes: Number of target classes (10 for EuroSAT).
        freeze: If True, freeze all backbone layers except the new head.
    """

    def __init__(self, num_classes: int = 10, freeze: bool = True) -> None:
        """Initialise backbone and classifier head."""
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._freeze_backbone = freeze

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the 1000-class ImageNet head.
        # nn.Linear always initialises with requires_grad=True, so the new
        # head remains trainable even when the rest of the backbone is frozen.
        in_features = self.backbone.fc.in_features  # 512 for ResNet-18
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        return self.backbone(x)

    def set_train_mode(self) -> None:
        """Switch to training mode with BatchNorm handled correctly.

        When the backbone is frozen we keep its BatchNorm layers in eval()
        so that their accumulated running statistics are not corrupted by the
        new data distribution.  Only the replacement head goes into train().

        Without this, leaving frozen BN layers in train() mode causes them to
        update their running mean/var with EuroSAT batch statistics, which
        degrades the pretrained representations silently.
        """
        if self._freeze_backbone:
            # Whole model to eval first — this also handles all BN layers.
            self.backbone.eval()
            # Switch only the new head back to train.
            self.backbone.fc.train()
        else:
            self.train()

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter (used for switching to full fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._freeze_backbone = False

    def trainable_parameter_count(self) -> tuple[int, int]:
        """Return (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
