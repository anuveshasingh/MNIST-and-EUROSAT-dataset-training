"""Dataset wrapper for EuroSAT split loading."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import EuroSAT


class EuroSATDataset(Dataset):
    """EuroSAT split dataset.

    Returns:
        Tuple[image_tensor, label_int]
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform,
        split_file: str = "splits.json",
    ) -> None:
        """Initialize a split view of EuroSAT."""
        super().__init__()
        self.base = EuroSAT(root=data_dir, download=False, transform=transform)
        with Path(data_dir, split_file).open("r", encoding="utf-8") as f:
            split_data = json.load(f)
        self.indices = split_data[split]

    def __len__(self) -> int:
        """Return split length."""
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Return transformed image and integer class label."""
        return self.base[self.indices[idx]]
