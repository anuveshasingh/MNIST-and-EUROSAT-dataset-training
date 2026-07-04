"""Model skeleton for A01."""

from __future__ import annotations

import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  
        x = self.pool(x)               
        x = self.relu(self.conv2(x))   
        x = self.pool(x)              
        x = x.flatten(1)               
        x = self.relu(self.fc1(x))     
        x = self.fc2(x)                
        return x
