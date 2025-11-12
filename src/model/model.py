from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from base.base_model import BaseModel


class ResNet18Classifier(BaseModel):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        weights: Optional[str] = None,
        in_channels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if weights is not None and pretrained:
            raise ValueError('Specify either pretrained=True or weights=<name>, not both.')

        if weights is not None:
            weights_enum = ResNet18_Weights[weights]
        elif pretrained:
            weights_enum = ResNet18_Weights.DEFAULT
        else:
            weights_enum = None

        self.model = resnet18(weights=weights_enum)
        if in_channels != 3:
            self._adapt_input_layer(in_channels)

        if dropout > 0.0:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def _adapt_input_layer(self, in_channels: int) -> None:
        weight = self.model.conv1.weight.data
        if in_channels == 1:
            new_weight = weight.mean(dim=1, keepdim=True)
        else:
            new_weight = weight[:, :in_channels, :, :].clone()
            if new_weight.shape[1] < in_channels:
                repeat = -(-in_channels // new_weight.shape[1])
                new_weight = new_weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
        self.model.conv1 = nn.Conv2d(
            in_channels,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False,
        )
        self.model.conv1.weight.data = new_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = ['ResNet18Classifier']
