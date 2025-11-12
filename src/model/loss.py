from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from data_loader.data_loaders import compute_class_weights


class CrossEntropyLossWithWeights(nn.Module):
    def __init__(
        self,
        class_weights_from_csv: Optional[str] = None,
        weight: Optional[list[float]] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if class_weights_from_csv is not None and weight is not None:
            raise ValueError('Specify either class_weights_from_csv or weight, not both.')

        if class_weights_from_csv is not None:
            weights_tensor = compute_class_weights(class_weights_from_csv)
        elif weight is not None:
            weights_tensor = torch.tensor(weight, dtype=torch.float32)
        else:
            weights_tensor = None

        self.loss = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=ignore_index)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(output, target)


__all__ = ['CrossEntropyLossWithWeights']
