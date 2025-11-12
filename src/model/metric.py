from __future__ import annotations

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == target).sum().item()
        return correct / target.size(0)


def macro_f1(output: torch.Tensor, target: torch.Tensor, num_classes: int | None = None) -> float:
    with torch.no_grad():
        preds = output.argmax(dim=1)
        target = target.view(-1)
        if num_classes is None:
            num_classes = output.size(1)
        f1_scores = []
        for cls in range(num_classes):
            pred_positive = preds == cls
            true_positive = target == cls
            tp = (pred_positive & true_positive).sum().item()
            fp = (pred_positive & ~true_positive).sum().item()
            fn = (~pred_positive & true_positive).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
        return sum(f1_scores) / len(f1_scores)


__all__ = ['accuracy', 'macro_f1']
