import torch
import torch.nn as nn
import pandas as pd

def class_weights_from_csv(csv_path, split="train"):
    df = pd.read_csv(csv_path)
    counts = df[df["split"] == split]["label"].value_counts().sort_index()
    total = counts.sum()
    # inverse-frequency normalised weights
    w = torch.tensor([ total / (len(counts) * c) for c in counts ], dtype=torch.float32)
    w = w / w.mean()
    return w

class CrossEntropyWeighted(nn.Module):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self._weights = None

    def build(self, device):
        self._weights = class_weights_from_csv(self.csv_path).to(device)

    def __call__(self, logits, targets):
        if self._weights is None:
            # fallback unweighted
            return nn.functional.cross_entropy(logits, targets)
        return nn.functional.cross_entropy(logits, targets, weight=self._weights)