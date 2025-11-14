import torch.nn as nn
from torchvision import models

def get_model(arch_type="resnet18", pretrained=True, num_classes=4):
    if arch_type == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown arch_type {arch_type}")