import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def balanced_accuracy(y_true, y_pred):
    # y_pred are predicted labels
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return float(np.mean(per_class))

def macro_auc(y_true, y_score, num_classes):
    # y_score shape: (N, C) softmax logits or probabilities
    y_true_onehot = np.zeros((len(y_true), num_classes))
    for i, c in enumerate(y_true):
        y_true_onehot[i, c] = 1.0
    try:
        return roc_auc_score(y_true_onehot, y_score, average='macro', multi_class='ovo')
    except Exception:
        return float('nan')
