import os, yaml, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from tqdm import tqdm

from dataset import OCTDataset
from utils import set_seed, balanced_accuracy, macro_auc

def build_model(arch, num_classes, pretrained=True):
    if arch == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif arch == 'resnet34':
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    # adapt to 1-channel input
    conv1 = model.conv1
    if conv1.in_channels != 1:
        with torch.no_grad():
            w = conv1.weight
            w = w.sum(dim=1, keepdim=True)  # sum RGB to gray
            conv1.weight = nn.Parameter(w)
            conv1.in_channels = 1
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(conv1.weight)

    # final classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_loader(cfg, split):
    ds = OCTDataset(csv_path=cfg['data']['labels_csv'],
                    split=split,
                    img_size=cfg['data']['img_size'],
                    augment=cfg['augment'] if split == cfg['data']['train_split'] else {})
    loader = DataLoader(ds,
                        batch_size=cfg['train']['batch_size'],
                        shuffle=(split == cfg['data']['train_split']),
                        num_workers=cfg['data']['num_workers'],
                        pin_memory=True)
    return ds, loader

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    train_ds, train_loader = make_loader(cfg, cfg['data']['train_split'])
    val_ds,   val_loader   = make_loader(cfg, cfg['data']['val_split'])

    # Model
    model = build_model(cfg['model']['arch'], cfg['model']['num_classes'], cfg['model'].get('pretrained', True)).to(device)

    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg['train']['lr'],
                            weight_decay=cfg['train']['weight_decay'])
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg['train'].get('amp', True)))

    best_bal = -1.0
    patience = cfg['train'].get('early_stop_patience', 7)
    patience_ctr = 0

    for epoch in range(cfg['train']['max_epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['max_epochs']} [train]")
        for x, y, _ in pbar:
            x, y = x.to(device), torch.as_tensor(y, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- validation ---
        model.eval()
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['train']['max_epochs']} [val]"):
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred = probs.argmax(axis=1)
                y_score.append(probs)
                y_pred.append(pred)
                y_true.append(y.numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)

        bal = balanced_accuracy(y_true, y_pred)
        auc = macro_auc(y_true, y_score, cfg['model']['num_classes'])
        print(f"[val] balanced_acc={bal:.4f}  macro_auc={auc:.4f}")

        # early stopping + checkpoint
        ckpt_path = cfg['log']['ckpt_path']
        if bal > best_bal:
            best_bal = bal
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
