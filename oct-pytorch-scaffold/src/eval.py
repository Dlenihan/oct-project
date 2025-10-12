import yaml, os, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix
from dataset import OCTDataset
from utils import balanced_accuracy, macro_auc

def build_model(num_classes, ckpt, device, pretrained=False):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    # adapt conv1 to 1-channel
    conv1 = model.conv1
    if conv1.in_channels != 1:
        with torch.no_grad():
            w = conv1.weight.sum(dim=1, keepdim=True)
            conv1.weight = nn.Parameter(w)
            conv1.in_channels = 1
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(conv1.weight)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model.to(device).eval()

def main(config_path, ckpt_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset/loader
    test_ds = OCTDataset(cfg['data']['labels_csv'], cfg['data']['test_split'], cfg['data']['img_size'])
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=cfg['data']['num_workers'], pin_memory=True)

    model = build_model(cfg['model']['num_classes'], ckpt_path, device)

    ys, ps, scores = [], [], []
    with torch.no_grad():
        for x,y,_ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(1)
            ys.append(y.numpy())
            ps.append(pred)
            scores.append(probs)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps); y_score = np.concatenate(scores)
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=3))
    print(f"Balanced accuracy: {balanced_accuracy(y_true, y_pred):.4f}")
    print(f"Macro AUC: {macro_auc(y_true, y_score, cfg['model']['num_classes']):.4f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt)
