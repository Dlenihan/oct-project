import os, yaml, torch, torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Simple Grad-CAM for last conv layer of ResNet
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, scores, class_idx=None):
        # scores: model output logits for one image
        if class_idx is None:
            class_idx = scores.argmax().item()
        self.model.zero_grad()
        scores[0, class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2,3), keepdim=True)  # GAP over HxW
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def build_model(num_classes, ckpt, device):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # adapt to 1ch
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

def preprocess(img_path, img_size):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(img_path).convert('L')
    tensor = t(img).unsqueeze(0)
    return img, tensor

def overlay_cam(img_pil, cam, alpha=0.35):
    img = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
    heat = (np.uint8(255 * cam))
    heat = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=-1)  # red heatmap
    heat = heat.astype(np.float32) / 255.0
    heat = np.array(Image.fromarray((heat*255).astype(np.uint8)).resize((img.shape[1], img.shape[0]))).astype(np.float32)/255.0
    ov = (1-alpha)*img + alpha*heat
    ov = np.clip(ov*255, 0, 255).astype(np.uint8)
    return Image.fromarray(ov)

def main(config_path, ckpt_path, out_dir, sample_paths=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(cfg['model']['num_classes'], ckpt_path, device)
    target_layer = model.layer4[-1].conv2  # last conv layer
    cam = GradCAM(model, target_layer)

    if sample_paths is None:
        print("Provide specific sample image paths with --samples to visualise.")
        return

    for p in sample_paths:
        img_pil, x = preprocess(p, cfg['data']['img_size'])
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
        cam_map = cam.generate(logits, class_idx=None)
        ov = overlay_cam(img_pil, cam_map)
        out_path = os.path.join(out_dir, os.path.basename(p).rsplit('.',1)[0] + "_cam.png")
        ov.save(out_path)
        print("Saved:", out_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--samples", nargs="+", help="Paths to sample images")
    args = ap.parse_args()
    main(args.config, args.ckpt, args.out, args.samples)
