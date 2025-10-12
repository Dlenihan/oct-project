# OCT Image Classification (PyTorch + HPC)

This repository is a **starter scaffold** for classifying OCT (or other medical) images using CNNs in PyTorch.
It includes:
- Dataset loader reading a `labels.csv` (with **patient-level** splits to avoid leakage)
- Baseline **ResNet-18** fine-tuning for **1‑channel** images (e.g., OCT B-scans)
- Training loop with **mixed precision** and **balanced accuracy** metric
- Evaluation script and **Grad-CAM** visualisation
- **SLURM** script for HPC submission
- Minimal **YAML config** example

> ⚠️ **Important:** Ensure your train/val/test splits are by **patient_id**, not by image, to avoid leakage.

## Directory Structure

```
oct-project/
  data/                  # your images live elsewhere (e.g., HPC $SCRATCH); this folder is a placeholder
  experiments/
    exp001_baseline.yaml
  logs/
  runs/                  # tensorboard logs
  slurm/
    train_resnet18.sbatch
  src/
    dataset.py
    train.py
    eval.py
    gradcam.py
    utils.py
  labels.csv             # path,label,split,patient_id (template provided)
  requirements.txt
  README.md
```

## Quick Start (local or HPC)
1. Create environment (example):
   ```bash
   conda create -n oct python=3.10 -y
   conda activate oct
   pip install -r requirements.txt
   # On HPC, install torch/torchvision matching the CUDA of the cluster.
   ```

2. Prepare `labels.csv` (see template below). **Paths** can be absolute or relative.
3. Edit `experiments/exp001_baseline.yaml` to set:
   - `num_classes`, `img_size`, `batch_size`, `max_epochs`
   - dataset paths
4. Run training locally:
   ```bash
   python -u src/train.py --config experiments/exp001_baseline.yaml
   ```
   Or submit to HPC with SLURM:
   ```bash
   sbatch slurm/train_resnet18.sbatch
   ```

5. Evaluate:
   ```bash
   python -u src/eval.py --config experiments/exp001_baseline.yaml --ckpt best_resnet18.pt
   ```

6. Grad-CAM visualisation for a few samples:
   ```bash
   python -u src/gradcam.py --config experiments/exp001_baseline.yaml --ckpt best_resnet18.pt --out cam_examples
   ```

## `labels.csv` Template
CSV with header: `patient_id,path,label,split`.
```
patient_id,path,label,split
P001,/path/to/P001/scan_01.png,0,train
P001,/path/to/P001/scan_02.png,0,train
P002,/path/to/P002/scan_01.png,1,val
P003,/path/to/P003/scan_01.png,0,test
```
- `label` should be integer-coded: `0..(num_classes-1)`
- **Ensure no patient_id appears in more than one split**.

## Tips
- For OCT B-scans, avoid vertical flips (they invert anatomy).
- Use **small rotations** (±5°), mild brightness/contrast jitter, and light Gaussian noise.
- Always validate with **balanced accuracy**/**macro AUC** when classes are imbalanced.
- Start 2‑D slice classification; later aggregate to **patient-level** (mean logits or attention pooling).

---

## Citation
If you use this scaffold in a publication or report, you can cite it informally as:
> "OCT PyTorch scaffold (2025), starter code prepared by an AI assistant for undergraduate research."

