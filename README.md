# OCT Retinal Disease Classification (PyTorch)

This repository adapts and extends the [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)
framework for 4-class retinal OCT image classification (CNV, DME, DRUSEN, NORMAL),
based on the dataset from *Kermany et al., Cell 2018* ([Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3)).

---

## Overview

| Component | Description |
|------------|-------------|
| **Base template** | Adapted from `victoresque/pytorch-template` for training, evaluation, and logging. |
| **Dataset loader** | CSV-driven loader for OCT data (`patient_id`, `path`, `label`, `split`). |
| **Models** | ResNet-18/50, EfficientNet-B0, ViT-Tiny (ImageNet-pretrained). |
| **Loss** | Cross-Entropy with optional class weighting. |
| **Metrics** | Accuracy, sensitivity, specificity, F1-score, ROC-AUC per class. |
| **Explainability** | Grad-CAM visualisation for model interpretability. |
| **Environment** | Local CPU testing → GPU training on University of Liverpool HPC. |

---

## Repository structure
OCT_Project/
├── data/
│    ├── metadata/
│    │    └── labels.csv
│    └── raw/              # train/val/test folder structure
├── notebooks/
│    ├── OCT_Dataset_Inspection.ipynb
│    └── 01_DatasetLoader_Sanity.ipynb
├── src/
│    ├── data/
│    │    └── oct_dataset.py
│    ├── models/
│    │    ├── resnet.py
│    │    ├── efficientnet.py
│    │    └── vit.py
│    ├── train.py
│    ├── evaluate.py
│    └── gradcam.py
├── experiments/
│    ├── exp000_repro_baseline.yaml
│    ├── exp101_aug.yaml
│    ├── exp102_effnet.yaml
│    └── exp103_gradcam.yaml
├── results/
│    ├── inspection_2025-10-12/
│    └── loader_sanity/
├── reports/
│    └── OCT_Dataset_Inspection_2025-10-12.html
├── CHANGELOG.md
└── README.md

---

## Setup

```bash
git clone https://github.com/Dlenihan/oct-project.git
cd oct-project
conda env create -f environment.yml
conda activate oct


# Baseline reproduction (ResNet-18)
python -u src/train.py --config experiments/exp000_repro_baseline.yaml

# Augmented run
python -u src/train.py --config experiments/exp101_aug.yaml

# Evaluate
python src/evaluate.py --config experiments/exp101_aug.yaml


Base training framework: victoresque/pytorch-template
OCT dataset: Kermany et al., Cell 2018, Mendeley Data
Grad-CAM implementation adapted from jacobgil/pytorch-grad-cam
