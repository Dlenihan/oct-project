---

## CHANGELOG.md

```markdown
# Changelog

## v0.1 – 2025-10-12
- Completed dataset inspection (`OCT_Dataset_Inspection.ipynb`)
- Verified train/val/test structure (109 309 images)
- Added inspection plots and metadata

## v0.2 – Dataset & loader
- Implemented CSV-based `OCTDataset` class
- Added augmentation options (flip, rotation, brightness)
- Created loader sanity notebook and batch preview grid

## v0.3 – Training pipeline
- Integrated ResNet-18 backbone (ImageNet pretrained)
- Added YAML config system
- Implemented training/validation loop with checkpointing

## v0.4 – Evaluation & explainability
- Added metrics: accuracy, F1, sensitivity, specificity, ROC-AUC
- Integrated Grad-CAM visualisation

## v0.5 – HPC deployment
- Added SLURM submission scripts
- GPU training verified on University HPC cluster