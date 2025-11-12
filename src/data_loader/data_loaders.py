from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from base.base_data_loader import BaseDataLoader


class OCTDataset(Dataset):
    """Dataset that loads OCT images with patient-level split awareness."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f'CSV file not found: {self.csv_path}')

        self.transform = transform
        self.split = split

        dataframe = pd.read_csv(self.csv_path)
        if 'split' not in dataframe.columns:
            raise ValueError("CSV must contain a 'split' column")
        self.dataframe = dataframe[dataframe['split'] == split].reset_index(drop=True)
        if self.dataframe.empty:
            raise ValueError(f'No samples found for split={split}')

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = Path(row['path'])
        if not image_path.is_file():
            raise FileNotFoundError(f'Image file not found: {image_path}')

        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = int(row['label'])
        patient_id = row.get('patient_id')
        if pd.isna(patient_id):
            patient_id = ''
        else:
            patient_id = str(patient_id)
        return image, label, patient_id


@dataclass
class DataLoaderConfig:
    csv_path: str
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    pin_memory: bool = False
    drop_last: bool = False
    mean: float = 0.5
    std: float = 0.5
    train_augmentations: Optional[Dict[str, Dict]] = None


class OCTDataModule:
    """Creates train/val/test data loaders for the OCT dataset."""

    def __init__(self, **kwargs) -> None:
        self.config = DataLoaderConfig(**kwargs)
        self.transforms = self._build_transforms()
        self._data_loaders: Dict[str, BaseDataLoader] = {}
        self._build_dataloaders()

    def _build_transforms(self) -> Dict[str, transforms.Compose]:
        base_transforms = [transforms.Resize((self.config.image_size, self.config.image_size))]
        normalization = transforms.Normalize(mean=[self.config.mean], std=[self.config.std])

        train_transforms: List = base_transforms.copy()
        if self.config.train_augmentations:
            for name, args in self.config.train_augmentations.items():
                if not hasattr(transforms, name):
                    raise AttributeError(f'Unknown transform {name}')
                transform_cls = getattr(transforms, name)
                train_transforms.append(transform_cls(**args))
        train_transforms.extend([transforms.ToTensor(), normalization])

        eval_transforms = base_transforms + [transforms.ToTensor(), normalization]

        return {
            'train': transforms.Compose(train_transforms),
            'val': transforms.Compose(eval_transforms),
            'test': transforms.Compose(eval_transforms),
        }

    def _build_dataloaders(self) -> None:
        for split in ('train', 'val', 'test'):
            try:
                dataset = OCTDataset(self.config.csv_path, split=split, transform=self.transforms[split])
            except ValueError:
                if split == 'test':
                    # Allow missing test split in the CSV.
                    continue
                raise
            shuffle = split == 'train'
            self._data_loaders[split] = BaseDataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last if shuffle else False,
            )

    def get_loader(self, split: str) -> BaseDataLoader:
        if split not in self._data_loaders:
            raise KeyError(f'Split {split} not available')
        return self._data_loaders[split]

    @property
    def train_loader(self) -> BaseDataLoader:
        return self.get_loader('train')

    @property
    def val_loader(self) -> BaseDataLoader:
        return self.get_loader('val')

    @property
    def test_loader(self) -> Optional[BaseDataLoader]:
        return self._data_loaders.get('test')


def compute_class_weights(csv_path: str | Path) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    class_counts = df['label'].value_counts().sort_index()
    counts = torch.tensor(class_counts.values, dtype=torch.float32)
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.sum() * len(counts)
    return weights


__all__ = ['OCTDataset', 'OCTDataModule', 'compute_class_weights']
