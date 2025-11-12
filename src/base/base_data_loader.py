from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


class BaseDataLoader(DataLoader):
    """Base class for all data loaders."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn: Optional = None,
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

    @property
    def device(self) -> torch.device:
        if hasattr(self.dataset, 'device'):
            return self.dataset.device
        return torch.device('cpu')
