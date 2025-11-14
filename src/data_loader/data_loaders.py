import pandas as pd
from torch.utils.data import DataLoader
from .oct_dataset import OCTDataset

class OCTDataLoaders:
    """
    Factory that builds train/val/test DataLoaders from a patient-level CSV.
    CSV columns required: path, label (int 0..3), split in {'train','val','test'}, patient_id.
    """
    def __init__(self, csv_path, img_size=224, batch_size=32, num_workers=4, augment=None):
        self.csv_path = csv_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment or {}

        self.train_ds = OCTDataset(csv_path, split="train", img_size=img_size, augment=augment)
        self.val_ds   = OCTDataset(csv_path, split="val",   img_size=img_size, augment=False)
        self.test_ds  = OCTDataset(csv_path, split="test",  img_size=img_size, augment=False)

        # Mac-friendly DataLoaders
        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0,            # mac: start with 0
            pin_memory=False,         # disable on CPU/MPS
            persistent_workers=False  # avoid worker hang
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False, persistent_workers=False
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False, persistent_workers=False
        )

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader