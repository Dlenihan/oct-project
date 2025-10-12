import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OCTDataset(Dataset):
    """
    Reads a labels.csv with columns: patient_id, path, label, split
    and returns grayscale images as 1xHxW tensors.
    """
    def __init__(self, csv_path, split, img_size=224, augment=None):
        df = pd.read_csv(csv_path, comment='#')
        df = df[df['split'] == split].reset_index(drop=True)
        assert len(df) > 0, f"No samples found for split={split} in {csv_path}"
        self.df = df
        self.img_size = img_size
        self.augment = augment or {}
        t_list = [transforms.ToTensor(),
                  transforms.Resize((img_size, img_size), antialias=True),
                  transforms.Normalize(mean=[0.5], std=[0.5])]
        self.base_transform = transforms.Compose(t_list)

        # light augmentations for training
        rot_deg = self.augment.get('rotation_deg', 0)
        hflip = self.augment.get('hflip', False)
        bright = float(self.augment.get('brightness', 0.0))
        contrast = float(self.augment.get('contrast', 0.0))

        aug_list = []
        if rot_deg and rot_deg > 0:
            aug_list.append(transforms.RandomRotation(degrees=rot_deg))
        if hflip:
            aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if bright > 0 or contrast > 0:
            aug_list.append(transforms.ColorJitter(brightness=bright, contrast=contrast))

        self.aug_transform = transforms.Compose(aug_list) if aug_list else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('L')  # grayscale
        if self.aug_transform is not None:
            img = self.aug_transform(img)
        img = self.base_transform(img)  # (1, H, W)
        label = int(row['label'])
        pid = str(row['patient_id'])
        return img, label, pid
