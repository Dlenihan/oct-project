import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OCTDataset(Dataset):
    """
    Reads a labels CSV with columns:
    path (str), label (int 0..3), split in {'train','val','test'}, patient_id (str)
    Returns grayscale images as 1xHxW tensors.
    """
    def __init__(self, csv_path, split, img_size=224, augment=None):
        df = pd.read_csv(csv_path, comment="#")
        df = df[df["split"] == split].reset_index(drop=True)
        assert len(df) > 0, f"No samples found for split={split} in {csv_path}"
        self.df = df
        self.img_size = img_size
        self.augment = augment or {}

        # Base pipeline: PIL -> tensor -> resize -> normalise (grayscale)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),  # converts to CxHxW, scales to [0,1]
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        # Optional light augmentation for training
        rot_deg = self.augment.get("rotation_deg", 0)
        hflip = bool(self.augment.get("hflip", False))
        bright = float(self.augment.get("brightness", 0.0))
        contrast = float(self.augment.get("contrast", 0.0))

        aug = []
        if rot_deg and rot_deg > 0:
            aug.append(transforms.RandomRotation(degrees=rot_deg))
        if hflip:
            aug.append(transforms.RandomHorizontalFlip(p=0.5))
        if bright > 0 or contrast > 0:
            aug.append(transforms.ColorJitter(brightness=bright, contrast=contrast))
        self.aug_transform = transforms.Compose(aug) if aug else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('L')  # grayscale
        img = img.convert('RGB')  # expand to 3 channels for pretrained models
        if self.aug_transform is not None:
            img = self.aug_transform(img)
        img = self.base_transform(img)  # (1, H, W)
        label = int(row["label"])
        pid = str(row["patient_id"])
        return img, label, pid