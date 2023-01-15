import os
import pandas as pd
import torch
import ile

class NHAMCSDataset(torch.Dataset):
    def __init__(self, year: int, transform=None, target_transform=None):
        self.dataset = ile.io.NHAMCS(year)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label