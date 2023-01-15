import os
import pandas as pd
import torch
import ile
from torch.utils.data import Dataset


class NHAMCSDataset(Dataset):
    def __init__(self, year: int, transform=None, target_transform=None):
        self._dataset = ile.io.NHAMCS(year)
        
        self._x = self._generate_x()
        self._y = self._generate_y()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self._y)

    def __getitem__(self, index):
        x = self._x[index, :]
        y = self._y[index]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            waittime = self.target_transform(waittime)
        return x, y

    def _generate_x(self):
        columns = []
        # RFV3D
        columns += [f'RFV{i}3D' for i in range(1, 6)]
        # RFV
        columns += [f'RFV{i}' for i in range(1, 6)]

        df = self._dataset.data[columns]
        for column in columns:
            # 1) create and concat new columns
            df_dummies = pd.get_dummies(df[column], prefix=f'{column}_')
            df = pd.concat([df, df_dummies], axis=1)
            # 2) drop the original column
            df.drop([column], axis=1, inplace=True)
        
        return df.to_numpy()

    def _generate_y(self):
        return self._dataset.data['WAITTIME'].values
