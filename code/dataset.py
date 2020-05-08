from pathlib import Path
import json
from typing import List
import pandas as pd
from PIL import Image

import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, metadata_paths: List[Path], class_idx_path: Path, transform=None):
        self._transform = transform
        self._class_idx = json.load(class_idx_path.open())

        metadata = pd.DataFrame()
        for p in metadata_paths:
            metadata = metadata.append(pd.read_csv(p), ignore_index=True, sort=False)
        self._metadata = metadata

    def __len__(self) -> int:
        return self._metadata.shape[0]

    def __getitem__(self, idx):
        metadata = self._metadata.iloc[idx]

        image_path = metadata.loc['path']
        image = Image.open(image_path)
        if self._transform:
            image = self._transform(image)

        label = metadata.loc['label']
        label_idx = self._class_idx[label]

        return image, label_idx
