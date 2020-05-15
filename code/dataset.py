from pathlib import Path
import json
from typing import List
import pandas as pd
import numpy as np
from PIL import Image

import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, class_idx_path: Path, patch_size: int, metadata_paths: List[Path] = None, metadata: pd.DataFrame = None,
                 transform=None):
        self._patch_size = patch_size
        self._transform = transform
        self._class_idx = json.load(class_idx_path.open())

        if metadata is not None:
            self._metadata = metadata
        else:
            metadata = pd.DataFrame()
            for p in metadata_paths:
                metadata = metadata.append(pd.read_csv(p), ignore_index=True, sort=False)
            self._metadata = metadata

    def __len__(self) -> int:
        return self._metadata.shape[0]

    def __getitem__(self, idx):
        patches_metadata = self._metadata.iloc[idx]

        patch_path = patches_metadata.loc['path']
        coodrs_corner = np.asarray(Path(patch_path).stem.rsplit('_', 2))[1:].astype(int)
        coords_central = coodrs_corner + self._patch_size // 2
        x_coord, y_coord = coords_central
        patch = Image.open(patch_path)
        if self._transform:
            patch = self._transform(patch)

        label = patches_metadata.loc['label']
        label_idx = self._class_idx[label]

        return {"patch": patch, "x_coord": x_coord, "y_coord": y_coord}, label_idx
