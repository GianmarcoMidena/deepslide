import logging
from pathlib import Path
from typing import List
from shutil import copyfile
import pandas as pd


class ClassBalancer:
    """
    Balancing class distribution.
    """
    def __init__(self, wsis_info: pd.DataFrame, partition_name: str):
        self._wsis_info = wsis_info
        self._partition_name = partition_name

    def balance(self) -> None:
        logging.info("Balancing the learning patches...")

        # Find the class with the most images.
        biggest_size = self._wsis_info['label'].value_counts().max()

        classes = self._wsis_info['label'].unique()
        for class_i in classes:
            image_paths = self._wsis_info.loc[self._wsis_info['label'] == class_i, 'path'].apply(Path).tolist()
            self._duplicate_until_n(image_paths=image_paths, n=biggest_size)

        logging.info(f"balanced all {self._partition_name} classes to have {biggest_size} images\n")

    @staticmethod
    def _duplicate_until_n(image_paths: List[Path], n: int) -> None:
        """
        Duplicate the underrepresented classes to balance class distributions.

        Args:
            image_paths: Image paths to check for balance.
            n: Desired number of images.
        """
        num_dupls = n - len(image_paths)

        logging.info(f"balancing {image_paths[0].parent} by duplicating {num_dupls}")

        for i in range(num_dupls):
            image_path = image_paths[i % len(image_paths)]

            xys = image_path.name.split("_")
            x = xys[:-2]
            y = xys[-2:]

            copyfile(src=image_path,
                     dst=Path(
                         image_path.parent, f"{'_'.join(x)}dup"
                                            f"{(i // len(image_paths)) + 2}_"
                                            f"{'_'.join(y)}"))
