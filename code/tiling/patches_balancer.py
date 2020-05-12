import logging
from pathlib import Path
from typing import List
from shutil import copyfile
import pandas as pd

from code.utils import extract_subfolder_paths, search_folder_file_paths, search_all_image_paths


class PatchesBalancer:
    """
    Balancing class distribution.
    """
    def __init__(self, image_dir: Path, partition_name: str):
        self._image_dir = image_dir
        self._partition_name = partition_name
        self._patches_info = None

    def balance_by_class(self) -> None:
        logging.info(f"Balancing the {self._partition_name} patches...")

        self._patches_info = self._extract_patches_info()

        biggest_class_size = self._count_the_biggest_class_items()

        for class_i in self._extract_classes():
            self._balance_class(class_i, biggest_class_size)

        logging.info(f"balanced all {self._partition_name} classes to have {biggest_class_size} images\n")

    def _extract_patches_info(self):
        subfolders = extract_subfolder_paths(folder=self._image_dir)
        return pd.concat([pd.Series(search_all_image_paths(subfolder)).to_frame('path').assign(label=subfolder)
                          for subfolder in subfolders],
                         ignore_index=True, sort=False)

    def _balance_class(self, class_i: str, biggest_class_size: int):
        class_i_image_paths = self._search_class_image_paths(class_i)
        n_lacking_images = biggest_class_size - len(class_i_image_paths)
        self._duplicate(image_paths=class_i_image_paths,
                        n_duplicates=n_lacking_images)

    def _count_the_biggest_class_items(self):
        """
        Find the class with the most images.
        """
        return self._patches_info['label'].value_counts().max()

    def _extract_classes(self):
        return self._patches_info['label'].unique()

    def _search_class_image_paths(self, class_i: str):
        return self._patches_info.loc[self._patches_info['label'] == class_i, 'path'].apply(Path).tolist()

    @staticmethod
    def _duplicate(image_paths: List[Path], n_duplicates: int) -> None:
        """
        Duplicate the underrepresented classes to balance class distributions.

        Args:
            image_paths: Image paths to check for balance.
            n_duplicates: Desired number of images.
        """
        logging.info(f"balancing {image_paths[0].parent} by duplicating {n_duplicates}")

        for i in range(n_duplicates):
            image_path = image_paths[i % len(image_paths)]

            patch_id_parts = image_path.name.split("_")
            image_id = patch_id_parts[:-2]
            patch_coords = patch_id_parts[-2:]

            copyfile(src=image_path,
                     dst=Path(
                         image_path.parent, f"{'_'.join(image_id)}dup"
                                            f"{(i // len(image_paths)) + 2}_"
                                            f"{'_'.join(patch_coords)}"))
