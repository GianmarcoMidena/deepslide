import logging
import math
from pathlib import Path
from typing import Tuple
import pandas as pd


class StepSizeCalculator:
    def __init__(self, patch_size: int, n_patches_per_class: int,
                 wsis_info: pd.DataFrame):
        """
        Args:
            patch_size: Size of the patches extracted from the WSI.
            n_patches_per_class: Desired number of patches per class.
        """
        self._patch_size = patch_size
        self._wsis_info = wsis_info
        self._n_patches_per_class = n_patches_per_class

    def calc_class_step_size(self, class_name: str) -> int:
        """
        Find how much the step size should be for each folder so that
        the class distributions are approximately equal.

        Returns:
            class step size
        """
        overlap_factor = self._calc_overlap_factor(class_name,
                                                   self._n_patches_per_class,
                                                   self._wsis_info)
        return int(self._patch_size / overlap_factor)

    def _calc_overlap_factor(self, class_: str, n_patches_per_class: int,
                             wsi_info: pd.DataFrame) -> float:
        """
        Find how much the inverse overlap factor should be for each folder so that
        the class distributions are approximately equal.

        Args:
            n_patches_per_class: Desired number of patches per class.

        Returns:
            A dictionary mapping classes to inverse overlap factor.
        """
        wsi_info_i = wsi_info.loc[wsi_info['label'] == class_]
        n_images = self._count_images(wsi_info_i)
        total_images_size = self._calc_total_image_size(wsi_info_i)
        # Each image is 13KB = 0.013MB, idk I just added two randomly.
        patch_size = 0.013
        min_n_patches_per_class = total_images_size / patch_size
        overlap_factor = max(1.0, math.sqrt(n_patches_per_class / min_n_patches_per_class) ** 1.5)
        logging.info(f"{class_}: {total_images_size}MB, "
                     f"{n_images} images, "
                     f"overlap_factor={overlap_factor:.2f}")
        return overlap_factor

    @staticmethod
    def _count_images(wsi_info: pd.DataFrame) -> Tuple[int, float]:
        """
        Finds the number of images.
        Used to decide how much to slide windows.

        Returns:
            A tuple containing the total size of the images and the number of images.
        """
        return wsi_info['id'].drop_duplicates().shape[0]

    @staticmethod
    def _calc_total_image_size(wsi_info: pd.DataFrame) -> float:
        """
        Finds the size of images.
        Used to decide how much to slide windows.

        Returns:
            A tuple containing the total size of the images and the number of images.
        """
        file_size = 0
        image_paths = wsi_info['path'].tolist()
        for image_path in image_paths:
            file_size += Path(image_path).stat().st_size

        file_size_mb = file_size / 1e6
        return file_size_mb

    @staticmethod
    def _extract_classes(wsis_info) -> str:
        return wsis_info['label'].unique()
