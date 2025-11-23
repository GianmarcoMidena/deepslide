import itertools
import numpy as np
from skimage.measure import block_reduce

from .patch_extractor import PatchExtractor


class PurplePatchExtractor(PatchExtractor):
    def __init__(self, patch_size: int, purple_threshold: int,
                 purple_scale_size: int, image_ext: str,
                 num_workers: int, logging_level=None):
        """
        Args:
            patch_size: Size of the patches extracted from the WSI.
            purple_threshold: Number of purple points for region to be considered purple.
            purple_scale_size: Scalar to use for reducing image to check for purple.
            image_ext: Image extension for saving patches.
            num_workers: Number of workers to use for IO.
        """
        super().__init__(patch_size=patch_size, image_ext=image_ext,
                         num_workers=num_workers,
                         logging_level=logging_level)
        self._purple_threshold = purple_threshold
        self._purple_scale_size = purple_scale_size

    def check_patch(self, crop: np.ndarray) -> bool:
        """
        Determines if a given portion of an image is an admissible patch.

        Args:
            crop: Portion of the image to check for being an admissible patch.

        Returns:
            A boolean representing whether the image is an admissible patch or not.
        """
        return self._is_purple(crop)

    def _is_purple(self, crop: np.ndarray) -> bool:
        """
        Determines if a given portion of an image is purple.

        Args:
            crop: Portion of the image to check for being purple.

        Returns:
            A boolean representing whether the image is purple or not.
        """
        block_size = (crop.shape[0] // self._purple_scale_size,
                      crop.shape[1] // self._purple_scale_size, 1)
        pooled = block_reduce(image=crop, block_size=block_size, func=np.average)

        # Calculate boolean arrays for determining if portion is purple.
        r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
        cond1 = r > g - 10
        cond2 = b > g - 10
        cond3 = ((r + b) / 2) > g + 20

        # Find the indexes of pooled satisfying all 3 conditions.
        pooled = pooled[cond1 & cond2 & cond3]
        num_purple = pooled.shape[0]

        return num_purple > self._purple_threshold
