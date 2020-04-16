import functools
import itertools
import logging
import math
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from imageio import (imsave, imread)
from skimage.measure import block_reduce

Image.MAX_IMAGE_PIXELS = None


class PatchExtractor:
    def __init__(self, patch_size: int, purple_threshold: int,
                 purple_scale_size: int, image_ext: str, type_histopath: bool,
                 num_workers: int):
        """
        Args:
            num_workers: Number of workers to use for IO.
            patch_size: Size of the patches extracted from the WSI.
            purple_threshold: Number of purple points for region to be considered purple.
            purple_scale_size: Scalar to use for reducing image to check for purple.
            image_ext: Image extension for saving patches.
            type_histopath: Only look for purple histopathology images and filter whitespace.
        """
        self._patch_size = patch_size
        self._purple_threshold = purple_threshold
        self._purple_scale_size = purple_scale_size
        self._image_ext = image_ext
        self._type_histopath = type_histopath
        self._num_workers = num_workers

    def gen_train_patches(self, wsis_info: pd.DataFrame, output_folder: Path,
                          n_patches_per_class: int) -> None:
        """
        Generates all patches for classes in the training set.

        Args:
            output_folder: Folder to save the patches to.
            n_patches_per_class: The desired number of training patches per class.
        """
        logging.info("Generating training patches...")

        step_size_by_class = self._calc_step_size_by_class(wsis_info=wsis_info,
                                                           n_patches_per_class=n_patches_per_class)

        for class_i in self._extract_classes(wsis_info):
            self._produce_patches_by_class(class_i=class_i,
                                           wsis_info=wsis_info,
                                           partition_name='training',
                                           output_folder=output_folder,
                                           step_size=step_size_by_class[class_i],
                                           by_folder=False)

    def gen_val_patches(self, wsis_info: pd.DataFrame, output_folder: Path,
                        step_size: int) -> None:
        """
        Generates all validation patches.

        Args:
            output_folder: Folder to save the patches to.
        """
        logging.info("Generating validation patches...")

        for class_i in self._extract_classes(wsis_info):
            self._produce_patches_by_class(class_i=class_i,
                                           wsis_info=wsis_info,
                                           partition_name='validation',
                                           output_folder=output_folder,
                                           step_size=step_size,
                                           by_folder=False)

    def _produce_patches_by_class(self, class_i: str, wsis_info: pd.DataFrame,
                                  output_folder: Path, partition_name: str,
                                  step_size: int, by_folder: bool):
        class_wsis_info = wsis_info.loc[wsis_info['label'] == class_i]
        self.produce_patches(wsis_info=class_wsis_info,
                             partition_name=partition_name,
                             output_folder=output_folder.joinpath(class_i),
                             step_size=step_size,
                             by_folder=by_folder)

    def produce_patches(self, wsis_info: pd.DataFrame, output_folder: Path,
                        partition_name: str, step_size: int,
                        by_folder: bool = True) -> None:
        """
        Produce the patches from the WSI in parallel.

        Args:
            output_folder: Folder to save the patches to.
            by_folder: Whether to generate the patches by folder or by image.
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        image_paths = wsis_info['path'].tolist()
        outputted_patches = 0

        logging.info(f"\ngetting small crops from {len(image_paths)} "
                     f"{partition_name} images "
                     f"with step size {step_size} "
                     f"outputting in {output_folder}")

        start_time = time.time()
        pool = ThreadPool(processes=self._num_workers)
        for image_path in image_paths:
            num_patches = self._produce_patches_by_image(image_path,
                                                         output_folder,
                                                         step_size,
                                                         by_folder, pool)
            if by_folder:
                print(f"{image_path}: num outputted windows: {num_patches}")
            else:
                outputted_patches += num_patches

        if not by_folder:
            logging.info(
                f"finished {partition_name} patches "
                f"with step size {step_size} "
                f"in {time.time() - start_time:.2f} seconds "
                f"outputting in {output_folder} "
                f"for {outputted_patches} patches")

    def _produce_patches_by_image(self, image_path: Path, output_folder: Path,
                                  step_size: int, by_folder: bool,
                                  pool: ThreadPool):
        """
        Args:
            output_folder: Folder to save the patches to.
            by_folder: Whether to generate the patches by folder or by image.
        """
        image = imread(uri=image_path)

        coords = self._calc_coords(image, step_size)

        patches_found = pool.imap_unordered(functools.partial(
                                            self._find_patch,
                                            output_folder=output_folder,
                                            image=image,
                                            image_loc=Path(image_path),
                                            by_folder=by_folder), coords)
        num_patches = sum([1 for patch_found in patches_found if patch_found])
        return num_patches

    def _calc_coords(self, image: np.ndarray,
                     step_size: int):
        x_steps = self._count_x_steps(image, step_size)
        y_steps = self._count_y_steps(image, step_size)

        return itertools.product(range(0, x_steps * step_size, step_size),
                                 range(0, y_steps * step_size, step_size))

    def _count_x_steps(self, image: np.ndarray,
                       step_size: int):
        return int((image.shape[0] - self._patch_size) / step_size) + 1

    def _count_y_steps(self, image: np.ndarray,
                       step_size: int):
        return int((image.shape[1] - self._patch_size) / step_size) + 1

    def _find_patch(self, xy_start: Tuple[int, int], output_folder: Path,
                    image: np.ndarray, image_loc: Path, by_folder: bool) -> int:
        """
        Find the patches for a WSI.

        Args:
            output_folder: Folder to save the patches to.
            image: WSI to extract patches from.
            xy_start: Starting coordinates of the patch.
            image_loc: Location of the image to use for creating output filename.
            by_folder: Whether to generate the patches by folder or by image.

        Returns:
            The number 1 if the image was saved successfully and a 0 otherwise.
            Used to determine the number of patches produced per WSI.
        """
        x_start, y_start = xy_start

        patch = image[x_start:x_start + self._patch_size,
                      y_start:y_start + self._patch_size, :]
        # Sometimes the images are RGBA instead of RGB. Only keep RGB channels.
        patch = patch[..., [0, 1, 2]]

        if by_folder:
            output_subsubfolder = output_folder.joinpath(image_loc.stem)
            output_subsubfolder = output_subsubfolder.joinpath(output_subsubfolder.name)
            output_subsubfolder.mkdir(parents=True, exist_ok=True)
            output_path = output_subsubfolder.joinpath(
                f"{image_loc.stem}_{x_start}_{y_start}.{self._image_ext}")
        else:
            output_path = output_folder.joinpath(
                f"{image_loc.stem}_{x_start}_{y_start}.{self._image_ext}")

        if self._type_histopath:
            if self._is_purple(crop=patch):
                imsave(uri=output_path, im=patch)
            else:
                return False
        else:
            imsave(uri=output_path, im=patch)
        return True

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

    def _calc_step_size_by_class(self, wsis_info: pd.DataFrame, n_patches_per_class: int) -> Dict[Path, int]:
        """
        Find how much the step size should be for each folder so that
        the class distributions are approximately equal.

        Args:
            n_patches_per_class: Desired number of patches per class.

        Returns:
            A dictionary mapping classes to step size.
        """
        step_size_by_class = {}
        for class_i in self._extract_classes(wsis_info):
            overlap_factor = self._calc_overlap_factor(class_i, n_patches_per_class, wsis_info)
            step_size_by_class[class_i] = int(self._patch_size / overlap_factor)
        return step_size_by_class

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
    def _extract_classes(wsis_info):
        return wsis_info['label'].unique()
