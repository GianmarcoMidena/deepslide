import functools
import itertools
import logging
import time
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
from imageio import (imsave, imread)
from .step_size_finder import StepSizeFinder


class PatchExtractor(ABC):
    def __init__(self, patch_size: int, image_ext: str, num_workers: int,
                 logging_level=None):
        """
        Args:
            patch_size: Size of the patches extracted from the WSI.
            image_ext: Image extension for saving patches.
            num_workers: Number of workers to use for IO.
        """
        self._patch_size = patch_size
        self._image_ext = image_ext
        self._pool = self._create_process_pool(num_workers=num_workers)
        self._logger = logging.getLogger(f"patch_extractor_{id(self)}")
        if logging_level is None:
            logging_level = logging.INFO
        self._logger.setLevel(logging_level)

    def extract_all(self, image_paths: List[Path], step_size: int, partition_name: str, output_folder: Path = None,
                    by_wsi: bool = False) -> int:
        """
        Args:
            output_folder: Folder to save the patches to.
            by_wsi: Whether to generate the patches by folder or by image.
        """
        self._logger.info(f"\ngetting small crops from {len(image_paths)} "
                          f"{partition_name} images "
                          f"with step size {step_size} "
                          f"outputting in {output_folder}")

        total_patches_found = 0
        start_time = time.time()
        for image_path in image_paths:
            total_patches_found += self._extract_all_from_image(image_path=image_path, step_size=step_size,
                                                                output_folder=output_folder, by_wsi=by_wsi)

        if not by_wsi:
            self._logger.info(
                f"finished {partition_name} patches "
                f"with step size {step_size} "
                f"in {time.time() - start_time:.2f} seconds "
                f"outputting in {output_folder} "
                f"for {total_patches_found} patches")

        return total_patches_found

    def extract_all_by_class(self, wsis_info: pd.DataFrame, partition_name: str, output_folder: Path,
                             step_size: int = None, step_size_finder: StepSizeFinder = None):
        """
        Args:
            output_folder: Folder to save the patches to.
        """
        for class_name in self._extract_classes(wsis_info):
            self._extract_all_from_class(class_name=class_name, wsis_info=wsis_info, partition_name=partition_name,
                                         output_folder=output_folder,
                                         step_size=step_size, step_size_finder=step_size_finder)

    def _extract_all_from_class(self, class_name: str, wsis_info: pd.DataFrame, partition_name: str,
                                output_folder: Path, step_size: int = None,
                                step_size_finder: StepSizeFinder = None):
        class_wsis_info = wsis_info.loc[wsis_info['label'] == class_name]
        class_image_paths = self._search_image_paths(class_wsis_info)
        if step_size is None:
            step_size = self._calc_class_step_size(image_paths=class_image_paths,
                                                   step_size_finder=step_size_finder)
        self.extract_all(image_paths=class_image_paths, step_size=step_size, partition_name=partition_name,
                         output_folder=output_folder.joinpath(class_name))

    def _extract_all_from_image(self, image_path: Path, step_size: int, output_folder: Path = None,
                                by_wsi: bool = False) -> int:
        """
        Args:
            output_folder: Folder to save the patches to.
            by_wsi: Whether to generate the patches by folder or by image.
        """
        image = imread(uri=image_path)

        coords = self._calc_coords(image, step_size)

        patches_found = self._pool.imap_unordered(
            functools.partial(
                self._extract_one_from_coords,
                output_folder=output_folder,
                image=image,
                image_loc=image_path,
                by_wsi=by_wsi),
            coords)
        num_patches = sum([1 for patch_found in patches_found if patch_found])
        if by_wsi:
            self._logger.info(f"{image_path}: num outputted windows: {num_patches}")
        return num_patches

    def _extract_one_from_coords(self, xy_start: Tuple[int, int], image: np.ndarray, image_loc: Path,
                                 output_folder: Path = None, by_wsi: bool = False) -> int:
        """
        Args:
            output_folder: Folder to save the patches to.
            image: WSI to extract patches from.
            xy_start: Starting coordinates of the patch.
            image_loc: Location of the image to use for creating output filename.
            by_wsi: Whether to generate the patches by folder or by image.

        Returns:
            The number 1 if the image was saved successfully and a 0 otherwise.
            Used to determine the number of patches produced per WSI.
        """
        x_start, y_start = xy_start

        patch = image[x_start:x_start + self._patch_size,
                      y_start:y_start + self._patch_size, :]
        # Sometimes the images are RGBA instead of RGB. Only keep RGB channels.
        patch = patch[..., [0, 1, 2]]

        if self.check_patch(crop=patch):
            if output_folder is not None:
                self._save_patch(patch=patch, x_start=x_start, y_start=y_start, image_name=image_loc.stem,
                                 output_folder=output_folder, by_wsi=by_wsi)
            return True
        return False

    @abstractmethod
    def check_patch(self, crop: np.ndarray) -> bool:
        """
        Determines if a given portion of an image is an admissible patch.

        Args:
            crop: Portion of the image to check for being an admissible patch.

        Returns:
            A boolean representing whether the image is an admissible patch or not.
        """

    def _save_patch(self, patch: np.ndarray, x_start: int, y_start: int, image_name: str, output_folder: Path,
                    by_wsi: bool = False):
        output_folder.mkdir(parents=True, exist_ok=True)
        if by_wsi:
            output_subsubfolder = output_folder.joinpath(image_name)
            output_subsubfolder.mkdir(parents=True, exist_ok=True)
            output_path = output_subsubfolder.joinpath(
                f"{image_name}_{x_start}_{y_start}.{self._image_ext}")
        else:
            output_path = output_folder.joinpath(
                f"{image_name}_{x_start}_{y_start}.{self._image_ext}")
        imsave(uri=output_path, im=patch)

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

    @staticmethod
    def _calc_class_step_size(image_paths: List[Path], step_size_finder: StepSizeFinder) -> int:
        return step_size_finder.search(image_paths)

    @staticmethod
    def _create_process_pool(num_workers: int):
        """
        Args:
            num_workers: Number of workers to use for IO.
        """
        return ThreadPool(processes=num_workers)

    @staticmethod
    def _extract_classes(wsis_info):
        return wsis_info['label'].unique()

    @staticmethod
    def _search_image_paths(wsis_info: pd.DataFrame) -> List[Path]:
        return wsis_info['path'].apply(Path).tolist()

    def set_logger_level(self, level):
        self._logger.setLevel(level)
