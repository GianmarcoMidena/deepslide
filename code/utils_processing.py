"""
DeepSlide
Contains all functions for processing.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour
"""

import functools
import itertools
import logging
import math
import time
from multiprocessing import (Process, Queue, RawArray)
from pathlib import Path
from shutil import copyfile
from typing import (Callable, Dict, List, Tuple)

import numpy as np
import pandas as pd
from PIL import Image
from imageio import (imsave, imread)
from skimage.measure import block_reduce

Image.MAX_IMAGE_PIXELS = None


def produce_patches(wsis_info: pd.DataFrame, output_folder: Path, model_stage: str,
                    inverse_overlap_factor: float,
                    num_workers: int, patch_size: int, purple_threshold: int,
                    purple_scale_size: int, image_ext: str,
                    type_histopath: bool) -> None:
    """
    Produce the patches from the WSI in parallel.

    Args:
        output_folder: Folder to save the patches to.
        inverse_overlap_factor: Overlap factor used in patch creation.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
    """
    logging.info(f"Generating {model_stage} evaluation patches...")

    output_folder.mkdir(parents=True, exist_ok=True)
    image_paths = wsis_info['path'].tolist()
    outputted_patches = 0

    logging.info(f"\ngetting small crops from {len(image_paths)} "
          f"images for {model_stage} "
          f"with inverse overlap factor {inverse_overlap_factor:.2f} "
          f"outputting in {output_folder}")

    start_time = time.time()

    for image_path in image_paths:
        image = imread(uri=image_path)

        # Sources:
        # 1. https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
        # 2. https://stackoverflow.com/questions/33247262/the-corresponding-ctypes-type-of-a-numpy-dtype
        # 3. https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
        img = RawArray(
            typecode_or_type=np.ctypeslib.as_ctypes_type(dtype=image.dtype),
            size_or_initializer=image.size)
        img_np = np.frombuffer(buffer=img,
                               dtype=image.dtype).reshape(image.shape)
        np.copyto(dst=img_np, src=image)

        # Number of x starting points.
        x_steps = int((image.shape[0] - patch_size) / patch_size *
                      inverse_overlap_factor) + 1
        # Number of y starting points.
        y_steps = int((image.shape[1] - patch_size) / patch_size *
                      inverse_overlap_factor) + 1
        # Step size, same for x and y.
        step_size = int(patch_size / inverse_overlap_factor)

        # Create the queues for passing data back and forth.
        in_queue = Queue()
        out_queue = Queue(maxsize=-1)

        # Create the processes for multiprocessing.
        processes = [
            Process(target=_find_patch_mp,
                    args=(functools.partial(
                        _find_patch,
                        output_folder=output_folder,
                        image=img_np,
                        image_loc=Path(image_path),
                        purple_threshold=purple_threshold,
                        purple_scale_size=purple_scale_size,
                        image_ext=image_ext,
                        type_histopath=type_histopath,
                        patch_size=patch_size), in_queue, out_queue))
            for __ in range(num_workers)
        ]
        for p in processes:
            p.daemon = True
            p.start()

        # Put the (x, y) coordinates in the input queue.
        for xy in itertools.product(range(0, x_steps * step_size, step_size),
                                    range(0, y_steps * step_size, step_size)):
            in_queue.put(obj=xy)

        # Store num_workers None values so the processes exit when not enough jobs left.
        for __ in range(num_workers):
            in_queue.put(obj=None)

        num_patches = sum([out_queue.get() for __ in range(x_steps * y_steps)])

        # Join the processes as they finish.
        for p in processes:
            p.join(timeout=1)

        outputted_patches += num_patches

    logging.info(
        f"finished patches for {model_stage} "
        f"with inverse overlap factor {inverse_overlap_factor:.2f} in {time.time() - start_time:.2f} seconds "
        f"outputting in {output_folder} "
        f"for {outputted_patches} patches")


def balance_classes(wsis_info: pd.DataFrame, model_stage: str) -> None:
    """
    Balancing class distribution.
    """
    logging.info("Balancing the training patches...")

    # Find the class with the most images.
    biggest_size = wsis_info['label'].value_counts().max()

    classes = wsis_info['label'].unique()
    for class_i in classes:
        image_paths = wsis_info.loc[wsis_info['label'] == class_i, 'path'].apply(Path).tolist()
        _duplicate_until_n(image_paths=image_paths, n=biggest_size)

    logging.info(f"balanced all {model_stage} classes to have {biggest_size} images\n")


def gen_train_patches(wsis_info: pd.DataFrame, output_folder: Path,
                      n_patches_per_class: int, num_workers: int,
                      patch_size: int, purple_threshold: int,
                      purple_scale_size: int, image_ext: str,
                      type_histopath: bool) -> None:
    """
    Generates all patches for classes in the training set.

    Args:
        output_folder: Folder to save the patches to.
        n_patches_per_class: The desired number of training patches per class.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
    """
    logging.info("Generating training patches...")

    # Find how much patches should overlap for each class.
    class_to_overlap_factor = _calc_class_to_overlap_factor(wsi_info=wsis_info,
                                                            n_patches_per_class=n_patches_per_class)

    # Produce the patches.
    classes = wsis_info['label'].unique()
    for class_i in classes:
        class_wsis_info = wsis_info.loc[wsis_info['label'] == class_i]
        produce_patches(wsis_info=class_wsis_info,
                        model_stage='training',
                        output_folder=output_folder.joinpath(class_i),
                        inverse_overlap_factor=class_to_overlap_factor[class_i],
                        num_workers=num_workers,
                        patch_size=patch_size,
                        purple_threshold=purple_threshold,
                        purple_scale_size=purple_scale_size,
                        image_ext=image_ext,
                        type_histopath=type_histopath)


def gen_val_patches(wsis_info: pd.DataFrame, output_folder: Path,
                    overlap_factor: float, num_workers: int, patch_size: int,
                    purple_threshold: int, purple_scale_size: int,
                    image_ext: str, type_histopath: bool) -> None:
    """
    Generates all validation patches.

    Args:
        output_folder: Folder to save the patches to.
        overlap_factor: The amount of overlap between patches.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
    """
    logging.info("Generating validation patches...")

    # Produce the patches.
    classes = wsis_info['label'].unique()
    for class_i in classes:
        class_wsis_info = wsis_info.loc[wsis_info['label'] == class_i]
        produce_patches(wsis_info=class_wsis_info, model_stage='validation',
                        output_folder=output_folder.joinpath(class_i),
                        inverse_overlap_factor=overlap_factor,
                        num_workers=num_workers,
                        patch_size=patch_size,
                        purple_threshold=purple_threshold,
                        purple_scale_size=purple_scale_size,
                        image_ext=image_ext,
                        type_histopath=type_histopath)


def _find_patch(xy_start: Tuple[int, int], output_folder: Path,
                image: np.ndarray, image_loc: Path,
                patch_size: int, image_ext: str, type_histopath: bool,
                purple_threshold: int, purple_scale_size: int) -> int:
    """
    Find the patches for a WSI.

    Args:
        output_folder: Folder to save the patches to.
        image: WSI to extract patches from.
        xy_start: Starting coordinates of the patch.
        image_loc: Location of the image to use for creating output filename.
        patch_size: Size of the patches extracted from the WSI.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.

    Returns:
        The number 1 if the image was saved successfully and a 0 otherwise.
        Used to determine the number of patches produced per WSI.
    """
    x_start, y_start = xy_start

    patch = image[x_start:x_start + patch_size, y_start:y_start +
                  patch_size, :]
    # Sometimes the images are RGBA instead of RGB. Only keep RGB channels.
    patch = patch[..., [0, 1, 2]]

    output_subsubfolder = output_folder.joinpath(
        Path(image_loc.name).with_suffix(""))
    output_subsubfolder = output_subsubfolder.joinpath(
        output_subsubfolder.name)
    output_subsubfolder.mkdir(parents=True, exist_ok=True)
    output_path = output_subsubfolder.joinpath(
        f"{image_loc.stem}_{x_start}_{y_start}.{image_ext}")

    if type_histopath:
        if _is_purple(crop=patch,
                      purple_threshold=purple_threshold,
                      purple_scale_size=purple_scale_size):
            imsave(uri=output_path, im=patch)
        else:
            return 0
    else:
        imsave(uri=output_path, im=patch)
    return 1


def _is_purple(crop: np.ndarray, purple_threshold: int,
               purple_scale_size: int) -> bool:
    """
    Determines if a given portion of an image is purple.

    Args:
        crop: Portion of the image to check for being purple.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.

    Returns:
        A boolean representing whether the image is purple or not.
    """
    block_size = (crop.shape[0] // purple_scale_size,
                  crop.shape[1] // purple_scale_size, 1)
    pooled = block_reduce(image=crop, block_size=block_size, func=np.average)

    # Calculate boolean arrays for determining if portion is purple.
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    cond1 = r > g - 10
    cond2 = b > g - 10
    cond3 = ((r + b) / 2) > g + 20

    # Find the indexes of pooled satisfying all 3 conditions.
    pooled = pooled[cond1 & cond2 & cond3]
    num_purple = pooled.shape[0]

    return num_purple > purple_threshold


def _calc_class_to_overlap_factor(wsi_info: pd.DataFrame, n_patches_per_class: int) -> Dict[Path, float]:
    """
    Find how much the inverse overlap factor should be for each folder so that
    the class distributions are approximately equal.

    Args:
        n_patches_per_class: Desired number of patches per class.

    Returns:
        A dictionary mapping classes to inverse overlap factor.
    """
    class_to_overlap_factor = {}
    classes = wsi_info['label'].unique()
    for class_i in classes:
        overlap_factor = _calc_overlap_factor(class_i, n_patches_per_class, wsi_info)
        class_to_overlap_factor[class_i] = overlap_factor

    return class_to_overlap_factor


def _calc_overlap_factor(class_: str, n_patches_per_class: int, wsi_info: pd.Series) -> float:
    """
    Find how much the inverse overlap factor should be for each folder so that
    the class distributions are approximately equal.

    Args:
        n_patches_per_class: Desired number of patches per class.

    Returns:
        A dictionary mapping classes to inverse overlap factor.
    """
    wsi_info_i = wsi_info.loc[wsi_info['label'] == class_]
    n_images = _count_images(wsi_info_i)
    total_size = _calc_total_image_size(wsi_info_i)
    # Each image is 13KB = 0.013MB, idk I just added two randomly.
    overlap_factor = max(1.0, math.sqrt(n_patches_per_class / (total_size / 0.013)) ** 1.5)
    logging.info(f"{class_}: {total_size}MB, "
                 f"{n_images} images, "
                 f"overlap_factor={overlap_factor:.2f}")
    return overlap_factor


def _count_images(wsi_info: pd.DataFrame) -> Tuple[int, float]:
    """
    Finds the number of images.
    Used to decide how much to slide windows.

    Returns:
        A tuple containing the total size of the images and the number of images.
    """
    return wsi_info['id'].drop_duplicates().shape[0]


def _calc_total_image_size(wsi_info: pd.DataFrame) -> Tuple[int, float]:
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

###########################################
#       BALANCING CLASS DISTRIBUTION      #
###########################################


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


def _find_patch_mp(func: Callable[[Tuple[int, int]], int], in_queue: Queue,
                   out_queue: Queue) -> None:
    """
    Find the patches from the WSI using multiprocessing.
    Helper function to ensure values are sent to each process
    correctly.

    Args:
        func: Function to call in multiprocessing.
        in_queue: Queue containing input data.
        out_queue: Queue to put output in.
    """
    while True:
        xy = in_queue.get()
        if xy is None:
            break
        out_queue.put(obj=func(xy))
