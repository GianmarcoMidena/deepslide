"""
DeepSlide
General helper methods used in other functions.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour
"""

import datetime
from pathlib import Path
from typing import List


def get_classes(folder: Path) -> List[str]:
    """
    Find the classes for classification.

    Args:
        folder: Folder containing the subfolders named by class.

    Returns:
        A list of strings corresponding to the class names.
    """
    return sorted([
        f.name for f in folder.iterdir()
        if ((folder.joinpath(f.name).is_dir()) and (".DS_Store" not in f.name))
    ],
                  key=str)


def get_log_csv_name(log_folder: Path) -> Path:
    """
    Find the name of the CSV file for logging.

    Args:
        log_folder: Folder to save logging CSV file in.

    Returns:
        The path including the filename of the logging CSV file with date information.
    """
    now = datetime.datetime.now()

    return log_folder.joinpath(f"log_{now.month}{now.day}{now.year}"
                               f"_{now.hour}{now.minute}{now.second}.csv")


def search_image_paths(folder: Path) -> List[Path]:
    """
    Find the full paths of the images in a folder.

    Args:
        folder: Folder containing images (assume folder only contains images).

    Returns:
        A list of the full paths to the images in the folder.
    """
    return sorted([
        folder.joinpath(f.name) for f in folder.iterdir() if ((
            folder.joinpath(f.name).is_file()) and (".DS_Store" not in f.name))
    ],
                  key=str)


def extract_subfolder_paths(folder: Path) -> List[Path]:
    """
    Find the paths of subfolders.

    Args:
        folder: Folder to look for subfolders in.

    Returns:
        A list containing the paths of the subfolders.
    """
    return sorted([
        folder.joinpath(f.name) for f in folder.iterdir()
        if ((folder.joinpath(f.name).is_dir()) and (".DS_Store" not in f.name))
    ],
                  key=str)


def get_all_image_paths(master_folder: Path) -> List[Path]:
    """
    Finds all image paths in subfolders.

    Args:
        master_folder: Root folder containing subfolders.

    Returns:
        A list of the paths to the images found in the folder.
    """
    all_paths = []
    subfolders = extract_subfolder_paths(folder=master_folder)
    if len(subfolders) > 1:
        for subfolder in subfolders:
            all_paths += search_image_paths(folder=subfolder)
    else:
        all_paths = search_image_paths(folder=master_folder)
    return all_paths
