"""
DeepSlide
General helper methods used in other functions.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour
"""

import datetime
import logging
from glob import glob
from pathlib import Path
from typing import Dict, List

import pandas as pd


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
        if (folder.joinpath(f.name).is_dir()) and (not f.name.startswith('.'))
    ], key=str)


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


def report_predictions(patches_pred_folder: Path, output_folder: Path,
                       conf_thresholds: Dict[str, float],
                       classes: List[str], image_ext: str) -> None:
    """
    Report the predictions for the WSI into a CSV file.

    Args:
        patches_pred_folder: Folder containing the predicted classes for each patch.
        output_folder: Folder to save the predicted classes for each WSI for each threshold.
        conf_thresholds: The confidence thresholds for determining membership in a class (filter out noise).
        classes: Names of the classes in the dataset.
        image_ext: Image extension for saving patches.
    """
    logging.info("Outputting predictions...")

    # Open a new CSV file for each set of confidence thresholds used on each set of WSI.
    output_file = "".join([
        f"{_class}{str(conf_thresholds[_class])[1:]}_"
        for _class in conf_thresholds
    ])[:-1]

    output_csv_path = output_folder.joinpath(f"{output_file}.csv")

    # Confirm the output directory exists.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["img", "predicted"] + \
              [f'percent_{_class}' for _class in classes] + \
              [f'count_{_class}' for _class in classes]
    report = pd.DataFrame(columns=columns)

    csv_paths = sorted(glob(str(patches_pred_folder.joinpath("*.csv"))))
    for csv_path in csv_paths:
        prediction = _get_prediction(patches_pred_file=Path(csv_path),
                                     conf_thresholds=conf_thresholds,
                                     classes=classes, image_ext=image_ext)
        report = report.append(prediction, sort=False, ignore_index=True)

    report.to_csv(output_csv_path, index=False)


def _get_prediction(patches_pred_file: Path, conf_thresholds: Dict[str, float],
                    classes: List[str], image_ext: str) -> str:
    """
    Find the predicted class for a single WSI.

    Args:
        patches_pred_file: File containing the predicted classes for the patches that make up the WSI.
        conf_thresholds: Confidence thresholds to determine membership in a class (filter out noise).
        image_ext: Image extension for saving patches.

    Returns:
        A string containing the accuracy of classification for each class using the thresholds.
    """

    patches_pred = pd.read_csv(patches_pred_file)

    conf_thresholds = pd.Series(conf_thresholds)

    conf_over_th = patches_pred['confidence'].gt(conf_thresholds.loc[patches_pred['prediction']].values)
    class_to_count = patches_pred.loc[conf_over_th, 'prediction']\
                                 .value_counts(normalize=True)\
                                 .reindex(index=classes, fill_value=0)
    class_to_percent = class_to_count.divide(class_to_count.sum())

    # Creating the line for output to CSV.
    arg_max = class_to_percent.sort_values(ascending=False).index[0]
    return {
        "img": Path(patches_pred_file.name).with_suffix(f'.{image_ext}'),
        "predicted": arg_max,
        **{f'percent_{_class}': class_to_percent.loc[_class] for _class in classes},
        **{f'count_{_class}': class_to_count.loc[_class] for _class in classes}
    }
