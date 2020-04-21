import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np


def report_predictions(patches_pred_folder: Path, output_folder: Path,
                       confidence_th: float,
                       classes: List[str]) -> None:
    """
    Report the predictions for the WSI into a CSV file.

    Args:
        patches_pred_folder: Folder containing the predicted classes for each patch.
        output_folder: Folder to save the predicted classes for each WSI for each threshold.
        confidence_th: The confidence threshold for determining membership in a class (filter out noise).
        classes: Names of the classes in the dataset.
        image_ext: Image extension for saving patches.
    """
    logging.info(f"Outputting predictions with confidence > {confidence_th}")

    report = _create_report(classes)

    for prediction in _extract_predictions(patches_pred_folder=patches_pred_folder,
                                           confidence_th=confidence_th, classes=classes):
        report = report.append(prediction, sort=False, ignore_index=True)

    _save_report(report, output_folder, confidence_th)


def _save_report(report: pd.DataFrame, output_folder: Path, confidence_th: float):
    output_folder.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_folder.joinpath(f"confidence_gt_{confidence_th*100:.0f}_perc.csv")
    report.to_csv(output_csv_path, index=False)


def _create_report(classes) -> pd.DataFrame:
    columns = ["image_id", "prediction"] + \
              [f'perc_{_class}_patch_preds' for _class in classes] + \
              [f'count_{_class}_patch_preds' for _class in classes]
    return pd.DataFrame(columns=columns)


def _extract_predictions(patches_pred_folder: Path, confidence_th: float,
                         classes: List[str]):
    """
    Find the predicted class for each WSI.

    Args:
        patches_pred_folder: Folder containing the predicted classes for each patch.
        confidence_th: Confidence threshold to determine membership in a class (filter out noise).
        classes: Names of the classes in the dataset.
        image_ext: Image extension for saving patches.
    """
    patches_pred_paths = sorted(glob(str(patches_pred_folder.joinpath("*.csv"))))
    for patches_pred_path in patches_pred_paths:
        yield _extract_prediction(Path(patches_pred_path), confidence_th, classes)


def _extract_prediction(patches_pred_path: Path, confidence_th: float,
                        classes: List[str]) -> Dict[str, Any]:
    """
    Find the predicted class for a single WSI.

    Args:
        patches_pred_path: File containing the predicted classes for the patches that make up the WSI.
        confidence_th: Confidence threshold to determine membership in a class (filter out noise).
        classes: Names of the classes in the dataset.

    Returns:
        A string containing the accuracy of classification for each class using the confidence threshold.
    """
    patches_pred = pd.read_csv(patches_pred_path)
    class_to_count = patches_pred.loc[patches_pred['confidence'] > confidence_th,
                                      'prediction'] \
        .value_counts() \
        .reindex(index=classes, fill_value=0)
    class_to_percent = class_to_count.div(np.maximum(class_to_count.sum(), 1))

    # Creating the line for output to CSV.
    arg_max = class_to_percent.sort_values(ascending=False).index[0]
    return {
        "image_id": f"{patches_pred_path.stem}",
        "prediction": arg_max,
        **{f'perc_{_class}_patch_preds': class_to_percent.loc[_class] for _class in classes},
        **{f'count_{_class}_patch_preds': class_to_count.loc[_class] for _class in classes}
    }
