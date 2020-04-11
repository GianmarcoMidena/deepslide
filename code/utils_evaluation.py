"""
DeepSlide
Functions for evaluation.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour
"""
import logging
import operator
from glob import glob
from pathlib import Path
from typing import (Dict, List, Tuple)

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


###########################################
#         THRESHOLD GRID SEARCH           #
###########################################


def get_prediction(patches_pred_file: Path, conf_thresholds: Dict[str, float],
                   image_ext: str) -> str:
    """
    Find the predicted class for a single WSI.

    Args:
        patches_pred_file: File containing the predicted classes for the patches that make up the WSI.
        conf_thresholds: Confidence thresholds to determine membership in a class (filter out noise).
        image_ext: Image extension for saving patches.

    Returns:
        A string containing the accuracy of classification for each class using the thresholds.
    """
    classes = list(conf_thresholds.keys())
    # Predicted class distribution per slide.
    class_to_count = {_class: 0 for _class in classes}

    # Looping through all the lines in the file and adding predictions.
    with patches_pred_file.open(mode="r") as patches_pred:

        patches_pred_lines = patches_pred.readlines()[1:]

        for line in patches_pred_lines:
            line_items = line[:-1].split(",")
            line_class = line_items[2]
            line_conf = float(line_items[3])
            if line_class in classes and line_conf > conf_thresholds[
                    line_class]:
                class_to_count[line_class] += 1
        if sum(class_to_count.values()) > 0:
            class_to_percent = {
                _class: class_to_count[_class] / sum(class_to_count.values())
                for _class in class_to_count
            }
        else:
            class_to_percent = {_class: 0 for _class in class_to_count}

    # Creating the line for output to CSV.
    return f"{Path(patches_pred_file.name).with_suffix(f'.{image_ext}')}," \
           f"{max(class_to_percent.items(), key=operator.itemgetter(1))[0]}," \
           f"{','.join([f'{class_to_percent[_class]:.5f}' for _class in classes])}," \
           f"{','.join([f'{class_to_count[_class]:.5f}' for _class in classes])}"


def output_all_predictions(patches_pred_folder: Path, output_folder: Path,
                           conf_thresholds: Dict[str, float],
                           classes: List[str], image_ext: str) -> None:
    """
    Output the predictions for the WSI into a CSV file.

    Args:
        patches_pred_folder: Folder containing the predicted classes for each patch.
        output_folder: Folder to save the predicted classes for each WSI for each threshold.
        conf_thresholds: The confidence thresholds for determining membership in a class (filter out noise).
        classes: Names of the classes in the dataset.
        image_ext: Image extension for saving patches.
    """
    logging.info("Outputting all predictions...")

    # Open a new CSV file for each set of confidence thresholds used on each set of WSI.
    output_file = "".join([
        f"{_class}{str(conf_thresholds[_class])[1:]}_"
        for _class in conf_thresholds
    ])

    output_csv_path = output_folder.joinpath(f"{output_file[:-1]}.csv")

    # Confirm the output directory exists.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to the output CSV.
    with output_csv_path.open(mode="w") as writer:
        writer.write(
            f"img,predicted,"
            f"{','.join([f'percent_{_class}' for _class in classes])},"
            f"{','.join([f'count_{_class}' for _class in classes])}\n")

        csv_paths = sorted(glob(str(patches_pred_folder.joinpath("*.csv"))))
        for csv_path in csv_paths:
            writer.write(
                f"{get_prediction(patches_pred_file=Path(csv_path), conf_thresholds=conf_thresholds, image_ext=image_ext)}\n"
            )


def grid_search(pred_folder: Path, inference_folder: Path, classes: List[str],
                threshold_search: Tuple[float], image_ext: str) -> None:
    """
    Main function for performing the grid search over the confidence thresholds. Initially outputs
    predictions for each threshold.

    Args:
        pred_folder: Path containing the predictions.
        inference_folder: Path to write predictions to.
        classes: Names of the classes in the dataset.
        threshold_search: Threshold values to search.
        image_ext: Image extension for saving patches.
    """
    logging.info("Running grid search...")

    for threshold in threshold_search:
        output_all_predictions(
            patches_pred_folder=pred_folder,
            output_folder=inference_folder,
            conf_thresholds={_class: threshold
                             for _class in classes},
            classes=classes,
            image_ext=image_ext)


###########################################
#        FINDING BEST THRESHOLDS          #
###########################################
def get_scores(true_labels: pd.DataFrame, prediction_labels: pd.DataFrame,
               classes: List[str]) -> Tuple[float, np.ndarray]:
    """
    Find the average class accuracy of the predictions.

    Args:
        true_labels: Ground truth label dictionary from filenames to label strings.
        prediction_labels: Predicted label dictionary from filenames to label strings.
        classes: Names of the classes in the dataset.

    Returns:
        A tuple containing the average class accuracy and a confusion matrix.
    """
    labels = true_labels.rename(columns={'label': 'true_label'})\
                        .merge(prediction_labels.rename(columns={'predicted': 'pred_label'}),
                               on='image_id', how='inner', sort=False)\
                        .sort_values(by='image_id')

    n_right_preds_by_label = labels.loc[labels['true_label'].eq(labels['pred_label']), 'true_label']\
                                   .value_counts()\
                                   .reindex(index=classes, fill_value=0)
    n_examples_by_true_label = labels['true_label'].value_counts()\
                                                   .reindex(index=classes, fill_value=0)
    acc_by_class = n_right_preds_by_label.div(n_examples_by_true_label)
    avg_class_acc = acc_by_class.mean()

    conf_matrix = confusion_matrix(y_true=labels['true_label'], y_pred=labels['pred_label'])
    return avg_class_acc, conf_matrix


def parse_thresholds(csv_path: Path) -> Dict[str, float]:
    """
    Parse the CSV filename to find the classes for each threshold.

    Args:
        csv_path: Path to the CSV file containing the classes.

    Returns:
        A dictionary mapping class names to thresholds.
    """
    class_to_threshold = {}
    items = Path(csv_path).with_suffix("").name.split("_")

    for item in items:
        subitems = item.split(".")
        _class = subitems[0]
        class_to_threshold[_class] = float(f"0.{subitems[1]}")

    return class_to_threshold


def find_best_acc_and_thresh(wsis_info: pd.DataFrame,
                             inference_folder: Path, classes: List[str]) -> \
        Dict[str, float]:
    """
    Find the best accuracy and threshold for the given images.

    Args:
        inference_folder: Folder containing the predicted labels.
        classes: Names of the classes in the dataset.

    Returns:
        A dictionary mapping class names to the best thresholds.
    """
    logging.info("Finding best thresholds...")

    prediction_csv_paths = sorted(glob(str(inference_folder.joinpath("*.csv"))))
    best_acc = 0
    best_thresholds = None
    best_csv = None
    for prediction_csv_path in prediction_csv_paths:
        predictions = pd.read_csv(prediction_csv_path)
        predictions['image_id'] = predictions['img'].str.rsplit(".", expand=True).iloc[:, 0]
        true_labels = wsis_info.loc[wsis_info['id'].isin(predictions['image_id'].values), ['id', 'label']]\
                               .rename(columns={'id': 'image_id'})
        prediction_labels = predictions[['image_id', 'predicted']]
        avg_class_acc, conf_matrix = get_scores(true_labels=true_labels, prediction_labels=prediction_labels,
                                                classes=classes)
        logging.info(f"thresholds {parse_thresholds(csv_path=prediction_csv_path)} "
                     f"has average class accuracy {avg_class_acc:.5f}")
        if best_acc < avg_class_acc:
            best_acc = avg_class_acc
            best_csv = prediction_csv_path
            best_thresholds = parse_thresholds(csv_path=prediction_csv_path)
    logging.info(f"view these predictions in {best_csv}")
    return best_thresholds


def print_final_test_results(wsis_info: pd.DataFrame, inference_folder: Path,
                             classes: List[str]) -> None:
    """
    Print the final accuracy and confusion matrix.

    Args:
        inference_folder: Folder containing the predicted labels.
        classes: Names of the classes in the dataset.
    """
    logging.info("Printing final test results...")

    prediction_csv_paths = sorted(glob(str(inference_folder.joinpath("*.csv"))))
    for prediction_csv_path in prediction_csv_paths:
        predictions = pd.read_csv(prediction_csv_path)
        predictions['image_id'] = predictions['img'].str.rsplit(".", expand=True).iloc[:, 0]
        true_labels = wsis_info.loc[wsis_info['id'].isin(predictions['image_id'].values), ['id', 'label']] \
                               .rename(columns={'id': 'image_id'})
        prediction_labels = predictions[['image_id', 'predicted']]
        avg_class_acc, conf_matrix = get_scores(true_labels=true_labels, prediction_labels=prediction_labels,
                                                classes=classes)
        logging.info(f"test set has final avg class acc: {avg_class_acc:.5f}"
              f"\n{conf_matrix}")
