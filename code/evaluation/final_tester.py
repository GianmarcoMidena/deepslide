import logging
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class FinalTester:
    def __init__(self, classes: List[str]):
        """
        Args:
            classes: Names of the classes in the dataset.
        """
        self._classes = classes

    def find_best_acc_and_thresh(self, wsis_info: pd.DataFrame,
                                 inference_folder: Path) \
            -> Dict[str, float]:
        """
        Find the best accuracy and threshold for the given images.

        Args:
            inference_folder: Folder containing the predicted labels.
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
            true_labels = wsis_info.loc[wsis_info['id'].isin(predictions['image_id'].values), ['id', 'label']] \
                .rename(columns={'id': 'image_id'})
            prediction_labels = predictions[['image_id', 'predicted']]
            avg_class_acc, conf_matrix = self._get_scores(true_labels=true_labels,
                                                          prediction_labels=prediction_labels)
            logging.info(f"thresholds {self._parse_thresholds(csv_path=prediction_csv_path)} "
                         f"has average class accuracy {avg_class_acc:.5f}")
            if best_acc < avg_class_acc:
                best_acc = avg_class_acc
                best_csv = prediction_csv_path
                best_thresholds = self._parse_thresholds(csv_path=prediction_csv_path)
        logging.info(f"view these predictions in {best_csv}")
        return best_thresholds

    def print_final_test_results(self, wsis_info: pd.DataFrame,
                                 inference_folder: Path) -> None:
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
            avg_class_acc, conf_matrix = self._get_scores(true_labels=true_labels,
                                                          prediction_labels=prediction_labels)
            logging.info(f"test set has final avg class acc: {avg_class_acc:.5f}"
                         f"\n{conf_matrix}")

    def _get_scores(self, true_labels: pd.DataFrame,
                    prediction_labels: pd.DataFrame) -> Tuple[float, np.ndarray]:
        """
        Find the average class accuracy of the predictions.

        Args:
            true_labels: Ground truth label dictionary from filenames to label strings.
            prediction_labels: Predicted label dictionary from filenames to label strings.

        Returns:
            A tuple containing the average class accuracy and a confusion matrix.
        """
        labels = true_labels.rename(columns={'label': 'true_label'}) \
            .merge(prediction_labels.rename(columns={'predicted': 'pred_label'}),
                   on='image_id', how='inner', sort=False) \
            .sort_values(by='image_id')

        n_right_preds_by_label = labels.loc[labels['true_label'].eq(labels['pred_label']), 'true_label'] \
            .value_counts() \
            .reindex(index=self._classes, fill_value=0)
        n_examples_by_true_label = labels['true_label'].value_counts() \
            .reindex(index=self._classes, fill_value=0)
        acc_by_class = n_right_preds_by_label.div(n_examples_by_true_label)
        avg_class_acc = acc_by_class.mean()

        conf_matrix = confusion_matrix(y_true=labels['true_label'], y_pred=labels['pred_label'])
        return avg_class_acc, conf_matrix

    @staticmethod
    def _parse_thresholds(csv_path: Path) -> Dict[str, float]:
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
