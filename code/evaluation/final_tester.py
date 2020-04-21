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

    def find_best_confidence_threshold(self, wsis_info: pd.DataFrame, inference_folder: Path) -> float:
        """
        Find the best accuracy and threshold for the given images.

        Args:
            inference_folder: Folder containing the predicted labels.
        Returns:
            The best confidence threshold.
        """
        logging.info("Finding best confidence threshold...")

        best_macro_avg_acc = 0
        best_confidence_th = None
        for preds_i, confidence_th_i in self._search_predictions_and_confidence_th(inference_folder):
            true_labels_i = self._extract_true_labels(wsis_info)
            pred_labels_i = self._extract_pred_labels(preds_i, true_labels_i)
            macro_avg_acc_i = self._calc_macro_avg_class_acc(true_labels=true_labels_i, pred_labels=pred_labels_i)
            micro_avg_acc_i = self._calc_micro_avg_class_acc(true_labels=true_labels_i, pred_labels=pred_labels_i)
            partial_macro_avg_acc_i = self._calc_macro_avg_class_acc(true_labels=true_labels_i,
                                                                     pred_labels=pred_labels_i, dropna=True)
            partial_micro_avg_acc_i = self._calc_micro_avg_class_acc(true_labels=true_labels_i,
                                                                     pred_labels=pred_labels_i, dropna=True)
            logging.info(f"predictions with confidence > {confidence_th_i} "
                         f"has macro average class accuracy {macro_avg_acc_i:.5f} "
                         f"({partial_macro_avg_acc_i:.5f} without unknowns), "
                         f"micro average class accuracy {micro_avg_acc_i:.5f} "
                         f"({partial_micro_avg_acc_i:.5f} without unknowns)")
            if macro_avg_acc_i > best_macro_avg_acc:
                best_macro_avg_acc = macro_avg_acc_i
                best_confidence_th = confidence_th_i
        logging.info(f"best confidence threshold: {best_confidence_th}")
        return best_confidence_th

    def print_final_test_results(self, wsis_info: pd.DataFrame,
                                 inference_folder: Path) -> None:
        """
        Print the final accuracy and confusion matrix.

        Args:
            inference_folder: Folder containing the predicted labels.
        """
        logging.info("Printing final test results...")

        for predictions_i in self._search_predictions(inference_folder):
            true_labels_i = self._extract_true_labels(wsis_info)
            pred_labels_i = self._extract_pred_labels(predictions_i, true_labels_i)
            macro_avg_acc_i = self._calc_macro_avg_class_acc(true_labels=true_labels_i, pred_labels=pred_labels_i)
            micro_avg_acc_i = self._calc_micro_avg_class_acc(true_labels=true_labels_i, pred_labels=pred_labels_i)
            partial_macro_avg_acc_i = self._calc_macro_avg_class_acc(true_labels=true_labels_i,
                                                                     pred_labels=pred_labels_i, dropna=True)
            partial_micro_avg_acc_i = self._calc_micro_avg_class_acc(true_labels=true_labels_i,
                                                                     pred_labels=pred_labels_i, dropna=True)
            conf_matrix_i = self._conf_matrix(true_labels_i, pred_labels_i)
            logging.info(f"test set has final "
                         f"macro avg acc: {macro_avg_acc_i:.5f} ({partial_macro_avg_acc_i:.5f} without unknowns), "
                         f"micro avg acc: {micro_avg_acc_i:.5f} ({partial_micro_avg_acc_i:.5f} without unknowns)"
                         f"\n{conf_matrix_i}")

    @staticmethod
    def _extract_true_labels(wsis_info) -> pd.Series:
        return wsis_info.loc[:, ['id', 'label']] \
                        .rename(columns={'id': 'image_id'}) \
                        .set_index('image_id') \
                        .squeeze() \
                        .sort_index()

    @staticmethod
    def _extract_pred_labels(predictions, true_labels: pd.Series) -> pd.Series:
        return predictions.rename(columns={'prediction': 'label'}) \
                            .loc[:, ['image_id', 'label']] \
                            .set_index('image_id', drop=True) \
                            .squeeze() \
                            .reindex(true_labels.index) \
                            .sort_index()

    def _calc_micro_avg_class_acc(self, true_labels: pd.Series, pred_labels: pd.Series,
                                  dropna: bool=False) -> Tuple[float, np.ndarray]:
        """
        Find the micro average class accuracy of the predictions.

        Args:
            true_labels: Ground truth label dictionary from filenames to label strings.
            pred_labels: Predicted label dictionary from filenames to label strings.

        Returns:
            A tuple containing the micro average class accuracy and a confusion matrix.
        """
        if dropna:
            pred_labels = pred_labels.dropna()
            true_labels = true_labels.loc[pred_labels.index]
        n_right_preds = pred_labels.eq(true_labels).sum()
        n_examples = true_labels.shape[0]
        return n_right_preds / n_examples

    def _calc_macro_avg_class_acc(self, true_labels: pd.Series, pred_labels: pd.Series,
                                  dropna: bool=False) -> Tuple[float, np.ndarray]:
        """
        Find the macro average class accuracy of the predictions.

        Args:
            true_labels: Ground truth label dictionary from filenames to label strings.
            pred_labels: Predicted label dictionary from filenames to label strings.

        Returns:
            A tuple containing the macro average class accuracy and a confusion matrix.
        """
        if dropna:
            pred_labels = pred_labels.dropna()
            true_labels = true_labels.loc[pred_labels.index]
        n_right_preds_by_label = true_labels.loc[pred_labels.eq(true_labels)] \
                                            .value_counts() \
                                            .reindex(index=self._classes, fill_value=0)
        n_examples_by_true_label = true_labels.value_counts() \
                                              .reindex(index=self._classes, fill_value=0)
        acc_by_class = n_right_preds_by_label.div(np.maximum(n_examples_by_true_label, 1))
        return acc_by_class.mean()

    def _conf_matrix(self, true_labels: pd.Series,
                     pred_labels: pd.Series) -> Tuple[float, np.ndarray]:
        """
        Calculate the confusion matrix.

        Args:
            true_labels: Ground truth label dictionary from filenames to label strings.
            pred_labels: Predicted label dictionary from filenames to label strings.

        Returns:
            A tuple containing the confusion matrix.
        """
        conf_matrix = confusion_matrix(y_true=true_labels, y_pred=pred_labels.fillna('Unknown'))
        return conf_matrix

    @staticmethod
    def _parse_confidence_th(csv_path: Path) -> float:
        """
        Parse the CSV filename to find the confidence threshold.

        Args:
            csv_path: Path to the CSV file containing the classes.

        Returns:
            A condifence threshold.
        """
        confidence_th_perc = Path(csv_path).stem\
                                           .lstrip("confidence_gt_")\
                                           .rstrip("_perc")
        return float(confidence_th_perc)/100

    def _search_predictions(self, inference_folder):
        prediction_csv_paths = sorted(glob(str(inference_folder.joinpath("*.csv"))))
        for prediction_csv_path_i in prediction_csv_paths:
            yield pd.read_csv(prediction_csv_path_i)

    def _search_predictions_and_confidence_th(self, inference_folder):
        prediction_csv_paths = sorted(glob(str(inference_folder.joinpath("*.csv"))))
        for prediction_csv_path_i in prediction_csv_paths:
            predictions_i = pd.read_csv(prediction_csv_path_i)
            confidence_th_i = self._parse_confidence_th(csv_path=prediction_csv_path_i)
            yield predictions_i, confidence_th_i
