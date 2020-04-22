import logging
from glob import glob
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,
                             balanced_accuracy_score)


class FinalTester:
    _UNKNOWN_CLASS = 'unknown'

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

        best_score = 0
        best_confidence_th = None
        for preds_i, confidence_th_i in self._search_predictions_and_confidence_th(inference_folder):
            true_labels_i = self._extract_true_labels(wsis_info)
            pred_labels_i = self._extract_pred_labels(preds_i, true_labels_i)
            metrics_i = self._calc_metrics(true_labels_i, pred_labels_i)
            conf_matrix_i = self._conf_matrix(true_labels_i, pred_labels_i)
            logging.info(f"predictions with confidence > {confidence_th_i}, "
                         f"\n{metrics_i} "
                         f"\n{conf_matrix_i}")
            f_score_avg = metrics_i['f_score_avg']
            if f_score_avg > best_score:
                best_score = f_score_avg
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
            metrics_i = self._calc_metrics(true_labels_i, pred_labels_i)
            conf_matrix_i = self._conf_matrix(true_labels_i, pred_labels_i)
            logging.info(f"test set has final "
                         f"\n{metrics_i} "
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
                                  dropna: bool = False) -> Tuple[float, np.ndarray]:
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
                                  dropna: bool = False) -> Tuple[float, np.ndarray]:
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

    def _conf_matrix(self, true_labels: pd.Series, pred_labels: pd.Series) -> pd.Series:
        """
        Calculate the confusion matrix.

        Args:
            true_labels: Ground truth label dictionary from filenames to label strings.
            pred_labels: Predicted label dictionary from filenames to label strings.

        Returns:
            A tuple containing the confusion matrix.
        """
        conf_matrix = confusion_matrix(y_true=true_labels, y_pred=pred_labels.fillna(self._UNKNOWN_CLASS),
                                       labels=self._classes + [self._UNKNOWN_CLASS])
        return pd.DataFrame(conf_matrix[:len(self._classes)], index=self._classes,
                            columns=self._classes + [self._UNKNOWN_CLASS])\
                 .rename_axis(index='ground truth') \
                 .rename_axis(columns='predictions')

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

    def _calc_metrics(self, y_true, y_pred):
        y_pred = y_pred.fillna(self._UNKNOWN_CLASS)
        f_score_per_class = self._score_per_class(f1_score, y_true, y_pred)
        f_score_micro = f1_score(y_true=y_true, y_pred=y_pred, labels=self._classes, average='micro')
        f_score_macro = f1_score(y_true=y_true, y_pred=y_pred, labels=self._classes, average='macro')
        f_score_avg = (f_score_micro + f_score_macro) / 2
        precision_per_class = self._score_per_class(precision_score, y_true, y_pred)
        precision_micro = precision_score(y_true=y_true, y_pred=y_pred, labels=self._classes, average='micro')
        precision_macro = precision_score(y_true=y_true, y_pred=y_pred, labels=self._classes, average='macro')
        precision_avg = (precision_micro + precision_macro) / 2
        recall_per_class = self._score_per_class(recall_score, y_true, y_pred)
        recall_micro = recall_score(y_true=y_true, y_pred=y_pred, labels=self._classes, average='micro')
        recall_macro = recall_score(y_true=y_true, y_pred=y_pred, labels=self._classes, average='macro')
        recall_avg = (recall_micro + recall_macro) / 2
        accuracy_macro = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        accuracy_micro = accuracy_score(y_true=y_true, y_pred=y_pred)
        accuracy_avg = (accuracy_micro + accuracy_macro) / 2
        return pd.Series({
            **{f"f_score_{label}": score for label, score in f_score_per_class.items()},
            'f_score_macro': f_score_macro, 'f_score_micro': f_score_micro, 'f_score_avg': f_score_avg,
            **{f"precision_{label}": score for label, score in precision_per_class.items()},
            'precision_macro': precision_macro, 'precision_micro': precision_micro, 'precision_avg': precision_avg,
            **{f"recall_{label}": score for label, score in recall_per_class.items()},
            'recall_macro': recall_macro, 'recall_micro': recall_micro, 'recall_avg': recall_avg,
            'accuracy_micro': accuracy_micro, 'accuracy_macro': accuracy_macro, 'accuracy_avg': accuracy_avg
        })

    def _score_per_class(self, score_fn, y_true, y_pred):
        score = score_fn(y_true=y_true, y_pred=y_pred, labels=self._classes, average=None)
        return {label_i: score_i for label_i, score_i in zip(self._classes, score)}
