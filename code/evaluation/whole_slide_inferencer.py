import logging
from glob import glob
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,
                             balanced_accuracy_score)


class WholeSlideInferencer:
    _UNKNOWN_CLASS = 'unknown'

    # Threshold values to search.
    _CONFIDENCE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, wsi_metadata_paths: List[Path], patches_pred_folder: Path, inference_folder: Path,
                 classes: List[str]):
        """
        Args:
            patches_pred_folder: Path containing the predictions.
            inference_folder: Folder containing the predicted labels.
            classes: Names of the classes in the dataset.
        """
        self._patches_pred_folder = patches_pred_folder
        self._inference_folder = inference_folder
        self._classes = classes
        self._true_labels = self._extract_true_labels(wsi_metadata_paths)

    def search_confidence_thesholds(self) -> None:
        """
        Performing the grid search over the confidence thresholds.
        Outputs predictions for each threshold.
        """
        logging.info("Running grid search...")

        for confidence_th in self._CONFIDENCE_THRESHOLDS:
            self.report_predictions(confidence_th=confidence_th)

    def find_best_confidence_threshold(self) -> float:
        """
        Find the best accuracy and threshold for the given images.

        Returns:
            The best confidence threshold.
        """
        logging.info("Finding best confidence threshold...")

        best_score = 0
        best_confidence_th = None
        for preds_i, confidence_th_i in self._search_predictions_and_confidence_th():
            pred_labels_i = self._extract_pred_labels(predictions=preds_i, true_labels=self._true_labels)
            metrics_i = self._calc_metrics(y_true=self._true_labels, y_pred=pred_labels_i)
            conf_matrix_i = self._conf_matrix(true_labels=self._true_labels, pred_labels=pred_labels_i)
            logging.info(f"predictions with confidence > {confidence_th_i}, "
                         f"\n{metrics_i} "
                         f"\n{conf_matrix_i}")
            f_score = metrics_i['f_score']
            if f_score > best_score or best_confidence_th is None:
                best_score = f_score
                best_confidence_th = confidence_th_i
        logging.info(f"best confidence threshold: {best_confidence_th}")
        return best_confidence_th

    def final_test_results(self):
        """
        Print the final accuracy and confusion matrix.
        """
        logging.info("Computing final test results...")

        for predictions_i in self._search_predictions():
            pred_labels_i = self._extract_pred_labels(predictions=predictions_i, true_labels=self._true_labels)
            metrics_i = self._calc_metrics(y_true=self._true_labels, y_pred=pred_labels_i)
            conf_matrix_i = self._conf_matrix(true_labels=self._true_labels, pred_labels=pred_labels_i)
            logging.info(f"test set has final "
                         f"\n{metrics_i} "
                         f"\n{conf_matrix_i}")
            return metrics_i, conf_matrix_i

    def report_predictions(self, confidence_th: float) -> None:
        """
        Report the predictions for the WSI into a CSV file.

        Args:
            confidence_th: The confidence threshold for determining membership in a class (filter out noise).
        """
        logging.info(f"Outputting predictions with confidence > {confidence_th}")

        report = self._create_report()

        for prediction in self._extract_predictions(confidence_th=confidence_th):
            report = report.append(prediction, sort=False, ignore_index=True)

        self._save_report(report, confidence_th)

    def _save_report(self, report: pd.DataFrame, confidence_th: float):
        self._inference_folder.mkdir(parents=True, exist_ok=True)
        output_csv_path = self._inference_folder.joinpath(f"confidence_gt_{confidence_th * 100:.0f}_perc.csv")
        report.to_csv(output_csv_path, index=False)

    def _create_report(self) -> pd.DataFrame:
        columns = ["image_id", "prediction"] + \
                  [f'perc_{_class}_patch_preds' for _class in self._classes] + \
                  [f'count_{_class}_patch_preds' for _class in self._classes]
        return pd.DataFrame(columns=columns)

    def _extract_predictions(self, confidence_th: float):
        """
        Find the predicted class for each WSI.

        Args:
            confidence_th: Confidence threshold to determine membership in a class (filter out noise).
        """
        patches_pred_paths = sorted(glob(str(self._patches_pred_folder.joinpath("*.csv"))))
        for patches_pred_path in patches_pred_paths:
            yield self._extract_prediction(Path(patches_pred_path), confidence_th)

    def _extract_prediction(self, patches_pred_path: Path, confidence_th: float) -> Dict[str, Any]:
        """
        Find the predicted class for a single WSI.

        Args:
            patches_pred_path: File containing the predicted classes for the patches that make up the WSI.
            confidence_th: Confidence threshold to determine membership in a class (filter out noise).

        Returns:
            A string containing the accuracy of classification for each class using the confidence threshold.
        """
        patches_pred = pd.read_csv(patches_pred_path)
        class_to_count = patches_pred.loc[patches_pred['confidence'] > confidence_th, 'prediction'] \
                                     .value_counts() \
                                     .reindex(index=self._classes, fill_value=0)
        class_to_percent = class_to_count.div(np.maximum(class_to_count.sum(), 1))

        # Creating the line for output to CSV.
        arg_max = class_to_percent.sort_values(ascending=False).index[0]
        return {
            "image_id": f"{patches_pred_path.stem}",
            "prediction": arg_max,
            **{f'perc_{_class}_patch_preds': class_to_percent.loc[_class] for _class in self._classes},
            **{f'count_{_class}_patch_preds': class_to_count.loc[_class] for _class in self._classes}
        }

    def _extract_true_labels(self, wsi_metadata_paths) -> pd.Series:
        wsi_metadata = pd.DataFrame()
        for p in wsi_metadata_paths:
            wsi_metadata = wsi_metadata.append(pd.read_csv(p), ignore_index=True, sort=False)

        return wsi_metadata.loc[:, ['id', 'label']] \
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

    def _search_predictions(self):
        prediction_csv_paths = sorted(glob(str(self._inference_folder.joinpath("*.csv"))))
        for prediction_csv_path_i in prediction_csv_paths:
            yield pd.read_csv(prediction_csv_path_i)

    def _search_predictions_and_confidence_th(self):
        prediction_csv_paths = sorted(glob(str(self._inference_folder.joinpath("*.csv"))))
        for prediction_csv_path_i in prediction_csv_paths:
            predictions_i = pd.read_csv(prediction_csv_path_i)
            confidence_th_i = self._parse_confidence_th(csv_path=prediction_csv_path_i)
            yield predictions_i, confidence_th_i

    @staticmethod
    def _parse_confidence_th(csv_path: Path) -> float:
        """
        Parse the CSV filename to find the confidence threshold.

        Args:
            csv_path: Path to the CSV file containing the classes.

        Returns:
            A confidence threshold.
        """
        confidence_th_perc = Path(csv_path).stem \
            .lstrip("confidence_gt_") \
            .rstrip("_perc")
        return float(confidence_th_perc) / 100

    def _calc_metrics(self, y_true, y_pred):
        if len(self._classes) > 2:
            return self._calc_multiclass_metrics(y_true, y_pred)
        else:
            return self._calc_binary_metrics(y_true, y_pred)

    def _calc_binary_metrics(self, y_true, y_pred):
        y_true = y_true[~y_pred.isna()]
        y_pred = y_pred[~y_pred.isna()]
        f_score = self._binary_f_score(y_true=y_true, y_pred=y_pred)
        recall = self._binary_recall(y_true=y_true, y_pred=y_pred)
        specificity = self._binary_specificity(y_true=y_true, y_pred=y_pred)
        precision = self._binary_precision(y_true=y_true, y_pred=y_pred)
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        return pd.Series({
            'f_score': f_score,
            'recall': recall,
            'specificity': specificity,
            'precision': precision,
            'accuracy': accuracy
        })

    def _calc_multiclass_metrics(self, y_true, y_pred):
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
            'f_score_macro': f_score_macro, 'f_score_micro': f_score_micro, 'f_score': f_score_avg,
            **{f"precision_{label}": score for label, score in precision_per_class.items()},
            'precision_macro': precision_macro, 'precision_micro': precision_micro, 'precision': precision_avg,
            **{f"recall_{label}": score for label, score in recall_per_class.items()},
            'recall_macro': recall_macro, 'recall_micro': recall_micro, 'recall': recall_avg,
            'accuracy_micro': accuracy_micro, 'accuracy_macro': accuracy_macro, 'accuracy': accuracy_avg
        })

    def _score_per_class(self, score_fn, y_true, y_pred):
        score = score_fn(y_true=y_true, y_pred=y_pred, labels=self._classes, average=None)
        return {label_i: score_i for label_i, score_i in zip(self._classes, score)}

    def _binary_f_score(self, y_true, y_pred):
        return f1_score(y_true=y_true, y_pred=y_pred, pos_label=self._classes[1])

    def _binary_recall(self, y_true, y_pred):
        return recall_score(y_true=y_true, y_pred=y_pred, pos_label=self._classes[1])

    def _binary_specificity(self, y_true, y_pred):
        return recall_score(y_true=y_true, y_pred=y_pred, pos_label=self._classes[0])

    def _binary_precision(self, y_true, y_pred):
        return precision_score(y_true=y_true, y_pred=y_pred, pos_label=self._classes[1])
