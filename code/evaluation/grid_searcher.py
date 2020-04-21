import logging
from pathlib import Path
from typing import List

from code.evaluation.utils import report_predictions


class ConfidenceThresholdFinder:
    """
        Performing the grid search over the confidence thresholds.
        Outputs predictions for each threshold.
    """

    # Find the best threshold for filtering noise (discard patches with a confidence less than this threshold).
    # Threshold values to search.
    _CONFIDENCE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, pred_folder: Path, inference_folder: Path,
                 classes: List[str]):
        """
        Args:
            pred_folder: Path containing the predictions.
            inference_folder: Path to write predictions to.
            classes: Names of the classes in the dataset.
        """
        self._pred_folder = pred_folder
        self._inference_folder = inference_folder
        self._classes = classes

    def search(self) -> None:
        logging.info("Running grid search...")

        for confidence_th in self._CONFIDENCE_THRESHOLDS:
            report_predictions(
                patches_pred_folder=self._pred_folder,
                output_folder=self._inference_folder,
                confidence_th=confidence_th,
                classes=self._classes)
