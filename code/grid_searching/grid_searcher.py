import logging
from pathlib import Path
from typing import List, Tuple

from code.utils import report_predictions


class GridSearcher:
    """
        Performing the grid search over the confidence thresholds.
        Initially outputs predictions for each threshold.
    """

    def __init__(self, pred_folder: Path, inference_folder: Path, classes: List[str],
                 thresholds: Tuple[float], image_ext: str):
        """
        Args:
            pred_folder: Path containing the predictions.
            inference_folder: Path to write predictions to.
            classes: Names of the classes in the dataset.
            thresholds: Threshold values to search.
            image_ext: Image extension for saving patches.
        """
        self._pred_folder = pred_folder
        self._inference_folder = inference_folder
        self._classes = classes
        self._thresholds = thresholds
        self._image_ext = image_ext

    def search(self) -> None:
        logging.info("Running grid search...")

        for threshold in self._thresholds:
            threshold_by_class = {_class: threshold for _class in self._classes}
            report_predictions(
                patches_pred_folder=self._pred_folder,
                output_folder=self._inference_folder,
                conf_thresholds=threshold_by_class,
                classes=self._classes,
                image_ext=self._image_ext)
