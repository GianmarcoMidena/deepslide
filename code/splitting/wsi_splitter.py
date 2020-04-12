import logging
from pathlib import Path
import pandas as pd
from typing import List
from code.utils import search_image_paths, extract_subfolder_paths


class WSISplitter:
    """
    Splits the data into learning, validation, and evaluation sets.
    """

    _IMAGE_ATTRIBUTES = ['id', 'path', 'label']

    def __init__(self, all_wsi: Path, val_wsi_per_class: int,
                 test_wsi_per_class: int, wsis_train: Path,
                 wsis_test: Path, wsis_val: Path):
        """
        Args:
            all_wsi: Location of the WSIs organized in subfolders by class.
            val_wsi_per_class: Number of WSI per class to use in the validation set.
            test_wsi_per_class: Number of WSI per class to use in the test set.
            wsis_train: Location to store the CSV file labels for learning.
            wsis_test: Location to store the CSV file labels for evaluation.
            wsis_val: Location to store the CSV file labels for validation.
        """
        self._all_wsi = all_wsi
        self._val_wsi_per_class = val_wsi_per_class
        self._test_wsi_per_class = test_wsi_per_class
        self._wsis_train_path = wsis_train
        self._wsis_val_path = wsis_val
        self._wsis_test_path = wsis_test
        self._wsis_train_report = None
        self._wsis_val_report = None
        self._wsis_test_report = None

    def split(self) -> None:
        logging.info("Splitting the slides into learning, validation and test...")

        self._create_report()

        class_paths = self._search_class_paths()
        for class_path in class_paths:
            self._split_per_class(class_path)

        self._save_report()

    def _split_per_class(self, class_path):
        class_image_paths = search_image_paths(class_path)
        self._check_n_slides_per_class(class_image_paths)

        train_image_paths, val_image_paths, test_image_paths = self._split_train_val_test_images(class_image_paths)

        logging.info(f"class {class_path.name} "
                     f"#train={len(train_image_paths)} "
                     f"#val={len(val_image_paths)} "
                     f"#test={len(test_image_paths)}")

        self._report_splits(train_image_paths, val_image_paths, test_image_paths)

    def _split_train_val_test_images(self, image_paths) -> (List[Path], List[Path], List[Path]):
        test_start_idx = len(image_paths) - self._test_wsi_per_class
        val_start_idx = test_start_idx - self._val_wsi_per_class
        train_image_paths = image_paths[:val_start_idx]
        val_image_paths = image_paths[val_start_idx:test_start_idx]
        test_image_paths = image_paths[test_start_idx:]
        return train_image_paths, val_image_paths, test_image_paths

    def _search_class_paths(self) -> List[Path]:
        return extract_subfolder_paths(self._all_wsi)

    def _check_n_slides_per_class(self, image_paths) -> None:
        assert len(image_paths) > self._val_wsi_per_class + self._test_wsi_per_class, \
            "Not enough slides in each class."

    def _create_report(self) -> None:
        self._wsis_train_report = pd.DataFrame(columns=self._IMAGE_ATTRIBUTES)
        self._wsis_val_report = pd.DataFrame(columns=self._IMAGE_ATTRIBUTES)
        self._wsis_test_report = pd.DataFrame(columns=self._IMAGE_ATTRIBUTES)

    def _report_splits(self, train_image_paths: List[Path],
                       val_image_paths: List[Path],
                       test_image_paths: List[Path]) -> None:
        self._wsis_train_report = self._increment_report(self._wsis_train_report, train_image_paths)
        self._wsis_val_report = self._increment_report(self._wsis_val_report, val_image_paths)
        self._wsis_test_report = self._increment_report(self._wsis_test_report, test_image_paths)

    @staticmethod
    def _increment_report(images_info: pd.DataFrame, image_paths: List[Path]) -> pd.DataFrame:
        for image_path in image_paths:
            image_info = {'id': str(image_path.with_suffix("").name),
                          'path': str(image_path),
                          'label': image_path.parent.name}
            images_info = images_info.append(image_info, ignore_index=True, sort=False)
        return images_info

    def _save_report(self) -> None:
        self._wsis_train_report.to_csv(self._wsis_train_path, index=False)
        self._wsis_val_report.to_csv(self._wsis_val_path, index=False)
        self._wsis_test_report.to_csv(self._wsis_test_path, index=False)
