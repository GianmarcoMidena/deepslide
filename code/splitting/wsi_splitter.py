import logging
from pathlib import Path
from glob import glob
import pandas as pd
from typing import List
from code.utils import search_image_paths, extract_subfolder_paths


class WSISplitter:
    """
    Splits the data into learning, validation, and evaluation sets.
    """

    _IMAGE_ATTRIBUTES = ['id', 'path', 'label']

    def __init__(self, all_wsi: Path, by_patient: bool, wsis_info: Path, val_wsi_per_class: int,
                 test_wsi_per_class: int, wsis_train: Path,
                 wsis_test: Path, wsis_val: Path):
        """
        Args:
            all_wsi: Location of the WSIs organized in subfolders by class.
            val_wsi_per_class: Number of WSI per class to use in the validation set.
            test_wsi_per_class: Number of WSI per class to use in the test set.
        """
        self._all_wsi = all_wsi
        self._by_patient = by_patient
        self._wsis_info = wsis_info
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

        if self._by_patient and self._wsis_info is not None:
            self._split_by_patient()
        else:
            class_paths = self._search_class_paths()
            for class_path in class_paths:
                self._split_per_class(class_path)

        self._save_report()

    def _split_by_patient(self):
        wsis_info = pd.read_csv(self._wsis_info)
        labels = wsis_info['label'].unique()
        wsis_info = wsis_info.assign(v=1).pivot_table(index='patient_id', columns='label', values='v', aggfunc='sum',
                                                      fill_value=0)
        wsis_info['std'] = wsis_info[labels].std(axis=1)
        wsis_info = wsis_info.sort_values('std', ascending=True).drop('std', axis=1)

        min_images_per_class_cumsum = wsis_info[labels].min(axis=1).cumsum()
        last_test_patient_id = \
            min_images_per_class_cumsum.loc[min_images_per_class_cumsum >= self._test_wsi_per_class].index[0]
        last_test_patient_loc = min_images_per_class_cumsum.index.get_loc(last_test_patient_id)
        test_set_patient_ids = min_images_per_class_cumsum.iloc[:last_test_patient_loc + 1].index.values

        min_images_per_class_cumsum = min_images_per_class_cumsum.iloc[last_test_patient_loc + 1:]\
            .subtract(min_images_per_class_cumsum.iloc[last_test_patient_loc])
        last_val_patient_id = \
            min_images_per_class_cumsum.loc[min_images_per_class_cumsum >= self._val_wsi_per_class].index[0]
        last_val_patient_loc = min_images_per_class_cumsum.index.get_loc(last_val_patient_id)
        val_set_patient_ids = min_images_per_class_cumsum.iloc[:last_val_patient_loc + 1].index.values

        train_set_patient_ids = min_images_per_class_cumsum.iloc[last_val_patient_loc + 1:].index.values

        self._search_image_paths_by_patient_ids(train_set_patient_ids, val_set_patient_ids, test_set_patient_ids)

    def _search_image_paths_by_patient_ids(self, train_set_patient_ids, val_set_patient_ids, test_set_patient_ids):
        paths = glob(str(self._all_wsi.joinpath("**").joinpath("*.*")), recursive=True)
        images_info = pd.Series(paths).to_frame('path')
        images_info['filename'] = images_info['path'].str.rsplit("/", n=1, expand=True)[1]
        images_info['patient_id'] = images_info['filename'].str.rsplit("-", n=1, expand=True)[0]
        images_info['path'] = images_info['path'].apply(Path)
        train_image_paths = images_info.loc[images_info['patient_id'].isin(train_set_patient_ids), 'path']
        val_image_paths = images_info.loc[images_info['patient_id'].isin(val_set_patient_ids), 'path']
        test_image_paths = images_info.loc[images_info['patient_id'].isin(test_set_patient_ids), 'path']
        self._report_splits(train_image_paths, val_image_paths, test_image_paths)

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
