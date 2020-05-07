import logging
from pathlib import Path

from img2ds.building import ImageDatasetBuilder

from code.splitting.csv_writer import CSVWriter


class WSISplitter:
    """
    Splits the data into learning, validation, and evaluation sets.
    """

    def __init__(self, wsi_root: Path, group: str, wsi_metadata: Path, n_splits: int, output_dir: Path,
                 path_column: str, seed: int = None):
        """
        Args:
            wsi_root: Location of the WSIs organized in subfolders by class.
        """
        self._wsi_root = wsi_root
        self._wsi_metadata = wsi_metadata
        self._group = group
        self._path_column = path_column
        self._n_splits = n_splits
        self._output_dir = output_dir
        self._seed = seed
        self._wsis_reports = None

    def split(self) -> None:
        logging.info(f"Splitting the slides into {self._n_splits} subsets...")

        partitions = ImageDatasetBuilder(data_root=self._wsi_root, n_splits=self._n_splits,
                                         with_shuffle=True, with_stratify=True, group=self._group,
                                         metadata=self._wsi_metadata, path_column=self._path_column,
                                         seed=self._seed).build()

        CSVWriter(n_splits=self._n_splits, output_dir=self._output_dir, output_file_name="wsi").write(partitions)
