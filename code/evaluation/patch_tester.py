import logging
import operator
import time
from pathlib import Path
from typing import List
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, transforms
from ..compute_stats import online_mean_and_std

from code.utils import extract_subfolder_paths, search_folder_image_paths
from code.models import resnet


class PatchTester:
    def __init__(self, checkpoints_folder: Path, auto_select: bool,
                 eval_model: Path, device: torch.device,
                 classes: List[str], num_classes: int,
                 num_layers: int, pretrain: bool,
                 batch_size: int, num_workers: int):
        self._auto_select = auto_select
        self._batch_size = batch_size
        self._checkpoints_folder = checkpoints_folder
        self._classes = pd.Series(classes)
        self._device = device
        self._eval_model = eval_model
        self._num_classes = num_classes
        self._num_layers = num_layers
        self._num_workers = num_workers
        self._pretrain = pretrain
        self._model = None
        """
        Args:
            patches_eval_folder: Folder containing patches to evaluate on.
            output_folder: Folder to save the model results to.
            checkpoints_folder: Directory to save model checkpoints to.
            auto_select: Automatically select the model with the highest validation accuracy,
            eval_model: Path to the model with the highest validation accuracy.
            device: Device to use for running model.
            classes: Names of the classes in the dataset.
            num_classes: Number of classes in the dataset.
            path_mean: Means of the WSIs for each dimension.
            path_std: Standard deviations of the WSIs for each dimension.
            num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
            pretrain: Use pretrained ResNet weights.
            batch_size: Mini-batch size to use for learning.
            num_workers: Number of workers to use for IO.
        """

    def predict(self, patches_eval_folder: Path, wsis_info: pd.DataFrame,
                partition_name: str, output_folder: Path) -> None:
        """
        Args:
            patches_eval_folder: Folder containing patches to evaluate on.
            output_folder: Folder to save the model results to.
        """
        logging.info(f"Predicting on {partition_name} patches...")

        self._load_model()

        start = time.time()

        # Load the data for each folder.
        image_folders = extract_subfolder_paths(folder=patches_eval_folder)

        # Where we want to write out the predictions.
        # Confirm the output directory exists.
        output_folder.mkdir(parents=True, exist_ok=True)

        # For each WSI.
        for image_folder in image_folders:
            try:
                self._predict_for_wsi(image_folder, output_folder, wsis_info)
            except self._EmptyImageDirectoryException as e:
                logging.error(str(e))

        logging.info(f"time for {patches_eval_folder}: {time.time() - start:.2f} seconds")

    def _predict_for_wsi(self, patches_dir_path, output_folder, wsis_info: pd.DataFrame):
        logging.info(patches_dir_path)

        data = self._load_data(patches_dir_path, wsis_info)

        num_test_image_windows = len(data) * self._batch_size

        # Load the image names so we know the coordinates of the patches we are predicting.
        patches_dir_path = patches_dir_path.joinpath(patches_dir_path.name)
        window_names = pd.Series(search_folder_image_paths(folder=patches_dir_path))

        logging.info(f"testing on {num_test_image_windows} crops from {patches_dir_path}")

        report = pd.DataFrame(columns=["x", "y", "prediction", "confidence"])

        for batch_index, (test_inputs, test_labels) in enumerate(data):
            report = self._predict_for_batch_of_patches(batch_index, test_inputs, window_names, report)

        report.to_csv(output_folder.joinpath(f"{patches_dir_path.name}.csv"), index=False, float_format='%.5f')

    def _predict_for_batch_of_patches(self, batch_index: int, test_inputs, window_names: pd.Series,
                                      report: pd.DataFrame):
        batch_window_names = window_names.iloc[batch_index * self._batch_size:
                                               batch_index * self._batch_size + self._batch_size]
        batch_window_names = batch_window_names.astype(str).str.rsplit(".", n=1, expand=True).iloc[:, 0]
        x, y = batch_window_names.str.split("_", expand=True).iloc[:, -2:].values.T.tolist()

        confidences, test_preds = \
            torch.max(nn.Softmax(dim=1)(self._model(test_inputs.to(device=self._device))), dim=1)

        batch_report = pd.DataFrame({"x": x, "y": y,
                                     "prediction": self._to_class_name(id=test_preds.tolist()),
                                     "confidence": confidences.tolist()})

        return report.append(batch_report,
                             sort=False, ignore_index=True)

    def _load_model(self):
        model_path = self._get_best_model(
            checkpoints_folder=self._checkpoints_folder) if self._auto_select else self._eval_model
        model = resnet(num_classes=self._num_classes,
                       num_layers=self._num_layers,
                       pretrain=self._pretrain)
        ckpt = torch.load(f=model_path)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        model = model.to(device=self._device)
        model.train(mode=False)
        logging.info(f"model loaded from {model_path}")
        self._model = model

    def _get_best_model(self, checkpoints_folder: Path) -> str:
        """
        Finds the model with the best validation accuracy.

        Args:
            checkpoints_folder: Folder containing the models to test.

        Returns:
            The location of the model with the best validation accuracy.
        """
        return max({
                       model: self._parse_val_acc(model_path=model)
                       for model in search_folder_image_paths(folder=checkpoints_folder)
                   }.items(),
                   key=operator.itemgetter(1))[0]

    def _to_class_name(self, id):
        return self._classes[id]

    def _load_data(self, patches_dir_path, wsis_info: pd.DataFrame):
        mean, std = online_mean_and_std(paths=wsis_info['path'].apply(Path).tolist())
        # Temporary fix. Need not to make folders with no crops.
        try:
            dataloader = torch.utils.data.DataLoader(
                dataset=datasets.ImageFolder(
                    root=str(patches_dir_path),
                    transform=transforms.Compose(transforms=[
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                    ])),
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers)
        except RuntimeError:
            raise self._EmptyImageDirectoryException
        return dataloader

    @staticmethod
    def _parse_val_acc(model_path: Path) -> float:
        """
        Parse the validation accuracy from the filename.

        Args:
            model_path: The model path to parse for the validation accuracy.

        Returns:
            The parsed validation accuracy.
        """
        return float(
            f"{('.'.join(model_path.name.split('.')[:-1])).split('_')[-1][2:]}")

    class _EmptyImageDirectoryException(Exception):
        _MESSAGE = "WARNING: One of the image directories is empty. Skipping this directory."

        def __init__(self):
            super().__init__(self._MESSAGE)
