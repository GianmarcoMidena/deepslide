import logging
import operator
import time
from pathlib import Path
from typing import List
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from ..compute_stats import online_mean_and_std

from code.utils import search_folder_file_paths
from code.models import resnet
from ..dataset import Dataset


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

    def predict(self, patches_metadata_paths: List[Path], slides_metadata_paths: List[Path], class_idx_path: Path,
                partition_name: str, output_folder: Path) -> None:
        """
        Args:
            patches_eval_folder: Folder containing patches to evaluate on.
            output_folder: Folder to save the model results to.
        """
        logging.info(f"Predicting on {partition_name} patches...")

        self._load_model()

        start = time.time()

        # Where we want to write out the predictions.
        # Confirm the output directory exists.
        output_folder.mkdir(parents=True, exist_ok=True)

        patches_metadata = pd.DataFrame()
        for p in patches_metadata_paths:
            patches_metadata = patches_metadata.append(pd.read_csv(p), ignore_index=True, sort=False)

        slide_paths = [Path(slide_path) for mp in slides_metadata_paths
                       for slide_path in pd.read_csv(mp)['path']]
        mean, std = online_mean_and_std(image_paths=slide_paths)

        slide_ids = self._extract_slide_ids(patches_metadata)
        for slide_id in slide_ids:
            patches_metadata_i = patches_metadata.loc[patches_metadata['path'].str.contains(slide_id)]
            self._predict_for_wsi(patches_metadata=patches_metadata_i, class_idx_path=class_idx_path,
                                  slide_id=slide_id, mean=mean, std=std, output_folder=output_folder)

        logging.info(f"time for {patches_metadata_paths[0].parent.name}: {time.time() - start:.2f} seconds")

    def _predict_for_wsi(self, patches_metadata: pd.DataFrame, class_idx_path: Path,
                         slide_id: str, mean, std, output_folder: Path):
        eval_data = self._load_data(patches_metadata=patches_metadata, class_idx_path=class_idx_path,
                                    mean=mean, std=std)

        num_test_image_windows = len(eval_data) * self._batch_size

        logging.info(f"testing on {num_test_image_windows} crops for {slide_id} slide")

        report = pd.DataFrame(columns=["x", "y", "prediction", "confidence"])

        patch_names = patches_metadata['path'].str.rsplit("/", n=1, expand=True)[1] \
                                              .str.rsplit(".", n=1, expand=True)[0]
        for batch_index, (eval_patches, _) in enumerate(eval_data):
            report = self._predict_for_batch_of_patches(batch_index, eval_patches, patch_names, report)

        report.to_csv(output_folder.joinpath(f"{slide_id}.csv"), index=False, float_format='%.5f')

    def _predict_for_batch_of_patches(self, batch_index: int, test_inputs, patch_names: pd.Series,
                                      report: pd.DataFrame):
        batch_window_names = patch_names.iloc[batch_index * self._batch_size:
                                              batch_index * self._batch_size + self._batch_size]
        x, y = batch_window_names.str.split("_", expand=True).iloc[:, -2:].values.T.tolist()

        test_preds, confidences = self._extract_pred_labels_and_confidences(test_inputs)

        batch_report = pd.DataFrame({"x": x, "y": y,
                                     "prediction": self._to_class_name(id=test_preds.tolist()),
                                     "confidence": confidences.tolist()})

        return report.append(batch_report,
                             sort=False, ignore_index=True)

    def _extract_pred_labels_and_confidences(self, inputs):
        logits = self._model(inputs.to(device=self._device)).squeeze(dim=1)
        if self._num_classes > 2:
            probs = nn.Softmax(dim=1)(logits)
        else:
            positive_class_probs = nn.Sigmoid()(logits)
            negative_class_probs = 1 - positive_class_probs
            probs = torch.stack([negative_class_probs, positive_class_probs], dim=1)
        pred_confidences, pred_labels = torch.max(probs, dim=1)
        return pred_labels, pred_confidences

    def _load_model(self):
        if self._auto_select:
            model_path = self._get_best_model(checkpoints_folder=self._checkpoints_folder)
        else:
            model_path = self._eval_model
        model = resnet(num_classes=self._num_classes,
                       num_layers=self._num_layers,
                       pretrain=self._pretrain)
        ckpt = torch.load(f=model_path)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        model = model.to(device=self._device)
        model.train(mode=False)
        logging.info(f"model loaded from {model_path}")
        self._model = model

    def _get_best_model(self, checkpoints_folder: Path) -> Path:
        """
        Finds the model with the best validation accuracy.

        Args:
            checkpoints_folder: Folder containing the models to test.

        Returns:
            The location of the model with the best validation accuracy.
        """
        return max({
                       model: self._parse_val_acc(model_path=model)
                       for model in search_folder_file_paths(folder=checkpoints_folder)
                   }.items(),
                   key=operator.itemgetter(1))[0]

    def _to_class_name(self, id):
        return self._classes[id]

    def _load_data(self, patches_metadata: pd.DataFrame, mean, std,
                   class_idx_path: Path):
        # Temporary fix. Need not to make folders with no crops.
        try:
            dataset = Dataset(
                metadata=patches_metadata,
                class_idx_path=class_idx_path,
                transform=transforms.Compose(transforms=[
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ]))
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
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

    def _extract_slide_ids(self, patches_metadata: pd.DataFrame) -> List[str]:
        return patches_metadata['path'].str.rsplit('/', n=1, expand=True)[1]\
                                       .str.rsplit('_', n=2, expand=True)[0]\
                                       .unique().tolist()
