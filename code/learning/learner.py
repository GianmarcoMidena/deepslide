import logging
import time
from pathlib import Path
from typing import (Dict, IO, List)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import (transforms)

from code.compute_stats import online_mean_and_std
from code.dataset import Dataset
from code.learning.early_stopper import EarlyStopper
from code.learning.random90rotation import Random90Rotation
from code.models import resnet


class Learner:
    def __init__(self, batch_size: int, num_workers: int,
                 device: torch.device, classes: List[str], learning_rate: float,
                 weight_decay: float, learning_rate_decay: float,
                 resume_checkpoint: bool, resume_checkpoint_path: Path, log_csv: Path,
                 color_jitter_brightness: float, color_jitter_contrast: float,
                 color_jitter_hue: float, color_jitter_saturation: float, num_classes: int,
                 num_layers: int, pretrain: bool, checkpoints_folder: Path,
                 num_epochs: int, early_stopping_patience: int,
                 train_slides_metadata_paths: List[Path], val_slides_metadata_paths: List[Path],
                 train_patch_metadata_paths: List[Path], val_patch_metadata_paths: List[Path],
                 class_idx_path: Path):
        """
        Args:
            batch_size: Mini-batch size to use for learning.
            num_workers: Number of workers to use for IO.
            device: Device to use for running model.
            classes: Names of the classes in the dataset.
            learning_rate: Learning rate to use for gradient descent.
            weight_decay: Weight decay (L2 penalty) to use in optimizer.
            learning_rate_decay: Learning rate decay amount per epoch.
            resume_checkpoint: Resume model from checkpoint file.
            resume_checkpoint_path: Path to the checkpoint file for resuming learning.
            log_csv: Name of the CSV file containing the logs.
            color_jitter_brightness: Random brightness jitter to use in data augmentation for ColorJitter() transform.
            color_jitter_contrast: Random contrast jitter to use in data augmentation for ColorJitter() transform.
            color_jitter_hue: Random hue jitter to use in data augmentation for ColorJitter() transform.
            color_jitter_saturation: Random saturation jitter to use in data augmentation for ColorJitter() transform.
            num_classes: Number of classes in the dataset.
            num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
            pretrain: Use pretrained ResNet weights.
            checkpoints_folder: Directory to save model checkpoints to.
            num_epochs: Number of epochs for learning.
        """
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._device = device
        self._classes = classes
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._learning_rate_decay = learning_rate_decay
        self._resume_checkpoint = resume_checkpoint
        self._resume_checkpoint_path = resume_checkpoint_path
        self._log_csv = log_csv
        self._color_jitter_brightness = color_jitter_brightness
        self._color_jitter_contrast = color_jitter_contrast
        self._color_jitter_hue = color_jitter_hue
        self._color_jitter_saturation = color_jitter_saturation
        self._num_classes = num_classes
        self._num_layers = num_layers
        self._pretrain = pretrain
        self._checkpoints_folder = checkpoints_folder
        self._num_epochs = num_epochs
        self._early_stopping_patience = early_stopping_patience
        self._train_slides_metadata_paths = train_slides_metadata_paths
        self._val_slides_metadata_paths = val_slides_metadata_paths
        self._train_patch_metadata_paths = train_patch_metadata_paths
        self._val_patch_metadata_paths = val_patch_metadata_paths
        self._class_idx_path = class_idx_path

    def train(self) -> None:
        # Loading in the data.
        data_transforms = self._get_data_transforms()

        image_datasets = {
            'train': Dataset(metadata_paths=self._train_patch_metadata_paths, transform=data_transforms['train'],
                             class_idx_path=self._class_idx_path),
            'val': Dataset(metadata_paths=self._val_patch_metadata_paths, transform=data_transforms['val'],
                           class_idx_path=self._class_idx_path)
        }

        dataloaders = {
            x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                           batch_size=self._batch_size,
                                           shuffle=(x is "train"),
                                           num_workers=self._num_workers)
            for x in ("train", "val")
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ("train", "val")}

        logging.info(f"{self._num_classes} classes: {self._classes}\n"
                     f"num train images {len(dataloaders['train']) * self._batch_size}\n"
                     f"num val images {len(dataloaders['val']) * self._batch_size}\n"
                     f"CUDA is_available: {torch.cuda.is_available()}")

        model = resnet(num_classes=self._num_classes,
                       num_layers=self._num_layers,
                       pretrain=self._pretrain)
        model = model.to(device=self._device)
        optimizer = optim.Adam(params=model.parameters(),
                               lr=self._learning_rate,
                               weight_decay=self._weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer,
                                               gamma=self._learning_rate_decay)

        # Initialize the model.
        if self._resume_checkpoint:
            ckpt = torch.load(f=self._resume_checkpoint_path)
            model.load_state_dict(state_dict=ckpt["model_state_dict"])
            optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(state_dict=ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            logging.info(f"model loaded from {self._resume_checkpoint_path}")
        else:
            start_epoch = 0

        # Print the model hyperparameters.
        self._print_params()

        # Logging the model after every epoch.
        # Confirm the output directory exists.
        self._log_csv.parent.mkdir(parents=True, exist_ok=True)

        with self._log_csv.open(mode="w") as writer:
            writer.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
            # Train the model.
            self._train_helper(model=model,
                               dataloaders=dataloaders,
                               dataset_sizes=dataset_sizes,
                               loss_fn=self._search_loss_fn(),
                               optimizer=optimizer,
                               scheduler=scheduler,
                               start_epoch=start_epoch,
                               writer=writer)

    @staticmethod
    def _calculate_confusion_matrix(all_labels: np.ndarray,
                                    all_predicts: np.ndarray, classes: List[str],
                                    num_classes: int) -> None:
        """
        Prints the confusion matrix from the given data.

        Args:
            all_labels: The ground truth labels.
            all_predicts: The predicted labels.
            classes: Names of the classes in the dataset.
            num_classes: Number of classes in the dataset.
        """
        remap_classes = {x: classes[x] for x in range(num_classes)}

        # Set print options.
        # Sources:
        #   1. https://stackoverflow.com/questions/42735541/customized-float-formatting-in-a-pandas-dataframe
        #   2. https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
        #   3. https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
        pd.options.display.float_format = "{:.2f}".format
        pd.options.display.width = 0

        actual = pd.Series(pd.Categorical(
            pd.Series(all_labels).replace(remap_classes), categories=classes),
            name="Actual")

        predicted = pd.Series(pd.Categorical(
            pd.Series(all_predicts).replace(remap_classes), categories=classes),
            name="Predicted")

        cm = pd.crosstab(index=actual, columns=predicted, normalize="index", dropna=False)

        cm.style.hide_index()
        logging.info(cm)

    def _get_data_transforms(self) -> Dict[str, torchvision.transforms.Compose]:
        """
        Sets up the dataset transforms for learning and validation.

        Returns:
            A dictionary mapping learning and validation strings to data transforms.
        """
        train_image_paths = [Path(slide_path) for mp in self._train_slides_metadata_paths
                             for slide_path in pd.read_csv(mp)['path']]
        val_image_paths = [Path(slide_path) for mp in self._val_slides_metadata_paths
                           for slide_path in pd.read_csv(mp)['path']]
        train_mean, train_std = online_mean_and_std(image_paths=train_image_paths)
        val_mean, val_std = online_mean_and_std(image_paths=val_image_paths)

        return {
            "train":
                transforms.Compose(transforms=[
                    transforms.ColorJitter(brightness=self._color_jitter_brightness,
                                           contrast=self._color_jitter_contrast,
                                           saturation=self._color_jitter_saturation,
                                           hue=self._color_jitter_hue),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    Random90Rotation(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=train_mean, std=train_std)
                ]),
            "val":
                transforms.Compose(transforms=[
                    transforms.ToTensor(),
                    transforms.Normalize(mean=val_mean, std=val_std)
                ])
        }

    def _print_params(self) -> None:
        """
        Print the configuration of the model.
        """
        logging.info(f"num_epochs: {self._num_epochs}\n"
                     f"num_layers: {self._num_layers}\n"
                     f"learning_rate: {self._learning_rate}\n"
                     f"batch_size: {self._batch_size}\n"
                     f"weight_decay: {self._weight_decay}\n"
                     f"learning_rate_decay: {self._learning_rate_decay}\n"
                     f"resume_checkpoint: {self._resume_checkpoint}\n"
                     f"resume_checkpoint_path (only if resume_checkpoint is true): "
                     f"{self._resume_checkpoint_path}\n"
                     f"output in checkpoints_folder: {self._checkpoints_folder}\n"
                     f"pretrain: {self._pretrain}\n"
                     f"log_csv: {self._log_csv}\n\n")

    def _train_helper(self, model: torchvision.models.resnet.ResNet,
                      dataloaders: Dict[str, torch.utils.data.DataLoader],
                      dataset_sizes: Dict[str, int],
                      loss_fn, optimizer: torch.optim,
                      scheduler: torch.optim.lr_scheduler, start_epoch: int,
                      writer: IO) -> None:
        """
        Function for learning ResNet.

        Args:
            model: ResNet model for learning.
            dataloaders: Dataloaders for IO pipeline.
            dataset_sizes: Sizes of the learning and validation dataset.
            loss_fn: Metric used for calculating loss.
            optimizer: Optimizer to use for gradient descent.
            scheduler: Scheduler to use for learning rate decay.
            start_epoch: Starting epoch for learning.
            writer: Writer to write logging information.
        """
        learning_init_time = time.time()

        # Initialize all the tensors to be used in learning and validation.
        # Do this outside the loop since it will be written over entirely at each
        # epoch and doesn't need to be reallocated each time.
        train_all_labels = torch.empty(size=(dataset_sizes["train"],),
                                       dtype=torch.long).cpu()
        train_all_predicts = torch.empty(size=(dataset_sizes["train"],),
                                         dtype=torch.long).cpu()
        val_all_labels = torch.empty(size=(dataset_sizes["val"],),
                                     dtype=torch.long).cpu()
        val_all_predicts = torch.empty(size=(dataset_sizes["val"],),
                                       dtype=torch.long).cpu()
        early_stopper = EarlyStopper(patience=self._early_stopping_patience)

        best_val_acc = 0.

        # Train for specified number of epochs.
        for epoch in range(start_epoch, self._num_epochs):
            epoch_init_time = time.time()

            # Training phase.
            model.train(mode=True)

            train_running_loss = 0.0
            train_running_corrects = 0

            # Train over all learning data.
            for idx, (train_inputs, true_labels) in enumerate(dataloaders["train"]):
                train_inputs = train_inputs.to(device=self._device)
                true_labels = true_labels.to(device=self._device)
                optimizer.zero_grad()

                # Forward and backpropagation.
                with torch.set_grad_enabled(mode=True):
                    train_logits = model(train_inputs).squeeze(dim=1)
                    train_loss = loss_fn(logits=train_logits,
                                         target=true_labels)
                    train_loss.backward()
                    optimizer.step()

                # Update learning diagnostics.
                train_running_loss += train_loss.item() * train_inputs.size(0)
                pred_labels = self._extract_pred_labels(train_logits)
                train_running_corrects += torch.sum(
                    pred_labels == true_labels.data, dtype=torch.double)

                start = idx * self._batch_size
                end = start + self._batch_size

                train_all_labels[start:end] = true_labels.detach().cpu()
                train_all_predicts[start:end] = pred_labels.detach().cpu()

            self._calculate_confusion_matrix(all_labels=train_all_labels.numpy(),
                                             all_predicts=train_all_predicts.numpy(),
                                             classes=self._classes,
                                             num_classes=self._num_classes)

            # Store learning diagnostics.
            train_loss = train_running_loss / dataset_sizes["train"]
            train_acc = train_running_corrects / dataset_sizes["train"]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Validation phase.
            model.train(mode=False)

            val_running_loss = 0.0
            val_running_corrects = 0

            # Feed forward over all the validation data.
            for idx, (val_inputs, val_labels) in enumerate(dataloaders["val"]):
                val_inputs = val_inputs.to(device=self._device)
                val_labels = val_labels.to(device=self._device)

                # Feed forward.
                with torch.set_grad_enabled(mode=False):
                    val_logits = model(val_inputs).squeeze(dim=1)
                    val_loss = loss_fn(logits=val_logits, target=val_labels)

                # Update validation diagnostics.
                val_running_loss += val_loss.item() * val_inputs.size(0)
                pred_labels = self._extract_pred_labels(val_logits)
                val_running_corrects += torch.sum(pred_labels == val_labels.data,
                                                  dtype=torch.double)

                start = idx * self._batch_size
                end = start + self._batch_size

                val_all_labels[start:end] = val_labels.detach().cpu()
                val_all_predicts[start:end] = pred_labels.detach().cpu()

            self._calculate_confusion_matrix(all_labels=val_all_labels.numpy(),
                                             all_predicts=val_all_predicts.numpy(),
                                             classes=self._classes,
                                             num_classes=self._num_classes)

            # Store validation diagnostics.
            val_loss = val_running_loss / dataset_sizes["val"]
            val_acc = val_running_corrects / dataset_sizes["val"]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            scheduler.step()

            current_lr = None
            for group in optimizer.param_groups:
                current_lr = group["lr"]

            # Remaining things related to learning.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_ckpt_path = self._checkpoints_folder.joinpath(
                    f"resnet{self._num_layers}_e{epoch}_va{val_acc:.5f}.pt")

                # Confirm the output directory exists.
                best_model_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the model as a state dictionary.
                torch.save(obj={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1
                },
                    f=str(best_model_ckpt_path))

                self._clean_ckpt_folder(best_model_ckpt_path)

            writer.write(f"{epoch},{train_loss:.4f},"
                         f"{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

            # Print the diagnostics for each epoch.
            logging.info(f"Epoch {epoch} "
                         f"with lr {current_lr:.15f}: "
                         f"{self._format_time_period(epoch_init_time, time.time())} "
                         f"t_loss: {train_loss:.4f} "
                         f"t_acc: {train_acc:.4f} "
                         f"v_loss: {val_loss:.4f} "
                         f"v_acc: {val_acc:.4f}\n")

            early_stopper.update(val_loss)
            if early_stopper.is_stopping():
                logging.info("Early stopping")
                break

        # Print learning information at the end.
        logging.info(f"\nlearning complete in "
                     f"{self._format_time_period(learning_init_time, time.time())}")

    def _extract_pred_labels(self, logits):
        if self._num_classes > 2:
            __, pred_labels = torch.max(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)
            pred_labels = (probs > 0.5).float()
        return pred_labels

    @staticmethod
    def _format_time_period(start, end) -> str:
        period = end - start
        if period > 60:
            return f"{period / 60:.2f} minutes"
        return f"{period:.0f}s"

    def _search_loss_fn(self):
        if self._num_classes > 2:
            loss_fn = lambda logits, target: nn.CrossEntropyLoss()(input=logits, target=target)
        else:
            loss_fn = self._binary_cross_entropy
        return loss_fn

    def _binary_cross_entropy(self, logits, target):
        probs = nn.Sigmoid()(logits)
        target = target.float()
        return nn.BCELoss()(input=probs, target=target)

    def _clean_ckpt_folder(self, best_model_ckpt_path):
        all_ckpt_paths = best_model_ckpt_path.parent.glob(f"resnet{self._num_layers}_e*_va*.pt")
        for ckpt_path in all_ckpt_paths:
            if ckpt_path.name != best_model_ckpt_path.name:
                ckpt_path.unlink()
