import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from imageio import imread, imsave


class Viewer:
    def __init__(self, classes: List[str], num_classes: int,
                 colors: Tuple[str], patch_size: int):
        """
        Args:
            classes: Names of the classes in the dataset.
            num_classes: Number of classes in the dataset.
            colors: Colors to use for visualization.
            patch_size: Size of the patches extracted from the WSI.
        """
        self._classes = classes
        self._num_classes = num_classes
        self._colors = colors
        self._patch_size = patch_size

    def visualize(self, wsis_info: pd.DataFrame, partition_name: str,
                  preds_folder: Path, vis_folder: Path) -> None:
        """
        Args:
            preds_folder: Path containing the predicted classes.
            vis_folder: Path to output the WSI with overlaid classes to.
        """
        logging.info(f"Visualizing {partition_name} set...")

        n_slides = wsis_info.shape[0]
        # Find list of WSI.
        logging.info(f"{n_slides} {partition_name} whole slides found")
        prediction_to_color = {
            self._classes[i]: self._color_to_np_color(color=self._colors[i])
            for i in range(self._num_classes)
        }
        # Go over all of the WSI.
        for _, wsi_info in wsis_info.iterrows():
            # Read in the image.
            whole_slide_numpy = imread(uri=wsi_info['path'])[..., [0, 1, 2]]
            logging.info(f"visualizing {wsi_info['id']} "
                         f"of shape {whole_slide_numpy.shape}")

            assert whole_slide_numpy.shape[2] == 3, \
                f"Expected 3 channels while your image has {whole_slide_numpy.shape[2]} channels."

            # Save it.
            output_file_name = f"{wsi_info['id']}_predictions.jpg"
            output_path = vis_folder.joinpath(output_file_name)

            # Confirm the output directory exists.
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Temporary fix. Need not to make folders with no crops.
            try:
                # Add the predictions to the image and save it.
                imsave(uri=output_path,
                       im=self._add_predictions_to_image(
                           xy_to_pred_class=self._get_xy_to_pred_class(
                               window_prediction_folder=preds_folder,
                               img_name=wsi_info['id']),
                           image=whole_slide_numpy,
                           prediction_to_color=prediction_to_color,
                           patch_size=self._patch_size))
            except FileNotFoundError:
                logging.info(
                    "WARNING: One of the image directories is empty. Skipping this directory"
                )
                continue

        logging.info(f"find the visualizations in {vis_folder}")

    @staticmethod
    def _color_to_np_color(color: str) -> np.ndarray:
        """
        Convert strings to NumPy colors.

        Args:
            color: The desired color as a string.

        Returns:
            The NumPy ndarray representation of the color.
        """
        colors = {
            "white": np.array([255, 255, 255]),
            "pink": np.array([255, 108, 180]),
            "black": np.array([0, 0, 0]),
            "red": np.array([255, 0, 0]),
            "purple": np.array([225, 225, 0]),
            "yellow": np.array([255, 255, 0]),
            "orange": np.array([255, 127, 80]),
            "blue": np.array([0, 0, 255]),
            "green": np.array([0, 255, 0])
        }
        return colors[color]

    @staticmethod
    def _add_predictions_to_image(
            xy_to_pred_class: Dict[Tuple[str, str], Tuple[str, float]],
            image: np.ndarray, prediction_to_color: Dict[str, np.ndarray],
            patch_size: int) -> np.ndarray:
        """
        Overlay the predicted dots (classes) on the WSI.

        Args:
            xy_to_pred_class: Dictionary mapping coordinates to predicted class along with the confidence.
            image: WSI to add predicted dots to.
            prediction_to_color: Dictionary mapping string color to NumPy ndarray color.
            patch_size: Size of the patches extracted from the WSI.

        Returns:
            The WSI with the predicted class dots overlaid.
        """
        for x, y in xy_to_pred_class.keys():
            prediction, __ = xy_to_pred_class[x, y]
            x = int(x)
            y = int(y)

            # Enlarge the dots so they are visible at larger scale.
            start = round((0.9 * patch_size) / 2)
            end = round((1.1 * patch_size) / 2)
            image[x + start:x + end, y + start:y +
                                               end, :] = prediction_to_color[prediction]

        return image

    @staticmethod
    def _get_xy_to_pred_class(window_prediction_folder: Path, img_name: str) \
            -> Dict[Tuple[str, str], Tuple[str, float]]:
        """
        Find the dictionary of predictions.

        Args:
            window_prediction_folder: Path to the folder containing a CSV file with the predicted classes.
            img_name: Name of the image to find the predicted classes for.

        Returns:
            A dictionary mapping image coordinates to the predicted class and the confidence of the prediction.
        """
        xy_to_pred_class = {}

        with window_prediction_folder.joinpath(img_name).with_suffix(".csv").open(
                mode="r") as csv_lines_open:
            csv_lines = csv_lines_open.readlines()[1:]

            predictions = [line[:-1].split(",") for line in csv_lines]
            for prediction in predictions:
                x = prediction[0]
                y = prediction[1]
                pred_class = prediction[2]
                confidence = float(prediction[3])
                # Implement thresholding.
                xy_to_pred_class[(x, y)] = (pred_class, confidence)
        return xy_to_pred_class
