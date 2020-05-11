import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from imageio import imread, imsave
import cv2 as cv


class Viewer:
    _DEFAULT_COLORS = ("red", "white", "blue", "green", "purple",
                       "orange", "black", "pink", "yellow")

    def __init__(self, patch_size: int, classes: List[str] = None, num_classes: int = None, colors_path: Path = None):
        """
        Args:
            patch_size: Size of the patches extracted from the WSI.
            classes: Names of the classes in the dataset.
            num_classes: Number of classes in the dataset.
            colors_path: Location of a JSON file that maps each class with a color.
        """
        self._classes = classes
        self._num_classes = num_classes
        self._colors_path = colors_path
        self._patch_size = patch_size

    def visualize(self, slides_info: pd.DataFrame, partition_name: str,
                  preds_folder: Path, vis_folder: Path) -> None:
        """
        Args:
            preds_folder: Path containing the predicted classes.
            vis_folder: Path to output the WSI with overlaid classes to.
        """
        logging.info(f"Visualizing {partition_name} set...")

        class_colors = self._load_class_colors()

        n_slides = slides_info.shape[0]
        # Find list of WSI.
        logging.info(f"{n_slides} {partition_name} whole slides found")

        # Go over all of the WSI.
        for _, slide_info in slides_info.iterrows():
            # Read in the image.
            whole_slide = imread(uri=slide_info['path'])[..., [0, 1, 2]]
            logging.info(f"visualizing {slide_info['id']} "
                         f"of shape {whole_slide.shape}")

            assert whole_slide.shape[2] == 3, \
                f"Expected 3 channels while your image has {whole_slide.shape[2]} channels."

            # Save it.
            output_file_name = f"{slide_info['id']}_predictions.jpg"
            output_path = vis_folder.joinpath(output_file_name)

            # Confirm the output directory exists.
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Temporary fix. Need not to make folders with no crops.
            try:
                # Add the predictions to the image and save it.
                patch_preds = pd.read_csv(preds_folder.joinpath(f"{slide_info['id']}.csv"))
                slide_with_patch_preds = self._decorate_slide_with_patch_preds(patch_predictions=patch_preds,
                                                                               slide=whole_slide,
                                                                               class_colors=class_colors,
                                                                               patch_size=self._patch_size)
                imsave(output_path, slide_with_patch_preds)
            except FileNotFoundError:
                logging.info(
                    "WARNING: One of the image directories is empty. Skipping this directory"
                )
                continue

        logging.info(f"find the visualizations in {vis_folder}")

    @staticmethod
    def _color_name_to_tuple(color: str) -> Tuple[int, int, int]:
        """
        Convert strings to NumPy colors.

        Args:
            color: The desired color as a string.

        Returns:
            The NumPy ndarray representation of the color.
        """
        colors = {
            "white": (255, 255, 255),
            "pink": (255, 108, 180),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "purple": (225, 225, 0),
            "yellow": (255, 255, 0),
            "orange": (255, 127, 80),
            "blue": (0, 0, 255),
            "green": (0, 255, 0)
        }
        return colors[color]

    @staticmethod
    def _decorate_slide_with_patch_preds(
            patch_predictions: pd.DataFrame,
            slide: np.ndarray, class_colors: pd.Series,
            patch_size: int) -> np.ndarray:
        """
        Overlay the predicted dots (classes) on the WSI.

        Args:
            slide: WSI to add predicted dots to.
            class_colors: Dictionary mapping string color to NumPy ndarray color.
            patch_size: Size of the patches extracted from the WSI.

        Returns:
            The WSI with the predicted class dots overlaid.
        """
        slide = cv.UMat(slide)
        for _, r in patch_predictions.iterrows():
            x = r['x']
            y = r['y']
            prediction = r['prediction']
            # Enlarge the dots so they are visible at larger scale.
            confidence = (r['confidence'] - .5) * 2 * .7 + .3
            half_patch_size = patch_size // 2
            radius = round(.06 * patch_size * confidence)
            center = (y + half_patch_size, x + half_patch_size)
            slide = cv.circle(slide, center, radius, class_colors[prediction], cv.FILLED)
        return slide.get()

    def _load_class_colors(self) -> pd.Series():
        if self._colors_path is not None:
            if self._colors_path.is_file():
                class_colors = pd.read_json(self._colors_path, typ='series') \
                    .apply(self._color_name_to_tuple)
            else:
                raise Exception(f'"{self._colors_path}" file does not exist!')
        else:
            class_colors = pd.Series({
                self._classes[i]: self._color_name_to_tuple(color=self._DEFAULT_COLORS[i])
                for i in range(self._num_classes)
            })
        return class_colors
