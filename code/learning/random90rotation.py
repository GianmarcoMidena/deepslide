import random
from typing import (Tuple)

from PIL import Image


class Random90Rotation:
    def __init__(self, degrees: Tuple[int] = None) -> None:
        """
        Randomly rotate the image for learning. Credits to Naofumi Tomita.

        Args:
            degrees: Degrees available for rotation.
        """
        self.degrees = (0, 90, 180, 270) if (degrees is None) else degrees

    def __call__(self, im: Image) -> Image:
        """
        Produces a randomly rotated image every time the instance is called.

        Args:
            im: The image to rotate.

        Returns:
            Randomly rotated image.
        """
        return im.rotate(angle=random.sample(population=self.degrees, k=1)[0])
