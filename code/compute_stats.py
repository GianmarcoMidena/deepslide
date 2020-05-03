from pathlib import Path
from typing import (List, Tuple)

import torch
from PIL import Image
from torchvision.transforms import ToTensor

Image.MAX_IMAGE_PIXELS = None


def online_mean_and_std(paths: List[Path]) -> Tuple[List[float], List[float]]:
    """
    Compute the mean and standard deviation of the images found in paths.

    Args:
        paths: List of paths of image files.

    Returns:
        A tuple containing the mean and standard deviation for the images over the channel, height, and width axes.

    This implementation is based on the discussion from: 
        https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9
    """

    class MyDataset(torch.utils.data.Dataset):
        """
        Creates a dataset by reading images.

        Attributes:
            data: List of the string image filenames.
        """

        def __init__(self, paths: List[Path]) -> None:
            """
            Create the MyDataset object.

            Args:
                paths: List of paths of image files.
            """
            self._paths = paths

        def __getitem__(self, index: int) -> torch.Tensor:
            """
            Finds the specified image and outputs in correct format.

            Args:
                index: Index of the desired image.

            Returns:
                A PyTorch Tensor in the correct color space.
            """
            return ToTensor()(Image.open(self._paths[index]).convert("RGB"))

        def __len__(self) -> int:
            return len(self._paths)

    def _online_mean_and_sd(loader: torch.utils.data.DataLoader
                           ) -> Tuple[List[float], List[float]]:
        """
        Computes the mean and standard deviation online.
            Var[x] = E[X^2] - (E[X])^2

        Args:
            loader: The PyTorch DataLoader containing the images to iterate over.

        Returns:
            A tuple containing the mean and standard deviation for the images over the channel, height, and width axes.
        """
        last_tot_pixels = 0
        fst_moment = torch.zeros(3)
        snd_moment = torch.zeros(3)

        for data in loader:
            b, __, h, w = data.shape
            current_n_pixels = b * h * w
            tot_pixels = last_tot_pixels + current_n_pixels
            fst_moment = ((last_tot_pixels * fst_moment +
                           torch.sum(data, dim=[0, 2, 3]))
                          / tot_pixels)
            snd_moment = ((last_tot_pixels * snd_moment +
                           torch.sum(data ** 2, dim=[0, 2, 3]))
                          / tot_pixels)
            last_tot_pixels = tot_pixels
        mean = fst_moment.numpy().tolist()
        std = torch.sqrt(snd_moment - fst_moment ** 2).numpy().tolist()
        return mean, std

    return _online_mean_and_sd(
        loader=torch.utils.data.DataLoader(dataset=MyDataset(
            paths=paths),
            batch_size=1,
            num_workers=1,
            shuffle=False))
