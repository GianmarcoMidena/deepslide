import logging
import math
import traceback
from multiprocessing.pool import ThreadPool
from pathlib import Path
import numpy as np
import pandas as pd

import PIL
from PIL import Image
from openslide import OpenSlide

from code.utils import search_folder_file_paths, extract_subfolder_paths


class SlidesDownscaler:
    _RGB_EXTENSIONS = ['jpeg', 'jpg']

    def __init__(self, original_slides_root: Path, new_slides_root: Path,
                 original_slide_ext: str, new_slide_ext: str,
                 downscale_factor: int, min_dim_size: int, num_workers: int):
        self._original_slides_root = original_slides_root
        self._new_slides_root = new_slides_root
        self._original_slide_ext = original_slide_ext
        self._new_slide_ext = new_slide_ext
        self._downscale_factor = downscale_factor
        self._min_dim_size = min_dim_size
        self._num_workers = num_workers

    def downscale(self):
        slide_paths, n_downscaled_slides = self._search_slide_paths_for_downscaling()
        tot_slides_to_downscale = len(slide_paths)
        logging.info(f"{tot_slides_to_downscale} slides to downscale")
        tot_slides = tot_slides_to_downscale + n_downscaled_slides
        n_discarded_slides = 0
        pool = ThreadPool(processes=self._num_workers)
        for downscaled in pool.imap_unordered(self._downscale_slide, slide_paths):
            if downscaled:
                n_downscaled_slides += 1
            else:
                n_discarded_slides += 1
            logging.info(f"{n_downscaled_slides} downscaled slides, "
                         f"{n_discarded_slides} discarded slides "
                         f"out of {tot_slides} total slides")

    def _search_slide_paths_for_downscaling(self):
        class_paths = extract_subfolder_paths(self._original_slides_root)
        slide_paths = []
        for class_path in class_paths:
            slide_paths += search_folder_file_paths(class_path)
        n_downscaled_slides = 0
        downscaled_slide_names = pd.Series(self._new_slides_root.rglob(f"*.{self._new_slide_ext}")).apply(str)\
                                   .str.rsplit("/", n=1, expand=True)[1]\
                                   .str.rsplit(".", n=1, expand=True)[0]
        filtered_slide_paths = []
        for slide_path in slide_paths:
            if not downscaled_slide_names.str.startswith(slide_path.stem).any():
                filtered_slide_paths.append(slide_path)
            else:
                n_downscaled_slides += 1
        return filtered_slide_paths, n_downscaled_slides

    # def _is_downscaled(self, slide_path):
    #     downscaled_slide_path = self._calc_new_wsi_path(slide_path)
    #     if list(downscaled_slide_path.parent
    #                                  .rglob(f"{downscaled_slide_path.stem}*{downscaled_slide_path.suffix}")):
    #         downscaled = True
    #     else:
    #         downscaled = False
    #     return downscaled, slide_path

    def _downscale_slide(self, slide_path: Path) -> bool:
        try:
            slide = OpenSlide(str(slide_path))
            self.__downscale_slide(slide, slide_path)
            return True
        except:  # OpenSlideError, FileNotFoundError
            logging.info("Error...")
            logging.error(f"{slide_path.stem}: {traceback.format_exc()}")
            return False

    def _calc_new_wsi_path(self, original_wsi_path, part: str = None):
        dir_path = self._new_slides_root.joinpath(original_wsi_path.parent.name)
        if part:
            path = dir_path.joinpath(f"{original_wsi_path.stem}_{part}")
        else:
            path = dir_path.joinpath(f"{original_wsi_path.stem}")
        return path.with_suffix(f".{self._new_slide_ext}")

    def __downscale_slide(self, slide: OpenSlide, slide_path: Path):
        # logging.info("Downscaling a slide...")
        original_width, original_height = slide.dimensions
        # logging.info(f"original dims: {(original_width, original_height)}")

        if self._min_dim_size:
            downscale_factor = self._calc_downscale_factor(slide)
        else:
            downscale_factor = self._downscale_factor

        # logging.info(f"downscale factor: {downscale_factor}")

        level = slide.get_best_level_for_downsample(downscale_factor)

        # logging.info(f"level: {level}")

        level_width, level_height = slide.level_dimensions[level]
        # logging.info(f"level width={level_width}, height={level_height}")

        # logging.info("calc target dims...")
        target_width, target_height = self._calc_downscaled_sizes(slide, downscale_factor)
        # logging.info(f"target dims = {(target_width, target_height)}")

        if level_width * level_height < 1e9:
            # logging.info("Reading region...")
            whole_slide_image = slide.read_region(location=(0, 0), level=level, size=(level_width, level_height))

            # logging.info("Converting to RGB...")
            if self._new_slide_ext.lower() in self._RGB_EXTENSIONS:
                whole_slide_image = whole_slide_image.convert("RGB")

            # logging.info("Resizing...")
            whole_slide_image = whole_slide_image.resize((target_width, target_height), PIL.Image.ANTIALIAS)

            # logging.info("Calculating path...")
            output_path = self._calc_new_wsi_path(slide_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # logging.info("Saving...")
            whole_slide_image.save(output_path)
        else:
            n_vertical_splits = self._find_best_n_splits(level_height)
            n_horizontal_splits = self._find_best_n_splits(level_width)

            logging.info(f"\"{slide_path.stem}\" slide: {n_vertical_splits} vertical splits, {n_horizontal_splits} horizontal splits")

            original_tile_height = math.ceil(original_height / n_vertical_splits)
            original_tile_width = math.ceil(original_width / n_horizontal_splits)

            level_tile_height = math.ceil(level_height / n_vertical_splits)
            level_tile_width = math.ceil(level_width / n_horizontal_splits)

            # logging.info(f"level_tile_width:{level_width / n_horizontal_splits} "
            #              f"approx:{math.ceil(level_width / n_horizontal_splits)}")

            # target_tile_height = math.ceil(target_height / n_vertical_splits)
            # target_tile_width = math.ceil(target_width / n_horizontal_splits)

            for i in range(n_vertical_splits):
                for j in range(n_horizontal_splits):
                    y_i = original_tile_height * i
                    x_j = original_tile_width * j
                    # logging.info(f"x_j={x_j}, y_i={y_i}")

                    level_height_i = min(level_tile_height, level_height - level_tile_height * i)
                    # target_height_i = min(target_tile_height, target_tile_height * (n_vertical_splits - 1))

                    level_width_j = min(level_tile_width, level_width - level_tile_width * j)
                    # target_width_j = min(target_tile_width, target_tile_width * (n_horizontal_splits - 1))

                    # logging.info(f"level height_i={level_height_i}, width_j={level_width_j}")
                    # logging.info(f"target height_i={target_height_i}, width_j={target_width_j}")

                    # logging.info(f"Reading region {i}_{j}...")
                    whole_slide_image_tile = slide.read_region(location=(x_j, y_i), level=level,
                                                               size=(level_width_j, level_height_i))

                    # logging.info(f"Converting to RGB {i}_{j}...")
                    whole_slide_image_tile = whole_slide_image_tile.convert("RGB")

                    # logging.info("Resizing...")
                    # whole_slide_image_tile = whole_slide_image_tile.resize((target_width_j, target_height_i),
                    #                                                        PIL.Image.ANTIALIAS)

                    # logging.info("Calculating path...")
                    output_path = self._calc_new_wsi_path(slide_path, part=f"{i}_{j}")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    # logging.info("Saving...")
                    whole_slide_image_tile.save(output_path)

    def _calc_downscale_factor(self, slide: OpenSlide):
        width, height = slide.dimensions
        smallest_dim = min(width, height)
        scale_factor = math.floor(smallest_dim / self._min_dim_size)
        return scale_factor

    @staticmethod
    def _calc_downscaled_sizes(slide: OpenSlide, downscale_factor: int):
        width, height = slide.dimensions
        downscale_weight = math.floor(width / downscale_factor)
        downscale_height = math.floor(height / downscale_factor)
        return downscale_weight, downscale_height

    def _find_best_n_splits(self, full_length):
        length_divisors = np.asarray(self._find_divisors(full_length))
        final_lengths = full_length / length_divisors
        target_length = 5e3
        distances_to_target_length = np.abs(final_lengths - target_length)
        best_length_divisor = length_divisors[distances_to_target_length.argmin()]
        return best_length_divisor

    @staticmethod
    def _find_divisors(v: int):
        divisors = [1, v]
        for d in range(2, math.floor(math.sqrt(v)) + 1):
            if v % d == 0:
                a, b = d, int(v / d)
                if a != b:
                    divisors += [a, b]
                else:
                    divisors.append(a)
        divisors.sort()
        return divisors
