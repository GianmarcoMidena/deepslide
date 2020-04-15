import logging
import math
from multiprocessing.pool import ThreadPool
from pathlib import Path

import PIL
from PIL import Image
from openslide import OpenSlide, OpenSlideError

from code.utils import search_image_paths, extract_subfolder_paths


class SlidesDownscaler:
    _RGB_EXTENSIONS = ['jpeg', 'jpg']

    def __init__(self, original_wsi_root: Path, new_wsi_root: Path,
                 original_wsi_ext: str, new_wsi_ext: str,
                 downscale_factor: int, num_workers: str):
        self._original_wsis_root = original_wsi_root
        self._new_wsis_root = new_wsi_root
        self._original_wsi_ext = original_wsi_ext
        self._new_wsi_ext = new_wsi_ext
        self._downscale_factor = downscale_factor
        self._num_workers = num_workers

    def downscale(self):
        slide_paths = self._search_slide_paths_for_downscaling()
        tot_slides_to_downscale = len(slide_paths)
        print(f"{tot_slides_to_downscale} slides to downscale")
        pool = ThreadPool(processes=self._num_workers)
        n_downscaled_slides = 0
        n_discarded_slides = 0
        for downscaled in pool.imap_unordered(self._downscale_slide, slide_paths):
            if downscaled:
                n_downscaled_slides += 1
            else:
                n_discarded_slides += 1
            logging.info(f"{n_downscaled_slides} downscaled slides, "
                         f"{n_discarded_slides} discarded slides "
                         f"out of {tot_slides_to_downscale} total slides")

    def _search_slide_paths_for_downscaling(self):
        class_paths = extract_subfolder_paths(self._original_wsis_root)
        slide_paths = []
        for class_path in class_paths:
            slide_paths += search_image_paths(class_path)
        return [p for p in slide_paths if not self._is_downscaled(p)]

    def _downscale_slide(self, slide_path: Path) -> bool:
        try:
            slide = OpenSlide(str(slide_path))
            downscaled_slide = self.__downscale_slide(slide)
            output_path = self._calc_new_wsi_path(slide_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            downscaled_slide.save(output_path)
            return True
        except Exception as e:  # OpenSlideError, FileNotFoundError
            logging.error(f"{slide_path.stem}: {e}")
            return False

    def _is_downscaled(self, original_wsi_path):
        downscaled_slide_path = self._calc_new_wsi_path(original_wsi_path)
        return downscaled_slide_path.is_file()

    def _calc_new_wsi_path(self, original_wsi_path):
        return self._new_wsis_root.joinpath(original_wsi_path.parent.name) \
                                  .joinpath(f"{original_wsi_path.stem}.{self._new_wsi_ext}")

    def __downscale_slide(self, slide: OpenSlide):
        level = slide.get_best_level_for_downsample(self._downscale_factor)
        whole_slide_image = slide.read_region(location=(0, 0), level=level, size=slide.level_dimensions[level])
        if self._new_wsi_ext.lower() in self._RGB_EXTENSIONS:
            whole_slide_image = whole_slide_image.convert("RGB")
        new_dimensions = self._calc_downscaled_sizes(slide)
        return whole_slide_image.resize(new_dimensions, PIL.Image.ANTIALIAS)

    def _calc_downscaled_sizes(self, slide: OpenSlide):
        width, height = slide.dimensions
        downscale_weight = math.floor(width / self._downscale_factor)
        downscale_height = math.floor(height / self._downscale_factor)
        return downscale_weight, downscale_height
