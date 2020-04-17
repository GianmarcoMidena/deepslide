import logging
import math
from pathlib import Path
from typing import Tuple, Set, List

from .step_size import StepSize


class StepSizeFinder:
    _MIN_STEP_SIZE = 1

    def __init__(self, target_n_patches: int, patch_size: int, patch_extractor):
        """
        Args:
            target_n_patches: Desired number of patches.
        """
        self._target_n_patches = target_n_patches
        self._patch_size = patch_size
        self._patch_extractor = patch_extractor

    def search(self, image_paths: List[Path]) -> int:
        starting_step_size = self._init_step_size(image_paths)
        starting_step_size_range = (1, self._patch_size)
        step_size_range = self.update_step_size_range(best_step_size=starting_step_size,
                                                      new_step_size=starting_step_size,
                                                      step_size_range=starting_step_size_range)
        step_size_range = next(iter(step_size_range))
        step_size = self._search(image_paths, starting_step_size, step_size_range)
        if step_size is not None:
            step_size = step_size.value
        return step_size

    def _init_step_size(self, image_paths):
        starting_step_size = self._patch_size // 2
        starting_n_patches = self._patch_extractor.extract_all(image_paths, step_size=starting_step_size,
                                                               partition_name='training')
        starting_step_size = StepSize(starting_step_size, starting_n_patches)
        return starting_step_size

    def _search(self, image_paths: List[Path], starting_step_size: StepSize,
                step_size_range: Tuple[int, int]) -> StepSize:
        if self._is_empty_range(step_size_range):
            return starting_step_size

        step_size_adjustment = self.calc_step_size_adjustment(starting_step_size=starting_step_size,
                                                              step_size_range=step_size_range)

        starting_n_patches = starting_step_size.n_patches
        if step_size_adjustment == 0:
            logging.debug(f'step size={starting_step_size.value}, '
                          f'n_patches={starting_n_patches} '
                          f'=> no step size change')
            return starting_step_size

        new_step_size = starting_step_size + step_size_adjustment
        new_n_patches = self._patch_extractor.extract_all(image_paths, step_size=new_step_size,
                                                          partition_name='training')
        new_n_patches_mean_distance = abs(new_n_patches - self._target_n_patches)
        logging.debug(f'step size={starting_step_size.value}->{new_step_size}, '
                      f'n_patches={starting_n_patches}->{new_n_patches}')
        new_step_size = StepSize(new_step_size, n_patches=new_n_patches)

        starting_n_patches_mean_distance = abs(starting_n_patches - self._target_n_patches)
        if (starting_n_patches == 0) and (new_n_patches == 0):
            return None
        elif new_n_patches_mean_distance <= starting_n_patches_mean_distance:
            if new_n_patches_mean_distance == 0:
                return new_step_size
            best_step_size = new_step_size
            if new_n_patches_mean_distance < starting_n_patches_mean_distance:
                logging.debug('new best')
        else:  # new_n_patches_mean_distance > starting_n_patches_mean_distance
            best_step_size = starting_step_size

        new_step_size_ranges = self.update_step_size_range(best_step_size=best_step_size, new_step_size=new_step_size,
                                                           step_size_range=step_size_range)

        best_distance_to_target = math.inf
        for new_step_size_range in new_step_size_ranges:
            new_step_size = self._search(image_paths, best_step_size, new_step_size_range)
            new_distance_to_target = abs(new_step_size.n_patches - self._target_n_patches)
            if new_distance_to_target == 0:
                return new_step_size
            if new_distance_to_target < best_distance_to_target:
                best_step_size = new_step_size
                best_distance_to_target = new_distance_to_target
        return best_step_size

    def calc_step_size_adjustment(self, starting_step_size: StepSize, step_size_range: Tuple[int, int]) -> int:
        min_step_size, max_step_size = step_size_range
        starting_step_size, starting_n_patches = starting_step_size.value, starting_step_size.n_patches
        if min_step_size == max_step_size:
            step_size_adjustment = min_step_size - starting_step_size
        else:
            step_size_adjustment = 0
            if starting_n_patches > self._target_n_patches:
                if starting_step_size < max_step_size:
                    starting_step_size_adjustment = max(1, min_step_size - starting_step_size)
                    step_size_adjustment = (max_step_size - (starting_step_size + starting_step_size_adjustment)) // 2 \
                                           + starting_step_size_adjustment
            elif starting_n_patches < self._target_n_patches:
                if (starting_step_size > min_step_size) and (starting_step_size > self._MIN_STEP_SIZE):
                    starting_step_size_adjustment = min(-1, max_step_size - starting_step_size)
                    step_size_adjustment = (min_step_size - (starting_step_size + starting_step_size_adjustment)) // 2 \
                                           + starting_step_size_adjustment
        return step_size_adjustment

    def update_step_size_range(self, best_step_size: StepSize, new_step_size: StepSize,
                               step_size_range: Tuple[int, int]) -> Set[Tuple[int, int]]:
        min_step_size, max_step_size = step_size_range
        step_size_ranges = set()
        if new_step_size.n_patches > self._target_n_patches:
            if new_step_size <= best_step_size:
                step_size_ranges.add((max(new_step_size + 1, min_step_size), max_step_size))
            else:  # (new_step_size > best_step_size)
                step_size_ranges.add((max(best_step_size + 1, min_step_size), min(new_step_size - 1, max_step_size)))
                step_size_ranges.add((max(new_step_size + 1, min_step_size), max_step_size))
        elif new_step_size.n_patches < self._target_n_patches:
            if new_step_size >= best_step_size:
                step_size_ranges.add((min_step_size, min(new_step_size - 1, max_step_size)))
            else:  # (new_step_size < best_step_size)
                step_size_ranges.add((min_step_size, min(new_step_size - 1, max_step_size)))
                step_size_ranges.add((max(new_step_size + 1, min_step_size), min(best_step_size - 1, max_step_size)))
        else:  # new_n_patches == target_n_patches
            step_size_ranges.add(step_size_range)
        return step_size_ranges

    @staticmethod
    def _is_empty_range(range):
        min_val, max_value = range
        return min_val > max_value
