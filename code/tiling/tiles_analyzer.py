import json
from collections import Iterable
from pathlib import Path

import pandas as pd


class TilesAnalyzer:
    def __init__(self, train_patches_root: Path, eval_patches_root: Path, image_ext: str,
                 slides_metadata: Path, train_slides_metadata: Path, test_slides_metadata: Path,
                 output_report_dir: Path):
        self._train_patches_root = train_patches_root
        self._eval_patches_root = eval_patches_root
        self._image_ext = image_ext
        self._slides_metadata = slides_metadata
        self._train_slides_metadata = train_slides_metadata
        self._test_slides_metadata = test_slides_metadata
        self._output_report_dir = output_report_dir

    def analyze(self):
        report = {}

        slides_metadata = pd.DataFrame()

        if self._slides_metadata:
            slides_metadata = slides_metadata.append(pd.read_csv(str(self._slides_metadata)), ignore_index=True, sort=False)
        else:
            slides_metadata = slides_metadata.append(pd.read_csv(str(self._train_slides_metadata)), ignore_index=True, sort=False)
            slides_metadata = slides_metadata.append(pd.read_csv(str(self._test_slides_metadata)), ignore_index=True, sort=False)

        all_slide_ids = slides_metadata['id']
        tiled_slide_ids = self._extract_slide_ids()
        untiled_slide_ids = all_slide_ids[~all_slide_ids.isin(tiled_slide_ids)]

        report['n_slides'] = len(all_slide_ids)
        report['n_untiled_slides'] = len(untiled_slide_ids)

        train_report = self._calc_stats(self._train_patches_root)
        for p in self._train_patches_root.iterdir():
            if p.is_dir():
                report_p = self._calc_stats(p)
                for p2 in p.iterdir():
                    if p2.is_dir():
                        report_p2 = self._calc_stats(p2)
                        for c in p2.iterdir():
                            if c.is_dir():
                                report_p2[c.name] = self._calc_stats(c)
                        report_p[p2.name] = report_p2
                train_report[p.name] = report_p
        report['train'] = train_report

        eval_report = {}
        for p in self._eval_patches_root.iterdir():
            if p.is_dir():
                report_p = self._calc_stats(p)
                report_p.update(self._calc_stats_by_class(p, slides_metadata))
                if p.name == 'val':
                    for p2 in p.iterdir():
                        if p2.is_dir():
                            report_p2 = self._calc_stats(p2)
                            report_p2.update(self._calc_stats_by_class(p2, slides_metadata))
                            report_p[p2.name] = report_p2
                eval_report[p.name] = report_p
        report['eval'] = eval_report

        self._output_report_dir.mkdir(parents=True, exist_ok=True)
        with self._output_report_dir.joinpath("output_analysis.json").open(mode="w") as fp:
            json.dump(report, fp, indent=True)

    def _extract_slide_ids(self, *paths: Path):
        tile_paths = []

        if not paths:
            paths = [self._train_patches_root, self._eval_patches_root]

        for p in paths:
            tile_paths += p.rglob(f"*.{self._image_ext}")

        tiled_slide_ids = pd.Series(tile_paths).apply(str) \
                                               .str.rsplit("/", n=1, expand=True)[1] \
                                               .str.rsplit("_", n=2, expand=True)[0] \
                                               .unique()
        return tiled_slide_ids

    def _calc_stats(self, paths = None, ids = None):
        tile_paths = []

        if ids:
            if not isinstance(ids, Iterable):
                ids = [ids]

        if paths:
            if not isinstance(paths, Iterable):
                paths = [paths]
        else:
            paths = [self._train_patches_root, self._eval_patches_root]

        for p in paths:
            if ids:
                for id in ids:
                    tile_paths += p.rglob(f"{id}*.{self._image_ext}")
            else:
                if p.is_dir():
                    tile_paths += p.rglob(f"*.{self._image_ext}")

        slide_ids = pd.Series(tile_paths).apply(str) \
                                         .str.rsplit("/", n=1, expand=True)[1] \
                                         .str.rsplit("_", n=2, expand=True)[0]

        distinct_slide_ids = slide_ids.unique()
        n_tiles_per_slide = slide_ids.value_counts()

        return {
            'n_slides': len(distinct_slide_ids),
            'n_tiles': int(n_tiles_per_slide.sum()),
            'mean_n_tiles_per_slide': n_tiles_per_slide.mean(),
            'std_n_tiles_per_slide': n_tiles_per_slide.std(),
            'min_n_tiles_per_slide': int(n_tiles_per_slide.min()),
            '25%_n_tiles_per_slide': n_tiles_per_slide.quantile(q=0.25),
            'median_n_tiles_per_slide': n_tiles_per_slide.median(),
            '75%_n_tiles_per_slide': n_tiles_per_slide.quantile(q=0.75),
            'max_n_tiles_per_slide': int(n_tiles_per_slide.max())
        }

    def _calc_stats_by_class(self, p: Path, slides_metadata: pd.DataFrame):
        report = {}
        tiled_slide_ids = self._extract_slide_ids(p)
        tiled_slides_metadata = slides_metadata[slides_metadata['id'].isin(tiled_slide_ids)]
        for c in tiled_slides_metadata['label'].unique():
            slide_ids_c = tiled_slides_metadata.loc[tiled_slides_metadata['label'] == c, 'id'].tolist()
            report[c] = self._calc_stats(ids=slide_ids_c, paths=p)
        return report
