from pathlib import Path
import pandas as pd
import json

from typing import List


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

        train_report = {
            'n_tiled_slides': len(self._extract_slide_ids(self._train_patches_root))
        }
        for p in self._train_patches_root.iterdir():
            if p.is_dir():
                report_p = {
                    'n_tiled_slides': len(self._extract_slide_ids(p))
                }
                for p2 in p.iterdir():
                    if p2.is_dir():
                        report_p2 = {
                            'n_tiled_slides': len(self._extract_slide_ids(p2))
                        }
                        for c in p2.iterdir():
                            if c.is_dir():
                                report_p2[f'n_tiled_slides_{c.name}'] = len(self._extract_slide_ids(c))
                        report_p[p2.name] = report_p2
                train_report[p.name] = report_p
        report['train'] = train_report

        eval_report = {}
        for p in self._eval_patches_root.iterdir():
            if p.is_dir():
                tiled_slides_p = self._extract_slide_ids(p)
                report_p = {
                    'n_tiled_slides': len(tiled_slides_p)
                }
                slides_metadata_p = slides_metadata[slides_metadata['id'].isin(tiled_slides_p)]
                for c in slides_metadata_p['label'].unique():
                    tiled_slides_c = slides_metadata_p.loc[slides_metadata_p['label'] == c, 'label']
                    report_p[f'n_tiled_slides_{c}'] = len(tiled_slides_c)
                if p.name == 'val':
                    for p2 in p.iterdir():
                        if p2.is_dir():
                            report_p2 = {}
                            tiled_slides_p2 = self._extract_slide_ids(p2)
                            report_p2['n_tiled_slides'] = len(tiled_slides_p2)
                            slides_metadata_p2 = slides_metadata[slides_metadata['id'].isin(tiled_slides_p2)]
                            for c in slides_metadata_p2['label'].unique():
                                tiled_slides_c = slides_metadata_p2.loc[slides_metadata_p2['label'] == c,
                                                                        'label']
                                report_p2[f'n_tiled_slides_{c}'] = len(tiled_slides_c)
                            report_p[p2.name] = report_p2
                eval_report[p.name] = report_p
        report['eval'] = eval_report

        self._output_report_dir.mkdir(parents=True, exist_ok=True)
        with self._output_report_dir.joinpath("output_analysis.json").open(mode="w") as fp:
            json.dump(report, fp, indent=True)

    def _extract_slide_ids(self, root: Path = None):
        tile_paths = []

        if not root:
            roots = [self._train_patches_root, self._eval_patches_root]
        else:
            roots = [root]

        for r in roots:
            tile_paths += r.rglob(f"*.{self._image_ext}")

        tiled_slide_ids = pd.Series(tile_paths).apply(str) \
                                               .str.rsplit("/", n=1, expand=True)[1] \
                                               .str.rsplit("_", n=2, expand=True)[0] \
                                               .unique()
        return tiled_slide_ids
