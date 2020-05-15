import logging
from pathlib import Path
from typing import List
import pandas as pd

from .patches_balancer import PatchesBalancer
from .purple_patch_extractor import PurplePatchExtractor
from .step_size_finder import StepSizeFinder
from ..configurer import Configurer


def tile(args):
    args = Configurer(args).build()

    patch_extractor = PurplePatchExtractor(patch_size=args.patch_size,
                                           purple_threshold=args.purple_threshold,
                                           purple_scale_size=args.purple_scale_size,
                                           image_ext=args.image_ext,
                                           num_workers=args.num_workers)

    logging.info("Generating training patches for training...")
    train_patches_folder = args.train_patches_root.joinpath("train")
    val_patches_folder = args.train_patches_root.joinpath("val")

    val_step_size = int(args.patch_size / args.val_patch_overlap_factor)
    eval_step_size = int(args.patch_size / args.test_patch_overlap_factor)

    splits_metadata_paths = sorted(list(args.slides_splits_dir.glob("*part_*.csv")))
    tot_splits = len(splits_metadata_paths)
    n_test_splits = 1
    n_train_splits = tot_splits - n_test_splits
    num_train_patches_per_class_per_split = args.num_train_patches_per_class // n_train_splits
    step_size_finder = StepSizeFinder(target_n_patches=num_train_patches_per_class_per_split,
                                      patch_size=args.patch_size, patch_extractor=patch_extractor)

    if args.slides_metadata:
        slides_metadata = pd.read_csv(args.slides_metadata)
    else:
        slides_metadata = pd.read_csv(args.train_slides_metadata)

    slides_metadata = slides_metadata.set_index('id', drop=True)

    if args.test_slides_metadata:
        test_metadata = pd.read_csv(args.test_slides_metadata)
        test_eval_dir = args.eval_patches_root.joinpath('test')

        patch_extractor.extract_all(image_paths=_extract_image_paths(test_metadata), step_size=eval_step_size,
                                    partition_name='testing', output_folder=test_eval_dir, by_wsi=True)

        report_test_eval_part_i = pd.DataFrame([str(x) for c in test_eval_dir.iterdir()
                                                       for x in c.iterdir()], columns=['path'])
        report_test_eval_part_i['id'] = report_test_eval_part_i['path'].str.rsplit('/', n=1, expand=True)[1] \
                                                                       .str.rsplit('_', n=2, expand=True)[0]
        report_test_eval_part_i = report_test_eval_part_i.set_index('id', drop=True)
        report_test_eval_part_i = report_test_eval_part_i.join(test_metadata['label'], how='inner', sort=False)
        report_test_eval_part_i.to_csv(test_eval_dir.joinpath(f"test_eval_patches.csv"), index=False)

    for metadata_path_i in splits_metadata_paths:
        part_id = metadata_path_i.stem.split("_")[-1]
        part_name = f"part_{part_id}"
        metadata_i = pd.read_csv(metadata_path_i)
        train_part_name = f'training (part {part_id})'
        train_dir_i = train_patches_folder.joinpath(part_name)
        patch_extractor.extract_all_by_class(slides_info=metadata_i, partition_name=train_part_name,
                                             output_folder=train_dir_i, step_size_finder=step_size_finder)
        PatchesBalancer(image_dir=train_dir_i, partition_name=train_part_name).balance_by_class()

        report_train_part_i = pd.DataFrame([str(x) for c in train_dir_i.iterdir()
                                            for x in c.iterdir()], columns=['path'])
        report_train_part_i['label'] = report_train_part_i['path'].str.rsplit('/', n=2, expand=True)[1]
        report_train_part_i = report_train_part_i.sample(frac=1., replace=False, random_state=3)
        report_train_part_i.to_csv(train_patches_folder.joinpath(f"train_patches_part_{part_id}.csv"), index=False)

        logging.info("Generating validation patches for training...")
        val_part_name = f'validation (part {part_id})'
        val_dir_i = val_patches_folder.joinpath(part_name)
        patch_extractor.extract_all_by_class(slides_info=metadata_i, partition_name=val_part_name,
                                             output_folder=val_dir_i, step_size=val_step_size)

        report_val_part_i = pd.DataFrame([str(x) for c in val_dir_i.iterdir()
                                          for x in c.iterdir()], columns=['path'])
        report_val_part_i['label'] = report_val_part_i['path'].str.rsplit('/', n=2, expand=True)[1]
        report_val_part_i.to_csv(val_patches_folder.joinpath(f"val_patches_part_{part_id}.csv"), index=False)

        logging.info(f"Generating patches for evaluation...")
        eval_part_name = f'evaluation (part {part_id})'

        if args.test_slides_metadata:
            eval_patches_folder = args.eval_patches_root.joinpath('val')
        else:
            eval_patches_folder = args.eval_patches_root
        eval_dir_i = eval_patches_folder.joinpath(part_name)

        patch_extractor.extract_all(image_paths=_extract_image_paths(metadata_i), step_size=eval_step_size,
                                    partition_name=eval_part_name, output_folder=eval_dir_i, by_wsi=True)

        report_eval_part_i = pd.DataFrame([str(x) for c in eval_dir_i.iterdir()
                                           for x in c.iterdir()], columns=['path'])
        report_eval_part_i['id'] = report_eval_part_i['path'].str.rsplit('/', n=1, expand=True)[1] \
                                                             .str.rsplit('_', n=2, expand=True)[0]
        report_eval_part_i = report_eval_part_i.set_index('id', drop=True)
        report_eval_part_i = report_eval_part_i.join(slides_metadata['label'], how='inner', sort=False)
        report_eval_part_i.to_csv(eval_patches_folder.joinpath(f"eval_patches_part_{part_id}.csv"), index=False)


def _extract_image_paths(slides_info: pd.DataFrame) -> List[Path]:
    return slides_info['path'].apply(Path).tolist()


def add_parser(subparsers):
    subparsers.add_parser("tile") \
        .with_slides_metadata() \
        .with_train_slides_metadata() \
        .with_test_slides_metadata() \
        .with_slides_splits_dir() \
        .with_train_patches_root() \
        .with_num_train_patches_per_class() \
        .with_num_workers() \
        .with_patch_size() \
        .with_purple_threshold() \
        .with_purple_scale_size() \
        .with_image_ext() \
        .with_val_patch_overlap_factor() \
        .with_eval_patches_root() \
        .with_test_patch_overlap_factor() \
        .set_defaults(func=tile)
