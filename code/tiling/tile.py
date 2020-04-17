import logging

import pandas as pd

from .patches_balancer import PatchesBalancer
from .purple_patch_extractor import PurplePatchExtractor
from ..configurer import Configurer


def tile(args):
    args = Configurer(args).build()

    patch_extractor = PurplePatchExtractor(patch_size=args.patch_size,
                                           purple_threshold=args.purple_threshold,
                                           purple_scale_size=args.purple_scale_size,
                                           image_ext=args.image_ext,
                                           num_workers=args.num_workers)

    train_wsis_info = pd.read_csv(args.wsis_train)
    val_wsis_info = pd.read_csv(args.wsis_val)
    test_wsis_info = pd.read_csv(args.wsis_test)

    # This is the input for model learning, automatically built.
    train_patches = args.train_folder.joinpath("train")
    val_patches = args.train_folder.joinpath("val")

    val_step_size = int(args.patch_size / args.val_patch_overlap_factor)
    test_step_size = int(args.patch_size / args.test_patch_overlap_factor)

    logging.info("Generating training patches...")
    patch_extractor.extract_all_by_class(wsis_info=train_wsis_info, partition_name='training',
                                         output_folder=train_patches,
                                         n_patches_per_class=args.num_train_patches_per_class)

    PatchesBalancer(image_dir=train_patches, partition_name='training').balance_by_class()

    logging.info("Generating validation patches...")
    patch_extractor.extract_all_by_class(wsis_info=val_wsis_info, partition_name='validation',
                                         output_folder=val_patches, step_size=val_step_size)

    logging.info(f"Generating validation evaluation patches...")
    patch_extractor.extract_all(wsis_info=val_wsis_info, step_size=test_step_size, partition_name='validation',
                                output_folder=args.patches_eval_val, by_wsi=True)

    logging.info(f"Generating test evaluation patches...")
    patch_extractor.extract_all(wsis_info=test_wsis_info, step_size=test_step_size, partition_name='test',
                                output_folder=args.patches_eval_test, by_wsi=True)


def add_parser(subparsers):
    subparsers.add_parser("tile") \
        .with_wsis_train() \
        .with_wsis_val() \
        .with_wsis_test() \
        .with_train_folder() \
        .with_num_train_patches_per_class() \
        .with_num_workers() \
        .with_patch_size() \
        .with_purple_threshold() \
        .with_purple_scale_size() \
        .with_image_ext() \
        .with_val_patch_overlap_factor() \
        .with_patches_eval_val() \
        .with_patches_eval_test() \
        .with_test_patch_overlap_factor() \
        .set_defaults(func=tile)
