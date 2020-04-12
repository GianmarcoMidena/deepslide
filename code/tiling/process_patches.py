import pandas as pd

from .class_balancer import ClassBalancer
from .patch_extractor import PatchExtractor
from ..configurer import Configurer


def process_patches(args):
    args = Configurer(args).build()

    patch_extractor = PatchExtractor()

    train_wsis_info = pd.read_csv(args.wsis_train)
    val_wsis_info = pd.read_csv(args.wsis_val)
    test_wsis_info = pd.read_csv(args.wsis_test)

    # This is the input for model learning, automatically built.
    train_patches = args.train_folder.joinpath("train")
    val_patches = args.train_folder.joinpath("val")

    patch_extractor.gen_train_patches(wsis_info=train_wsis_info,
                                      output_folder=train_patches,
                                      n_patches_per_class=args.num_train_patches_per_class,
                                      num_workers=args.num_workers,
                                      patch_size=args.patch_size,
                                      purple_threshold=args.purple_threshold,
                                      purple_scale_size=args.purple_scale_size,
                                      image_ext=args.image_ext,
                                      type_histopath=args.type_histopath)

    ClassBalancer(wsis_info=train_wsis_info, partition_name='learning').balance()

    patch_extractor.gen_val_patches(wsis_info=val_wsis_info,
                                    output_folder=val_patches,
                                    overlap_factor=args.gen_val_patches_overlap_factor,
                                    num_workers=args.num_workers,
                                    patch_size=args.patch_size,
                                    purple_threshold=args.purple_threshold,
                                    purple_scale_size=args.purple_scale_size,
                                    image_ext=args.image_ext,
                                    type_histopath=args.type_histopath)

    patch_extractor.produce_patches(wsis_info=val_wsis_info,
                                    partition_name='validation',
                                    output_folder=args.patches_eval_val,
                                    inverse_overlap_factor=args.slide_overlap,
                                    num_workers=args.num_workers,
                                    patch_size=args.patch_size,
                                    purple_threshold=args.purple_threshold,
                                    purple_scale_size=args.purple_scale_size,
                                    image_ext=args.image_ext,
                                    type_histopath=args.type_histopath)

    patch_extractor.produce_patches(wsis_info=test_wsis_info,
                                    partition_name='evaluation',
                                    output_folder=args.patches_eval_test,
                                    inverse_overlap_factor=args.slide_overlap,
                                    num_workers=args.num_workers,
                                    patch_size=args.patch_size,
                                    purple_threshold=args.purple_threshold,
                                    purple_scale_size=args.purple_scale_size,
                                    image_ext=args.image_ext,
                                    type_histopath=args.type_histopath)


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
        .with_type_histopath() \
        .with_gen_val_patches_overlap_factor() \
        .with_patches_eval_val() \
        .with_patches_eval_test() \
        .with_slide_overlap() \
        .set_defaults(func=process_patches)