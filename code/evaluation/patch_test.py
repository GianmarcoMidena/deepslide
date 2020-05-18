from .patch_tester import PatchTester
from ..configurer import Configurer


def patch_evaluate(args):
    args = Configurer(args)\
        .with_device()\
        .with_classes()\
        .with_num_classes()\
        .build()

    slides_metadata_paths = sorted(list(args.slides_splits_dir.glob("*part_*.csv")))

    if args.test_slides_metadata:
        nested_cross_validation = False
    else:
        nested_cross_validation = True

    patch_metadata_paths = sorted(list(args.eval_patches_root.rglob("*part_*.csv")))

    tot_splits = len(slides_metadata_paths)

    outer_part_ids = list(range(tot_splits))

    for i in outer_part_ids:
        if nested_cross_validation:
            test_slides_metadata_paths = [slides_metadata_paths[i]]
            test_patch_metadata_paths = [patch_metadata_paths[i]]
            inner_part_ids = outer_part_ids[:i] + outer_part_ids[i + 1:]
            testing_partition_name = f'testing (part {i+1})'
        else:
            test_slides_metadata_paths = [args.test_slides_metadata]
            test_patch_metadata_paths = [args.eval_patches_root.joinpath('test').joinpath("test_eval_patches.csv")]
            inner_part_ids = outer_part_ids
            testing_partition_name = f'testing'

        for j in inner_part_ids:
            if nested_cross_validation:
                part_id = f'{i+1}_{j+1}'
            else:
                part_id = f'{j + 1}'
            part_name = f'part_{part_id}'

            val_slides_metadata_paths = [slides_metadata_paths[j]]
            val_patch_metadata_paths = [patch_metadata_paths[j]]

            checkpoints_folder = args.checkpoints_root.joinpath(part_name)
            checkpoint_path = checkpoints_folder.joinpath(args.checkpoint_file)

            tester = PatchTester(auto_select=args.auto_select,
                                 batch_size=args.batch_size,
                                 checkpoints_folder=checkpoints_folder,
                                 classes=args.classes,
                                 device=args.device,
                                 eval_model=checkpoint_path,
                                 num_classes=args.num_classes,
                                 num_layers=args.num_layers,
                                 num_workers=args.num_workers,
                                 pretrain=args.pretrain,
                                 patch_size=args.patch_size,
                                 spatial_sensitive=args.spatial_sensitive,
                                 n_spatial_features=args.n_spatial_features)

            # Apply the model to the validation patches.
            tester.predict(patches_metadata_paths=val_patch_metadata_paths,
                           slides_metadata_paths=val_slides_metadata_paths,
                           partition_name=f'validation (part {j+1})',
                           output_folder=args.preds_val.joinpath(part_name),
                           class_idx_path=args.class_idx)

            # Apply the model to the test patches.
            tester.predict(patches_metadata_paths=test_patch_metadata_paths,
                           slides_metadata_paths=test_slides_metadata_paths,
                           partition_name=testing_partition_name,
                           output_folder=args.preds_test.joinpath(part_name),
                           class_idx_path=args.class_idx)

        if not nested_cross_validation:
            break


def add_parser(subparsers):
    subparsers.add_parser("test_on_patches") \
        .with_slides_root() \
        .with_slides_splits_dir() \
        .with_eval_patches_root() \
        .with_preds_val() \
        .with_preds_test() \
        .with_auto_select() \
        .with_batch_size() \
        .with_num_layers() \
        .with_num_workers() \
        .with_pretrain() \
        .with_checkpoints_root() \
        .with_checkpoint_file() \
        .with_class_idx() \
        .with_test_slides_metadata() \
        .with_patch_size() \
        .with_spatial_sensitivity() \
        .with_n_spatial_features() \
        .set_defaults(func=patch_evaluate)
