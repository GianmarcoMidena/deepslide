from .patch_tester import PatchTester
from ..configurer import Configurer


def patch_evaluate(args):
    args = Configurer(args)\
        .with_device()\
        .with_classes()\
        .with_num_classes()\
        .build()

    wsi_metadata_paths = sorted(list(args.wsi_splits_dir.glob("*.csv")))
    patch_metadata_paths = sorted(list(args.eval_patches_root.glob("*.csv")))
    tot_splits = len(wsi_metadata_paths)
    n_test_splits = 1
    n_train_splits = tot_splits - n_test_splits
    train_wsi_metadata_paths = wsi_metadata_paths[:-n_test_splits]
    val_patch_metadata_paths = patch_metadata_paths[:-n_test_splits]
    test_wsi_metadata_paths = wsi_metadata_paths[-n_test_splits:]
    test_patch_metadata_paths = patch_metadata_paths[-n_test_splits:]

    for i in range(n_train_splits):
        part_id = i + 1
        part_name = f'part_{part_id}'

        val_wsi_metadata_paths_i = [train_wsi_metadata_paths[i]]
        val_patch_metadata_paths_i = [val_patch_metadata_paths[i]]

        checkpoints_folder_i = args.checkpoints_root.joinpath(part_name)
        checkpoint_path_i = checkpoints_folder_i.joinpath(args.checkpoint_file)

        tester = PatchTester(auto_select=args.auto_select,
                             batch_size=args.batch_size,
                             checkpoints_folder=checkpoints_folder_i,
                             classes=args.classes,
                             device=args.device,
                             eval_model=checkpoint_path_i,
                             num_classes=args.num_classes,
                             num_layers=args.num_layers,
                             num_workers=args.num_workers,
                             pretrain=args.pretrain)

        # Apply the model to the validation patches.
        tester.predict(patches_metadata_paths=val_patch_metadata_paths_i,
                       wsi_metadata_paths=val_wsi_metadata_paths_i,
                       partition_name=f'validation (part {part_id})',
                       output_folder=args.preds_val.joinpath(part_name),
                       class_idx_path=args.class_idx)

        # Apply the model to the test patches.
        tester.predict(patches_metadata_paths=test_patch_metadata_paths,
                       wsi_metadata_paths=test_wsi_metadata_paths,
                       partition_name='testing',
                       output_folder=args.preds_test.joinpath(part_name),
                       class_idx_path=args.class_idx)


def add_parser(subparsers):
    subparsers.add_parser("test_on_patches") \
        .with_all_wsi() \
        .with_wsi_splits_dir() \
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
        .set_defaults(func=patch_evaluate)
