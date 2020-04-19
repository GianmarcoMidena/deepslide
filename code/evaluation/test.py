import pandas as pd
from .tester import Tester
from ..configurer import Configurer


def test(args):
    args = Configurer(args)\
        .with_device()\
        .with_classes()\
        .with_num_classes()\
        .build()

    # Does nothing if auto_select is True.
    eval_model = args.checkpoints_folder.joinpath(args.checkpoint_file)

    tester = Tester(auto_select=args.auto_select,
                    batch_size=args.batch_size,
                    checkpoints_folder=args.checkpoints_folder,
                    classes=args.classes,
                    device=args.device,
                    eval_model=eval_model,
                    num_classes=args.num_classes,
                    num_layers=args.num_layers,
                    num_workers=args.num_workers,
                    pretrain=args.pretrain)

    val_wsis_info = pd.read_csv(args.wsis_val)
    test_wsis_info = pd.read_csv(args.wsis_test)

    # Run the ResNet on the generated patches.
    # Validation patches.
    tester.predict(patches_eval_folder=args.patches_eval_val,
                   wsis_info=val_wsis_info,
                   partition_name='validation',
                   output_folder=args.preds_val)

    # Test patches.
    tester.predict(patches_eval_folder=args.patches_eval_test,
                   wsis_info=test_wsis_info,
                   partition_name='evaluation',
                   output_folder=args.preds_test)


def add_parser(subparsers):
    subparsers.add_parser("test") \
        .with_all_wsi() \
        .with_patches_eval_val() \
        .with_patches_eval_test() \
        .with_preds_val() \
        .with_preds_test() \
        .with_auto_select() \
        .with_batch_size() \
        .with_num_layers() \
        .with_num_workers() \
        .with_pretrain() \
        .with_checkpoints_folder() \
        .with_checkpoint_file() \
        .with_wsis_val() \
        .with_wsis_test() \
        .set_defaults(func=test)