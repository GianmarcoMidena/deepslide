from code.configurer import Configurer
from code.learning.learner import Learner
from code.utils import get_log_csv_name


def train(args):
    args = Configurer(args).with_device() \
        .with_classes() \
        .with_num_classes() \
        .build()

    wsi_metadata_paths = sorted(list(args.wsi_splits_dir.glob("*.csv")))
    patch_metadata_paths = sorted(list(args.train_patches_root.joinpath('train').glob("*.csv")))
    tot_splits = len(wsi_metadata_paths)
    n_test_splits = 1
    n_train_splits = tot_splits - n_test_splits
    train_wsi_metadata_paths = wsi_metadata_paths[:-n_test_splits]
    train_patch_metadata_paths = patch_metadata_paths[:-n_test_splits]

    for i in range(n_train_splits):
        part_name = f'part_{i+1}'

        val_wsi_metadata_paths_i = [train_wsi_metadata_paths[i]]
        val_patch_metadata_paths_i = [train_patch_metadata_paths[i]]
        train_wsi_metadata_paths_i = train_wsi_metadata_paths[:i] + train_wsi_metadata_paths[i+1:]
        train_patch_metadata_paths_i = train_patch_metadata_paths[:i] + train_patch_metadata_paths[i + 1:]

        checkpoints_folder_i = args.checkpoints_root.joinpath(part_name)
        # Only used is resume_checkpoint is True.
        resume_checkpoint_path_i = checkpoints_folder_i.joinpath(args.checkpoint_file)

        # Named with date and time.
        log_csv_i = get_log_csv_name(log_folder=args.log_root.joinpath(part_name))

        # Training the ResNet.
        Learner(batch_size=args.batch_size,
                checkpoints_folder=checkpoints_folder_i,
                classes=args.classes,
                color_jitter_brightness=args.color_jitter_brightness,
                color_jitter_contrast=args.color_jitter_contrast,
                color_jitter_hue=args.color_jitter_hue,
                color_jitter_saturation=args.color_jitter_saturation,
                device=args.device,
                learning_rate=args.learning_rate,
                learning_rate_decay=args.learning_rate_decay,
                log_csv=log_csv_i,
                num_classes=args.num_classes,
                num_layers=args.num_layers,
                num_workers=args.num_workers,
                pretrain=args.pretrain,
                resume_checkpoint=args.resume_checkpoint,
                resume_checkpoint_path=resume_checkpoint_path_i,
                save_interval=args.save_interval,
                num_epochs=args.num_epochs,
                weight_decay=args.weight_decay,
                early_stopping_patience=args.early_stopping,
                train_wsi_metadata_paths=train_wsi_metadata_paths_i,
                val_wsi_metadata_paths=val_wsi_metadata_paths_i,
                train_patch_metadata_paths=train_patch_metadata_paths_i,
                val_patch_metadata_paths=val_patch_metadata_paths_i,
                class_idx_path=args.class_idx).train()


def add_parser(subparsers):
    subparsers.add_parser("train") \
              .with_all_wsi() \
              .with_wsi_splits_dir() \
              .with_batch_size() \
              .with_color_jitter_brightness() \
              .with_color_jitter_contrast() \
              .with_color_jitter_hue() \
              .with_color_jitter_saturation() \
              .with_learning_rate() \
              .with_learning_rate_decay() \
              .with_num_layers() \
              .with_num_workers() \
              .with_pretrain() \
              .with_resume_checkpoint() \
              .with_save_interval() \
              .with_num_epochs() \
              .with_weight_decay() \
              .with_early_stopping() \
              .with_train_patches_root() \
              .with_checkpoints_root() \
              .with_checkpoint_file() \
              .with_log_root() \
              .with_class_idx() \
              .set_defaults(func=train)
