from code.configurer import Configurer
from code.learning.learner import Learner
from code.utils import get_log_csv_name


def train(args):
    args = Configurer(args).with_device() \
        .with_classes() \
        .with_num_classes() \
        .build()

    slides_metadata_paths = sorted(list(args.slides_splits_dir.glob("*part_*.csv")))
    train_patch_metadata_paths = sorted(list(args.train_patches_root.joinpath('train').glob("*part_*.csv")))
    val_patch_metadata_paths = sorted(list(args.train_patches_root.joinpath('val').glob("*part_*.csv")))
    tot_splits = len(slides_metadata_paths)

    outer_part_ids = list(range(tot_splits))

    fixed_folds = args.fixed_folds

    for i in outer_part_ids:

        if args.nested_cross_validation:
            inner_part_ids = outer_part_ids[:i] + outer_part_ids[i+1:]
        else:
            inner_part_ids = outer_part_ids

        if fixed_folds:
            if args.nested_cross_validation:
                inner_part_ids = [j for j in inner_part_ids
                                  if (f"{i+1}" in fixed_folds) or (f"{i+1}_{j+1}" in fixed_folds)]
            else:
                inner_part_ids = [j for j in inner_part_ids if f"{j + 1}" in fixed_folds]

        for j in inner_part_ids:
            if args.nested_cross_validation:
                part_name = f'part_{i+1}_{j+1}'
            else:
                part_name = f'part_{j + 1}'

            val_slides_metadata_paths = [slides_metadata_paths[j]]
            val_patch_metadata_paths_j = [val_patch_metadata_paths[j]]
            train_slides_metadata_paths = slides_metadata_paths[:j] + slides_metadata_paths[j+1:]
            train_patch_metadata_paths_j = train_patch_metadata_paths[:j] + train_patch_metadata_paths[j+1:]

            checkpoints_folder = args.checkpoints_root.joinpath(part_name)
            # Only used is resume_checkpoint is True.
            resume_checkpoint_path = checkpoints_folder.joinpath(args.checkpoint_file)

            # Named with date and time.
            log_csv = get_log_csv_name(log_folder=args.log_root.joinpath(part_name))

            _train(args, checkpoints_folder, log_csv, resume_checkpoint_path, train_patch_metadata_paths_j,
                   train_slides_metadata_paths, val_patch_metadata_paths_j, val_slides_metadata_paths)

        if not args.nested_cross_validation:
            break


def _train(args, checkpoints_folder, log_csv, resume_checkpoint_path, train_patch_metadata_paths_j,
           train_slides_metadata_paths, val_patch_metadata_paths_j, val_slides_metadata_paths):
    # Training the ResNet.
    Learner(batch_size=args.batch_size,
            checkpoints_folder=checkpoints_folder,
            classes=args.classes,
            color_jitter_brightness=args.color_jitter_brightness,
            color_jitter_contrast=args.color_jitter_contrast,
            color_jitter_hue=args.color_jitter_hue,
            color_jitter_saturation=args.color_jitter_saturation,
            device=args.device,
            learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay,
            log_csv=log_csv,
            num_classes=args.num_classes,
            num_layers=args.num_layers,
            num_workers=args.num_workers,
            pretrain=args.pretrain,
            resume_checkpoint=args.resume_checkpoint,
            resume_checkpoint_path=resume_checkpoint_path,
            last_val_acc=args.last_val_acc,
            num_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping,
            train_slides_metadata_paths=train_slides_metadata_paths,
            val_slides_metadata_paths=val_slides_metadata_paths,
            train_patch_metadata_paths=train_patch_metadata_paths_j,
            val_patch_metadata_paths=val_patch_metadata_paths_j,
            class_idx_path=args.class_idx,
            spatial_sensitive=args.spatial_sensitive,
            n_spatial_features=args.n_spatial_features,
            patch_size=args.patch_size).train()


def add_parser(subparsers):
    subparsers.add_parser("train") \
              .with_slides_root() \
              .with_slides_splits_dir() \
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
              .with_num_epochs() \
              .with_weight_decay() \
              .with_early_stopping() \
              .with_train_patches_root() \
              .with_checkpoints_root() \
              .with_checkpoint_file() \
              .with_last_val_acc() \
              .with_log_root() \
              .with_class_idx() \
              .with_fixed_folds() \
              .with_nested_cross_validation() \
              .with_spatial_sensitivity() \
              .with_n_spatial_features() \
              .with_patch_size() \
              .set_defaults(func=train)
