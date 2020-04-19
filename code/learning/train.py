import pandas as pd
from code.configurer import Configurer
from code.learning.learner import Learner
from code.utils import get_log_csv_name


def train(args):
    args = Configurer(args).with_device() \
        .with_classes() \
        .with_num_classes() \
        .build()

    # Only used is resume_checkpoint is True.
    resume_checkpoint_path = args.checkpoints_folder.joinpath(args.checkpoint_file)

    # Named with date and time.
    log_csv = get_log_csv_name(log_folder=args.log_folder)

    train_wsis_info = pd.read_csv(args.wsis_train)
    val_wsis_info = pd.read_csv(args.wsis_val)

    # Training the ResNet.
    Learner(batch_size=args.batch_size,
            checkpoints_folder=args.checkpoints_folder,
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
            save_interval=args.save_interval,
            num_epochs=args.num_epochs,
            train_folder=args.train_folder,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping,
            train_wsis_info=train_wsis_info, val_wsis_info=val_wsis_info).train()


def add_parser(subparsers):
    subparsers.add_parser("train") \
        .with_all_wsi() \
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
        .with_train_folder() \
        .with_weight_decay() \
        .with_early_stopping() \
        .with_checkpoint_file() \
        .with_checkpoints_folder() \
        .with_log_folder() \
        .with_wsis_train() \
        .with_wsis_val() \
        .set_defaults(func=train)
