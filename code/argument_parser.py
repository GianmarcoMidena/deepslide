import argparse
from pathlib import Path


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, prog=None, usage=None, description=None, epilog=None,
                 parents=[], formatter_class=argparse.HelpFormatter,
                 prefix_chars='-', fromfile_prefix_chars=None,
                 argument_default=None, conflict_handler='error',
                 add_help=True, allow_abbrev=True):
        super().__init__(prog=prog, usage=usage, description=description,
                         epilog=epilog, parents=parents,
                         formatter_class=formatter_class,
                         prefix_chars=prefix_chars,
                         fromfile_prefix_chars=fromfile_prefix_chars,
                         argument_default=argument_default,
                         conflict_handler=conflict_handler,
                         add_help=add_help, allow_abbrev=allow_abbrev)

    def with_slides_root(self):
        # Input folders for learning images.
        # Must contain subfolders of images labelled by class.
        # If your two classes are 'a' and 'n', you must have a/*.jpg with the images in class a and
        # n/*.jpg with the images in class n.
        self.add_argument(
            "--slides",
            type=Path,
            default=Path("slides"),
            help="Location of the WSIs organized in subfolders by class")
        return self

    def with_positive_class(self):
        self.add_argument(
            "--positive_class",
            type=str,
            required=False
        )
        return self

    def with_negative_class(self):
        self.add_argument(
            "--negative_class",
            type=str,
            required=False
        )
        return self

    def with_original_slides(self):
        self.add_argument(
            "--original_slides",
            type=Path,
            default=Path("original_slides"),
            help="Location of the original WSIs organized in subfolders by class")
        return self

    def with_downscale_factor(self):
        self.add_argument("--downscale_factor",
                          type=int, required=False)
        return self

    def with_min_dim_size(self):
        self.add_argument("--min_dim_size",
                          type=int, required=False)
        return self

    def with_slides_metadata(self):
        self.add_argument(
            "--slides_metadata",
            type=Path,
            required=False,
            help="Location of a CSV file containing the wsi metadata"
        )
        return self

    def with_train_slides_metadata(self):
        self.add_argument(
            "--train_slides_metadata",
            type=Path,
            required=False,
            help="Location of a CSV file containing the train slide metadata"
        )
        return self

    def with_test_slides_metadata(self):
        self.add_argument(
            "--test_slides_metadata",
            type=Path,
            required=False,
            help="Location of a CSV file containing the train slide metadata"
        )
        return self

    def with_nested_cross_validation(self):
        self.add_argument(
            "--nested_cross_validation",
            action="store_true"
        )
        return self

    def with_spatial_sensitivity(self):
        self.add_argument(
            "--spatial_sensitive",
            action="store_true"
        )
        return self

    def with_n_spatial_features(self):
        self.add_argument(
            "--n_spatial_features",
            type=int,
            default=64
        )
        return self

    def with_path_column(self):
        self.add_argument(
            "--path_column",
            type=str,
            default='path',
            help="The path column name"
        )
        return self

    def with_group(self):
        self.add_argument(
            "--group",
            type=str,
            required=False,
            help="Keeps the samples belonging to the same group in the same partition")
        return self

    def with_n_splits(self):
        self.add_argument(
            "--n_splits",
            type=int,
            required=False,
            default=1)
        return self

    def with_fixed_folds(self):
        self.add_argument(
            "--fixed_folds",
            type=str,
            nargs='*',
            required=False)
        return self

    def with_slides_splits_dir(self):
        self.add_argument(
            "--slides_splits_dir",
            type=Path,
            default=Path("slides"),
            help="path to the wsi splits directory")
        return self

    def with_num_workers(self):
        # Number of processes to use.
        self.add_argument("--num_workers",
                          type=int,
                          default=8,
                          help="Number of workers to use for IO")
        return self

    def with_patch_size(self):
        # Default shape for ResNet in PyTorch.
        self.add_argument("--patch_size",
                          type=int,
                          default=224,
                          help="Size of the patches extracted from the WSI")
        return self

    def with_slides_val(self):
        self.add_argument("--slides_val",
                          type=Path,
                          default=Path("slides_val.csv"),
                          help="Location to store the CSV file info for validation wsis")
        return self

    def with_slides_test(self):
        self.add_argument("--slides_test",
                          type=Path,
                          default=Path("slides_test.csv"),
                          help="Location to store the CSV file info for evaluation wsis")
        return self

    def with_train_patches_root(self):
        # This is the input for model learning, automatically built.
        self.add_argument(
            "--train_patches_root",
            type=Path,
            default=Path("patches/train"),
            help="Location of the automatically built learning input folder")
        return self

    def with_eval_patches_root(self):
        # Folders of patches by WSI in evaluation set, used for finding validation/test accuracy at WSI level.
        self.add_argument(
            "--eval_patches_root",
            type=Path,
            default=Path("patches/eval"),
            help=
            "Folders of patches by WSI in evaluation set, used for finding validation/test accuracy at WSI level"
        )
        return self

    def with_num_train_patches_per_class(self):
        # Target number of learning patches per class.
        self.add_argument("--num_train_patches_per_class",
                          type=int,
                          default=80000,
                          help="Target number of learning samples per class")
        return self

    def with_purple_threshold(self):
        # Number of purple points for region to be considered purple.
        self.add_argument(
            "--purple_threshold",
            type=int,
            default=100,
            help="Number of purple points for region to be considered purple.")
        return self

    def with_purple_scale_size(self):
        # Scalar to use for reducing image to check for purple.
        self.add_argument(
            "--purple_scale_size",
            type=int,
            default=15,
            help="Scalar to use for reducing image to check for purple.")
        return self

    def with_test_patch_overlap_factor(self):
        # Sliding window overlap factor (for evaluation).
        # For generating patches during the learning phase, we slide a window to overlap by some factor.
        # Must be an integer. 1 means no overlap, 2 means overlap by 1/2, 3 means overlap by 1/3.
        # Recommend 2 for very high resolution, 3 for medium, and 5 not extremely high resolution images.
        self.add_argument("--test_patch_overlap_factor",
                          type=int,
                          default=3,
                          help="Sliding window overlap factor for the evaluation phase")
        return self

    def with_val_patch_overlap_factor(self):
        # Overlap factor to use when generating validation patches.
        self.add_argument(
            "--val_patch_overlap_factor",
            type=float,
            default=1.5,
            help="Overlap factor to use when generating validation patches.")
        return self

    def with_image_ext(self):
        self.add_argument("--image_ext",
                          type=str,
                          default="jpg",
                          help="Image extension for saving patches")
        return self

    def with_original_slide_ext(self):
        self.add_argument("--original_slide_ext",
                          type=str,
                          default="svs",
                          help="Original WSI extension")
        return self

    def with_color_jitter_brightness(self):
        self.add_argument(
            "--color_jitter_brightness",
            type=float,
            default=0.5,
            help=
            "Random brightness jitter to use in data augmentation for ColorJitter() transform"
        )
        return self

    def with_color_jitter_contrast(self):
        self.add_argument(
            "--color_jitter_contrast",
            type=float,
            default=0.5,
            help=
            "Random contrast jitter to use in data augmentation for ColorJitter() transform"
        )
        return self

    def with_color_jitter_saturation(self):
        self.add_argument(
            "--color_jitter_saturation",
            type=float,
            default=0.5,
            help=
            "Random saturation jitter to use in data augmentation for ColorJitter() transform"
        )
        return self

    def with_color_jitter_hue(self):
        self.add_argument(
            "--color_jitter_hue",
            type=float,
            default=0.2,
            help=
            "Random hue jitter to use in data augmentation for ColorJitter() transform"
        )
        return self

    def with_num_epochs(self):
        # Model hyperparameters.
        self.add_argument("--num_epochs",
                          type=int,
                          default=20,
                          help="Number of epochs for learning")
        return self

    def with_num_layers(self):
        # Choose from [18, 34, 50, 101, 152].
        self.add_argument(
            "--num_layers",
            type=int,
            default=18,
            help=
            "Number of layers to use in the ResNet model from [18, 34, 50, 101, 152]")
        return self

    def with_learning_rate(self):
        self.add_argument("--learning_rate",
                          type=float,
                          default=0.001,
                          help="Learning rate to use for gradient descent")
        return self

    def with_batch_size(self):
        self.add_argument("--batch_size",
                          type=int,
                          default=16,
                          help="Mini-batch size to use for learning")
        return self

    def with_weight_decay(self):
        self.add_argument("--weight_decay",
                          type=float,
                          default=1e-4,
                          help="Weight decay (L2 penalty) to use in optimizer")
        return self

    def with_learning_rate_decay(self):
        self.add_argument("--learning_rate_decay",
                          type=float,
                          default=0.85,
                          help="Learning rate decay amount per epoch")
        return self

    def with_early_stopping(self):
        self.add_argument("--early_stopping",
                          type=int,
                          default=15,
                          help="Early stopping patience")
        return self

    def with_resume_checkpoint(self):
        self.add_argument("--resume_checkpoint",
                          action="store_true",
                          help="Resume model from checkpoint file")
        return self

    def with_checkpoints_root(self):
        # Where models are saved.
        self.add_argument("--checkpoints_root",
                          type=Path,
                          default=Path("checkpoints"),
                          help="Directory to save model checkpoints to")
        return self

    def with_checkpoint_file(self):
        # Name of checkpoint file to load from.
        self.add_argument(
            "--checkpoint_file",
            type=Path,
            default=Path("xyz.pt"),
            help="Checkpoint file to load if resume_checkpoint_path is True")
        return self

    def with_last_val_acc(self):
        self.add_argument("--last_val_acc",
                          type=float,
                          required=False)
        return self

    def with_pretrain(self):
        # ImageNet pretrain?
        self.add_argument("--pretrain",
                          default=False,
                          action="store_true",
                          help="Use pretrained ResNet weights")
        return self

    def with_log_root(self):
        self.add_argument("--log_root",
                          type=Path,
                          default=Path("logs"),
                          help="Directory to save logs to")
        return self

    def with_auto_select(self):
        # Selecting the best model.
        # Automatically select the model with the highest validation accuracy.
        self.add_argument(
            "--auto_select",
            type=bool,
            default=True,
            help="Automatically select the model with the highest validation accuracy")
        return self

    def with_preds_train(self):
        # Where to put the learning prediction CSV files.
        self.add_argument(
            "--preds_train",
            type=Path,
            default=Path("preds/train"),
            help="Directory for outputting learning prediction CSV files")
        return self

    def with_preds_val(self):
        # Where to put the validation prediction CSV files.
        self.add_argument(
            "--preds_val",
            type=Path,
            default=Path("preds/val"),
            help="Directory for outputting validation prediction CSV files")
        return self

    def with_preds_test(self):
        # Where to put the evaluation prediction CSV files.
        self.add_argument(
            "--preds_test",
            type=Path,
            default=Path("preds/test"),
            help="Directory for outputting evaluation prediction CSV files")
        return self

    def with_inference_train(self):
        # Folder for outputting WSI predictions based on each threshold.
        self.add_argument(
            "--inference_train",
            type=Path,
            default=Path("inference/train"),
            help=
            "Folder for outputting WSI learning predictions based on each threshold")
        return self

    def with_inference_root(self):
        self.add_argument(
            "--inference_root",
            type=Path,
            default=Path("inference"),
            help=
            "Folder for outputting WSI validation/testing predictions")
        return self

    def with_vis_train(self):
        # For visualization.
        self.add_argument(
            "--vis_train",
            type=Path,
            default=Path("vis/train"),
            help="Folder for outputting the WSI learning prediction visualizations")
        return self

    def with_vis_val(self):
        self.add_argument(
            "--vis_val",
            type=Path,
            default=Path("vis/val"),
            help="Folder for outputting the WSI validation prediction visualizations")
        return self

    def with_vis_test(self):
        self.add_argument(
            "--vis_test",
            type=Path,
            default=Path("vis/test"),
            help="Folder for outputting the WSI evaluation prediction visualizations")
        return self

    def with_class_idx(self):
        self.add_argument(
            "--class_idx",
            type=Path,
            default=Path("class_idx.json"),
            help="Location of a JSON file that maps each class with a numeric index"
        )
        return self

    def with_class_colors(self):
        self.add_argument(
            "--class_colors",
            type=Path,
            default=Path("class_colors.json"),
            help="Location of a JSON file that maps each class with a color"
        )
        return self

    def with_seed(self):
        self.add_argument(
            "--seed",
            type=int,
            default=3
        )
        return self
