import pandas as pd

from code.configurer import Configurer
from code.visualization.viewer import Viewer


def visualize(args):
    args = Configurer(args) \
        .with_classes() \
        .with_num_classes() \
        .build()

    viewer = Viewer(patch_size=args.patch_size, classes=args.classes, num_classes=args.num_classes,
                    colors_path=args.class_colors)

    val_slides_info = pd.read_csv(args.slides_val)
    test_slides_info = pd.read_csv(args.slides_test)

    # Visualizing patch predictions with overlaid dots.
    viewer.visualize(slides_info=val_slides_info,
                     partition_name='validation',
                     preds_folder=args.preds_val,
                     vis_folder=args.vis_val)

    viewer.visualize(slides_info=test_slides_info,
                     partition_name='evaluation',
                     preds_folder=args.preds_test,
                     vis_folder=args.vis_test)


def add_parser(subparsers):
    subparsers.add_parser("visualize") \
        .with_slides_root() \
        .with_slides_val() \
        .with_slides_test() \
        .with_patch_size() \
        .with_preds_val() \
        .with_preds_test() \
        .with_vis_val() \
        .with_vis_test() \
        .with_class_colors() \
        .set_defaults(func=visualize)
