import pandas as pd

from code.configurer import Configurer
from code.visualization.viewer import Viewer


def visualize(args):
    args = Configurer(args) \
        .with_classes() \
        .with_num_classes() \
        .build()

    # This order is the same order as your sorted classes.
    colors = ("red", "white", "blue", "green", "purple",
              "orange", "black", "pink", "yellow")

    viewer = Viewer(classes=args.classes,
                    colors=colors,
                    num_classes=args.num_classes,
                    patch_size=args.patch_size)

    val_wsis_info = pd.read_csv(args.wsis_val)
    test_wsis_info = pd.read_csv(args.wsis_test)

    # Visualizing patch predictions with overlaid dots.
    viewer.visualize(wsis_info=val_wsis_info,
                     partition_name='validation',
                     preds_folder=args.preds_val,
                     vis_folder=args.vis_val)

    viewer.visualize(wsis_info=test_wsis_info,
                     partition_name='evaluation',
                     preds_folder=args.preds_test,
                     vis_folder=args.vis_test)


def add_parser(subparsers):
    subparsers.add_parser("visualize") \
        .with_all_wsi() \
        .with_wsis_val() \
        .with_wsis_test() \
        .with_patch_size() \
        .with_preds_val() \
        .with_preds_test() \
        .with_vis_val() \
        .with_vis_test() \
        .set_defaults(func=visualize)
