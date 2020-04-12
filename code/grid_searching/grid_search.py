from code.configurer import Configurer
from code.grid_searching.grid_searcher import GridSearcher


def grid_search(args):
    args = Configurer(args)\
        .with_classes()\
        .build()

    # Find the best threshold for filtering noise (discard patches with a confidence less than this threshold).
    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    # Searching over thresholds for filtering noise.
    GridSearcher(pred_folder=args.preds_val,
                 inference_folder=args.inference_val,
                 classes=args.classes,
                 image_ext=args.image_ext,
                 thresholds=thresholds).search()


def add_parser(subparsers):
    subparsers.add_parser("grid_search") \
        .with_all_wsi() \
        .with_preds_val() \
        .with_inference_val() \
        .with_image_ext() \
        .set_defaults(func=grid_search)