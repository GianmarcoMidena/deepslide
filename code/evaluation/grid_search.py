from code.configurer import Configurer
from code.evaluation.grid_searcher import ConfidenceThresholdFinder


def grid_search(args):
    args = Configurer(args)\
        .with_classes()\
        .build()

    # Searching over thresholds for filtering noise.
    ConfidenceThresholdFinder(pred_folder=args.preds_val,
                              inference_folder=args.inference_val,
                              classes=args.classes).search()


def add_parser(subparsers):
    subparsers.add_parser("grid_search") \
        .with_all_wsi() \
        .with_preds_val() \
        .with_inference_val() \
        .set_defaults(func=grid_search)
