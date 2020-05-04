import pandas as pd

from code.configurer import Configurer
from code.evaluation.whole_slide_inferencer import WholeSlideInferencer


def final_test(args):
    args = Configurer(args) \
        .with_classes(binary_check=True) \
        .build()

    val_wsis_info = pd.read_csv(args.wsis_val)
    test_wsis_info = pd.read_csv(args.wsis_test)

    validator = WholeSlideInferencer(wsis_info=val_wsis_info,
                                     patches_pred_folder=args.preds_val,
                                     inference_folder=args.inference_val,
                                     classes=args.classes)
    tester = WholeSlideInferencer(wsis_info=test_wsis_info,
                                  patches_pred_folder=args.preds_test,
                                  inference_folder=args.inference_test,
                                  classes=args.classes)

    # Searching over thresholds for filtering noise.
    validator.search_confidence_thesholds()

    # Running the code on the evaluation set.
    best_confidence_th = validator.find_best_confidence_threshold()

    tester.report_predictions(confidence_th=best_confidence_th)

    tester.print_final_test_results()


def add_parser(subparsers):
    subparsers.add_parser("whole_slide_inference") \
        .with_all_wsi() \
        .with_wsis_val() \
        .with_wsis_test() \
        .with_inference_val() \
        .with_inference_test() \
        .with_preds_val() \
        .with_preds_test() \
        .with_positive_class() \
        .with_negative_class() \
        .set_defaults(func=final_test)
