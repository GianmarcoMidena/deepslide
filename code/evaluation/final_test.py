import pandas as pd

from code.configurer import Configurer
from code.evaluation.final_tester import FinalTester
from code.evaluation.utils import report_predictions


def final_test(args):
    args = Configurer(args) \
        .with_classes() \
        .build()

    tester = FinalTester(classes=args.classes)

    val_wsis_info = pd.read_csv(args.wsis_val)
    test_wsis_info = pd.read_csv(args.wsis_test)

    # Running the code on the evaluation set.
    best_confidence_th = tester.find_best_confidence_threshold(
        wsis_info=val_wsis_info,
        inference_folder=args.inference_val)

    report_predictions(patches_pred_folder=args.preds_test,
                       output_folder=args.inference_test,
                       confidence_th=best_confidence_th,
                       classes=args.classes)

    tester.print_final_test_results(wsis_info=test_wsis_info,
                                    inference_folder=args.inference_test)


def add_parser(subparsers):
    subparsers.add_parser("final_test") \
        .with_all_wsi() \
        .with_wsis_val() \
        .with_wsis_test() \
        .with_inference_val() \
        .with_inference_test() \
        .with_preds_test() \
        .set_defaults(func=final_test)
