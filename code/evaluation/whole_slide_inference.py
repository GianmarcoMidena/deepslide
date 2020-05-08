import pandas as pd
import logging

from code.configurer import Configurer
from code.evaluation.whole_slide_inferencer import WholeSlideInferencer


def final_test(args):
    args = Configurer(args) \
        .with_classes(binary_check=True) \
        .build()

    val_inference_root = args.inference_root.joinpath('val')
    test_inference_root = args.inference_root.joinpath('test')

    wsi_metadata_paths = sorted(list(args.wsi_splits_dir.glob("*.csv")))
    tot_splits = len(wsi_metadata_paths)
    n_test_splits = 1
    n_train_splits = tot_splits - n_test_splits
    train_wsi_metadata_paths = wsi_metadata_paths[:-n_test_splits]
    test_wsi_metadata_paths = wsi_metadata_paths[-n_test_splits:]

    all_test_metrics = pd.DataFrame()
    all_conf_matrices = None

    for i in range(n_train_splits):
        part_id = i + 1
        logging.info(f"fold {part_id}")
        part_name = f'part_{part_id}'

        val_patches_pred_folder_i = args.preds_val.joinpath(part_name)
        test_patches_pred_folder_i = args.preds_test.joinpath(part_name)
        val_inference_fold_i = val_inference_root.joinpath(part_name)
        test_inference_fold_i = test_inference_root.joinpath(part_name)

        val_wsi_metadata_paths_i = [train_wsi_metadata_paths[i]]

        validator = WholeSlideInferencer(wsi_metadata_paths=val_wsi_metadata_paths_i,
                                         patches_pred_folder=val_patches_pred_folder_i,
                                         inference_folder=val_inference_fold_i,
                                         classes=args.classes)

        tester = WholeSlideInferencer(wsi_metadata_paths=test_wsi_metadata_paths,
                                      patches_pred_folder=test_patches_pred_folder_i,
                                      inference_folder=test_inference_fold_i,
                                      classes=args.classes)

        # Searching over thresholds for filtering noise.
        validator.search_confidence_thesholds()

        # Running the code on the evaluation set.
        best_confidence_th = validator.find_best_confidence_threshold()

        tester.report_predictions(confidence_th=best_confidence_th)

        metrics_i, conf_matrix_i = tester.final_test_results()

        all_test_metrics = all_test_metrics.append(metrics_i, ignore_index=True, sort=False)
        if all_conf_matrices is None:
            all_conf_matrices = conf_matrix_i
            sample_size = conf_matrix_i.sum().sum()
        else:
            all_conf_matrices += conf_matrix_i

    logging.info("Overall final test metrics"
                 f"\n{all_test_metrics.mean(axis=0)} "
                 f"\n{(all_conf_matrices/all_conf_matrices.sum().sum())*sample_size}")


def add_parser(subparsers):
    subparsers.add_parser("whole_slide_inference") \
        .with_all_wsi() \
        .with_wsi_splits_dir() \
        .with_inference_root() \
        .with_preds_val() \
        .with_preds_test() \
        .with_positive_class() \
        .with_negative_class() \
        .set_defaults(func=final_test)
