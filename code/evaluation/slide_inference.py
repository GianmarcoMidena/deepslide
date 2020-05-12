import pandas as pd
import logging

from code.configurer import Configurer
from code.evaluation.slide_inferencer import SlideInferencer


def final_test(args):
    args = Configurer(args) \
        .with_classes(binary_check=True) \
        .build()

    val_inference_root = args.inference_root.joinpath('val')
    test_inference_root = args.inference_root.joinpath('test')

    slides_metadata_paths = sorted(list(args.slides_splits_dir.glob("*part_*.csv")))
    tot_splits = len(slides_metadata_paths)

    all_test_metrics = pd.DataFrame()
    all_conf_matrices = None

    outer_part_ids = list(range(tot_splits))

    for i in outer_part_ids:
        test_slides_metadata_paths = [slides_metadata_paths[i]]
        test_inference_folder = test_inference_root.joinpath(f'part_{i+1}')

        inner_part_ids = outer_part_ids[:i] + outer_part_ids[i+1:]

        best_confidence_th = None
        best_score = 0
        best_part_id = None

        for j in inner_part_ids:
            part_id = f"{i+1}_{j+1}"
            logging.info(f"fold {part_id}")
            part_name = f'part_{part_id}'

            val_patches_pred_folder = args.preds_val.joinpath(part_name)
            val_inference_folder = val_inference_root.joinpath(part_name)

            val_slides_metadata_paths = [slides_metadata_paths[j]]

            validator = SlideInferencer(slides_metadata_paths=val_slides_metadata_paths,
                                        patches_pred_folder=val_patches_pred_folder,
                                        inference_folder=val_inference_folder,
                                        classes=args.classes)

            # Searching over thresholds for filtering noise.
            validator.search_confidence_thesholds()

            # Running the code on the evaluation set.
            new_confidence_th, new_score = validator.find_best_confidence_threshold()

            if new_score > best_score:
                best_score = new_score
                best_confidence_th = new_confidence_th
                best_part_id = part_id

        test_patches_pred_folder = args.preds_test.joinpath(f'part_{best_part_id}')

        tester = SlideInferencer(slides_metadata_paths=test_slides_metadata_paths,
                                 patches_pred_folder=test_patches_pred_folder,
                                 inference_folder=test_inference_folder,
                                 classes=args.classes)

        tester.report_predictions(confidence_th=best_confidence_th)

        metrics, conf_matrix = tester.final_test_results()

        all_test_metrics = all_test_metrics.append(metrics, ignore_index=True, sort=False)
        if all_conf_matrices is None:
            all_conf_matrices = conf_matrix
        else:
            all_conf_matrices += conf_matrix

    logging.info("Overall final test metrics"
                 f"\n{all_test_metrics.mean(axis=0)} "
                 f"\n{all_conf_matrices/all_conf_matrices.sum().sum()}")


def add_parser(subparsers):
    subparsers.add_parser("slide_inference") \
        .with_slides_root() \
        .with_slides_splits_dir() \
        .with_inference_root() \
        .with_preds_val() \
        .with_preds_test() \
        .with_positive_class() \
        .with_negative_class() \
        .set_defaults(func=final_test)
