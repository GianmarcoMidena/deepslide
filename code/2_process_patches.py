import config
import pandas as pd

from utils_processing import balance_classes, gen_train_patches, \
    gen_val_patches, produce_patches

train_wsis_info = pd.read_csv(config.args.wsis_train)
val_wsis_info = pd.read_csv(config.args.wsis_val)
test_wsis_info = pd.read_csv(config.args.wsis_test)

gen_train_patches(wsis_info=train_wsis_info,
                  output_folder=config.train_patches,
                  n_patches_per_class=config.args.num_train_patches_per_class,
                  num_workers=config.args.num_workers,
                  patch_size=config.args.patch_size,
                  purple_threshold=config.args.purple_threshold,
                  purple_scale_size=config.args.purple_scale_size,
                  image_ext=config.args.image_ext,
                  type_histopath=config.args.type_histopath)

balance_classes(wsis_info=train_wsis_info, model_stage='training')

gen_val_patches(wsis_info=val_wsis_info,
                output_folder=config.val_patches,
                overlap_factor=config.args.gen_val_patches_overlap_factor,
                num_workers=config.args.num_workers,
                patch_size=config.args.patch_size,
                purple_threshold=config.args.purple_threshold,
                purple_scale_size=config.args.purple_scale_size,
                image_ext=config.args.image_ext,
                type_histopath=config.args.type_histopath)

produce_patches(wsis_info=val_wsis_info,
                model_stage='validation',
                output_folder=config.args.patches_eval_val,
                inverse_overlap_factor=config.args.slide_overlap,
                num_workers=config.args.num_workers,
                patch_size=config.args.patch_size,
                purple_threshold=config.args.purple_threshold,
                purple_scale_size=config.args.purple_scale_size,
                image_ext=config.args.image_ext,
                type_histopath=config.args.type_histopath)

produce_patches(wsis_info=test_wsis_info,
                model_stage='testing',
                output_folder=config.args.patches_eval_test,
                inverse_overlap_factor=config.args.slide_overlap,
                num_workers=config.args.num_workers,
                patch_size=config.args.patch_size,
                purple_threshold=config.args.purple_threshold,
                purple_scale_size=config.args.purple_scale_size,
                image_ext=config.args.image_ext,
                type_histopath=config.args.type_histopath)
