import config
import pandas as pd
from utils_evaluation import find_best_acc_and_thresh, output_all_predictions,\
                             print_final_test_results

val_wsis_info = pd.read_csv(config.args.wsis_val)
test_wsis_info = pd.read_csv(config.args.wsis_test)

# Running the code on the testing set.
best_thresholds = find_best_acc_and_thresh(
    wsis_info=val_wsis_info,
    inference_folder=config.args.inference_val,
    classes=config.classes)

output_all_predictions(patches_pred_folder=config.args.preds_test,
                       output_folder=config.args.inference_test,
                       conf_thresholds=best_thresholds,
                       classes=config.classes,
                       image_ext=config.args.image_ext)

print_final_test_results(wsis_info=test_wsis_info,
                         inference_folder=config.args.inference_test,
                         classes=config.classes)
