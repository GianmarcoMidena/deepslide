import config
from utils_evaluation import grid_search

# Searching over thresholds for filtering noise.
grid_search(pred_folder=config.args.preds_val,
            inference_folder=config.args.inference_val,
            classes=config.classes,
            image_ext=config.args.image_ext,
            threshold_search=config.threshold_search)
