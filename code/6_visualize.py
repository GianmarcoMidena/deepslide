import config
import pandas as pd
from viewer import Viewer


viewer = Viewer(classes=config.classes,
                colors=config.colors,
                num_classes=config.num_classes,
                patch_size=config.args.patch_size)

val_wsis_info = pd.read_csv(config.args.wsis_val)
test_wsis_info = pd.read_csv(config.args.wsis_test)

# Visualizing patch predictions with overlaid dots.
viewer.visualize(wsis_info=val_wsis_info,
                 partition_name='validation',
                 preds_folder=config.args.preds_val,
                 vis_folder=config.args.vis_val)

viewer.visualize(wsis_info=test_wsis_info,
                 partition_name='testing',
                 preds_folder=config.args.preds_test,
                 vis_folder=config.args.vis_test)
