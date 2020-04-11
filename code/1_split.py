import config
from wsi_splitter import WSISplitter

WSISplitter(all_wsi=config.args.all_wsi,
            wsis_train=config.args.wsis_train,
            wsis_test=config.args.wsis_test,
            wsis_val=config.args.wsis_val,
            test_wsi_per_class=config.args.test_wsi_per_class,
            val_wsi_per_class=config.args.val_wsi_per_class).split()
