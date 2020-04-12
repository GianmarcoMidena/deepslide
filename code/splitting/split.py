from code.configurer import Configurer
from code.splitting.wsi_splitter import WSISplitter


def split(args):
    args = Configurer(args).build()

    WSISplitter(all_wsi=args.all_wsi,
                wsis_train=args.wsis_train,
                wsis_test=args.wsis_test,
                wsis_val=args.wsis_val,
                test_wsi_per_class=args.test_wsi_per_class,
                val_wsi_per_class=args.val_wsi_per_class).split()


def add_parser(subparsers):
    subparsers.add_parser("split") \
        .with_all_wsi() \
        .with_wsis_train() \
        .with_wsis_val() \
        .with_wsis_test() \
        .with_val_wsi_per_class() \
        .with_test_wsi_per_class() \
        .set_defaults(func=split)