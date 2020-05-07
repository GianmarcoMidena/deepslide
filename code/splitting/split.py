from code.configurer import Configurer
from code.splitting.wsi_splitter import WSISplitter


def split(args):
    args = Configurer(args).build()

    WSISplitter(wsi_root=args.all_wsi,
                group=args.group,
                wsi_metadata=args.wsi_metadata,
                path_column=args.path_column,
                n_splits=args.n_splits,
                output_dir=args.wsi_splits_dir,
                seed=args.seed).split()


def add_parser(subparsers):
    subparsers.add_parser("split") \
        .with_all_wsi() \
        .with_n_splits() \
        .with_group() \
        .with_wsi_metadata() \
        .with_path_column() \
        .with_wsi_splits_dir() \
        .with_seed() \
        .set_defaults(func=split)
