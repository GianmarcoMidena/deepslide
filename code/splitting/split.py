from code.configurer import Configurer
from code.splitting.slides_splitter import SlidesSplitter


def split(args):
    args = Configurer(args).build()

    SlidesSplitter(slides_root=args.slides,
                   group=args.group,
                   slides_metadata=args.slides_metadata,
                   path_column=args.path_column,
                   n_splits=args.n_splits,
                   output_dir=args.slides_splits_dir,
                   seed=args.seed).split()


def add_parser(subparsers):
    subparsers.add_parser("split") \
        .with_slides_root() \
        .with_n_splits() \
        .with_group() \
        .with_slides_metadata() \
        .with_path_column() \
        .with_slides_splits_dir() \
        .with_seed() \
        .set_defaults(func=split)
