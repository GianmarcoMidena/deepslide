from code.configurer import Configurer
from .slides_downscaler import SlidesDownscaler


def downscale(args):
    args = Configurer(args).build()

    SlidesDownscaler(original_wsi_root=args.original_slides,
                     new_wsi_root=args.all_wsi,
                     original_wsi_ext=args.original_wsi_ext,
                     new_wsi_ext=args.image_ext,
                     downscale_factor=args.downscale_factor,
                     num_workers=args.num_workers).downscale()


def add_parser(subparsers):
    subparsers.add_parser("downscale") \
        .with_original_slides() \
        .with_all_wsi() \
        .with_original_wsi_ext() \
        .with_image_ext() \
        .with_downscale_factor() \
        .with_num_workers() \
        .set_defaults(func=downscale)
