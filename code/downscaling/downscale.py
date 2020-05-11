from code.configurer import Configurer
from .slides_downscaler import SlidesDownscaler


def downscale(args):
    args = Configurer(args).build()

    SlidesDownscaler(original_slides_root=args.original_slides,
                     new_slides_root=args.slides,
                     original_slide_ext=args.original_slide_ext,
                     new_slide_ext=args.image_ext,
                     downscale_factor=args.downscale_factor,
                     num_workers=args.num_workers).downscale()


def add_parser(subparsers):
    subparsers.add_parser("downscale") \
        .with_original_slides() \
        .with_slides_root() \
        .with_original_slide_ext() \
        .with_image_ext() \
        .with_downscale_factor() \
        .with_num_workers() \
        .set_defaults(func=downscale)
