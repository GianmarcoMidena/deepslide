import argparse
import logging

from .downscaling import downscale
from .splitting import split
from .tiling import tile
from .learning import train
from .evaluation import patch_test, slide_inference
from .visualization import visualize

from .argument_parser import ArgumentParser


def config():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="DeepSlide",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(parser_class=ArgumentParser)
    downscale.add_parser(subparsers)
    split.add_parser(subparsers)
    tile.add_parser(subparsers)
    train.add_parser(subparsers)
    patch_test.add_parser(subparsers)
    slide_inference.add_parser(subparsers)
    visualize.add_parser(subparsers)
    args = parser.parse_args()

    args.func(args)
